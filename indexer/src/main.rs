//! Builds a chesser game-position index shard from a Lichess PGN (.pgn or .pgn.zst).
//!
//! Usage: chesser-indexer INPUT OUT_DIR [--min-ply N] [--stride N] [--threads N] [--max-games N]
//!
//! Shard layout (all little-endian):
//!   blocks.bin   positions in blocks of 32, transposed: byte [b*1024 + j*32 + lane] is
//!                packed byte j of position 32b+lane. A packed byte holds two squares'
//!                option codes (0 empty, 1-6 white PNBRQK, 7-12 black), low nibble = even square.
//!   game.bin     u32 game id per position slot (u32::MAX for padding in the last block)
//!   ply.bin      u16 half-moves played per position slot
//!   games.jsonl  one JSON object per game; offsets.bin holds u64 byte offsets into it
//!   meta.json    counts and parameters

use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    hash::{BuildHasherDefault, Hasher},
    io::{self, BufWriter, Write},
    ops::ControlFlow,
    path::PathBuf,
    thread,
    time::Instant,
};

use pgn_reader::{
    RawTag, Reader, SanPlus, Visitor,
    shakmaty::{Board, Chess, Color, Position, san::San},
};

type Packed = [u8; 32];

#[derive(Default)]
struct Tags {
    site: String,
    white: String,
    black: String,
    white_elo: String,
    black_elo: String,
    date: String,
    result: String,
    event: String,
    has_fen: bool,
}

struct Game {
    tags: Tags,
    sans: Vec<San>,
}

struct Collector;

impl Visitor for Collector {
    type Tags = Tags;
    type Movetext = Game;
    type Output = Option<Game>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(Tags::default())
    }

    fn tag(&mut self, tags: &mut Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        let slot = match name {
            b"Site" => &mut tags.site,
            b"White" => &mut tags.white,
            b"Black" => &mut tags.black,
            b"WhiteElo" => &mut tags.white_elo,
            b"BlackElo" => &mut tags.black_elo,
            b"UTCDate" => &mut tags.date,
            b"Result" => &mut tags.result,
            b"Event" => &mut tags.event,
            b"FEN" => {
                tags.has_fen = true;
                return ControlFlow::Continue(());
            }
            _ => return ControlFlow::Continue(()),
        };
        *slot = value.decode_utf8_lossy().into_owned();
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Tags) -> ControlFlow<Self::Output, Game> {
        // Games from a custom starting position aren't reachable from the normal setup.
        if tags.has_fen {
            return ControlFlow::Break(None);
        }
        ControlFlow::Continue(Game { tags, sans: Vec::with_capacity(128) })
    }

    fn san(&mut self, game: &mut Game, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        game.sans.push(san_plus.san);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, game: Game) -> Self::Output {
        Some(game)
    }
}

fn encode(board: &Board) -> Packed {
    let mut out = [0u8; 32];
    for (sq, piece) in board.iter() {
        let s = sq.to_usize();
        let code = piece.role as u8 + if piece.color == Color::White { 0 } else { 6 };
        out[s / 2] |= code << (4 * (s % 2));
    }
    out
}

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn game_json(t: &Tags) -> String {
    let mut s = String::from("{");
    let fields = [
        ("Site", &t.site),
        ("White", &t.white),
        ("Black", &t.black),
        ("WhiteElo", &t.white_elo),
        ("BlackElo", &t.black_elo),
        ("UTCDate", &t.date),
        ("Result", &t.result),
        ("Event", &t.event),
    ];
    for (i, (k, v)) in fields.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        json_str(&mut s, k);
        s.push(':');
        json_str(&mut s, v);
    }
    s.push_str("}\n");
    s
}

struct Params {
    min_ply: usize,
    stride: usize,
}

/// Replay a game and keep every `stride`-th position from `min_ply` on.
/// The phase is shifted per game so both sides to move are represented.
fn positions(game: &Game, seq: u64, p: &Params) -> Vec<(Packed, u16)> {
    let mut pos = Chess::default();
    let start = p.min_ply + (seq as usize % p.stride);
    let mut out = Vec::new();
    for (i, san) in game.sans.iter().enumerate() {
        let Ok(m) = san.to_move(&pos) else { break };
        pos.play_unchecked(m);
        let ply = i + 1;
        if ply >= start && (ply - start) % p.stride == 0 {
            out.push((encode(pos.board()), ply as u16));
        }
    }
    out
}

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

fn hash(p: &Packed) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for chunk in p.chunks_exact(8) {
        h = (h ^ u64::from_le_bytes(chunk.try_into().unwrap())).wrapping_mul(0x9E3779B97F4A7C15);
        h ^= h >> 29;
    }
    h
}

/// Opening positions repeat across thousands of games; dedupe those (bounded memory).
const DEDUPE_BELOW_PLY: u16 = 30;
const DEDUPE_CAPACITY: usize = 150_000_000;

struct ShardWriter {
    blocks: BufWriter<File>,
    game: BufWriter<File>,
    ply: BufWriter<File>,
    games: BufWriter<File>,
    offsets: BufWriter<File>,
    pending: Vec<(Packed, u32, u16)>,
    seen: HashSet<u64, BuildHasherDefault<IdentityHasher>>,
    games_written: u32,
    games_bytes: u64,
    positions: u64,
    duplicates: u64,
}

impl ShardWriter {
    fn new(dir: &PathBuf) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let open = |name: &str| -> io::Result<BufWriter<File>> {
            Ok(BufWriter::with_capacity(1 << 20, File::create(dir.join(name))?))
        };
        Ok(ShardWriter {
            blocks: open("blocks.bin")?,
            game: open("game.bin")?,
            ply: open("ply.bin")?,
            games: open("games.jsonl")?,
            offsets: open("offsets.bin")?,
            pending: Vec::with_capacity(32),
            seen: HashSet::default(),
            games_written: 0,
            games_bytes: 0,
            positions: 0,
            duplicates: 0,
        })
    }

    fn add_game(&mut self, json: &str, positions: Vec<(Packed, u16)>) -> io::Result<()> {
        let id = self.games_written;
        let mut kept = 0;
        for (packed, ply) in positions {
            if ply < DEDUPE_BELOW_PLY {
                let h = hash(&packed);
                if self.seen.contains(&h) {
                    self.duplicates += 1;
                    continue;
                }
                if self.seen.len() < DEDUPE_CAPACITY {
                    self.seen.insert(h);
                }
            }
            self.pending.push((packed, id, ply));
            kept += 1;
            if self.pending.len() == 32 {
                self.flush_block()?;
            }
        }
        if kept > 0 {
            self.offsets.write_all(&self.games_bytes.to_le_bytes())?;
            self.games.write_all(json.as_bytes())?;
            self.games_bytes += json.len() as u64;
            self.games_written += 1;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let mut block = [0u8; 1024];
        for (lane, (packed, game, ply)) in self.pending.iter().enumerate() {
            for j in 0..32 {
                block[j * 32 + lane] = packed[j];
            }
            self.game.write_all(&game.to_le_bytes())?;
            self.ply.write_all(&ply.to_le_bytes())?;
        }
        // Padding slots: empty boards that are flagged invalid by game id.
        for _ in self.pending.len()..32 {
            self.game.write_all(&u32::MAX.to_le_bytes())?;
            self.ply.write_all(&0u16.to_le_bytes())?;
        }
        self.blocks.write_all(&block)?;
        self.positions += self.pending.len() as u64;
        self.pending.clear();
        Ok(())
    }

    fn finish(mut self, dir: &PathBuf, source: &str, p: &Params, games_read: u64) -> io::Result<u64> {
        if !self.pending.is_empty() {
            self.flush_block()?;
        }
        for w in [&mut self.blocks, &mut self.game, &mut self.ply, &mut self.games, &mut self.offsets] {
            w.flush()?;
        }
        let blocks = self.positions.div_ceil(32);
        let meta = format!(
            "{{\"layout\":\"blocks32\",\"blocks\":{},\"positions\":{},\"games\":{},\"games_read\":{},\
             \"duplicates_skipped\":{},\"min_ply\":{},\"stride\":{},\"source\":{:?}}}\n",
            blocks, self.positions, self.games_written, games_read, self.duplicates, p.min_ply, p.stride, source
        );
        fs::write(dir.join("meta.json"), meta)?;
        Ok(self.positions)
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: chesser-indexer INPUT OUT_DIR [--min-ply N] [--stride N] [--threads N] [--max-games N]");
        std::process::exit(2);
    }
    let input = args[1].clone();
    let out_dir = PathBuf::from(&args[2]);
    let mut params = Params { min_ply: 8, stride: 4 };
    let mut threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1);
    let mut max_games = u64::MAX;
    let mut i = 3;
    while i + 1 < args.len() {
        let v: u64 = args[i + 1].parse().expect("numeric option value");
        match args[i].as_str() {
            "--min-ply" => params.min_ply = v as usize,
            "--stride" => params.stride = (v as usize).max(1),
            "--threads" => threads = v as usize,
            "--max-games" => max_games = v,
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }

    let file = File::open(&input)?;
    let reader: Box<dyn io::Read + Send> = if input.ends_with(".zst") {
        Box::new(zstd::Decoder::new(file)?)
    } else {
        Box::new(file)
    };

    let started = Instant::now();
    let (work_tx, work_rx) = crossbeam_channel::bounded::<(u64, Vec<Game>)>(64);
    let (done_tx, done_rx) = crossbeam_channel::bounded::<Vec<(String, Vec<(Packed, u16)>)>>(64);
    let params = &params;

    let mut games_read = 0u64;
    let mut writer = ShardWriter::new(&out_dir)?;
    thread::scope(|scope| -> io::Result<()> {
        for _ in 0..threads {
            let work_rx = work_rx.clone();
            let done_tx = done_tx.clone();
            scope.spawn(move || {
                for (first_seq, batch) in work_rx {
                    let out = batch
                        .iter()
                        .enumerate()
                        .map(|(k, g)| (game_json(&g.tags), positions(g, first_seq + k as u64, params)))
                        .collect();
                    done_tx.send(out).unwrap();
                }
            });
        }
        drop(done_tx);

        let writer = &mut writer;
        let write_handle = scope.spawn(move || -> io::Result<()> {
            for batch in done_rx {
                for (json, positions) in batch {
                    writer.add_game(&json, positions)?;
                }
            }
            Ok(())
        });

        let mut pgn = Reader::new(reader);
        let mut batch = Vec::with_capacity(256);
        let mut first_seq = 0u64;
        while games_read < max_games {
            match pgn.read_game(&mut Collector)? {
                None => break,
                Some(None) => {}
                Some(Some(game)) => {
                    batch.push(game);
                    if batch.len() == 256 {
                        work_tx.send((first_seq, std::mem::take(&mut batch))).unwrap();
                        first_seq += 256;
                    }
                }
            }
            games_read += 1;
            if games_read % 1_000_000 == 0 {
                let secs = started.elapsed().as_secs_f64();
                eprintln!("  {}M games read, {:.0} games/s", games_read / 1_000_000, games_read as f64 / secs);
            }
        }
        if !batch.is_empty() {
            work_tx.send((first_seq, batch)).unwrap();
        }
        drop(work_tx);
        write_handle.join().unwrap()
    })?;

    let games = writer.games_written;
    let dups = writer.duplicates;
    let positions = writer.finish(&out_dir, &input, params, games_read)?;
    eprintln!(
        "{}: {} games read, {} games kept, {} positions ({} early duplicates skipped) in {:.0}s",
        input, games_read, games, positions, dups, started.elapsed().as_secs_f64()
    );
    Ok(())
}
