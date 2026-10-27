/* Nearest-position search over a game index shard (see indexer/src/main.rs).
 *
 * Positions are stored in blocks of 32, transposed so that packed byte j of all
 * 32 positions is one 32-byte AVX2 register. A byte holds two squares' option
 * codes (low nibble = even square). Each region's per-square costs are
 * quantized to 0..255 and laid out as 16-entry tables, so a square's cost for
 * 32 positions at once is a single vpshufb. Totals are accumulated in u16.
 *
 * The scores are approximate (quantized); callers keep the top k per region
 * and rescore those exactly. Positions with fewer than min_pieces pieces are
 * never returned (counted once per block, shared by all regions).
 *
 * A shard's blocks are either a plain array or stored as independent zstd units
 * of UNIT blocks (see compress_units); each thread then decompresses one unit at
 * a time into its own buffer, so the disk is read 3-4x less. The packed bytes of
 * every shortlisted position are copied out with it, so nothing is read again.
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

#define CHUNK 64 /* blocks scored together (64 KB of positions) */
#define UNIT 256 /* blocks per compressed unit and per scheduling unit */

typedef struct {
    uint16_t *s;
    int64_t *idx;
    uint8_t *pos; /* k * 32 packed bytes of the kept positions */
    int k;
    uint16_t worst;
} topk;

static void topk_insert(topk *t, uint16_t v, int64_t i, const uint8_t *blk, int p) {
    int w = 0;
    for (int j = 1; j < t->k; j++)
        if (t->s[j] > t->s[w]) w = j;
    t->s[w] = v;
    t->idx[w] = i;
    for (int j = 0; j < 32; j++) t->pos[w * 32 + j] = blk[j * 32 + p];
    uint16_t m = t->s[0];
    for (int j = 1; j < t->k; j++)
        if (t->s[j] > m) m = t->s[j];
    t->worst = m;
}

/* blocks: nblocks * 1024 bytes of plain blocks, or NULL when zdata/zoff hold zstd units
 * (zoff: byte offsets of the units, one more than there are units).
 * n: real positions (the last block may be padded).
 * qtables: R * 64 * 16 quantized costs [region][square][code].
 * out_s / out_i / out_p: R * k approximate totals, slot indices (unsorted, -1 = none)
 * and the 32 packed bytes of each kept position. */
int top_matches(const uint8_t *blocks, const uint8_t *zdata, const uint64_t *zoff, int64_t nblocks,
                int64_t n, const uint8_t *qtables, int R, int k, int min_pieces,
                uint16_t *out_s, int64_t *out_i, uint8_t *out_p) {
    int nt = omp_get_max_threads();
    uint16_t *ts = malloc(sizeof(uint16_t) * nt * R * k);
    int64_t *ti = malloc(sizeof(int64_t) * nt * R * k);
    uint8_t *tp = malloc((size_t)nt * R * k * 32);
    topk *tk = malloc(sizeof(topk) * nt * R);
    if (!ts || !ti || !tp || !tk) return -1;
    for (int64_t j = 0; j < (int64_t)nt * R * k; j++) { ts[j] = UINT16_MAX; ti[j] = -1; }
    for (int64_t j = 0; j < (int64_t)nt * R; j++)
        tk[j] = (topk){ts + j * k, ti + j * k, tp + j * k * 32, k, UINT16_MAX};
    int64_t nunits = (nblocks + UNIT - 1) / UNIT;
    int failed = 0;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        topk *mine = tk + (int64_t)t * R;
        const __m256i nib = _mm256_set1_epi8(0x0F);
        const __m256i ones = _mm256_set1_epi8(1);
        __m256i lut[64];
        uint8_t *buf = zdata ? aligned_alloc(32, (size_t)UNIT * 1024) : NULL;

        const __m256i zero = _mm256_setzero_si256();
        const __m256i max_empty = _mm256_set1_epi8((char)(64 - min_pieces));
        uint32_t eligible[CHUNK]; /* bit p set: position p of the block has enough pieces */

        #pragma omp for schedule(dynamic, 1)
        for (int64_t u = 0; u < nunits; u++) {
            int64_t u0 = u * UNIT, u1 = u0 + UNIT < nblocks ? u0 + UNIT : nblocks;
            const uint8_t *unit;
            if (zdata) {
#ifdef HAVE_ZSTD
                size_t got = ZSTD_decompress(buf, (size_t)(u1 - u0) * 1024, zdata + zoff[u], zoff[u + 1] - zoff[u]);
                if (ZSTD_isError(got) || got != (size_t)(u1 - u0) * 1024) {
                    #pragma omp atomic write
                    failed = 1;
                    continue;
                }
                unit = buf;
#else
                #pragma omp atomic write
                failed = 1;
                continue;
#endif
            } else {
                unit = blocks + u0 * 1024;
            }
            for (int64_t c0 = u0; c0 < u1; c0 += CHUNK) {
                int64_t c1 = c0 + CHUNK < u1 ? c0 + CHUNK : u1;
                for (int64_t b = c0; b < c1; b++) {
                    const uint8_t *blk = unit + (b - u0) * 1024;
                    __m256i empty = zero;
                    for (int j = 0; j < 32; j++) {
                        __m256i v = _mm256_load_si256((const __m256i *)(blk + j * 32));
                        empty = _mm256_sub_epi8(empty, _mm256_cmpeq_epi8(_mm256_and_si256(v, nib), zero));
                        empty = _mm256_sub_epi8(empty, _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(v, 4), nib), zero));
                    }
                    eligible[b - c0] = ~(uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(empty, max_empty));
                }
                for (int r = 0; r < R; r++) {
                    for (int sq = 0; sq < 64; sq++)
                        lut[sq] = _mm256_broadcastsi128_si256(
                            _mm_loadu_si128((const __m128i *)(qtables + ((int64_t)r * 64 + sq) * 16)));
                    topk *tr = mine + r;
                    for (int64_t b = c0; b < c1; b++) {
                        const uint8_t *blk = unit + (b - u0) * 1024;
                        /* Interleaving the two squares' u8 costs and multiplying by 1s with
                         * maddubs adds them straight into u16 lanes. unpacklo/hi work per
                         * 128-bit half, so acc0 holds positions 0-7,16-23 and acc1 8-15,24-31. */
                        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
                        for (int j = 0; j < 32; j++) {
                            __m256i v = _mm256_load_si256((const __m256i *)(blk + j * 32));
                            __m256i lo = _mm256_and_si256(v, nib);
                            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nib);
                            __m256i a = _mm256_shuffle_epi8(lut[2 * j], lo);
                            __m256i c = _mm256_shuffle_epi8(lut[2 * j + 1], hi);
                            acc0 = _mm256_add_epi16(acc0, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, c), ones));
                            acc1 = _mm256_add_epi16(acc1, _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, c), ones));
                        }
                        __m256i w = _mm256_set1_epi16((short)(tr->worst > 32767 ? 32767 : tr->worst));
                        uint32_t m0 = _mm256_movemask_epi8(_mm256_cmpgt_epi16(w, acc0));
                        uint32_t m1 = _mm256_movemask_epi8(_mm256_cmpgt_epi16(w, acc1));
                        uint32_t ok = eligible[b - c0];
                        if (!(m0 | m1) || !ok) continue;
                        uint16_t v0[16], v1[16];
                        _mm256_storeu_si256((__m256i *)v0, acc0);
                        _mm256_storeu_si256((__m256i *)v1, acc1);
                        int lanes = b == nblocks - 1 ? (int)(n - b * 32) : 32;
                        for (int e = 0; e < 16; e++) {
                            int p0 = e < 8 ? e : e + 8;       /* position held in acc0 lane e */
                            int p1 = e < 8 ? e + 8 : e + 16;  /* position held in acc1 lane e */
                            if (p0 < lanes && (ok >> p0 & 1) && v0[e] < tr->worst) topk_insert(tr, v0[e], b * 32 + p0, blk, p0);
                            if (p1 < lanes && (ok >> p1 & 1) && v1[e] < tr->worst) topk_insert(tr, v1[e], b * 32 + p1, blk, p1);
                        }
                    }
                }
            }
        }
        free(buf);
    }

    for (int r = 0; r < R; r++) {
        topk out = {out_s + (int64_t)r * k, out_i + (int64_t)r * k, out_p + (int64_t)r * k * 32, k, UINT16_MAX};
        for (int j = 0; j < k; j++) { out.s[j] = UINT16_MAX; out.idx[j] = -1; }
        for (int t = 0; t < nt; t++) {
            topk *src = tk + (int64_t)t * R + r;
            for (int j = 0; j < k; j++)
                if (src->idx[j] >= 0 && src->s[j] < out.worst) {
                    /* the kept bytes are already packed: write them as one 32-byte "column" */
                    uint8_t col[1024];
                    for (int q = 0; q < 32; q++) col[q * 32] = src->pos[j * 32 + q];
                    topk_insert(&out, src->s[j], src->idx[j], col, 0);
                }
        }
    }
    free(ts);
    free(ti);
    free(tp);
    free(tk);
    return failed ? -2 : 0;
}

/* Decompress unit u into out (UNIT * 1024 bytes); returns its block count, or -1. */
int decode_unit(const uint8_t *zdata, const uint64_t *zoff, int64_t nblocks, int64_t u, uint8_t *out) {
#ifdef HAVE_ZSTD
    int64_t u0 = u * UNIT, u1 = u0 + UNIT < nblocks ? u0 + UNIT : nblocks;
    size_t got = ZSTD_decompress(out, (size_t)(u1 - u0) * 1024, zdata + zoff[u], zoff[u + 1] - zoff[u]);
    return ZSTD_isError(got) || got != (size_t)(u1 - u0) * 1024 ? -1 : (int)(u1 - u0);
#else
    return -1;
#endif
}

/* Compress units u0 .. u0+nu-1 of plain blocks into dst, cap bytes apart (cap >= unit_bound()),
 * storing each size. Every unit is decompressed again and compared before it counts.
 * Returns 0, or -1 on any failure. */
int64_t unit_bound(void) {
#ifdef HAVE_ZSTD
    return (int64_t)ZSTD_compressBound((size_t)UNIT * 1024);
#else
    return -1;
#endif
}

int compress_units(const uint8_t *blocks, int64_t nblocks, int level, int64_t u0, int64_t nu,
                   uint8_t *dst, int64_t cap, int64_t *sizes) {
#ifdef HAVE_ZSTD
    int bad = 0;
    #pragma omp parallel
    {
        uint8_t *check = malloc((size_t)UNIT * 1024);
        #pragma omp for schedule(dynamic, 1)
        for (int64_t i = 0; i < nu; i++) {
            int64_t b0 = (u0 + i) * UNIT, b1 = b0 + UNIT < nblocks ? b0 + UNIT : nblocks;
            size_t raw = (size_t)(b1 - b0) * 1024;
            size_t got = ZSTD_compress(dst + i * cap, (size_t)cap, blocks + b0 * 1024, raw, level);
            int ok = !ZSTD_isError(got) && check &&
                     ZSTD_decompress(check, raw, dst + i * cap, got) == raw &&
                     memcmp(check, blocks + b0 * 1024, raw) == 0;
            if (!ok) {
                #pragma omp atomic write
                bad = 1;
            } else {
                sizes[i] = (int64_t)got;
            }
        }
        free(check);
    }
    return bad ? -1 : 0;
#else
    return -1;
#endif
}
