{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [ ps.numpy ps.pillow ps.chess ps.cairosvg ]))
    pkgs.cargo
    pkgs.rustc
    pkgs.zstd
    pkgs.zstd.dev
  ];
}
