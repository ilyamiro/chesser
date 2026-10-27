{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [ ps.numpy ps.pillow ps.chess ]))
    pkgs.cargo
    pkgs.rustc
  ];
}
