# {
#   description = "Python Development Enviornment";
#
#   inputs = {
#     nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
#   };
#
#   outputs = {nixpkgs, ...}: let
#     inherit (nixpkgs) lib;
#     forAllSystems = lib.genAttrs lib.systems.flakeExposed;
#   in {
#     devShells = forAllSystems (
#       system: let
#         pkgs = nixpkgs.legacyPackages.${system};
#       in {
#         default =
#           (pkgs.buildFHSEnv {
#             name = "python-dev";
#
#             targetPkgs = p:
#               with p; [
#                 stdenv.cc.cc
#                 glibc
#                 zlib
#
#                 python3
#                 uv
#               ];
#
#             profile = ''
#               unset PYTHONPATH
#               uv sync --inexact
#               uv pip install -e . -e "jernerics @ ./../jernerics"
#               . .venv/bin/activate
#             '';
#
#             runScript = "zsh";
#           }).env;
#       }
#     );
#   };
# }
{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              (python3.withPackages (
                ps: with ps; [
                  ruff
                  pytest
                  mypy
                ]
              ))
              pyright
              uv
            ];

            LD_LIBRARY_PATH = [
              "$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib"
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync --inexact
              uv pip install -e . -e "jernerics @ ./../jernerics"
              . .venv/bin/activate
            '';
          };

          hpc = pkgs.mkShell {
            packages = with pkgs; [
              python3
              uv
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync --inexact
              uv pip install -e . -e "jernerics @ ./../jernerics"
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
