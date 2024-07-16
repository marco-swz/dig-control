{
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = {nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachSystem flake-utils.lib.allSystems (system:
    let
        pkgs = import nixpkgs {
            inherit system;
        };
    in rec {
        devShell = (pkgs.buildFHSUserEnv {
            name = "julia-env";
            targetPkgs = pkgs: (with pkgs; [
                julia-bin
            ]);
            profile = ''
            '';
            runScript = "bash";
        }).env;
    });
}
