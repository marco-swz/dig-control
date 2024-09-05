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
        python33 = ((pkgs.python3.override {
            sourceVersion = {major = "3"; minor = "3" ; patch = "6"; suffix = ""; };
        }));
    in rec {
        devShell = (pkgs.buildFHSUserEnv {
            name = "julia-env";
            targetPkgs = pkgs: (with pkgs; [
                ipopt
                julia-bin
                (python33.withPackages(ps: with ps; [
                    virtualenv
                ]))
            ]);
            profile = ''
                export IPOPT_PATH=${pkgs.ipopt}

                if [ ! -d "venv" ]; then
                    virtualenv venv
                    source venv//bin/activate
                    pip install -r requirements.txt
                else
                    source venv//bin/activate
                fi
            '';
            runScript = "bash";
        }).env;
    });
}
