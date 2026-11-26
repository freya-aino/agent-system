# The dev environment docs/stack.md has always named and the repository has never
# held: the pinned toolchain, the microvm tools M7 needs, the Postgres client for
# store-schema.sql, and node for the management UI in web/.
{ pkgs, lib, inputs, ... }:

let
  # fromRustupToolchainFile reads rust-toolchain.toml, so the channel and the
  # components (rustfmt, clippy) are declared once, in the file CI also honours.
  rustPkgs = import inputs.nixpkgs {
    inherit (pkgs.stdenv.hostPlatform) system;
    overlays = [ (import inputs.rust-overlay) ];
  };
  toolchain = rustPkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
  # Impure by necessity: a devenv cannot take an argument, and the alternative is
  # two committed environments that drift.
  microvm = (builtins.getEnv "AGENT_DEVENV_MICROVM") == "1";
in
{
  packages =
    [
      toolchain
      # The management UI (web/README.md). Without this, `npm run build` cannot
      # run at all, which is why web/dist has never existed.
      pkgs.nodejs_22
      # psql, for applying store-schema.sql by hand against the compose Postgres.
      pkgs.postgresql_17
      # The compose stack (docker-compose.yml) and Garage's init script.
      pkgs.docker-compose
    ]
    # M7's tools, opt-in with `AGENT_DEVENV_MICROVM=1 devenv shell`.
    #
    # Not in the default set for two reasons: M7 is Linux-with-KVM only, and the
    # `firecracker` build is often uncached, so every developer who only wants to
    # run the suite would wait for a VMM to compile. Everything before M7 runs on
    # the process driver and needs none of this.
    ++ lib.optionals (microvm && pkgs.stdenv.hostPlatform.isLinux) [
      pkgs.firecracker
      # mkfs.ext4, for building the guest rootfs the driver needs.
      pkgs.e2fsprogs
    ];

  env.RUST_BACKTRACE = "1";

  enterShell = ''
    echo "agent-system devenv: $(rustc --version), node $(node --version)"
    echo "  cargo test --workspace            the suite; needs nothing external"
    echo "  cd web && npm install && npm run build   the management UI"
    echo "  docker compose up -d && sh garage-init.sh  the stack, for C4's --ignored tests"
    ${lib.optionalString (!microvm) ''
      echo "  AGENT_DEVENV_MICROVM=1 devenv shell        adds firecracker + mkfs.ext4 for M7"
    ''}
  '';

  # No git hooks: the suite is the gate (.github/workflows/ci.yml), and a hook that
  # runs on every commit would be a second, slower copy of it.
}
