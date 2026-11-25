# One image, both services. They share the whole workspace, so two images would be
# two copies of the same build for the sake of two different entrypoints.
#
# The base image tag matches rust-toolchain.toml, which is the same pin CI uses: an
# image built on a different compiler is a different artifact. `rustup` would honour
# the file anyway, but then the build would download a toolchain on every cache miss.
FROM rust:1.98-bookworm AS build
WORKDIR /src

# The manifests first, so a source change does not re-resolve the dependency graph.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
# The web build is the devenv's job (web/DEPS.md); the image serves what is there.
RUN cargo build --release --locked -p agent-s1 -p agent-s2 -p agent-seeds

FROM debian:bookworm-slim AS runtime
# ca-certificates for outbound HTTPS to providers and the object store; nothing
# else, because the smaller the image the less there is to keep patched.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# An unprivileged user: neither service needs root, and S2's drivers refuse to run
# the VMM as root anyway.
RUN useradd --system --create-home --uid 10001 agent
COPY --from=build /src/target/release/agent-s1 /usr/local/bin/agent-s1
COPY --from=build /src/target/release/agent-s2 /usr/local/bin/agent-s2
COPY --from=build /src/target/release/agent-seeds /usr/local/bin/agent-seeds
COPY seeds /srv/seeds
USER agent
WORKDIR /srv
