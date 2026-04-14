FROM rust:bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY rust ./rust
WORKDIR /src/rust
RUN cargo build --release -p rusty-claude-cli

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        procps \
        python3 \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/rust/target/release/claw /usr/local/bin/claw

ENV HOME=/root
WORKDIR /workspace
CMD ["bash"]
