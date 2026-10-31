# syntax=docker/dockerfile:1

# Stage 1: Build the Web SPA
FROM node:22-alpine AS web-builder
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: The search model (docs/SEARCH.md): SigLIP 2 exported to ONNX, with the ONNX
# Runtime library Studio loads it with. Downloaded here, at build time, at a pinned revision;
# this layer stays cached until export.py changes.
FROM python:3.12-slim-bookworm AS search-model
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers==5.22.1 huggingface_hub==2.4.0 safetensors==0.8.0 \
        onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.31.1 numpy==2.5.4
COPY reference/search/export.py /export.py
RUN python /export.py /search-model && rm -rf /root/.cache/huggingface

# Stage 3: Build the Rust backend binary
FROM rust:1.92-bookworm AS rust-builder
WORKDIR /app
# Install protobuf compiler (or build.rs will use protoc-bin-vendored)
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy workspace dependencies and source trees
COPY Cargo.toml Cargo.lock ./
COPY proto/ proto/
COPY studio/ studio/
# The cloud enhancers' rewriting prompt (studio/src/cloud.rs include_str!).
COPY backend/aries_worker/stages/system_prompt.txt backend/aries_worker/stages/system_prompt.txt

RUN cargo build --release --bin aries-image-studio -p aries-image-studio

# Stage 4: Minimal runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 1000 aries && useradd -u 1000 -g aries -s /bin/bash -m aries

WORKDIR /app
RUN mkdir -p /app/data /app/web/dist && chown -R aries:aries /app

COPY --from=rust-builder /app/target/release/aries-image-studio /usr/local/bin/aries-image-studio
COPY --from=web-builder /app/web/dist /app/web/dist
COPY --from=search-model /search-model /app/search-model

USER aries:aries

ENV ARIES_HTTP_ADDR="0.0.0.0:8080" \
    ARIES_GRPC_ADDR="0.0.0.0:50051" \
    ARIES_DATA_DIR="/app/data" \
    ARIES_STATIC_DIR="/app/web/dist" \
    ARIES_SEARCH_MODEL_DIR="/app/search-model" \
    RUST_LOG="info,aries_image_studio=debug"

# Expose HTTP (API & SPA) and gRPC (AgentGateway)
EXPOSE 8080 50051

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/metrics || exit 1

ENTRYPOINT ["/usr/local/bin/aries-image-studio"]
