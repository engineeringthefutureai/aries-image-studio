FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates curl git pciutils \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
        libze1 libze-intel-gpu1 intel-opencl-icd python3 python3-venv \
    && groupadd -f -r render \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv venv --python 3.12 /opt/venv

# Install torch & torchvision strictly from the Intel XPU index (pinned: the int4 enhancer
# checkpoint is tied to this torch + torchao pair)
RUN uv pip install "torch==2.14.1+xpu" torchvision --index-url https://download.pytorch.org/whl/xpu

# Supporting ML packages and git diffusers. torchao loads torchao-quantized (int4)
# checkpoints. No flash-linear-attention: measured +6% decode for ~67 s of Triton compile
# per fresh container and a C++ toolchain in the image (docs/HARDWARE.md).
RUN uv pip install "transformers==5.20.1" "torchao==0.18.0" accelerate safetensors pillow
RUN uv pip install "git+https://github.com/huggingface/diffusers.git"

# Install V3 thin agent and worker dependencies (gRPC, protobuf, httpx, prometheus-client)
RUN uv pip install "grpcio>=1.84.0" "protobuf>=7.35.1" "httpx>=0.27.0" "prometheus-client>=0.20.0" "safetensors>=0.4.0"

# Assert torch build is the XPU build
RUN python -c "import torch; assert '+xpu' in torch.__version__, f'Unexpected torch version: {torch.__version__}'"

WORKDIR /app
COPY pyproject.toml /app/
COPY aries/ /app/aries/
COPY aries_agent/ /app/aries_agent/
COPY aries_worker/ /app/aries_worker/

RUN uv pip install --no-deps -e /app

# Expose agent Prometheus metrics port (default 9464)
EXPOSE 9464

# Run the thin agent process
CMD ["python", "-X", "faulthandler", "-m", "aries_agent"]
