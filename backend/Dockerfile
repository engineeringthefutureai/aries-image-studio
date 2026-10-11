FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates curl git \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
        libze1 libze-intel-gpu1 intel-opencl-icd python3 python3-venv \
    && groupadd -f -r render \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv venv --python 3.12 /opt/venv

# Install torch & torchvision strictly from the Intel XPU index
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# Install supporting ML packages and git diffusers
RUN uv pip install "transformers>=5.17" accelerate safetensors pillow psutil
RUN uv pip install "git+https://github.com/huggingface/diffusers.git"

# Install API and telemetry packages
RUN uv pip install fastapi "uvicorn[standard]" prometheus-client

# Assert torch build is the XPU build
RUN python -c "import torch; assert '+xpu' in torch.__version__, f'Unexpected torch version: {torch.__version__}'"

WORKDIR /app
COPY aries_manager/ /app/aries_manager/
COPY aries_worker/ /app/aries_worker/

EXPOSE 8000

# faulthandler: a GPU fault can kill the process; leave a Python stack behind.
CMD ["python", "-X", "faulthandler", "-m", "aries_manager", "--host", "0.0.0.0", "--port", "8000"]
