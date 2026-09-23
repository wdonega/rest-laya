FROM python:3.11-slim

# gcc + libc6-dev: on CUDA, torch runs some ops (e.g. the matmul in
# ModernBERT's rotary embedding) through Triton, which compiles a small C
# helper against the CUDA driver on first use. Without a C compiler every
# /predict fails with "RuntimeError: Failed to find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Which torch build to install, as its suffix on PyTorch's wheel index
# (https://download.pytorch.org/whl/<variant>):
#   cpu   -- default; skips ~3GB of CUDA libraries.
#   cu132 -- CUDA 13.2, e.g. a Jetson Orin on JetPack 7.2 (its aarch64 wheel
#            has the sm_87 kernels the cu130 one lacks).
# A CUDA variant must match the host driver's CUDA version. Usually set
# through .env.cuda rather than by hand.
ARG TORCH_VARIANT=cpu
# Keep in sync with the torch pin in requirements.txt.
ARG TORCH_VERSION=2.14.0
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    "torch==${TORCH_VERSION}+${TORCH_VARIANT}"

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

COPY app ./app/

# Optional: download the weights at build time, so the first start in
# production doesn't pay for the download. Uncomment to enable. Pass the
# same checkpoint list you set in LAYA_MODELS, e.g.
#   docker compose -f docker-compose-dev.yml build --build-arg LAYA_MODELS=english,multilingual
# Caveat: docker-compose.yml mounts the hf-cache volume over the cache
# directory, and an existing volume hides whatever the image baked in.
# After changing the weights, drop it: docker volume rm rest-laya_hf-cache
# ARG LAYA_MODELS=english
# RUN python -c "import os; from laya import Router; Router().preload(names=os.environ['LAYA_MODELS'].split(','))"

# Tells the NVIDIA container runtime (`--runtime nvidia`) which GPUs and
# driver features to hand over; without them it exposes nothing and torch
# reports "Found no NVIDIA driver". Same defaults as NVIDIA's CUDA images.
# Ignored under Docker's default runtime, so harmless in the CPU build.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8055

# --workers 1: each uvicorn worker loads its own copy of the checkpoints
# into memory. Prefer scaling horizontally (more container replicas) over
# running multiple workers in the same process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8055", "--workers", "1"]
