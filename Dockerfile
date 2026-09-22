FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

COPY app ./app/

# Optional: download the weights at build time, so the first start in
# production doesn't pay for the download. Uncomment to enable. Pass the
# same checkpoint list you set in LAYA_MODELS, e.g.
#   docker compose build --build-arg LAYA_MODELS=english,multilingual
# Caveat: docker-compose.yml mounts the hf-cache volume over the cache
# directory, and an existing volume hides whatever the image baked in.
# After changing the weights, drop it: docker volume rm rest-laya_hf-cache
# ARG LAYA_MODELS=english
# RUN python -c "import os; from laya import Router; Router().preload(names=os.environ['LAYA_MODELS'].split(','))"

EXPOSE 8055

# --workers 1: each uvicorn worker loads its own copy of the checkpoints
# into memory. Prefer scaling horizontally (more container replicas) over
# running multiple workers in the same process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8055", "--workers", "1"]
