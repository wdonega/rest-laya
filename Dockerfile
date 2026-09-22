FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

COPY app ./app/

# Opcional: baixa os pesos já no build, para não pagar o download no
# primeiro start em produção. Comente se preferir baixar em runtime.
# RUN python -c "from laya import Router; Router(preload=True)"

EXPOSE 8055

# --workers 1: cada worker do uvicorn carrega sua própria cópia dos
# checkpoints em memória. Prefira escalar horizontalmente (mais réplicas
# do container) a subir múltiplos workers no mesmo processo.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8055", "--workers", "1"]
