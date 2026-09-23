<p align="center">
  <img src="logo.png" alt="rest-laya" width="480">
</p>

# rest-laya

A simple Python service that exposes the Laya model ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) over
REST, for consumption by any language or framework that can make an HTTP
request.

**Designed to be jev-compatible**: also serves `POST /v1/systemone` —
[jev](https://docs.typesafe.ai)'s own endpoint — with the same request/response
shape, `Authorization: Bearer` auth, and a jev-style error envelope
(`{"error": {"message", "type", "field_path"}}`). jev's docs don't publish the
exact error JSON, so that envelope is modeled on their documented SDK error
fields — test your client against it before relying on it. See
[jev compatibility](#jev-compatibility) below to toggle it off.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8055
```

## Running with Docker

```bash
docker compose up --build
```

By default this builds a **CPU-only** image: the Dockerfile installs the
`torch==…+cpu` wheel, which skips ~3GB of CUDA libraries.

## Running with CUDA

Needs a Linux host with an NVIDIA GPU, its driver, and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker Desktop on macOS can't pass a GPU through, so this doesn't apply there.

No file edits: `docker-compose.cuda.yml` layers the CUDA settings on top of
`docker-compose.yml` (CUDA torch build, the GPU reservation, more concurrent
predictions).

1. On the CUDA host, once: `cp .env.example .env` and uncomment the
   `COMPOSE_FILE` line.
2. `docker compose up -d --build`

Without a `.env`, the same thing in one line:
`docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build`.

The CUDA overlay installs `cu132` (CUDA 13.2, e.g. a Jetson Orin on JetPack
7.2). For a host on another CUDA version, set `TORCH_VARIANT` in `.env` to
the matching suffix from [PyTorch's wheel index](https://download.pytorch.org/whl/)
(e.g. `cu130`, `cu126`).

No app change is needed: laya picks `cuda` automatically when
`torch.cuda.is_available()` is true. To confirm, run
`docker compose exec rest-laya python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`.
It should print `True`, and the list should include your GPU's arch (e.g.
`sm_87` on an Orin).

## Trying it out

### Health

```bash
curl http://localhost:8055/health
```

Response: 
```json
{"status": "ok", "router_loaded": true, "models_loaded": ["laya/english"]}
```

### Predict

Available at both `/predict` and `/v1/systemone` (jev's own endpoint path) —
same handler, same request/response shape, pick whichever path your client
expects.

```bash
curl -X POST http://localhost:8055/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LAYA_API_TOKEN" \
  -d '{
    "state": {"body": "I was charged twice for invoice 4411. Please refund today."},
    "model": "english",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle `body`?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs and outages",
          "sales": "pricing"
        }
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the sender ask for money back?"
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgently does `body` need a response?",
        "criteria": [
          "can wait a week, no real impact",
          "handle within a few days, mildly annoyed customer",
          "handle today, customer is losing money or patience",
          "drop everything, customer is threatening to churn or escalate"
        ]
      }
    }
  }'
```

Checkpoint selection is the request body's `model` field:

- `"model": "english"` or `"model": "laya/english"` — the `laya/` prefix
  is optional, both forms work. Omit the field to use the first
  checkpoint in `LAYA_MODELS` (the configured default).
- An unknown value (not english/multilingual/typed-decisions, with or
  without the prefix) returns `422` with a JSON body.
- A known value that this instance didn't preload (not in `LAYA_MODELS`)
  returns `400` with a JSON body, e.g.:
  ```json
  {
    "error": {
      "message": "model 'multilingual' is not configured on this instance; available models: ['english']",
      "type": "invalid_request_error",
      "field_path": "model"
    }
  }
  ```

Response:

```json
{
  "model": "laya/english",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.9716,
        "technical": 0.0108,
        "sales": 0.0176
      },
      "confidence": 0.8654,
      "action": {
        "act_probability": 1.0
      }
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.8954,
      "confidence": 0.8954,
      "action": {
        "act_probability": 1.0
      }
    },
    "urgency": {
      "type": "score",
      "score": 1.9076,
      "legend": {
        "0": "can wait a week, no real impact",
        "1": "handle within a few days, mildly annoyed customer",
        "2": "handle today, customer is losing money or patience",
        "3": "drop everything, customer is threatening to churn or escalate"
      },
      "probabilities": {
        "0": 0.0067,
        "1": 0.1018,
        "2": 0.8688,
        "3": 0.0227
      },
      "confidence": 0.658,
      "action": {
        "act_probability": 1.0
      }
    }
  },
  "usage": {
    "input_tokens": 187,
    "output_tokens": 0
  }
}
```

### Errors

All error bodies use jev's envelope, `{"error": {"message", "type", "field_path"}}`
(`field_path` is only present for field-specific errors). Status/`type` pairs:

| Status | `type`                        | When |
|--------|-------------------------------|------|
| 400    | `invalid_request_error`       | e.g. `model` valid but not preloaded |
| 401    | `authentication_error`        | `LAYA_API_TOKEN` is set and the request's bearer token is missing/wrong |
| 422    | `unprocessable_entity_error`  | malformed body, or the model rejects the request |

### Auth

`/predict` and `/v1/systemone` check `Authorization: Bearer <token>` against
`LAYA_API_TOKEN` (see `docker-compose.yml`). If that env var is empty or
unset, auth is off and no header is required at all.

With auth on, `/docs`, `/redoc` and `/openapi.json` are disabled, so the API
schema isn't exposed to anonymous callers. `/health` stays open either way
(the container healthcheck needs it).

### jev compatibility

Controlled by `LAYA_JEV_COMPAT` (see `docker-compose.yml`) — **on by default**.
When on:

- `/v1/systemone` responds (same handler as `/predict`). It's a `404` when off.
- `model` also accepts any jev model id — anything starting with `jev-`,
  e.g. `jev-latest`, `jev-1.13.0` — which resolves to the default
  checkpoint (same as omitting `model`; Laya has no per-checkpoint
  equivalent of jev's versioned model ids).

Set `LAYA_JEV_COMPAT=false` to turn both off. An empty value keeps it on.

## Notes

- Which checkpoints get preloaded at startup is controlled by the
  `LAYA_MODELS` env var (comma-separated, e.g. `english` or
  `english,multilingual`) — see `docker-compose.yml`. Defaults to
  `english` only. Each checkpoint adds to memory usage, so only enable
  the ones you actually need.
- If a `/predict` or `/v1/systemone` request omits `model`, the sidecar uses
  the first checkpoint in `LAYA_MODELS` as the default — it does not
  hardcode `english`.
- `/health` returns `503` (not `200`) when the router isn't usable, so the
  container healthcheck fails.
- `usage.output_tokens` is always `0`: Laya is a classifier, it scores the
  options in one forward pass and generates no tokens. `usage.input_tokens`
  counts every token the model read, and each question is encoded together
  with the full `state`, so `state` is counted once **per question**
  (3 questions over a 50-token `state` ≈ 150+ tokens).
- Predictions run one at a time by default (`LAYA_MAX_CONCURRENT_PREDICTIONS=1`
  in `docker-compose.yml`); extra requests queue. On CPU that's faster
  overall than running them in parallel. Raise it on GPU.
- The model weights are cached in the `hf-cache` Docker volume, so restarts
  don't re-download them. To force a fresh download (e.g. after changing
  checkpoints), run `docker compose down && docker volume rm rest-laya_hf-cache`.
- Per-request latency after preloading: ~32ms (GPU) / 193-464ms (CPU).
- Recalibrating `temperature_by_options` should happen offline, as a
  separate script that overwrites the Router config — not on every request.
