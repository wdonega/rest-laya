<p align="center">
  <img src="logo.png" alt="rest-laya" width="480">
</p>

A small Python service that serves the Laya model ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) over
REST, so any language or framework that can make an HTTP request can use it.

It is also jev-compatible. It serves `POST /v1/systemone`, which is
[jev](https://docs.typesafe.ai)'s own endpoint, with the same request and
response shape, `Authorization: Bearer` auth, and a jev-style error envelope
(`{"error": {"message", "type", "field_path"}}`). jev's docs don't publish the
exact error JSON, so the envelope follows the error fields in their SDK docs.
Test your client against it before relying on it. To turn it off, see
[jev compatibility](#jev-compatibility).

# Contents

- [Running with Docker](#running-with-docker)
  - [With CUDA](#with-cuda)
- [Trying it out](#trying-it-out)
  - [Health](#health)
  - [Predict](#predict)
  - [Errors](#errors)
- [Configuration](#configuration)
  - [Auth](#auth)
  - [jev compatibility](#jev-compatibility)
- [Building your own image](#building-your-own-image)
  - [Published images](#published-images)
- [Running without Docker](#running-without-docker)
- [Notes](#notes)

# Running with Docker

```bash
docker run -d --name rest-laya -p 8055:8055 \
  -e LAYA_MODELS=english \
  -v rest-laya_hf-cache:/root/.cache/huggingface \
  ghcr.io/wdonega/rest-laya:cpu
```

This pulls the prebuilt CPU image (x86 and ARM) and serves on port 8055. The
first start downloads the model weights into the `rest-laya_hf-cache` volume,
so later starts are faster; drop the `-v` and every new container downloads
them again. To change what it runs (checkpoints, auth, ...), pass more `-e`
variables; see [Configuration](#configuration).

Or, from a clone of this repo, with Compose:

```bash
docker compose up -d
```

It runs the same image with the settings from `docker-compose.yml`, and
shares the same weights volume.

## With CUDA

Needs a Linux host with an NVIDIA GPU, its driver, and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker Desktop on macOS can't pass a GPU through, so this doesn't apply there.

```bash
docker compose --env-file .env.cuda up -d
```

`.env.cuda` switches to the CUDA image (`:cu132`, CUDA 13.2, e.g. a Jetson
Orin on JetPack 7.2), runs the container under the `nvidia` runtime (which
hands it the GPU) and runs more predictions at once. nvidia-container-toolkit
registers that runtime with Docker; `docker info | grep -i runtimes` should
list `nvidia`.

The app needs no change, since laya uses `cuda` whenever
`torch.cuda.is_available()` is true. To confirm, run
`docker exec rest-laya python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`.
It should print `True`, and the list should include your GPU's arch (e.g.
`sm_87` on an Orin).

Only `cpu` and `cu132` are prebuilt. For a host on another CUDA version,
[build your own image](#building-your-own-image).

# Trying it out

## Health

```bash
curl http://localhost:8055/health
```

Response:
```json
{"status": "ok", "router_loaded": true, "models_loaded": ["laya/english"]}
```

It returns `503` (not `200`) while the router isn't usable, so the container
healthcheck fails.

## Predict

`/predict` and `/v1/systemone` (jev's endpoint path) run the same handler
with the same request and response shape. Use whichever path your client
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

- `"model": "english"` and `"model": "laya/english"` both work, since the
  `laya/` prefix is optional. Omit the field to use the first checkpoint in
  `LAYA_MODELS`, which is the configured default.
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

## Errors

All error bodies use jev's envelope, `{"error": {"message", "type", "field_path"}}`
(`field_path` is only present for field-specific errors). Status/`type` pairs:

| Status | `type`                        | When |
|--------|-------------------------------|------|
| 400    | `invalid_request_error`       | e.g. `model` valid but not preloaded |
| 401    | `authentication_error`        | `LAYA_API_TOKEN` is set and the request's bearer token is missing/wrong |
| 422    | `unprocessable_entity_error`  | malformed body, or the model rejects the request |

# Configuration

All settings are environment variables. With `docker run`, pass each one
with `-e`:

```bash
docker run -d --name rest-laya -p 8055:8055 \
  -e LAYA_MODELS=english,multilingual -e LAYA_API_TOKEN=secret \
  -v rest-laya_hf-cache:/root/.cache/huggingface \
  ghcr.io/wdonega/rest-laya:cpu
```

With Compose, the defaults are in `docker-compose.yml`. Override them in the
shell or in an env file rather than editing that file:

```bash
LAYA_MODELS=english,multilingual docker compose up -d
```

`.env.cpu` and `.env.cuda` are ready-made env files for each build. The
shell wins over the env file.

| Variable | Default | What it does |
|---|---|---|
| `LAYA_MODELS` | `english` | Checkpoints to preload, comma-separated: `english`, `multilingual`, `typed-decisions`. The first one is the default when a request omits `model`. Each adds ~1GB of memory. |
| `LAYA_API_TOKEN` | empty | Bearer token for `/predict` and `/v1/systemone`; empty disables auth. See [Auth](#auth). Don't commit a real one. |
| `LAYA_JEV_COMPAT` | `true` | jev compatibility. See [jev compatibility](#jev-compatibility). |
| `LAYA_MAX_CONCURRENT_PREDICTIONS` | `1` (`4` in `.env.cuda`) | Predictions run at once; extra requests queue. 1 is fastest on CPU, where one forward pass already uses every core; raise it on GPU. |
| `TORCH_VARIANT` | `cpu` (`cu132` in `.env.cuda`) | Which image to run (or build): the torch build, CPU or a CUDA version. |
| `DOCKER_RUNTIME` | `runc` (`nvidia` in `.env.cuda`) | Docker runtime; `nvidia` hands the GPU to the container. |

## Auth

`/predict` and `/v1/systemone` check `Authorization: Bearer <token>` against
`LAYA_API_TOKEN`. If that variable is empty or unset, auth is off and no
header is required.

With auth on, `/docs`, `/redoc` and `/openapi.json` are disabled, so
anonymous callers can't read the API schema. `/health` stays open either
way, because the container healthcheck needs it.

## jev compatibility

`LAYA_JEV_COMPAT` controls this, and it is on by default. When on:

- `/v1/systemone` responds (same handler as `/predict`). It's a `404` when off.
- `model` also accepts any jev model id (anything starting with `jev-`,
  e.g. `jev-latest`, `jev-1.13.0`). It resolves to the default checkpoint,
  the same as omitting `model`, because Laya has nothing like jev's
  versioned model ids.

Set `LAYA_JEV_COMPAT=false` to turn both off. An empty value keeps it on.

# Building your own image

`docker-compose-dev.yml` runs the same service with an image built from this
checkout. It tags the build `rest-laya:<variant>-dev` so it never shadows the
GHCR image, and it takes the same env files:

```bash
docker compose -f docker-compose-dev.yml up -d --build                       # CPU
docker compose -f docker-compose-dev.yml --env-file .env.cuda up -d --build  # CUDA
```

The torch build comes from `TORCH_VARIANT`: `cpu` (skips ~3GB of CUDA
libraries), or the suffix from [PyTorch's wheel index](https://download.pytorch.org/whl/)
matching the host's CUDA version, e.g. `cu126`, `cu130`, `cu132`. Set it in
`.env.cuda`, or for a one-off:
`TORCH_VARIANT=cu130 docker compose -f docker-compose-dev.yml --env-file .env.cuda up -d --build`.

## Published images

GitHub Actions (`.github/workflows/docker.yml`) builds `cpu` and `cu132` for
x86 and ARM on every push to `main` and publishes them to GHCR
(`ghcr.io/wdonega/rest-laya`) with these tags:

| | CPU | CUDA |
|---|---|---|
| Newest build | `latest`, `latest-cpu`, `cpu` | `latest-cuda`, `cu132` |
| Pinned build | `revN`, `revN-cpu` | `revN-cuda` |
| Release (`v1.2.0` git tag) | `1.2.0-cpu` | `1.2.0-cuda` |

`N` is the workflow run number, which goes up on every run; the Actions
tab shows which commit each run built. The compose files use `cpu` and
`cu132`, the value of `TORCH_VARIANT`.

# Running without Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8055
```

The same `LAYA_*` variables apply; export them before starting.

# Notes

- `usage.output_tokens` is always `0`. Laya is a classifier that scores the
  options in one forward pass and generates no tokens. `usage.input_tokens`
  counts every token the model read. Each question is encoded together with
  the full `state`, so `state` is counted once per question (3 questions over
  a 50-token `state` come to 150+ tokens).
- The model weights are cached in the `hf-cache` Docker volume, so restarts
  don't re-download them. To force a fresh download (e.g. after changing
  checkpoints), run `docker compose down && docker volume rm rest-laya_hf-cache`.
- Per-request latency after preloading: ~32ms (GPU) / 193-464ms (CPU).
- Recalibrate `temperature_by_options` offline, in a separate script that
  overwrites the Router config, and not on every request.
