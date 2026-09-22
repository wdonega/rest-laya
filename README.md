# rest-laya

A simple Python service that exposes the Laya model (convaiinnovations/laya) over
REST, for consumption by any language or framework that can make an HTTP
request.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8055
```

## Running with Docker

```bash
docker compose up --build
```

## Trying it out

```bash
curl http://localhost:8055/health

curl -X POST http://localhost:8055/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"body": "I was charged twice for invoice 4411. Please refund today."},
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
    },
    "model": "laya"
  }'
```

Response:

```json
{
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
  }
}
```

## Notes

- Which checkpoints get preloaded at startup is controlled by the
  `LAYA_MODELS` env var (comma-separated, e.g. `english` or
  `english,multilingual`) — see `docker-compose.yml`. Defaults to
  `english` only. Each checkpoint adds to memory usage, so only enable
  the ones you actually need.
- If a `/predict` request omits `model`, the sidecar uses the first
  checkpoint in `LAYA_MODELS` as the default — it does not hardcode
  `laya`/`english`.
- Per-request latency after preloading: ~32ms (GPU) / 193-464ms (CPU).
- Recalibrating `temperature_by_options` should happen offline, as a
  separate script that overwrites the Router config — not on every request.
