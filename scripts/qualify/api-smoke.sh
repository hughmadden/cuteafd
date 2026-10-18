#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8000}"

curl -fsS "$url/health" | jq .
models="$(curl -fsS "$url/v1/models")"
printf '%s\n' "$models" | jq .
model="${2:-$(printf '%s\n' "$models" | jq -er '.data[0].id')}"
curl -fsS "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":16}" | jq .
