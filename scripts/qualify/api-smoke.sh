#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8000}"

curl -fsS "$url/health" | jq .
models="$(curl -fsS "$url/v1/models")"
printf '%s\n' "$models" | jq .
model="${2:-$(printf '%s\n' "$models" | jq -er '.data[0].id')}"
reply="$(curl -fsS "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":16}")"
printf '%s\n' "$reply" | jq .
# A reply carrying an error, or no text at all (answer or reasoning), fails the smoke.
printf '%s\n' "$reply" | jq -e '.error == null and ((.choices[0].message.content // "") + (.choices[0].message.reasoning_content // "") != "")' >/dev/null ||
  { echo "api-smoke: error or empty completion" >&2; exit 1; }
