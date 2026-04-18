#!/usr/bin/env bash
# Build and start the voscript service via docker compose.
# Requires .env in the repo root with at least HF_TOKEN (and ideally API_KEY).

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.example to .env and fill in secrets." >&2
    exit 1
fi

if ! grep -q '^HF_TOKEN=hf_' .env; then
    echo "ERROR: HF_TOKEN missing or placeholder in .env" >&2
    exit 1
fi

if ! grep -qE '^API_KEY=.+' .env || grep -q '^API_KEY=change-me' .env; then
    echo "WARNING: API_KEY is empty or still the placeholder. The service will" >&2
    echo "         accept unauthenticated requests — only safe on trusted LAN." >&2
fi

docker compose up -d --build

echo
echo "Waiting for /healthz..."
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${HOST_PORT:-8780}/healthz" >/dev/null; then
        echo "Service is up at http://localhost:${HOST_PORT:-8780}"
        exit 0
    fi
    sleep 2
done

echo "Service did not become healthy. Check: docker logs voscript" >&2
exit 1
