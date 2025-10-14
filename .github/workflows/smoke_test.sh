#!/usr/bin/env bash
set -euo pipefail

# Base URL to Core (FastAPI). Override in CI with:
#   CORE_BASE_URL="https://core.example.com"
CORE_BASE_URL="${CORE_BASE_URL:-http://localhost:8123}"

echo "[smoke] Pinging $CORE_BASE_URL/health ..."
curl -fsS "$CORE_BASE_URL/health" > /dev/null
echo "[smoke] /health OK"

# Sample question
Q='{"query":"Current asthma in adults by sex in the latest year available"}'

echo "[smoke] POST $CORE_BASE_URL/v1/widget_text"
RESP="$(curl -fsS -X POST "$CORE_BASE_URL/v1/widget_text" \
  -H 'Content-Type: application/json' \
  -d "$Q")"

# Basic assertions without jq
echo "$RESP" | grep -q '"answer_text"' || { echo "[smoke] missing answer_text in response"; echo "$RESP"; exit 1; }
echo "$RESP" | grep -q '"engine_version"' || { echo "[smoke] missing engine_version in response"; echo "$RESP"; exit 1; }

echo "[smoke] widget_text OK"
