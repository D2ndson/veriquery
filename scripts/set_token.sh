#!/usr/bin/env bash
# Paste your Hugging Face token without using an editor:   bash scripts/set_token.sh
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
read -rsp "Paste your Hugging Face token (input is hidden), then press Enter: " T; echo
[ -n "$T" ] || { echo "nothing entered"; exit 1; }
grep -q '^HF_TOKEN=' .env && sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$T|" .env || echo "HF_TOKEN=$T" >> .env
echo "saved to .env"
