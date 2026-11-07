#!/usr/bin/env bash
# Public https link to the app via a free Cloudflare quick tunnel (no account needed).
if ! command -v cloudflared >/dev/null; then
  echo "installing cloudflared ..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared || { echo "could not install cloudflared; use SSH port forwarding instead"; exit 1; }
fi
exec cloudflared tunnel --url http://localhost:8080 2>&1 | tee -a "$(dirname "$0")/../logs/tunnel.log"
