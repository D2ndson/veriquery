#!/usr/bin/env python
"""Ping every model in the chosen profile with a tiny prompt and print a pass/fail table.

    python scripts/check_models.py                    # profile from VERIQUERY_PROFILE or "hybrid"
    python scripts/check_models.py --profile hf_api

Run this first. It tells you immediately if your HF token is wrong, a hosted model is not available to
your account, or the local vLLM server is not up, before you spend GPU time on anything else.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.config import build_llms, hf_token, load_models_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_models_config()
    profile = args.profile or "hybrid"
    uses_hf = any(s["endpoint"] == "hf" for s in cfg["profiles"][profile].values())
    if uses_hf:
        tok = hf_token(required=False)
        print("HF_TOKEN:", "found" if tok and not tok.startswith("hf_paste") else "MISSING (edit .env)")
    llms = build_llms(profile, cfg, use_cache=False)
    bad = 0
    print(f"\nprofile = {profile}\n{'role':<10} {'endpoint':<8} {'model':<42} result")
    for role, llm in llms.items():
        ep = cfg["profiles"][profile][role]["endpoint"]
        t0 = time.time()
        try:
            llm.max_retries = 1
            out = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=12)
            print(f"{role:<10} {ep:<8} {llm.model:<42} OK  ({time.time() - t0:.1f}s) -> {out.strip()[:30]!r}")
        except Exception as e:
            bad += 1
            msg = str(e).replace("\n", " ")[:160]
            print(f"{role:<10} {ep:<8} {llm.model:<42} FAIL {msg}")
    print("\nAll models reachable." if not bad else
          f"\n{bad} model(s) failed. Fix: HF_TOKEN in .env; pick another model in configs/models.yaml; "
          "or start vLLM with scripts/serve_vllm.sh for the local ones.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
