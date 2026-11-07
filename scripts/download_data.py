#!/usr/bin/env python
"""Prepare all datasets.

    python scripts/download_data.py                       # ShopTrap demo data + BIRD Mini-Dev + Train
    python scripts/download_data.py --what minidev        # just Mini-Dev (the benchmark, ~500 questions)
    python scripts/download_data.py --what shoptrap       # offline demo data only (seconds, no network)
    python scripts/download_data.py --what minidev,dev,train

BIRD downloads resume if interrupted. Nothing is downloaded twice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.config import path  # noqa: E402
from veriquery.datasets import shoptrap  # noqa: E402
from veriquery.pipeline.bird import download_bird  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="shoptrap,minidev,train")
    args = ap.parse_args()
    what = [w.strip() for w in args.what.split(",") if w.strip()]
    if "shoptrap" in what:
        print(shoptrap.write_all(path("data/shoptrap")))
    bird = [w for w in what if w in ("minidev", "dev", "train")]
    if bird:
        download_bird(bird)
    print("Data ready.")


if __name__ == "__main__":
    main()
