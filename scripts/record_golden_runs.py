#!/usr/bin/env python
"""Record demo runs for the UI replay mode:  python scripts/record_golden_runs.py [--profile hybrid] [--writer writer-dpo] [ids...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.pipeline.golden import record_golden  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="*")
ap.add_argument("--profile")
ap.add_argument("--writer")
a = ap.parse_args()
record_golden(a.ids or None, a.profile, "final", a.writer)
