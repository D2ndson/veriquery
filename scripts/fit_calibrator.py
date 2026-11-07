#!/usr/bin/env python
"""Refit the calibrator from existing evaluation results (no GPU, no API):  python scripts/fit_calibrator.py minidev_dpo_final shoptrap_dpo_final"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.pipeline.calibrate import fit_calibrator  # noqa: E402

if __name__ == "__main__":
    fit_calibrator(sys.argv[1:] or ["minidev_dpo_final", "shoptrap_dpo_final"])
