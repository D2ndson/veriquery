"""Shim: the reporter now lives in veriquery/agents/reporter.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from veriquery.agents.reporter import build_report  # noqa: E402,F401
