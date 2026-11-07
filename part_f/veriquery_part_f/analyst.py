"""Shim: the Python analyst now lives in veriquery/agents/python_analyst.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from veriquery.agents.python_analyst import AnalysisResult, analyze  # noqa: E402,F401
