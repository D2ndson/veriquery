"""Shim: component B now lives in veriquery/schema_memory.py.

Kept so the commands in this folder's README still work, e.g.
    python schema_memory.py build --all
Data and outputs still default to this folder (person_b/data, person_b/outputs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriquery.schema_memory import *  # noqa: E402,F401,F403
from veriquery.schema_memory import main  # noqa: E402

if __name__ == "__main__":
    main()
