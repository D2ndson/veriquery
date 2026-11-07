"""Record "golden runs": real runs of the full system on demo questions, saved as event streams.

The web UI can REPLAY these instantly and deterministically (no GPU / no API needed), which is what you want for the
video: the replay shows exactly what the live system did, with a visible "RECORDED RUN" badge. Live mode works too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from veriquery.config import build_pipeline, path
from veriquery.events import JsonlRecorder

# ShopTrap showcase questions (see data/shoptrap/README.md for what each one demonstrates)
DEMO_IDS = ["a01", "a03", "f02", "u01", "w01", "t01", "t02", "p03", "d02"]


def record_golden(ids: list[str] | None = None, profile: str | None = None, system: str = "final",
                  writer: str | None = None) -> dict:
    qs = {q["id"]: q for q in (json.loads(l) for l in path("data/shoptrap/questions.jsonl").read_text().splitlines() if l.strip())}
    ids = [i for i in (ids or DEMO_IDS) if i in qs]
    out_dir = path("data/golden")
    out_dir.mkdir(parents=True, exist_ok=True)
    vq = build_pipeline(profile, system, writer, use_cache=False)
    db = str(path("data/shoptrap/shoptrap.sqlite"))
    index = []
    for i in tqdm(ids, desc="golden runs", unit="q"):
        q = qs[i]
        f = out_dir / f"{i}.jsonl"
        f.unlink(missing_ok=True)
        t0 = time.time()
        try:
            state = vq.ask(q["question"], db, q["evidence"], on_event=JsonlRecorder(f))
            from veriquery.server.runs import serialize_state
            (out_dir / f"{i}.result.json").write_text(json.dumps(serialize_state(state), default=str))
            ok = True
        except Exception as e:
            print(f"  {i} failed: {e}")
            ok = False
        index.append({"id": i, "question": q["question"], "evidence": q["evidence"], "trap": q["trap"], "ok": ok,
                      "seconds": round(time.time() - t0, 1), "file": f.name})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"[golden] recorded {sum(x['ok'] for x in index)}/{len(index)} runs in {out_dir}")
    return {"runs": index}
