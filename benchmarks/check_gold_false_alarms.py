"""Run the sanity checks on BIRD gold SQL. Gold answers should pass, so anything that fires
is either a false alarm (tune the check) or a label error (input for the corrected-labels study).

    python benchmarks/check_gold_false_alarms.py --data data/minidev/MINIDEV
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import veriquery.verifier.sanity as S  # noqa: E402
from veriquery.verifier import Severity, Verifier, VerifierInput  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4, help="examples printed per check")
    args = ap.parse_args()

    check_seconds = Counter()

    def timed(f):
        def wrapper(ctx):
            t0 = time.time()
            try:
                return f(ctx)
            finally:
                check_seconds[f.__name__] += time.time() - t0
        wrapper.__name__ = f.__name__
        return wrapper

    S.CHECKS[:] = [timed(f) for f in S.CHECKS]

    qs = json.loads((args.data / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    verifier = Verifier(timeout_s=60)
    fired, examples, decisions = Counter(), defaultdict(list), Counter()
    started = time.time()
    for i, q in enumerate(qs, 1):
        db = str(args.data / "dev_databases" / q["db_id"] / f"{q['db_id']}.sqlite")
        r = verifier.verify(VerifierInput(q["question"], q["SQL"], db, evidence=q.get("evidence", "")))
        decisions[r.decision.value] += 1
        for c in r.checks:
            if c.severity is not Severity.INFO or "crashed" in c.message:
                key = (c.name, c.severity.value)
                fired[key] += 1
                if len(examples[key]) < args.examples:
                    examples[key].append((q["question_id"], q["question"][:120], q["SQL"][:200], c.message[:240]))
        if i % 50 == 0:
            print(f"{i}/{len(qs)} done, {time.time() - started:.0f}s", flush=True)

    print(f"\n{len(qs)} gold queries in {time.time() - started:.0f}s; decisions: {dict(decisions)}")
    print("seconds per check:", {k: round(s, 1) for k, s in check_seconds.most_common()})
    for (name, sev), n in fired.most_common():
        print(f"\n== {name} [{sev}] x{n}")
        for qid, question, sql, msg in examples[(name, sev)]:
            print(f"  q{qid}: {question}\n    SQL: {sql}\n    -> {msg}")


if __name__ == "__main__":
    main()
