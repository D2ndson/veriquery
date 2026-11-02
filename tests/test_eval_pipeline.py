import json

from benchmarks.eval_pipeline import shard_questions, summarize


def test_shards_keep_databases_whole():
    qs = [{"question_id": i, "db_id": db} for i, db in enumerate("aaabbcd")]
    shards = shard_questions(qs, 2)
    assert sorted(len(s) for s in shards) == [3, 4]
    for db in "abcd":
        assert sum(any(q["db_id"] == db for q in s) for s in shards) == 1


def test_summary_scores_primary_reading_and_calibration(tmp_path):
    def rec(qid, readings):
        return {"question_id": qid, "db_id": "x", "difficulty": "simple", "question": "", "clarifier": "proceed",
                "readings": readings, "error": None, "seconds": 2.0, "usage": {"calls": 4}}
    rows = [rec(1, [{"correct": True, "confidence": "high"}]),
            rec(2, [{"correct": False, "confidence": "high"}]),
            rec(3, [{"correct": False, "confidence": "medium"}, {"correct": True, "confidence": "medium"}]),
            rec(4, [])]
    (tmp_path / "shard_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = summarize(tmp_path)
    assert s["questions"] == 4 and s["accuracy_primary"] == 0.25 and s["accuracy_any_reading"] == 0.5
    assert s["by_confidence"]["high"] == {"answers": 2, "share": 0.5, "accuracy": 0.5}
    assert s["confidently_wrong_rate"] == 0.25
