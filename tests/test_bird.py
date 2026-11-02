"""Synthetic database regression tests; these are not BIRD benchmark results."""
import json
import sqlite3

import pytest

from benchmarks.bird import (
    BirdExample, apply_corrections, database_path, fixed_subset, load_bird,
    score_prediction,
)


def write_json(tmp_path, rows, name="questions.json"):
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def row(qid=1, **changes):
    item = dict(question_id=qid, db_id="sales", question="How many sales?",
                evidence="Count rows", SQL="SELECT COUNT(*) FROM sales", difficulty="simple")
    item.update(changes)
    return item


def test_load_official_shape_and_ignore_metadata(tmp_path):
    path = write_json(tmp_path, [row(external_knowledge="must not enter prompts")])
    example = load_bird(path)[0]
    assert example == BirdExample(1, "sales", "How many sales?", "Count rows",
                                  "SELECT COUNT(*) FROM sales", "simple")
    assert not hasattr(example, "external_knowledge")


@pytest.mark.parametrize("bad", [
    [row(), row()], [row(), row("1")], [row(True)], [row(None)], [row("")],
    [row(SQL="")], [row(evidence=None)], [row(db_id="../sales")],
    {"data": [row()]}, ["not a record"], [dict(question_id=1)],
])
def test_reject_malformed_data(tmp_path, bad):
    with pytest.raises(ValueError):
        load_bird(write_json(tmp_path, bad))


def test_optional_metadata(tmp_path):
    item = row()
    del item["evidence"], item["difficulty"]
    example = load_bird(write_json(tmp_path, [item]))[0]
    assert example.evidence == ""
    assert example.difficulty == "unknown"


def test_subset_is_order_independent_and_seeded(tmp_path):
    examples = load_bird(write_json(tmp_path, [row(i) for i in range(150)]))
    sample = fixed_subset(examples)
    assert len(sample) == len({e.question_id for e in sample}) == 100
    assert sample == fixed_subset(reversed(examples))
    assert sample != fixed_subset(examples, seed=123)
    assert examples[0].question_id == 0  # input was not shuffled in place


@pytest.mark.parametrize("size", [0, -1, True, 1.5, 2])
def test_subset_rejects_invalid_sizes(tmp_path, size):
    examples = load_bird(write_json(tmp_path, [row()]))
    with pytest.raises(ValueError):
        fixed_subset(examples, size=size)


def test_database_resolution(tmp_path):
    directory = tmp_path / "sales"
    directory.mkdir()
    path = directory / "sales.sqlite"
    path.write_bytes(b"")
    assert database_path(tmp_path, "sales") == path
    with pytest.raises(FileNotFoundError):
        database_path(tmp_path, "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("name", ["../sales", "/sales", "a/b", "a\\b", "..", ".", "a\x00b"])
def test_database_rejects_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        database_path(tmp_path, name)


def test_database_rejects_escaping_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sales.sqlite").write_bytes(b"")
    (root / "sales").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        database_path(root, "sales")


def test_corrected_overlay_retains_original_and_input(tmp_path):
    examples = load_bird(write_json(tmp_path, [row(), row(2)]))
    path = write_json(tmp_path, [dict(question_id=1, db_id="sales", corrected_sql="SELECT 7",
                                    original_SQL=examples[0].gold_sql, source="review@abc")], "corrections.json")
    corrected = apply_corrections(examples, path)
    assert corrected[0].gold_sql == "SELECT 7"
    assert corrected[0].original_gold_sql == examples[0].gold_sql
    assert corrected[0].label_source == "review@abc"
    assert examples[0].original_gold_sql is None
    assert corrected[1] is examples[1]
    with pytest.raises(ValueError, match="already"):
        apply_corrections(corrected, path)


@pytest.mark.parametrize("changes", [
    {"question_id": 99}, {"db_id": "other"}, {"original_SQL": "wrong revision"},
    {"SQL": "SELECT 2"}, {"question": "A different task"}, {"evidence": "changed"},
    {"corrected_sql": ""}, {"source": ""},
])
def test_correction_rejects_conflicts(tmp_path, changes):
    examples = load_bird(write_json(tmp_path, [row()]))
    correction = dict(question_id=1, db_id="sales", corrected_sql="SELECT 1")
    correction.update(changes)
    with pytest.raises(ValueError):
        apply_corrections(examples, write_json(tmp_path, [correction], "corrections.json"))


def test_duplicate_correction_rejected(tmp_path):
    examples = load_bird(write_json(tmp_path, [row()]))
    correction = dict(question_id=1, db_id="sales", corrected_sql="SELECT 1")
    with pytest.raises(ValueError, match="Duplicate correction"):
        apply_corrections(examples, write_json(tmp_path, [correction, correction], "corrections.json"))


@pytest.fixture
def executor(tmp_path):
    from veriquery.sql_execution import SQLiteExecutor
    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE sales (value INTEGER); INSERT INTO sales VALUES (2),(1),(1),(NULL);")
    return SQLiteExecutor(path, max_rows=100)


def test_set_matches_bird_but_bag_detects_duplicates(executor):
    pred = "SELECT DISTINCT value FROM sales"
    gold = "SELECT value FROM sales"
    assert score_prediction(executor, pred, gold).correct is True
    bag = score_prediction(executor, pred, gold, mode="bag")
    assert bag.correct is False
    assert bag.predicted_rows == 3 and bag.gold_rows == 4


def test_ordered_mode_detects_order(executor):
    pred = "SELECT value FROM sales ORDER BY value DESC"
    gold = "SELECT value FROM sales ORDER BY value ASC"
    assert score_prediction(executor, pred, gold, "bag").correct is True
    assert score_prediction(executor, pred, gold, "ordered").correct is False


def test_column_order_is_not_ignored(executor):
    assert score_prediction(executor, "SELECT 1, 2", "SELECT 2, 1").correct is False


def test_empty_results_and_null_values(executor):
    assert score_prediction(executor, "SELECT value FROM sales WHERE 0", "SELECT value FROM sales WHERE 1=0").correct is True
    assert score_prediction(executor, "SELECT NULL", "SELECT value FROM sales WHERE value IS NULL").correct is True


def test_bad_prediction_is_incorrect(executor):
    result = score_prediction(executor, "SELECT no_such_column FROM sales", "SELECT 1")
    assert result.correct is False
    assert result.status == "prediction_error"
    assert result.prediction_error


def test_gold_error_is_unscorable(executor):
    result = score_prediction(executor, "SELECT 1", "SELECT no_such_column FROM sales")
    assert result.correct is None
    assert result.status == "unscorable_gold_error"
    assert result.gold_error


def test_truncation_never_receives_correct_credit(tmp_path):
    from veriquery.sql_execution import SQLiteExecutor
    path = tmp_path / "small.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE numbers (n)")
        conn.executemany("INSERT INTO numbers VALUES (?)", [(1,), (2,), (3,)])
    limited = SQLiteExecutor(path, max_rows=1)
    both = score_prediction(limited, "SELECT n FROM numbers", "SELECT n FROM numbers")
    assert both.correct is None
    assert both.status == "unscorable_gold_truncated"
    prediction = score_prediction(limited, "SELECT n FROM numbers", "SELECT 1")
    assert prediction.correct is False
    assert prediction.status == "prediction_truncated"


def test_invalid_mode_does_not_execute(executor):
    with pytest.raises(ValueError, match="mode"):
        score_prediction(executor, "SELECT 1", "SELECT 1", "fuzzy")
