from benchmarks.demo_sql import run_demo


def test_offline_demo_shows_feedback_and_explicit_final_answer():
    result = run_demo()
    assert result["kind"] == "synthetic_functional_demo"
    assert result["runs"]["once"]["status"] == "failed"
    assert result["runs"]["repair"]["status"] == "success"
    assert result["runs"]["repair"]["model_calls"] == 2
    assert result["runs"]["repair"]["execution"]["rows"] == [(55.0,)]
    assert [attempt["action"] for attempt in result["runs"]["react"]["attempts"]] == ["query", "final"]
    assert result["checks"]["repair_received_database_error"]
    assert result["checks"]["react_received_inspection_rows"]
