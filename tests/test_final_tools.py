"""Unit tests for the final-project tools (no LLM, no network)."""
import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from veriquery.agents.analysis_tools import (auto_analysis, contribution_analysis, detect_anomalies, forecast_series,
                                             pvm_decomposition, wants_decomposition)
from veriquery.agents.provenance import check_provenance
from veriquery.agents.sql_tools import SQLTools
from veriquery.datasets import shoptrap
from veriquery.fewshot import FewShotBank, tables_in_sql
from veriquery.pii import classify_column, mask_frame
from veriquery.prompts import build_text_prompt, extract_sql_flexible, split_plan
from veriquery.semantic_layer import SemanticLayer
from veriquery.verifier.calibrator import Calibrator, clopper_pearson_upper, ece, select_threshold
from veriquery.verifier.metamorphic import applicable_relation, scale_invariance_check


# ----------------------------------------------------------------------------- prompts
def test_text_prompt_roundtrip_with_plan():
    prompt = build_text_prompt("How many X?", "t(a INTEGER)", "X means a>1", reasoning="plan")
    assert "### Question\nHow many X?" in prompt and "<plan>" in prompt and "X means a>1" in prompt
    plan, sql = extract_sql_flexible("Sure.\n<plan>\ntables: t\nfilters: a>1\n</plan>\n```sql\nSELECT COUNT(*) FROM t WHERE a>1;\n```")
    assert plan.startswith("tables: t") and sql == "SELECT COUNT(*) FROM t WHERE a>1"
    assert split_plan("no plan here")[0] == ""


def test_extract_sql_flexible_variants():
    assert extract_sql_flexible("select 1")[1] == "select 1"
    assert extract_sql_flexible("```sql\nSELECT 1\n```")[1] == "SELECT 1"
    assert extract_sql_flexible("<think>hmm</think>```SELECT 2```" if False else "<think>x</think>SELECT 2")[1] == "SELECT 2"
    with pytest.raises(ValueError):
        extract_sql_flexible("I cannot do that")


def test_prompt_includes_feedback_and_examples():
    p = build_text_prompt("q", "s", examples=[{"question": "eq", "sql": "SELECT 1", "schema": "t(a)"}],
                          feedback="no such column", previous_sql="SELECT zz")
    assert "Example 1" in p and "no such column" in p and "SELECT zz" in p


# ----------------------------------------------------------------------------- sql tools
def test_literal_fix_int_div_and_null_order(db):
    t = SQLTools(db)
    sql, notes = t.postprocess("SELECT COUNT(*) FROM customers WHERE region = 'east'", "how many")
    assert "'East'" in sql and notes
    sql, notes = t.postprocess("SELECT COUNT(*) FROM customers WHERE region = 'East'", "how many")
    assert notes == []  # exact value exists: untouched
    sql, notes = t.postprocess("SELECT SUM(CASE WHEN region='East' THEN 1 ELSE 0 END)*100/COUNT(*) FROM customers",
                               "what percentage are in the east?")
    assert "CAST" in sql.upper() and "REAL" in sql.upper()
    conn = sqlite3.connect(db)
    assert conn.execute(sql).fetchone()[0] == pytest.approx(60.0)
    sql, notes = t.postprocess("SELECT name FROM products ORDER BY rating LIMIT 1", "lowest rated product")
    assert "NOT" in sql and "NULL" in sql
    assert conn.execute(sql).fetchone()[0] == "Chair"
    # not a ratio question -> no int-div rewrite
    assert t.postprocess("SELECT 1/2", "how many")[0] == "SELECT 1/2"


def test_fixes_are_selectable_and_never_raise(db):
    assert SQLTools(db, fixes=()).postprocess("SELECT 1", "")[1] == []
    assert SQLTools(db).postprocess("this is not sql", "")[0] == "this is not sql"
    sql, _ = SQLTools(db, fixes=("literals",)).postprocess("SELECT name FROM products ORDER BY rating LIMIT 1", "lowest")
    assert "NULL" not in sql


def test_hints_suggest_close_names(db):
    h = SQLTools(db).hints("select nme from customers", "no such column: nme")
    assert "customers.name" in h
    assert "customers" in SQLTools(db).hints("select 1 from custmers", "no such table: custmers")


# ----------------------------------------------------------------------------- few-shot
def test_fewshot_bank_retrieval_and_exclusion():
    bank = FewShotBank()
    bank.add("How many schools are in Fresno?", "SELECT COUNT(*) FROM schools WHERE City='Fresno'", "schools(City TEXT)", "ca")
    bank.add("What is the average salary of engineers?", "SELECT AVG(salary) FROM emp", "emp(salary REAL)", "hr")
    got = bank.retrieve("How many schools are located in Fresno county?", k=1)
    assert got[0].db_id == "ca"
    assert all(e.db_id != "ca" for e in bank.retrieve("How many schools are in Fresno?", k=2, exclude_db="ca"))
    assert [e.db_id for e in bank.retrieve("How many schools are in Fresno?", k=2, same_db_only="hr")] == []
    assert tables_in_sql("SELECT a FROM t1 JOIN t2 ON t1.i=t2.i") == ["t1", "t2"]


def test_fewshot_save_load(tmp_path):
    bank = FewShotBank()
    bank.add("q one", "SELECT 1", "t(a)", "d")
    bank.save(tmp_path / "b.jsonl")
    assert len(FewShotBank.load(tmp_path / "b.jsonl")) == 1 and len(FewShotBank.load(tmp_path / "none.jsonl")) == 0


# ----------------------------------------------------------------------------- semantic layer / pii / provenance
def test_semantic_layer_matches_and_persists(tmp_path):
    sl = SemanticLayer(tmp_path)
    sl.add("shop", "net revenue", "after discounts, delivered only")
    assert sl.match("shop", "What was net revenue in 2024?")[0]["definition"].startswith("after")
    assert sl.match("shop", "gross sales") == []
    assert SemanticLayer(tmp_path).all("shop")[0]["uses"] == 1    # survives a restart, use counted
    assert sl.remove("shop", "Net Revenue") and sl.all("shop") == []


def test_pii_classification_and_masking():
    assert classify_column("customer_email") == "email" and classify_column("city") is None
    assert classify_column("x", ["S1234567A", "T7654321B"]) == "national_id"
    df = pd.DataFrame({"email": ["alice@x.com"], "city": ["SG"], "phone": ["+65 9123 4567"]})
    out, masked = mask_frame(df)
    assert "alice" not in out["email"][0] and out["city"][0] == "SG" and set(masked) == {"email", "phone"}


def test_provenance():
    df = pd.DataFrame({"region": ["East"], "revenue": [1234567.891], "share": [0.25]})
    assert check_provenance("East earned 1,234,567.89.", df)[0]
    assert check_provenance("That is 1.23 million.", df)[0]
    assert check_provenance("A 25% share.", df)[0]
    ok, bad = check_provenance("Revenue was up 15%.", df)
    assert not ok and bad == ["15%"]
    assert check_provenance("In 2024 it was 1,234,568.", df, "revenue in 2024?")[0]


# ----------------------------------------------------------------------------- calibrator
def test_clopper_pearson_and_threshold():
    assert clopper_pearson_upper(0, 50, 0.1) == pytest.approx(1 - 0.1 ** (1 / 50), rel=1e-6)
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 1500)
    correct = rng.uniform(0, 1, 1500) < scores              # perfectly calibrated
    sel = select_threshold(scores, correct, alpha=0.1, delta=0.1)
    assert sel and sel["risk_upper_bound"] <= 0.1 and 0 < sel["coverage"] < 1
    assert select_threshold(scores, np.zeros(1500, bool), alpha=0.05) is None   # nothing is ever correct


def test_calibrator_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(1)
    n = 600
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 2, n),
                         rng.uniform(0, 1, n), rng.integers(0, 3, n), rng.integers(0, 2, n), rng.uniform(0, 5, n),
                         rng.integers(0, 2, n), rng.uniform(3, 6, n)]).astype(float)
    logit = 2.0 * X[:, 0] - 2.5 * X[:, 3] - 0.8 * X[:, 2] + 1.0 * X[:, 4] - 0.5
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    cal = Calibrator.fit(X, y, alpha=0.2)
    assert cal.params["meta"]["auroc"] > 0.75
    good = cal.predict([1, 0, 0, 0, 1.0, 0, 1, 1, 0, 4])
    bad = cal.predict([0, 1, 2, 1, 0.2, 2, 0, 1, 0, 4])
    assert good > bad
    cal.save(tmp_path / "c.json")
    assert Calibrator.load(tmp_path / "c.json").predict([1, 0, 0, 0, 1.0, 0, 1, 1, 0, 4]) == pytest.approx(good)
    assert not Calibrator.load(tmp_path / "missing.json").ready
    assert 0 <= ece(np.array([0.9, 0.1]), np.array([1, 0])) <= 0.2


# ----------------------------------------------------------------------------- metamorphic
def test_metamorphic_applicability_and_violation(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript("CREATE TABLE o(id INTEGER PRIMARY KEY, cust INT, amt REAL);"
                    "INSERT INTO o VALUES (1,1,10),(2,1,20),(3,2,60);")
    c.commit()
    assert applicable_relation("SELECT AVG(amt) FROM o") == "average"
    assert applicable_relation("SELECT SUM(amt) FROM o") is None
    assert applicable_relation("SELECT SUM(amt)*1.0/COUNT(DISTINCT cust) FROM o") is None
    assert applicable_relation("SELECT cust FROM o GROUP BY cust ORDER BY SUM(amt) DESC LIMIT 1") == "argmax"
    ok = pd.read_sql("SELECT AVG(amt) FROM o", c)
    assert scale_invariance_check(str(p), "SELECT AVG(amt) FROM o", ok) is None
    bad_sql = "SELECT AVG(amt) FROM o WHERE rowid <= 4"     # depends on row identity, changes when rows are duplicated
    bad = scale_invariance_check(str(p), bad_sql, pd.read_sql(bad_sql, c))
    assert bad is not None and bad.name == "metamorphic_scale"


# ----------------------------------------------------------------------------- analysis tools
def _pvm_frame():
    return pd.DataFrame([
        ("2024Q1", "A", 100, 1000.0), ("2024Q1", "B", 50, 1000.0), ("2024Q1", "C", 30, 300.0),
        ("2024Q2", "A", 90, 990.0), ("2024Q2", "B", 60, 1260.0), ("2024Q2", "D", 10, 200.0)],
        columns=["quarter", "category", "units", "revenue"])


def test_pvm_identity_is_exact():
    res = pvm_decomposition(_pvm_frame(), "category", "quarter", "units", "revenue")
    t = res["totals"]
    assert t["price"] + t["volume"] + t["mix"] + t["new_lost"] == pytest.approx(t["delta"], abs=1e-9)
    assert t["delta"] == pytest.approx(150.0)
    row = res["table"].set_index("category").loc["A"]
    assert row["price"] + row["volume"] + row["mix"] == pytest.approx(row["delta"], abs=1e-9)


def test_contribution_and_auto_selection():
    res = contribution_analysis(_pvm_frame(), "quarter", "category", "revenue")
    assert res["total_delta"] == pytest.approx(150.0) and res["table"].iloc[0]["category"] == "C"
    a = auto_analysis(_pvm_frame(), "Why did revenue change between Q1 and Q2?")
    assert a.operation == "pvm_decomposition" and "price effect" in a.answer
    only_rev = _pvm_frame().drop(columns="units")
    assert auto_analysis(only_rev, "Why did revenue drop?").operation == "contribution_analysis"
    assert auto_analysis(_pvm_frame(), "list all rows") is None
    assert wants_decomposition("What drove the drop in sales?") and not wants_decomposition("How many orders?")


def test_anomalies_and_forecast():
    s = pd.Series([10, 11, 12, 11, 13, 12, 60, 12, 11, 13])
    an = detect_anomalies(s)
    assert an["index"].tolist() == [6]
    assert detect_anomalies(pd.Series([5, 5, 5, 5])).empty
    fc = forecast_series(pd.Series(np.arange(1, 25, dtype=float)), horizon=3)
    assert len(fc) == 3 and (fc["lower"] <= fc["forecast"]).all() and fc["forecast"].iloc[0] > 20
    with pytest.raises(ValueError):
        forecast_series(pd.Series([1, 2]))


# ----------------------------------------------------------------------------- ShopTrap dataset
def test_shoptrap_builds_and_gold_runs(tmp_path):
    info = shoptrap.write_all(tmp_path, seed=7)
    assert info["problems"] == [] and info["questions"] >= 35
    again = shoptrap.build_db(tmp_path / "again.sqlite", seed=7)
    a, b = sqlite3.connect(tmp_path / "shoptrap.sqlite"), sqlite3.connect(again)
    assert a.execute("SELECT SUM(shipping_fee), COUNT(*) FROM orders").fetchone() == \
        b.execute("SELECT SUM(shipping_fee), COUNT(*) FROM orders").fetchone()   # deterministic
    qs = [json.loads(l) for l in (tmp_path / "questions.jsonl").read_text().splitlines()]
    assert {q["trap"] for q in qs} >= {"fanout", "units", "date_format", "literal_case", "ratio", "null_order", "ambiguous", "why"}
    assert all(len(q["readings"]) >= 2 for q in qs if q["trap"] == "ambiguous" and q["id"] != "a04" and q["readings"])
    # fan-out trap really exists: joining items repeats the shipping fee
    fan = a.execute("SELECT SUM(o.shipping_fee) FROM orders o JOIN order_items oi ON oi.order_id=o.order_id").fetchone()[0]
    assert fan > a.execute("SELECT SUM(shipping_fee) FROM orders").fetchone()[0] * 1.5
