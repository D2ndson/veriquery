import pandas as pd
import pytest

from veriquery.verifier import (
    Decision,
    FirstSuccessfulRoute,
    PandasSecondRoute,
    QuestionOutcome,
    RouteResult,
    Severity,
    SqlSecondRoute,
    Verifier,
    VerifierInput,
    calibration_table,
    execution_match,
    make_verifier_node,
    reliability_metrics,
    repair_loop,
    results_match,
)
from veriquery.verifier.db import run_sql
from veriquery.verifier.dual_route import extract_sql


def stub_llm(sql):
    return lambda prompt: f"Here you go:\n```sql\n{sql}\n```"


def names(report, severity=None):
    return {c.name for c in report.checks if severity is None or c.severity is severity}


# ------------------------------------------------------------- sanity checks

def test_correct_answer_accepted_when_routes_agree(db):
    v = Verifier(SqlSecondRoute(stub_llm("SELECT COUNT(id) FROM customers WHERE region = 'East'")))
    r = v.verify(VerifierInput("How many customers are in the East region?",
                               "SELECT COUNT(*) FROM customers WHERE region = 'East'", db))
    assert r.decision is Decision.ACCEPT
    assert r.routes_agree is True
    assert not r.failures and not r.warnings
    assert r.confidence == pytest.approx(0.9)


def test_join_fanout_inflates_sum(db):
    r = Verifier().verify(VerifierInput(
        "What is the total credit of East customers who placed orders?",
        "SELECT SUM(c.credit) FROM customers AS c JOIN orders AS o ON c.id = o.customer_id WHERE c.region = 'East'",
        db))
    assert r.decision is Decision.REPAIR
    assert "join_fanout" in names(r, Severity.FAIL)
    assert "orders.customer_id is not unique" in r.repair_feedback


def test_one_to_many_count_of_many_side_is_fine(db):
    r = Verifier().verify(VerifierInput(
        "How many orders were placed by East customers?",
        "SELECT COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.region = 'East'",
        db))
    assert "join_fanout" not in names(r)
    assert r.decision is Decision.ACCEPT


def test_count_distinct_is_not_flagged(db):
    r = Verifier().verify(VerifierInput(
        "How many East customers placed orders?",
        "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.region = 'East'",
        db))
    assert "join_fanout" not in names(r)


def test_integer_division(db):
    bad = Verifier().verify(VerifierInput(
        "What percentage of customers are in the East region?",
        "SELECT SUM(CASE WHEN region = 'East' THEN 1 ELSE 0 END) / COUNT(*) * 100 FROM customers", db))
    assert "integer_division" in names(bad, Severity.FAIL)

    good = Verifier().verify(VerifierInput(
        "What percentage of customers are in the East region?",
        "SELECT CAST(SUM(CASE WHEN region = 'East' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customers", db))
    assert "integer_division" not in names(good)
    assert good.decision is Decision.ACCEPT


def test_filter_value_case_mismatch(db):
    r = Verifier().verify(VerifierInput(
        "List the names of customers in the east region.",
        "SELECT name FROM customers WHERE region = 'east'", db))
    fv = next(c for c in r.checks if c.name == "filter_value")
    assert fv.severity is Severity.FAIL
    assert fv.evidence["suggestion"] == "East"
    assert "empty_result" in names(r, Severity.FAIL)


def test_date_format_mismatch(db):
    r = Verifier().verify(VerifierInput(
        "What was the total consumption in 2012?",
        "SELECT SUM(consumption) FROM yearmonth WHERE date BETWEEN '2012-01' AND '2012-12'", db))
    assert "date_format" in names(r, Severity.FAIL)
    assert "null_result" in names(r, Severity.FAIL)


def test_date_upper_bound_excludes_last_month(db):
    r = Verifier().verify(VerifierInput(
        "How many orders were placed in 2012?",
        "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2012-01' AND '2012-12'", db))
    assert "date_upper_bound" in names(r, Severity.WARN)
    assert "date_format" not in names(r)


def test_year_missing_from_sql(db):
    r = Verifier().verify(VerifierInput("How many orders were placed in 2013?", "SELECT COUNT(*) FROM orders", db))
    assert "year_missing" in names(r, Severity.WARN)


def test_duplicate_rows_after_join(db):
    r = Verifier().verify(VerifierInput(
        "List the names of customers who placed orders.",
        "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id", db))
    assert "duplicate_rows" in names(r, Severity.INFO)  # BIRD scores sets, so duplicates only inform


def test_execution_error_goes_to_repair(db):
    r = Verifier().verify(VerifierInput("How many customers?", "SELECT COUNT(*) FROM customer", db))
    assert r.decision is Decision.REPAIR
    assert "no such table" in r.repair_feedback


def test_precomputed_result_is_used(db):
    # When the pipeline already ran the SQL, the verifier checks that result.
    r = Verifier().verify(VerifierInput("How many customers are there?", "SELECT COUNT(*) FROM customers", db,
                                        result=pd.DataFrame({"n": [-1]})))
    assert "count_value" in names(r, Severity.FAIL)


# ----------------------------------------------------------------- dual route

def test_disagreement_repairs_then_flags_low_confidence(db):
    route = SqlSecondRoute(stub_llm("SELECT COUNT(*) FROM customers WHERE region = 'East'"))
    v = Verifier(route, max_repair_rounds=2)
    inp = VerifierInput("How many customers are in the East region?", "SELECT COUNT(*) FROM customers", db)

    r = v.verify(inp)
    assert r.decision is Decision.REPAIR
    assert r.routes_agree is False
    assert "region = 'East'" in r.repair_feedback

    inp.repair_round = 2
    assert v.verify(inp).decision is Decision.LOW_CONFIDENCE


def test_second_route_failure_does_not_block(db):
    v = Verifier(SqlSecondRoute(stub_llm("SELECT nope FROM nowhere")))
    r = v.verify(VerifierInput("How many customers are there?", "SELECT COUNT(*) FROM customers", db))
    assert r.decision is Decision.ACCEPT
    assert r.routes_agree is None
    assert r.confidence == pytest.approx(0.7)


def test_second_route_skipped_when_already_failing(db):
    calls = []

    def route(inp, first):
        calls.append(1)
        return RouteResult("", None, "unused")

    Verifier(route).verify(VerifierInput("How many customers?", "SELECT COUNT(*) FROM customer", db))
    assert not calls


def test_second_route_prompt_is_independent(db):
    route = SqlSecondRoute(stub_llm(""))
    first_sql = "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id ORDER BY o.amount DESC LIMIT 1"
    inp = VerifierInput("Who placed the largest order?", first_sql, db)
    prompt = route.build_prompt(inp, run_sql(db, first_sql))
    assert first_sql not in prompt
    assert "subqueries" in prompt and "MAX()/MIN()" in prompt
    assert "exactly 1 column(s), in this order: name" in prompt
    assert "CREATE TABLE customers" in prompt


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


# --------------------------------------------------------- loop + integration

def test_repair_loop_fixes_fanout(db):
    fixed = "SELECT SUM(credit) FROM customers WHERE region = 'East' AND id IN (SELECT customer_id FROM orders)"
    seen = []

    def writer(prev, feedback):
        seen.append(feedback)
        return fixed

    inp = VerifierInput(
        "What is the total credit of East customers who placed orders?",
        "SELECT SUM(c.credit) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.region = 'East'", db)
    sql, reports = repair_loop(inp, writer, Verifier())
    assert sql == fixed
    assert [r.decision for r in reports] == [Decision.REPAIR, Decision.ACCEPT]
    assert "join_fanout" in seen[0]
    assert run_sql(db, sql).iat[0, 0] == 600


def test_verifier_node(db):
    node = make_verifier_node(Verifier())
    out = node({"question": "How many customers are there?", "sql": "SELECT COUNT(*) FROM customers", "db_path": db})
    assert out["verification_status"] == "accept"
    assert out["verification"]["checks"] == []


# ---------------------------------------------------------- compare + metrics

def test_results_match_tolerant():
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = pd.DataFrame({"n": [2.0, 1.0000001], "s": ["b", "a"]})
    assert results_match(a, b)[0]
    assert not results_match(a, pd.DataFrame({"x": [1, 3], "y": ["a", "b"]}))[0]
    assert not results_match(a, a.head(1))[0]
    # Different column counts: compare shared columns only.
    assert results_match(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1], "extra": [9]}))[0]


def test_execution_match_bird_rule(db):
    gold = "SELECT name FROM customers WHERE region = 'West'"
    assert execution_match(db, "SELECT name FROM customers WHERE id > 3 ORDER BY id DESC", gold)
    assert not execution_match(db, "SELECT name FROM customers", gold)
    assert not execution_match(db, "SELECT broken", gold)


def test_reliability_metrics():
    outcomes = [
        QuestionOutcome("q1", True, False, True, False, 0.9),  # right, accepted
        QuestionOutcome("q2", False, True, True, False, 0.9),  # caught and fixed
        QuestionOutcome("q3", False, True, False, True, 0.3),  # caught, still wrong, flagged
        QuestionOutcome("q4", False, False, False, False, 0.9),  # confidently wrong
        QuestionOutcome("q5", True, True, False, False, 0.7),  # false alarm, broken by repair
    ]
    m = reliability_metrics(outcomes)
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["confidently_wrong_rate"] == pytest.approx(2 / 5)
    assert m["confidently_wrong_share_of_errors"] == pytest.approx(2 / 3)
    assert m["catch_rate"] == pytest.approx(2 / 3)
    assert m["catch_and_fix_rate"] == pytest.approx(1 / 3)
    assert m["false_alarm_rate"] == pytest.approx(1 / 2)
    assert m["broken_by_repair"] == 1
    assert m["flag_precision"] == pytest.approx(1.0)

    table = calibration_table(outcomes)
    assert sum(row["n"] for row in table) == 5


# ------------------------------------------------------- units, dates, limits

def test_cents_column_asked_in_dollars(db):
    bad = Verifier().verify(VerifierInput("What is the total price in dollars of all products?",
                                          "SELECT SUM(price_cents) FROM products", db))
    assert "units" in names(bad, Severity.FAIL)
    good = Verifier().verify(VerifierInput("What is the total price in dollars of all products?",
                                           "SELECT SUM(price_cents) / 100.0 FROM products", db))
    assert "units" not in names(good)


def test_unit_from_data_card(db):
    cards = {"orders": {"columns": {"amount": {"unit": "thousands"}}}}
    r = Verifier().verify(VerifierInput("What is the total order amount in millions?",
                                        "SELECT SUM(amount) FROM orders", db, data_cards=cards))
    assert "units" in names(r, Severity.WARN)


def test_strftime_on_non_iso_dates(db):
    r = Verifier().verify(VerifierInput("How many products launched in 2012?",
                                        "SELECT COUNT(*) FROM products WHERE strftime('%Y', launched) = '2012'", db))
    assert "strftime_format" in names(r, Severity.FAIL)


def test_ascending_order_returns_null(db):
    bad = Verifier().verify(VerifierInput("Which product has the lowest rating?",
                                          "SELECT name FROM products ORDER BY rating LIMIT 1", db))
    assert "order_nulls" in names(bad, Severity.FAIL)
    good = Verifier().verify(VerifierInput("Which product has the lowest rating?",
                                           "SELECT name FROM products WHERE rating IS NOT NULL ORDER BY rating LIMIT 1", db))
    assert "order_nulls" not in names(good)
    assert "limit_truncation" not in names(good)


def test_limit_truncation_and_top_n(db):
    r = Verifier().verify(VerifierInput("List the names of all products.", "SELECT name FROM products LIMIT 2", db))
    assert "limit_truncation" in names(r, Severity.WARN)
    r = Verifier().verify(VerifierInput("List the top two products by rating.", "SELECT name FROM products", db))
    assert "top_n" in names(r, Severity.WARN)


# --------------------------------------------------------------- pandas route

def test_pandas_route_agrees(db):
    code = "c = tables['customers']\nresult = pd.DataFrame({'n': [int((c['region'] == 'East').sum())]})"
    route = PandasSecondRoute(lambda prompt: f"```python\n{code}\n```")
    r = Verifier(route).verify(VerifierInput("How many customers are in the East region?",
                                             "SELECT COUNT(*) FROM customers WHERE region = 'East'", db))
    assert r.routes_agree is True
    assert r.decision is Decision.ACCEPT


def test_pandas_route_prompt(db):
    route = PandasSecondRoute(lambda p: "")
    sql = "SELECT COUNT(*) AS n FROM customers c JOIN orders o ON c.id = o.customer_id"
    prompt = route.build_prompt(VerifierInput("How many orders?", sql, db), run_sql(db, sql))
    assert sql not in prompt
    assert "Only these tables are relevant: customers, orders." in prompt
    assert "exactly 1 column(s),\nin this order: n." in prompt


def test_fallback_to_pandas_when_sql_route_fails(db):
    sql_route = SqlSecondRoute(stub_llm("SELECT nope FROM nowhere"))
    pandas_route = PandasSecondRoute(lambda p: "```python\nresult = pd.DataFrame({'n': [len(tables['customers'])]})\n```")
    r = Verifier(FirstSuccessfulRoute(sql_route, pandas_route)).verify(
        VerifierInput("How many customers are there?", "SELECT COUNT(*) FROM customers", db))
    assert r.routes_agree is True
    assert "tables['customers']" in r.second_route.code


def test_pandas_error_is_reported(db):
    route = PandasSecondRoute(lambda p: "```python\nresult = tables['missing']\n```")
    out = route(VerifierInput("q", "SELECT 1", db), pd.DataFrame({"x": [1]}))
    assert out.result is None and "no such table" in out.error


def test_sanity_checks_can_be_disabled(db):
    r = Verifier(sanity_checks=False).verify(VerifierInput(
        "List the names of customers in the east region.", "SELECT name FROM customers WHERE region = 'east'", db))
    assert r.checks == [] and r.decision is Decision.ACCEPT


# ------------------------------------------- regressions from BIRD gold SQL

def test_count_over_join_only_warns(db):
    # Counting joined rows is often intended ("how many orders..."), so COUNT(col) never forces a repair.
    r = Verifier().verify(VerifierInput(
        "How many orders did East customers place?",
        "SELECT COUNT(c.id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.region = 'East'", db))
    assert "join_fanout" in names(r, Severity.WARN)
    assert r.decision is not Decision.REPAIR


def test_superlatives_and_comparisons_allow_limit(db):
    for q in ["Which product is the cheapest?", "Which customer has the tallest order?",
              "Which is older, Lamp or Rug?", "What was the peak month?"]:
        r = Verifier().verify(VerifierInput(q, "SELECT name FROM products ORDER BY price_cents LIMIT 1", db))
        assert "limit_truncation" not in names(r), q


def test_year_inside_compact_date_literal(db):
    r = Verifier().verify(VerifierInput("What was the consumption in January 2012?",
                                        "SELECT SUM(consumption) FROM yearmonth WHERE date = '201201'", db))
    assert "year_missing" not in names(r)


def test_average_count_question_allows_fraction(db):
    r = Verifier().verify(VerifierInput("On average, how many orders does a customer place?",
                                        "SELECT COUNT(*) * 1.0 / 5 FROM orders", db))
    assert "count_value" not in names(r)


def test_partly_null_lookup_is_fine_but_all_null_fails(db):
    r = Verifier().verify(VerifierInput("What is the name and rating of the Desk?",
                                        "SELECT name, rating FROM products WHERE name = 'Desk'", db))
    assert "null_result" not in names(r)
    r = Verifier().verify(VerifierInput("What is the rating of the Desk?",
                                        "SELECT rating FROM products WHERE name = 'Desk'", db))
    assert "null_result" in names(r, Severity.FAIL)


def test_duplicate_column_names_do_not_crash(db):
    r = Verifier().verify(VerifierInput("List customer and product names.",
                                        "SELECT c.name, p.name FROM customers c JOIN products p ON c.id = p.id", db))
    assert not any("crashed" in c.message for c in r.checks)


def test_small_real_percentage_is_not_flagged(db):
    r = Verifier().verify(VerifierInput("What percentage of products are lamps?",
                                        "SELECT CAST(1 AS REAL) * 100 / 400 FROM products LIMIT 1", db))
    assert "percentage_scale" not in names(r)


def test_conditional_count_over_join_only_warns(db):
    r = Verifier().verify(VerifierInput(
        "How many more orders came from East customers than West customers?",
        "SELECT SUM(CASE WHEN c.region = 'East' THEN 1 ELSE 0 END) - SUM(IIF(c.region = 'West', 1, 0)) "
        "FROM customers c JOIN orders o ON c.id = o.customer_id", db))
    assert "join_fanout" in names(r, Severity.WARN)
    assert "join_fanout" not in names(r, Severity.FAIL)


def test_conditional_sum_of_duplicated_value_fails(db):
    r = Verifier().verify(VerifierInput(
        "What is the total credit of East customers who placed orders?",
        "SELECT SUM(IIF(c.region = 'East', c.credit, 0)) FROM customers c JOIN orders o ON c.id = o.customer_id", db))
    assert "join_fanout" in names(r, Severity.FAIL)


def test_like_pattern_with_wrong_date_format(db):
    r = Verifier().verify(VerifierInput("How many orders were placed in June 2012?",
                                        "SELECT COUNT(*) FROM orders WHERE order_date LIKE '201206%'", db))
    assert "date_format" in names(r, Severity.FAIL)
    r = Verifier().verify(VerifierInput("How many orders were placed in June 2012?",
                                        "SELECT COUNT(*) FROM orders WHERE order_date LIKE '2012-06%'", db))
    assert "date_format" not in names(r)


# ------------------------------------------- second route follows the rules too

def test_how_many_times_counts_as_ratio(db):
    r = Verifier().verify(VerifierInput("How many times more orders did customer 1 place than customer 2?",
                                        "SELECT SUM(IIF(customer_id = 1, 1, 0)) / SUM(IIF(customer_id = 2, 1, 0)) FROM orders", db))
    assert "integer_division" in names(r, Severity.FAIL)


def test_second_route_prompt_lists_pitfalls(db):
    sql = "SELECT COUNT(*) FROM customers"
    for route in (SqlSecondRoute(stub_llm("")), PandasSecondRoute(lambda p: "")):
        prompt = route.build_prompt(VerifierInput("How many customers?", sql, db), run_sql(db, sql))
        assert "CAST(... AS REAL)" in prompt and "exclude NULLs" in prompt


def test_second_route_that_breaks_a_rule_is_ignored(db):
    # The second query truncates the percentage to 0, so its disagreement must not count.
    bad_second = "SELECT SUM(CASE WHEN region = 'East' THEN 1 ELSE 0 END) * 100 / COUNT(*) / 100 FROM customers"
    r = Verifier(SqlSecondRoute(stub_llm(bad_second))).verify(VerifierInput(
        "What percentage of customers are in the East region?",
        "SELECT CAST(SUM(CASE WHEN region = 'East' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customers", db))
    assert r.routes_agree is None
    assert any("discarded" in c.message for c in r.checks)
    assert r.decision is Decision.ACCEPT


def test_pandas_second_route_with_empty_result_is_ignored(db):
    route = PandasSecondRoute(lambda p: "```python\nresult = pd.DataFrame({'n': []})\n```")
    r = Verifier(route).verify(VerifierInput("How many customers are there?", "SELECT COUNT(*) FROM customers", db))
    assert r.routes_agree is None
    assert r.second_route.kind == "pandas"
    assert r.decision is Decision.ACCEPT
