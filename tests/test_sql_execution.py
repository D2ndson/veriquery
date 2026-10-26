import sqlite3
import time

import pytest

from veriquery.sql_execution import SQLiteExecutor


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "business ?#.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript('''
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE sales(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount REAL);
        INSERT INTO customers VALUES(1, 'Ada'),(2, 'Bo');
        INSERT INTO sales VALUES(1,1,10.0),(2,1,15.0),(3,2,30.0);
        CREATE TABLE "odd""name" ("a space" TEXT);
        ''')
    return path


def test_select_and_cte(database):
    result = SQLiteExecutor(database).execute('''WITH totals AS (
        SELECT customer_id, SUM(amount) total FROM sales GROUP BY customer_id
    ) SELECT name,total FROM totals JOIN customers ON customer_id=customers.id ORDER BY name''')
    assert result.ok
    assert result.columns == ["name", "total"]
    assert result.rows == [("Ada", 25.0), ("Bo", 30.0)]
    assert not result.truncated


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales", "SELECT 1; DROP TABLE sales", "ATTACH DATABASE ':memory:' AS other",
    "PRAGMA writable_schema=ON", "CREATE TABLE evil(a)", "BEGIN", "VACUUM",
    "WITH x AS (DELETE FROM sales RETURNING *) SELECT * FROM x",
    "SELECT * INTO copied FROM sales", "SELECT * FROM pragma_table_info('sales')",
    "SELECT load_extension('/tmp/evil')", "SELECT writefile('/tmp/evil','bad')",
])
def test_mutations_and_escape_routes_blocked(database, sql):
    result = SQLiteExecutor(database).execute(sql)
    assert not result.ok
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT count(*) FROM sales").fetchone() == (3,)


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    result = SQLiteExecutor(path).execute("SELECT 1")
    assert result.error_type == "database"
    assert not path.exists()


def test_result_cap_is_explicit(database):
    result = SQLiteExecutor(database, max_rows=2).execute("SELECT id FROM sales ORDER BY id")
    assert result.ok and result.truncated
    assert result.rows == [(1,), (2,)]
    assert not SQLiteExecutor(database, max_rows=3).execute("SELECT id FROM sales").truncated


def test_error_is_repairable_and_no_partial_rows(database):
    result = SQLiteExecutor(database).execute("SELECT missing_column FROM sales")
    assert result.error_type == "execution"
    assert "missing_column" in result.error
    assert result.rows == []


def test_timeout(database):
    start = time.monotonic()
    result = SQLiteExecutor(database, timeout_seconds=.02).execute('''
        WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n)
        SELECT SUM(x) FROM n
    ''')
    assert result.error_type == "timeout"
    assert result.rows == []
    assert time.monotonic() - start < 2


def test_schema_cards_quoted_names_and_foreign_keys(database):
    cards = SQLiteExecutor(database).schema_cards()
    assert [c["table"] for c in cards] == ['customers', 'odd"name', 'sales']
    assert cards[1]["columns"][0]["name"] == "a space"
    assert cards[2]["foreign_keys"] == [{"from":"customer_id", "to_table":"customers", "to_column":"id"}]
    assert "Ada" not in repr(cards)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout(database, timeout):
    with pytest.raises(ValueError):
        SQLiteExecutor(database, timeout_seconds=timeout)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_row_limit(database, limit):
    with pytest.raises(ValueError):
        SQLiteExecutor(database, max_rows=limit)


def test_union_and_literals_are_safe(database):
    result = SQLiteExecutor(database).execute("SELECT 'DELETE FROM sales; --' UNION SELECT 'DROP TABLE sales'")
    assert result.ok and len(result.rows) == 2


def test_expensive_value_is_bounded(database):
    result = SQLiteExecutor(database).execute("SELECT zeroblob(999999999)")
    assert not result.ok
    assert "too big" in result.error


def test_accumulated_values_have_byte_cap(database, monkeypatch):
    import veriquery.sql_execution as module
    monkeypatch.setattr(module, "_MAX_RESULT_BYTES", 512)
    result = SQLiteExecutor(database).execute("SELECT zeroblob(200) FROM sales")
    assert result.ok and result.truncated
    assert len(result.rows) == 1
