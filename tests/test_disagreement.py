import sqlite3
import pytest
from veriquery.verifier.disagreement import collect_disagreement_evidence

A = 'SELECT c.id FROM customers c JOIN orders o ON c.id=o.customer_id'
B = 'SELECT c.id FROM customers c JOIN orders o ON c.id=o.id'

@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'probe.sqlite'
    with sqlite3.connect(path) as con:
        con.executescript('''CREATE TABLE customers(id INTEGER PRIMARY KEY);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL);
        INSERT INTO customers VALUES(1),(2);
        INSERT INTO orders VALUES(10,1,5),(11,1,7),(12,2,9),(13,NULL,1);''')
    return str(path)


def test_targets_changed_join_keys_and_counts(database):
    report = collect_disagreement_evidence(database, A, B)
    assert report.differences == ['joins']
    key = {(p['table'], p['column']): p for p in report.probes}['orders', 'customer_id']
    assert key['duplicate_nonnull_rows'] == 1 and key['null_rows'] == 1
    assert key['declared_foreign_keys'][0]['to_table'] == 'customers'
    assert report.sql_calls == 3 and report.schema_reads == 1
    assert 'proof' in report.feedback()


def test_alias_renaming_no_probes(database):
    b = 'SELECT x.id FROM customers x JOIN orders y ON x.id=y.customer_id'
    report = collect_disagreement_evidence(database, A, b)
    assert not report.differences and report.sql_calls == report.schema_reads == 0


def test_aggregation_probes_shared_join_grain_with_budget(database):
    base = ' FROM customers c JOIN orders o ON c.id=o.customer_id'
    report = collect_disagreement_evidence(database, 'SELECT SUM(o.total)'+base,
                                          'SELECT AVG(o.total)'+base, max_probes=1)
    assert report.differences == ['aggregation'] and report.sql_calls == 1
    assert any('budget' in note for note in report.notes)


def test_filter_only_no_probes(database):
    report = collect_disagreement_evidence(database, 'SELECT id FROM orders WHERE total>5',
                                          'SELECT id FROM orders WHERE total>7')
    assert report.differences == ['filters'] and report.sql_calls == report.schema_reads == 0
    assert 'not implemented' in ' '.join(report.notes)


@pytest.mark.parametrize('sql', ['DELETE FROM orders', 'SELECT * FROM orders; DROP TABLE orders',
    'WITH x AS (SELECT * FROM orders) SELECT * FROM x',
    'SELECT * FROM orders a JOIN orders b ON a.id=b.id',
    'SELECT * FROM customers c JOIN orders o ON c.id=o.customer_id OR c.id=o.id'])
def test_unsupported_and_mutation_never_probed(database, sql):
    report = collect_disagreement_evidence(database, sql, 'SELECT id FROM orders')
    assert report.sql_calls == report.schema_reads == 0 and report.notes
    with sqlite3.connect(database) as con:
        assert con.execute('SELECT COUNT(*) FROM orders').fetchone() == (4,)


def test_zero_budget_and_missing_column(database):
    b = 'SELECT c.id FROM customers c JOIN orders o ON c.id=o.missing'
    zero = collect_disagreement_evidence(database, A, b, max_probes=0)
    assert zero.schema_reads == zero.sql_calls == 0
    report = collect_disagreement_evidence(database, A, b)
    assert any('Unknown key' in n for n in report.notes)
    assert all(p['column'] != 'missing' for p in report.probes)


def test_failed_probe_costed_without_invented_evidence(database, monkeypatch):
    from veriquery.sql_execution import QueryResult, SQLiteExecutor
    monkeypatch.setattr(SQLiteExecutor, 'execute', lambda self, sql: QueryResult(sql, error='timed out', error_type='timeout'))
    report = collect_disagreement_evidence(database, A, B, max_probes=1)
    assert report.sql_calls == 1 and report.probes[0]['error'] == 'timed out'
    assert 'nonnull_rows' not in report.probes[0]
