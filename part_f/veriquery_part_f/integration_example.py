"""Integration adapter: call after teammate C's SQL execution and teammate E's checks."""
import pandas as pd
from analyst import analyze
from reporter import build_report


def run_part_f(question: str, sql_rows: list[dict], sql: str, verifier_result: dict) -> dict:
    # SQL is supplied for display only; this module never executes it.
    analysis = analyze(pd.DataFrame(sql_rows), question)
    return build_report(analysis, sql=sql, verification=verifier_result)

if __name__ == '__main__':
    rows = [{'region': 'East', 'sales': 100}, {'region': 'West', 'sales': 200}]
    report = run_part_f('total sales by region', rows, 'SELECT region, sales FROM orders', {'status': 'passed'})
    print(report['answer'], report['confidence'])
