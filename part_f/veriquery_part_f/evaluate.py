"""Small local regression evaluator, NOT official InfiAgent-DABench scoring.
Cases JSONL: {"question":"...", "expected_value": 123.0, "expected_operation":"sum"}
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from analyst import analyze


def evaluate(csv_path: str, cases_path: str, output_path: str = 'results.csv'):
    df = pd.read_csv(csv_path)
    rows = []
    for i, line in enumerate(Path(cases_path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        result = analyze(df, case['question'])
        value = None
        if result.table.shape == (1, 2) and 'value' in result.table:
            value = float(result.table['value'].iloc[0])
        expected = case.get('expected_value')
        correct = expected is not None and value is not None and abs(value - float(expected)) <= float(case.get('tolerance', 1e-6))
        rows.append({'case': i, 'question': case['question'], 'expected': expected, 'actual': value, 'correct': correct, 'warnings': '; '.join(result.warnings)})
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f'Local test accuracy: {results.correct.mean():.1%}' if len(results) else 'No test cases')
    print(f'Saved {output_path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--out', default='results.csv')
    a = p.parse_args()
    evaluate(a.csv, a.cases, a.out)
