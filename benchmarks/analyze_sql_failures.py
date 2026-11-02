"""Separate execution failures from wrong answers and compare paired runs.

Consumes run_sql JSONL output; never sends reference SQL to a model.
Usage: python -m benchmarks.analyze_sql_failures run.jsonl --compare new.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def index_records(records):
    indexed = {}
    for record in records:
        if 'readings' in record:
            reading = record['readings'][0] if record['readings'] else {}
            correct = reading.get('correct', False)
            record = {**record,
                      'confidence': reading.get('confidence'),
                      'original_score': {'correct': correct,
                          'status': 'correct' if correct else 'incorrect'},
                      'writer': {'sql': reading.get('sql'),
                          'execution': {'error': None, 'truncated': False}
                              if reading.get('executed') is True else None}}
        key = (record['db_id'], str(record['question_id']))
        if key in indexed:
            raise ValueError(f'Duplicate example: {key}')
        score = record['original_score']
        if score.get('correct') is not None and type(score['correct']) is not bool:
            raise ValueError('correct must be true, false or null')
        indexed[key] = record
    return indexed


def failure_kind(record):
    score = record['original_score']
    if score['correct'] is None:
        return 'unscorable_reference'
    if score['correct'] is True:
        return 'correct'
    writer = record['writer']
    execution = writer.get('execution')
    if not writer.get('sql'):
        return 'no_final_sql'
    if execution and execution.get('error'):
        return 'execution_' + (execution.get('error_type') or 'error')
    if execution and execution.get('truncated'):
        return 'result_truncated'
    if score['status'] == 'incorrect' and execution:
        return 'executes_but_wrong_answer'
    if score['status'] == 'incorrect':
        return 'incorrect_answer_execution_unknown'
    return score['status']


def analyze(records):
    rows = list(index_records(records).values())
    counts = Counter(failure_kind(row) for row in rows)
    scorable = len(rows) - counts['unscorable_reference']
    by_database = {}
    for row in rows:
        by_database.setdefault(row['db_id'], Counter())[failure_kind(row)] += 1
    failures = []
    for row in rows:
        category = failure_kind(row)
        if category == 'correct':
            continue
        failures.append({'question_id': row['question_id'], 'db_id': row['db_id'],
                         'question': row.get('question'), 'category': category,
                         'sql': row['writer'].get('sql'),
                         'error': (row['writer'].get('execution') or {}).get('error'),
                         'manual_cause': None})
    high = [r for r in rows if r.get('confidence') == 'high' and r['original_score']['correct'] is not None]
    high_correct = sum(r['original_score']['correct'] is True for r in high)
    return {'attempted': len(rows), 'scorable': scorable,
            'accuracy': counts['correct'] / scorable if scorable else None,
            'failure_rate': (scorable-counts['correct']) / scorable if scorable else None,
            'categories': dict(counts),
            'by_database': {key: dict(value) for key, value in by_database.items()},
            'failures': failures,
            'high_confidence': {'answers': len(high),
                'accuracy': high_correct/len(high) if high else None,
                'coverage': len(high)/scorable if scorable else None,
                'confidently_wrong_per_scorable': (len(high)-high_correct)/scorable if scorable else None}
                if any('confidence' in r for r in rows) else None,
            'note': 'Wrong answer is observed; join/filter/aggregation causes require manual review. '
                    'Execution success is not a confidence label. Confidently-wrong rate cannot '
                    'be inferred without verifier decisions.'}


def compare(before, after):
    left, right = index_records(before), index_records(after)
    if left.keys() != right.keys():
        raise ValueError('Paired comparison requires exactly the same database/question IDs')
    transitions = Counter()
    changed = []
    for key, old in left.items():
        new = right[key]
        if old.get('question') != new.get('question'):
            raise ValueError(f'Question changed for {key}')
        a, b = old['original_score']['correct'], new['original_score']['correct']
        if a is None or b is None:
            label = 'excluded_unscorable'
        elif a and b:
            label = 'both_correct'
        elif not a and b:
            label = 'fixed'
        elif a and not b:
            label = 'regressed'
        else:
            label = 'both_wrong'
        transitions[label] += 1
        if label in {'fixed', 'regressed'}:
            changed.append({'db_id': key[0], 'question_id': key[1], 'transition': label})
    paired = len(left) - transitions['excluded_unscorable']
    return {'paired_scorable': paired, 'transitions': dict(transitions),
            'net_accuracy_change': (transitions['fixed']-transitions['regressed'])/paired if paired else None,
            'changed_examples': changed,
            'note': 'Check run manifests for identical dataset, database hashes, model digest and '
                    'budgets before interpreting this as a controlled comparison.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = read(args.run)
    output = {'baseline': analyze(records)}
    if args.compare:
        candidate = read(args.compare)
        output.update(candidate=analyze(candidate), paired=compare(records, candidate))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
