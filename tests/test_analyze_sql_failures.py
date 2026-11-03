import pytest

from benchmarks.analyze_sql_failures import analyze, compare


def row(qid, correct, *, error=None, truncated=False):
    return {'question_id': qid, 'db_id': 'sales', 'question': f'Question {qid}',
            'original_score': {'correct': correct, 'status': 'incorrect' if correct is False else 'correct'},
            'writer': {'sql': 'SELECT 1', 'execution': {'error': error,
                       'error_type': 'execution' if error else None, 'truncated': truncated}}}


def test_distinguishes_silent_wrong_answer_from_execution_failure():
    report = analyze([row(1, True), row(2, False), row(3, False, error='missing column'),
                      row(4, None), row(5, False, truncated=True)])
    assert report['accuracy'] == .25
    assert report['failure_rate'] == .75
    assert report['scorable'] == 4
    assert report['categories']['executes_but_wrong_answer'] == 1
    assert report['categories']['execution_execution'] == 1
    assert report['categories']['result_truncated'] == 1


def test_paired_comparison_counts_regressions_and_excludes_unscorable():
    result = compare([row(1, False), row(2, True), row(3, None), row(4, False)],
                     [row(4, True), row(3, True), row(2, False), row(1, True)])
    assert result['transitions'] == {'fixed': 2, 'regressed': 1, 'excluded_unscorable': 1}
    assert result['net_accuracy_change'] == pytest.approx(1/3)


def test_refuses_mismatched_or_duplicated_examples():
    with pytest.raises(ValueError, match='same'):
        compare([row(1, False)], [row(2, True)])
    with pytest.raises(ValueError, match='Duplicate'):
        analyze([row(1, False), row(1, True)])


def test_pipeline_logs_measure_accuracy_and_coverage_without_guessing_execution():
    rows = [{'question_id': i, 'db_id':'x', 'question':str(i), 'readings':[
        {'sql':'SELECT 1', 'correct':i==0, 'confidence':'high' if i<2 else 'low'}]}
        for i in range(3)]
    out=analyze(rows)
    assert out['high_confidence']['accuracy']==.5
    assert out['high_confidence']['coverage']==pytest.approx(2/3)
    assert out['high_confidence']['confidently_wrong_per_scorable']==pytest.approx(1/3)
    assert out['categories']['incorrect_answer_execution_unknown']==2
