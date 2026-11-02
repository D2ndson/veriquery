"""Runner contract tests using synthetic SQLite and scripted outputs, not BIRD scores."""
import json
import sqlite3

import pytest

from benchmarks.bird import BirdExample
from benchmarks.run_sql import evaluate, load_few_shots, summarize


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []
        self.stats = {'calls': 0, 'prompt_tokens': 0}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.stats['calls'] += 1
        self.stats['prompt_tokens'] += 7
        return next(self.outputs)


@pytest.fixture
def benchmark(tmp_path):
    folder = tmp_path / 'synthetic'
    folder.mkdir()
    with sqlite3.connect(folder / 'synthetic.sqlite') as conn:
        conn.execute('CREATE TABLE items (value INTEGER)')
        conn.executemany('INSERT INTO items VALUES (?)', [(1,), (2,)])
    example = BirdExample(1, 'synthetic', 'List the item values.', '',
                          'SELECT value FROM items /* GOLD_ONLY_SECRET */', 'simple')
    return tmp_path, example


@pytest.mark.parametrize(('mode', 'outputs'), [
    ('once', ['SELECT value FROM items']),
    ('repair', ['SELECT missing FROM items', 'SELECT value FROM items']),
    ('react', [json.dumps({'action': 'query', 'sql': 'SELECT value FROM items'}),
               json.dumps({'action': 'final', 'sql': 'SELECT value FROM items'})]),
])
def test_evaluate_all_modes_without_gold_leakage(benchmark, mode, outputs):
    root, example = benchmark
    llm = ScriptedLLM(outputs)
    observed = []
    records = evaluate([example], root, llm, mode=mode, on_record=observed.append)
    assert records == observed
    record = records[0]
    assert record['original_score']['correct'] is True
    assert record['bag_diagnostic']['correct'] is True
    assert record['writer']['status'] == 'success'
    assert record['writer']['model_calls'] == len(outputs)
    assert record['usage'] == {'calls': len(outputs), 'prompt_tokens': 7 * len(outputs)}
    assert record['cost_usd'] is None
    assert all('GOLD_ONLY_SECRET' not in str(p) for p in llm.prompts)


def test_corrected_labels_rescore_same_generation(benchmark, tmp_path):
    root, original = benchmark
    correction = tmp_path / 'corrections.json'
    correction.write_text(json.dumps([{'question_id': 1, 'db_id': 'synthetic',
        'corrected_sql': 'SELECT 1 /* CORRECTED_GOLD_SECRET */', 'source': 'manual audit'}]))
    llm = ScriptedLLM(['SELECT 1'])
    result = evaluate([original], root, llm, mode='once', corrections=correction)[0]
    assert result['original_score']['correct'] is False
    assert result['corrected_score']['correct'] is True
    assert result['label_source'] == 'manual audit'
    assert len(llm.prompts) == 1
    assert 'GOLD_ONLY_SECRET' in original.gold_sql
    assert all('CORRECTED_GOLD_SECRET' not in str(p) for p in llm.prompts)


def test_gold_error_separate_from_model_failure(benchmark):
    root, example = benchmark
    bad = BirdExample(2, example.db_id, example.question, '', 'SELECT invalid', 'simple')
    llm = ScriptedLLM(['SELECT value FROM items', 'SELECT value FROM items'])
    records = evaluate([example, bad], root, llm, mode='once')
    report = summarize(records)
    assert report['attempted'] == 2
    assert report['scorable'] == 1
    assert report['unscorable'] == 1
    assert report['correct'] == 1
    assert report['execution_accuracy'] == 1
    assert report['correct_per_attempted'] == .5
    assert report['total_model_calls'] == 2
    assert records[1]['original_score']['correct'] is None


def test_summary_denominators_and_empty():
    def record(correct, status):
        return {'original_score': {'correct': correct, 'status': status},
                'writer': {'status': 'failed' if correct is False else 'success',
                           'model_calls': 1, 'elapsed_seconds': .5}}
    summary = summarize([record(True, 'correct'), record(False, 'incorrect'),
                         record(None, 'unscorable_gold_error')])
    assert summary['execution_accuracy'] == .5
    assert summary['correct_per_attempted'] == pytest.approx(1 / 3)
    assert summary['elapsed_seconds'] == 1.5
    assert summarize([])['execution_accuracy'] is None
    assert summarize([])['correct_per_attempted'] is None
    assert summarize([record(True, 'correct')], 'corrected_score')['attempted'] == 0


def test_load_few_shots_validated_provenance(tmp_path):
    path = tmp_path / 'shots.json'
    row = {'question': 'Count items', 'sql': 'SELECT COUNT(*) FROM items',
           'schema': 'items(value INTEGER)', 'db_id': 'training_database'}
    path.write_text(json.dumps({'source': 'bird_train', 'examples': [row]}))
    assert load_few_shots(path)[0].source == 'bird_train'
    assert load_few_shots(None) == ()
    path.write_text(json.dumps({'source': 'bird_dev', 'examples': [row]}))
    with pytest.raises(ValueError):
        load_few_shots(path)
    row['source'] = 'bird_dev'
    path.write_text(json.dumps({'source': 'bird_train', 'examples': [row]}))
    with pytest.raises(ValueError):
        load_few_shots(path)


def test_cli_subset_with_full_dataset_corrections(benchmark, tmp_path, monkeypatch):
    from benchmarks import run_sql

    root, example = benchmark
    data = tmp_path / 'data.json'
    data.write_text(json.dumps([{'question_id': n, 'db_id': example.db_id,
        'question': example.question, 'SQL': example.gold_sql} for n in (1, 2)]))
    corrections = tmp_path / 'overlay.json'
    corrections.write_text(json.dumps([{'question_id': n, 'db_id': example.db_id,
        'corrected_sql': 'SELECT value FROM items'} for n in (1, 2)]))
    llm = ScriptedLLM(['SELECT value FROM items'])
    llm.close = lambda: None
    monkeypatch.setattr(run_sql, 'CachedOllama', lambda *args, **kwargs: llm)
    monkeypatch.setattr(run_sql, 'preflight_model', lambda *args: {'name': 'scripted:latest', 'digest': 'synthetic-digest'})
    output = tmp_path / 'run'
    run_sql.main(['--data', str(data), '--db-root', str(root), '--model', 'scripted',
        '--output', str(output), '--mode', 'once', '--size', '1',
        '--corrections', str(corrections)])
    manifest = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((output / 'once_summary.json').read_text())
    assert manifest['model']['digest'] == 'synthetic-digest'
    assert len(manifest['database_sha256']['synthetic']) == 64
    assert 'benchmarks/run_sql.py' in manifest['code_sha256']
    assert manifest['source_rows'] == 2
    assert manifest['selected_count'] == 1
    assert summary['original']['correct'] == 1
    assert summary['corrected']['correct'] == 1
    assert len(llm.prompts) == 1
    assert json.loads((output / 'once_failures.json').read_text()) == []


@pytest.mark.parametrize('models, expected', [
    ([{'name': 'test:latest', 'digest': 'abc'}], 'abc'),
    ([{'name': 'different:latest', 'digest': 'abc'}], None),
    ([{'name': 'test:latest'}], None),
])
def test_model_preflight(monkeypatch, models, expected):
    import io
    from benchmarks import run_sql
    def respond(url, timeout):
        assert url == 'http://localhost:11434/api/tags'
        assert timeout == 10
        return io.BytesIO(json.dumps({'models': models}).encode())
    monkeypatch.setattr(run_sql.urllib.request, 'urlopen', respond)
    if expected:
        assert run_sql.preflight_model('http://localhost:11434/', 'test')['digest'] == expected
    else:
        with pytest.raises(RuntimeError, match='not installed'):
            run_sql.preflight_model('http://localhost:11434', 'test')


def test_model_preflight_unreachable(monkeypatch):
    from benchmarks import run_sql
    def fail(*args, **kwargs):
        raise OSError('connection refused')
    monkeypatch.setattr(run_sql.urllib.request, 'urlopen', fail)
    with pytest.raises(RuntimeError, match='preflight failed'):
        run_sql.preflight_model('http://localhost:11434', 'test')
