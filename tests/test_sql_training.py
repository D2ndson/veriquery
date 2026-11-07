import sqlite3
import pytest
from training.prepare_sql_data import prepare
from training.train_sql_lora import encode_example


def test_training_separates_databases_and_rejects_evaluation_overlap(tmp_path):
    rows=[]
    for name in ['a','b','c']:
        folder=tmp_path/name
        folder.mkdir()
        with sqlite3.connect(folder/f'{name}.sqlite') as conn:
            conn.execute('CREATE TABLE sales(amount REAL)')
        rows.append({'db_id':name,'question':'Total?', 'SQL':'SELECT sum(amount) FROM sales'})
    splits, manifest=prepare(rows,tmp_path,{'evaluation'})
    assert {r['db_id'] for r in splits['train']}.isdisjoint(r['db_id'] for r in splits['validation'])
    assert sum(map(len,splits.values())) == 3
    assert all('sum(amount)' not in r['prompt'] for split in splits.values() for r in split)
    with pytest.raises(ValueError,match='overlap'):
        prepare(rows,tmp_path,{'a'})


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return [1,2,3] if len(messages)==1 else [1,2,3,4,5]


def test_only_answer_tokens_are_trained_and_overlength_is_dropped():
    row={'prompt':'question','completion':'sql'}
    assert encode_example(Tokenizer(),row,5)['labels']==[-100,-100,-100,4,5]
    assert encode_example(Tokenizer(),row,4) is None


def test_revision_requires_immutable_commit():
    from training.train_sql_lora import validate_revision
    assert validate_revision('A'*40) == 'a'*40
    for revision in ['main', 'abc', '', 'x'*40]:
        with pytest.raises(ValueError, match='commit'):
            validate_revision(revision)


def test_resume_requires_matching_experiment_and_local_checkpoint(tmp_path):
    import json
    from training.train_sql_lora import validate_resume
    output = tmp_path/'run'
    validate_resume(output, None, {})
    output.mkdir()
    with pytest.raises(ValueError, match='new output'):
        validate_resume(output, None, {})
    checkpoint = output/'checkpoint-100'
    checkpoint.mkdir()
    (checkpoint/'trainer_state.json').write_text('{}')
    (output/'experiment.json').write_text(json.dumps({'model': 'model', 'revision': 'a'*40}))
    validate_resume(output, checkpoint, {'model': 'model', 'revision': 'a'*40})
    with pytest.raises(ValueError, match='mismatch'):
        validate_resume(output, checkpoint, {'revision': 'b'*40})
    with pytest.raises(ValueError, match='inside'):
        validate_resume(tmp_path/'other', checkpoint, {})


def test_adapter_must_match_training_base_and_revision(tmp_path):
    import json
    from training.train_sql_lora import validate_adapter
    adapter = tmp_path/'adapter'
    adapter.mkdir()
    (tmp_path/'experiment.json').write_text(json.dumps({'model':'qwen', 'revision':'a'*40}))
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'qwen'}))
    validate_adapter(adapter, 'qwen', 'a'*40)
    with pytest.raises(ValueError, match='revision'):
        validate_adapter(adapter, 'qwen', 'b'*40)
    with pytest.raises(ValueError, match='model'):
        validate_adapter(adapter, 'other', 'a'*40)
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'other'}))
    with pytest.raises(ValueError, match='configuration'):
        validate_adapter(adapter, 'qwen', 'a'*40)


def test_metadata_preparation_preserves_keys_and_labels_validation_scope(tmp_path):
    from training.prepare_sql_data import metadata_cards
    schema = {'table_names_original':['sales','items'],
              'column_names_original':[[-1,'*'],[0,'id'],[1,'sale_id']],
              'column_types':['text','number','number'], 'primary_keys':[[1]], 'foreign_keys':[[2,1]]}
    cards=metadata_cards(schema)
    assert cards[0]['columns'][0]['primary_key'] is True
    assert cards[1]['foreign_keys']==[{'from':'sale_id','to_table':'sales','to_column':'id'}]
    rows=[{'db_id':name,'question':'Count?', 'SQL':'SELECT count(*) FROM sales'} for name in ['a','b']]
    splits,manifest=prepare(rows,None,{'heldout'},schemas={'a':schema,'b':schema})
    assert manifest['reference_validation']=='syntax_and_safety_only'
    assert sum(map(len,splits.values()))==2


def test_extracts_nested_bird_database_archive(tmp_path):
    from training.extract_databases import extract_databases
    import io, zipfile
    inner=io.BytesIO()
    with zipfile.ZipFile(inner,'w') as z:
        z.writestr('dev_databases/sales/sales.sqlite',b'sqlite fixture')
    outer=tmp_path/'dev.zip'
    with zipfile.ZipFile(outer,'w') as z:
        z.writestr('dev_20240627/dev_databases.zip',inner.getvalue())
    paths=extract_databases(outer,tmp_path/'db')
    assert paths==[tmp_path/'db/sales/sales.sqlite']
    assert paths[0].read_bytes()==b'sqlite fixture'
    with pytest.raises(FileExistsError):
        extract_databases(outer,tmp_path/'db')


def test_smoke_scans_all_rows_and_uses_longest_fitting_examples():
    from training.train_sql_lora import select_training_examples
    class VariableTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            prefix = [1, 2]
            return prefix if len(messages) == 1 else prefix + [3]*int(messages[1]['content'])
    rows = [{'prompt': 'q', 'completion': str(n)} for n in [1, 2, 7, 20, 6]]
    selected, counts = select_training_examples(VariableTokenizer(), rows, 10, smoke=True)
    assert [len(item['input_ids']) for item in selected] == [9, 8]
    assert counts == {'source': 5, 'examined': 5, 'eligible': 4, 'retained': 2,
                      'overlength': 1, 'selected_token_lengths': [9, 8]}
    full, counts = select_training_examples(VariableTokenizer(), rows, 10)
    assert [len(item['input_ids']) for item in full] == [3, 4, 9, 8]


def test_runtime_budget_saves_at_step_boundary_without_changing_horizon():
    from types import SimpleNamespace
    from training.train_sql_lora import RuntimeBudget
    now = [10.0]
    budget = RuntimeBudget(60, clock=lambda: now[0])
    state = SimpleNamespace(global_step=3, max_steps=343)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    now[0] = 69
    assert budget.on_step_end(None, state, control) is control
    assert not control.should_training_stop
    now[0] = 70
    budget.on_step_end(None, state, control)
    assert control.should_save and control.should_training_stop
    assert state.max_steps == 343
    assert budget.stop_reason == 'runtime_budget'


def test_signal_waits_for_step_boundary_and_completed_run_is_not_paused():
    import signal
    from types import SimpleNamespace
    from training.train_sql_lora import RuntimeBudget
    budget = RuntimeBudget()
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    budget.request_stop(signal.SIGUSR1)
    assert not control.should_save
    budget.on_step_end(None, SimpleNamespace(global_step=25, max_steps=343), control)
    assert control.should_save and control.should_training_stop
    assert budget.stop_reason == f'signal_{signal.SIGUSR1}'
    complete = RuntimeBudget(1, clock=lambda: 0)
    complete.request_stop(signal.SIGTERM)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    complete.on_step_end(None, SimpleNamespace(global_step=343, max_steps=343), control)
    assert complete.stop_reason is None
    assert not control.should_save


def test_runtime_resume_still_rejects_changed_horizon(tmp_path):
    import json
    from training.train_sql_lora import validate_resume
    checkpoint = tmp_path/'checkpoint-25'
    checkpoint.mkdir()
    (checkpoint/'trainer_state.json').write_text('{}')
    identity = {'model': 'qwen', 'revision': 'a'*40, 'epochs': 1.0,
                'max_steps': -1, 'max_length': 4096, 'smoke': False}
    (tmp_path/'experiment.json').write_text(json.dumps(identity))
    validate_resume(tmp_path, checkpoint, identity)  # Legacy full-run manifest remains valid.
    for field, value in [('max_steps', 50), ('epochs', 2.0), ('max_length', 2048)]:
        with pytest.raises(ValueError, match=field):
            validate_resume(tmp_path, checkpoint, {**identity, field: value})
