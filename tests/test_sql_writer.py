"""Role C behavioural tests using actual SQLite execution and scripted LLMs."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from veriquery.agents.sql_writer import FewShotExample, SQLWriter, extract_sql, sql_writer_node
from veriquery.sql_execution import SQLiteExecutor


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'sales.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript('CREATE TABLE sales (id INTEGER, amount REAL); INSERT INTO sales VALUES (1, 10), (2, 20), (3, 30);')
    return path


class Scripted:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize('response', ['SELECT 1', '```sql\nSELECT 1\n```', '{"sql": "SELECT 1"}'])
def test_extract_supported_formats(response):
    assert extract_sql(response) == 'SELECT 1'


@pytest.mark.parametrize('response', ['', '```sql\nSELECT 1\n```\nexplanation', '{"sql":"SELECT 1","extra":1}', '{"sql": 1}'])
def test_extract_rejects_invalid_protocol(response):
    with pytest.raises(ValueError):
        extract_sql(response)


def test_once_does_not_repair_execution_error(db):
    llm = Scripted('SELECT absent FROM sales', 'SELECT sum(amount) FROM sales')
    result = SQLWriter(llm, SQLiteExecutor(db)).run('Total amount?', mode='once')
    assert not result.ok
    assert result.model_calls == len(llm.prompts) == len(result.attempts) == 1
    assert result.execution.error


def test_repairs_with_real_error_and_preserves_context(db):
    llm = Scripted('SELECT nonexistent FROM sales', 'SELECT sum(amount) FROM sales')
    result = SQLWriter(llm, SQLiteExecutor(db)).run('Total amount?', evidence='Amounts are dollars.', data_cards='sales(id, amount)')
    assert result.ok and result.model_calls == 2
    assert result.execution.rows[0][0] == 60
    payload = json.loads(llm.prompts[1].split('\n', 1)[1])
    assert payload['question'] == 'Total amount?'
    assert payload['evidence'] == 'Amounts are dollars.'
    assert payload['schema'] == 'sales(id, amount)'
    assert payload['previous_sql'] == 'SELECT nonexistent FROM sales'
    assert 'nonexistent' in payload['repair_feedback']


def test_repair_attempt_budget_counts_model_errors(db):
    llm = Scripted(ValueError('endpoint unavailable'), '', 'SELECT missing FROM sales')
    result = SQLWriter(llm, SQLiteExecutor(db), max_repairs=2).run('Total amount?')
    assert not result.ok and result.model_calls == 3 and len(result.attempts) == 3
    assert result.attempts[0].error


def test_stop_after_first_success(db):
    llm = Scripted('SELECT count(*) FROM sales')
    result = SQLWriter(llm, SQLiteExecutor(db), max_repairs=5).run('How many sales?')
    assert result.ok and result.model_calls == 1


def test_result_truncation_is_not_success(db):
    llm = Scripted('SELECT * FROM sales')
    result = SQLWriter(llm, SQLiteExecutor(db, max_rows=1)).run('Show all sales')
    assert result.execution.truncated
    assert result.status == 'truncated' and not result.ok


def test_b_data_card_to_prompt_contract_and_train_examples(db):
    cards = SimpleNamespace(to_prompt=lambda: 'TABLE sales: id INTEGER, amount REAL')
    example = FewShotExample('Count records', 'SELECT count(*) FROM train_table', 'train_table(id)', db_id='train_db')
    same_db = FewShotExample('Other question', 'SELECT 1', 'sales(id)', db_id='sales_db')
    llm = Scripted('SELECT count(*) FROM sales')
    SQLWriter(llm, SQLiteExecutor(db), few_shots=(example, same_db)).run('How many sales?', data_cards=cards, db_id='sales_db')
    payload = json.loads(llm.prompts[0].split('\n', 1)[1])
    assert payload['schema'] == cards.to_prompt()
    assert len(payload['training_examples']) == 1
    assert payload['training_examples'][0]['db_id'] == 'train_db'
    assert 'gold_sql' not in payload
    with pytest.raises(ValueError):
        FewShotExample('Question', 'SELECT 1', 'schema', source='bird_minidev')


def test_prompt_budget_fails_before_model_call(db):
    llm = Scripted('SELECT 1')
    with pytest.raises(ValueError, match='budget'):
        SQLWriter(llm, SQLiteExecutor(db), max_prompt_chars=1000).run('Q', data_cards='x' * 1500)
    assert not llm.prompts


def test_react_observes_before_explicit_final(db):
    llm = Scripted('{"action":"query","sql":"SELECT count(*) FROM sales"}', '{"action":"final","sql":"SELECT sum(amount) FROM sales"}')
    result = SQLWriter(llm, SQLiteExecutor(db)).run('Total amount?', mode='react')
    assert result.ok and result.execution.rows[0][0] == 60
    assert [a.action for a in result.attempts] == ['query', 'final']
    assert 'Tool history:' in llm.prompts[1] and '"rows": [[3]]' in llm.prompts[1]


def test_react_query_only_never_promoted_to_final(db):
    llm = Scripted('{"action":"query","sql":"SELECT 1"}')
    result = SQLWriter(llm, SQLiteExecutor(db), max_react_steps=1).run('Q', mode='react')
    assert not result.ok and result.status == 'no_final_answer'
    assert result.sql == '' and result.execution is None


@pytest.mark.parametrize('response', ['{"action":"shell","sql":"SELECT 1"}', '{"action":"final","sql":"SELECT 1","reason":"extra"}', 'SELECT 1'])
def test_react_rejects_unadvertised_actions_and_shapes(db, response):
    result = SQLWriter(Scripted(response), SQLiteExecutor(db), max_react_steps=1).run('Q', mode='react')
    assert not result.ok and result.attempts[0].action == 'invalid'
    assert result.attempts[0].result is None


def test_react_final_truncation_is_not_success(db):
    result = SQLWriter(Scripted('{"action":"final","sql":"SELECT * FROM sales"}'), SQLiteExecutor(db, max_rows=1)).run('All', mode='react')
    assert not result.ok and result.status == 'truncated'


def test_e_repair_callback_contract(db):
    llm = Scripted('SELECT sum(amount) FROM sales')
    writer = SQLWriter(llm, SQLiteExecutor(db))
    inp = SimpleNamespace(question='Total amount?', sql='SELECT count(*) FROM sales', db_path=str(db), evidence='Dollars', data_cards='sales(id,amount)', repair_round=0)
    assert writer.repair_callback(inp, 'Return a sum, not a count.') == 'SELECT sum(amount) FROM sales'
    payload = json.loads(llm.prompts[0].split('\n', 1)[1])
    assert payload['previous_sql'] == inp.sql
    assert payload['repair_feedback'] == 'Return a sum, not a count.'


def test_e_callback_preserves_prompt_budget(db):
    llm = Scripted('SELECT 1')
    writer = SQLWriter(llm, SQLiteExecutor(db), max_prompt_chars=1000)
    inp = SimpleNamespace(question='Q', sql='SELECT 2', db_path=str(db), evidence='', data_cards='x' * 1500)
    with pytest.raises(ValueError, match='budget'):
        writer.repair_callback(inp, 'Fix')
    assert not llm.prompts


def test_node_state_contract_and_input_immutability(db):
    pytest.importorskip('pandas')
    state = {'question': 'Total?', 'db_path': str(db), 'schema_prompt': 'sales(id,amount)', 'repair_round': 2}
    original = dict(state)
    updates = sql_writer_node(state, SQLWriter(Scripted('SELECT sum(amount) AS total FROM sales'), SQLiteExecutor(db)))
    assert state == original
    assert updates['verification_status'] == 'pending'
    assert updates['result'].iloc[0]['total'] == 60
    assert updates['repair_round'] == 2


def test_node_rejects_different_database(db, tmp_path):
    with pytest.raises(ValueError, match='database'):
        sql_writer_node({'question':'Q','db_path':str(tmp_path / 'other.sqlite')}, SQLWriter(Scripted('SELECT 1'), SQLiteExecutor(db)))
