import json
import sqlite3
import pytest
from training.evaluate_sql_pipeline import LocalPipelineClient, load_progress, summarize
from veriquery.agents.orchestrator import Config, VeriQuery


def test_client_sampling_budget_and_usage():
    calls=[]
    def backend(messages,limit,temp,seed):
        calls.append((messages,limit,temp,seed));return 'SELECT 1',5,3
    client=LocalPipelineClient(backend,'test',max_calls=3)
    checker=client.with_temperature(.3)
    with client.agent('writer'):
        assert client('question')=='SELECT 1'
    with checker.agent('verifier'):
        checker('question')
    client.chat([{'role':'user','content':'question'}],tag='writer')
    assert calls[0][3]==calls[2][3]
    assert checker.stats.calls==1 and client.stats.calls==2
    assert checker.budget is client.budget
    assert calls[1][2]==.3
    with pytest.raises(ValueError,match='budget exhausted'):
        checker('fourth')
    client.reset_question_budget();checker('fourth')
    assert client.usage()['by_agent']['writer']['calls']==2


def test_failed_call_is_counted():
    def fail(*args):raise ValueError('failure')
    client=LocalPipelineClient(fail,'test')
    with pytest.raises(ValueError,match='failure'):client('question')
    assert client.stats.calls==1 and client.budget['calls']==1


def test_resume_requires_identical_manifest_and_order(tmp_path):
    out=tmp_path/'run';manifest={'selected_ids':[{'db_id':'a','question_id':1},{'db_id':'a','question_id':2}]}
    assert load_progress(out,manifest)==[]
    record={'db_id':'a','question_id':1,'score':{'correct':True}}
    (out/'traces.jsonl').write_text(json.dumps(record)+'\n'+ '{"interrupted"')
    assert load_progress(out,manifest,True)==[record]
    assert (out/'traces.jsonl').read_text().endswith('\n')
    with pytest.raises(ValueError,match='identity mismatch'):load_progress(out,{'selected_ids':[]},True)
    record['question_id']=2;(out/'traces.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='prefix'):load_progress(out,manifest,True)


def test_summary_excludes_unscorable_references_and_reports_coverage():
    rows=[{'score':{'correct':x},'report_confidence':label} for x,label in [(True,'high'),(False,'high'),(False,'low'),(None,'high')]]
    summary=summarize(rows)
    assert summary['attempted']==4 and summary['scorable']==3
    assert summary['high_confidence_accuracy']==.5
    assert summary['high_confidence_coverage']==2/3
    assert summary['confidently_wrong_rate']==1/3
    assert summarize([])['accuracy'] is None


@pytest.mark.parametrize('mode',['generic','structure','evidence'])
def test_client_works_with_real_pipeline_without_gold(tmp_path,mode):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE items(id INTEGER PRIMARY KEY, amount REAL)')
        conn.execute('INSERT INTO items VALUES (1,5)')
    prompts=[]
    def backend(messages,*args):
        prompts.extend(m['content'] for m in messages)
        return 'SELECT SUM(amount) AS total FROM items',20,10
    client=LocalPipelineClient(backend,'test')
    pipeline=VeriQuery(client,Config(use_clarifier=False,second_route='sql',n_candidates=3,
        summarise=False,notebook_dir=str(tmp_path/'notebook'),disagreement_mode=mode))
    state=pipeline.ask('What is the total amount?',db)
    assert state['answers'][0]['sql']=='SELECT SUM(amount) AS total FROM items'
    assert state['usage']['total']['calls']==client.budget['calls']
    assert len(prompts)>=4


def test_truncated_generation_accounts_for_consumed_tokens():
    from training.evaluate_sql_pipeline import GenerationFailure
    def backend(*args):
        raise GenerationFailure('truncated',31,1024)
    client=LocalPipelineClient(backend,'test')
    with pytest.raises(GenerationFailure):client('question')
    assert client.stats.prompt_tokens==31
    assert client.stats.completion_tokens==1024
