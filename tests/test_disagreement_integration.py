import json
import sqlite3
import pandas as pd
import pytest
from veriquery.verifier.models import Decision, RouteResult, Severity, VerifierInput
from veriquery.verifier.verifier import Verifier
from veriquery.sql_execution import SQLiteExecutor

A = 'SELECT c.id FROM customers c JOIN orders o ON c.id=o.customer_id'
B = 'SELECT c.id FROM customers c JOIN orders o ON c.id=o.id'

@pytest.fixture
def db(tmp_path):
    p = tmp_path/'db.sqlite'
    with sqlite3.connect(p) as con:
        con.executescript('CREATE TABLE customers(id INTEGER); CREATE TABLE orders(id INTEGER, customer_id INTEGER); INSERT INTO customers VALUES(1); INSERT INTO orders VALUES(2,1),(3,1);')
    return str(p)


def verify(db, mode, *, repair_round=0, agree=False, kind='sql'):
    inp = VerifierInput(question='List customers', sql=A, db_path=db,
                        result=pd.DataFrame({'id':[1]}), repair_round=repair_round)
    route = lambda *_: RouteResult(B, pd.DataFrame({'id':[1 if agree else 2]}), kind=kind)
    return Verifier(second_route=route, sanity_checks=False, disagreement_mode=mode).verify(inp)


@pytest.mark.parametrize('mode',['generic','structure'])
def test_no_extra_database_calls_and_unchanged_confidence(db, monkeypatch, mode):
    def forbid(*_):
        raise AssertionError('Unexpected diagnostic database access')
    monkeypatch.setattr(SQLiteExecutor, 'schema_cards', forbid)
    monkeypatch.setattr(SQLiteExecutor, 'execute', forbid)
    report=verify(db, mode)
    assert report.decision is Decision.REPAIR and report.confidence == 0.3
    checks=[c for c in report.checks if c.name=='disagreement_diagnostics']
    if mode=='generic':
        assert not checks and 'Targeted disagreement' not in report.repair_feedback
    else:
        assert checks[0].severity is Severity.INFO
        assert checks[0].evidence['sql_calls']==checks[0].evidence['schema_reads']==0
        assert checks[0].evidence['differences']==['joins']


def test_evidence_appends_diagnostics_without_changing_decision(db):
    report=verify(db,'evidence')
    check=next(c for c in report.checks if c.name=='disagreement_diagnostics')
    assert check.evidence['sql_calls']==3 and check.evidence['schema_reads']==1
    assert report.decision is Decision.REPAIR and report.confidence==0.3
    assert 'duplicate_nonnull_rows' in report.repair_feedback
    assert all('gold' not in k for k in check.evidence)


@pytest.mark.parametrize('kwargs',[{'agree':True},{'repair_round':2},{'kind':'pandas'}])
def test_no_diagnostics_without_repairable_sql_disagreement(db,kwargs):
    report=verify(db,'evidence',**kwargs)
    assert not any(c.name=='disagreement_diagnostics' for c in report.checks)


def test_orchestrator_config_threads_mode(tmp_path):
    from veriquery.agents.orchestrator import Config, VeriQuery
    v=VeriQuery(lambda _: '', Config(disagreement_mode='structure', notebook_dir=str(tmp_path/'notebook')))
    assert v.verifier.disagreement_mode=='structure'


def test_cli_threads_mode_to_child_process(tmp_path,monkeypatch):
    from benchmarks import eval_pipeline as ep
    monkeypatch.setattr(ep,'load_questions',lambda _: [])
    monkeypatch.setattr(ep,'fixed_subset',lambda *args: [])
    calls=[]
    class Child:
        def __init__(self,args): calls.append(args)
        def wait(self): return 0
    monkeypatch.setattr(ep.subprocess,'Popen',Child)
    ep.main(['--data',str(tmp_path),'--out',str(tmp_path/'out'),'--disagreement-mode','evidence'])
    assert calls[0][calls[0].index('--disagreement-mode')+1]=='evidence'
    assert json.loads((tmp_path/'out/config.json').read_text())['disagreement_mode']=='evidence'


def test_summary_accounts_diagnostics(tmp_path):
    from benchmarks.eval_pipeline import summarize
    row={'question_id':1,'difficulty':'simple','clarifier':'proceed','error':None,'seconds':2,'usage':{'calls':3},
         'readings':[{'correct':False,'confidence':'low','verification_rounds':[{'checks':[{
             'name':'disagreement_diagnostics','evidence':{'sql_calls':2,'schema_reads':1,'elapsed_seconds':.5}}]}]}]}
    (tmp_path/'shard_0.jsonl').write_text(json.dumps(row)+'\n')
    assert summarize(tmp_path)['disagreement_diagnostics']=={'rounds':1,'sql_calls':2,'schema_reads':1,'seconds':.5}
