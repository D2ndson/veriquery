"""Reproducible Role C BIRD runs: python -m benchmarks.run_sql --help."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from benchmarks.bird import load_bird, fixed_subset, database_path, score_prediction, apply_corrections
from veriquery.agents.sql_writer import SQLWriter, FewShotExample
from veriquery.sql_client import CachedOllama
from veriquery.sql_execution import SQLiteExecutor


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def preflight_model(base_url, model):
    """Verify installed Ollama weights before creating run artifacts."""
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/api/tags', timeout=10) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError('model listing exceeds 4 MB')
        payload = json.loads(raw)
        models = payload.get('models') if isinstance(payload, dict) else None
        if not isinstance(models, list) or any(not isinstance(x, dict) for x in models):
            raise ValueError('endpoint returned an invalid model listing')
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Ollama preflight failed at {base_url}: {exc}') from exc
    requested = model if ':' in model.rsplit('/', 1)[-1] else model + ':latest'
    matches = [x for x in models if x.get('name') == requested or x.get('model') == requested]
    if len(matches) != 1 or not isinstance(matches[0].get('digest'), str) or not matches[0]['digest']:
        raise RuntimeError(f'Ollama model {requested!r} is not installed with a unique digest; install it before running')
    return {'name': requested, 'digest': matches[0]['digest']}


def code_fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = ['benchmarks/bird.py', 'benchmarks/run_sql.py', 'veriquery/agents/sql_writer.py',
             'veriquery/sql_execution.py', 'veriquery/sql_client.py']
    return {name: digest(root / name) for name in paths}


def load_few_shots(path):
    if path is None:
        return ()
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict) or obj.get('source') != 'bird_train':
        raise ValueError('Few-shot file must declare source=bird_train')
    rows = obj.get('examples')
    if not isinstance(rows, list) or any(not isinstance(x, dict) for x in rows):
        raise ValueError('Few-shot examples must be an array of objects')
    if any(x.get('source', 'bird_train') != 'bird_train' for x in rows):
        raise ValueError('Every few-shot example must come from bird_train')
    try:
        return tuple(FewShotExample(**dict(x, source='bird_train')) for x in rows)
    except TypeError as exc:
        raise ValueError(f'Invalid few-shot fields: {exc}') from exc


def evaluate(examples, db_root, llm, *, mode='repair', max_repairs=2,
             max_react_steps=6, timeout=5.0, max_rows=10000, few_shots=(), cards_dir=None,
             corrections=None, on_record=None, correction_scope=None):
    """One generation per condition; corrected labels only rescore predictions."""
    examples = list(examples)
    # A fixed subset can use an overlay reviewed against the entire dataset.
    # Validate unknown IDs against that full scope, then rescore selected rows.
    scope = list(correction_scope) if correction_scope is not None else examples
    if corrections:
        originals = {str(x.question_id): x for x in scope}
        if any(originals.get(str(x.question_id)) != x for x in examples):
            raise ValueError('Correction scope must contain every selected original example')
        corrected = {str(x.question_id): x for x in apply_corrections(scope, corrections)}
    else:
        corrected = {}
    records=[]
    for example in examples:
        path=database_path(db_root,example.db_id)
        executor=SQLiteExecutor(path,timeout_seconds=timeout,max_rows=max_rows)
        cards=None
        if cards_dir:
            cards=json.loads((Path(cards_dir)/f'{example.db_id}.json').read_text())
        writer=SQLWriter(llm,executor,max_repairs=max_repairs,max_react_steps=max_react_steps,few_shots=few_shots)
        before=dict(getattr(llm,'stats',{}))
        output=writer.run(example.question,data_cards=cards,evidence=example.evidence,
                          db_id=example.db_id,mode=mode)
        after=dict(getattr(llm,'stats',{}))
        score=score_prediction(executor,output.sql,example.gold_sql)
        bag=score_prediction(executor,output.sql,example.gold_sql,mode='bag')
        record={'question_id':example.question_id,'db_id':example.db_id,'difficulty':example.difficulty,
                'mode':mode,'question':example.question,'writer':output.to_dict(),
                'original_score':asdict(score),'bag_diagnostic':asdict(bag),
                'usage':{k:after[k]-before.get(k,0) for k in after},
                'cost_usd':None}
        if str(example.question_id) in corrected:
            alt=corrected[str(example.question_id)]
            record['corrected_score']=asdict(score_prediction(executor,output.sql,alt.gold_sql))
            record['label_source']=alt.label_source
        records.append(record)
        if on_record:on_record(record)
    return records


def summarize(records, key='original_score'):
    selected=[x for x in records if key in x]
    scores=[x[key] for x in selected]
    scorable=[x for x in scores if x['correct'] is not None]
    correct=sum(x['correct'] is True for x in scorable)
    return {'attempted':len(scores),'scorable':len(scorable),'unscorable':len(scores)-len(scorable),
            'correct':correct,'execution_accuracy':correct/len(scorable) if scorable else None,
            'correct_per_attempted':correct/len(scores) if scores else None,
            'status_counts':dict(Counter(x['status'] for x in scores)),
            'writer_status_counts':dict(Counter(x['writer']['status'] for x in selected)),
            'total_model_calls':sum(x['writer']['model_calls'] for x in selected),
            'elapsed_seconds':sum(x['writer']['elapsed_seconds'] for x in selected)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Original Mini-Dev SQLite question JSON')
    p.add_argument('--db-root',type=Path,required=True,help='Directory containing db_id/db_id.sqlite')
    p.add_argument('--model',required=True,help='Installed Ollama model tag; digest is recorded automatically')
    p.add_argument('--base-url',default='http://localhost:11434')
    p.add_argument('--output',type=Path,required=True,help='New run directory; existing runs are never overwritten')
    p.add_argument('--mode',choices=['once','repair','react','all'],default='all')
    p.add_argument('--size',type=int,default=100,help='Fixed subset size; use --full for all rows')
    p.add_argument('--full',action='store_true')
    p.add_argument('--seed',type=int,default=6130)
    p.add_argument('--max-repairs',type=int,default=2)
    p.add_argument('--max-react-steps',type=int,default=6)
    p.add_argument('--query-timeout',type=float,default=5.0)
    p.add_argument('--max-rows',type=int,default=10000)
    p.add_argument('--cache',type=Path,default=Path('runs/sql_calls.sqlite'))
    p.add_argument('--cards-dir',type=Path,help='B data-card JSONs named db_id.json; otherwise schema-only fallback')
    p.add_argument('--few-shots',type=Path,help='Audited BIRD train-only example JSON')
    p.add_argument('--corrections',type=Path,help='Validated external label overlay, scored separately')
    args=p.parse_args(argv)
    examples=load_bird(args.data)
    selected=fixed_subset(examples,size=len(examples) if args.full else args.size,seed=args.seed)
    shots=load_few_shots(args.few_shots)
    # Check every path before generating or creating partial run artifacts.
    for x in selected:database_path(args.db_root,x.db_id)
    if args.corrections:apply_corrections(examples,args.corrections)
    database_hashes = {x.db_id: digest(database_path(args.db_root, x.db_id))
                       for x in {row.db_id: row for row in selected}.values()}
    card_hashes = ({db_id: digest(args.cards_dir / f'{db_id}.json') for db_id in database_hashes}
                   if args.cards_dir else {})
    model_info = preflight_model(args.base_url, args.model)
    args.output.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest={'config':config,'dataset_sha256':digest(args.data),'source_rows':len(examples),
              'model':model_info,'database_sha256':database_hashes,'cards_sha256':card_hashes,
              'code_sha256':code_fingerprints(),'selected_count':len(selected),'question_ids':[x.question_id for x in selected],
              'python':platform.python_version(),'schema_source':'data_cards' if args.cards_dir else 'schema_only',
              'few_shots_sha256':digest(args.few_shots) if args.few_shots else None,
              'corrections_sha256':digest(args.corrections) if args.corrections else None,
              'comparison':'BIRD set semantics; bag comparison is diagnostic',
              'cost_note':'No price configured; cost_usd is null, never assumed zero.'}
    try:manifest['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):manifest['commit']=None
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    client=CachedOllama(args.model,base_url=args.base_url,cache_path=args.cache,model_digest=model_info['digest'])
    try:
        for mode in ['once','repair','react'] if args.mode=='all' else [args.mode]:
            with (args.output/f'{mode}.jsonl').open('w') as f:
                def save(record):
                    f.write(json.dumps(record,default=str)+'\n');f.flush()
                records=evaluate(selected,args.db_root,client,mode=mode,max_repairs=args.max_repairs,
                                 max_react_steps=args.max_react_steps,timeout=args.query_timeout,
                                 max_rows=args.max_rows,few_shots=shots,cards_dir=args.cards_dir,
                                 corrections=args.corrections,on_record=save,correction_scope=examples)
            summary={'original':summarize(records)}
            if args.corrections:summary['corrected']=summarize(records,'corrected_score')
            (args.output/f'{mode}_summary.json').write_text(json.dumps(summary,indent=2))
            first={str(x['question_id']):next((a['sql'] for a in x['writer']['attempts'] if a['sql']), '') for x in records}
            (args.output/f'{mode}_first_sql.json').write_text(json.dumps(first,indent=2))
            failed=[{'question_id':x['question_id'],'db_id':x['db_id'],
                     'status':x['original_score']['status'],'sql':x['writer']['sql'],
                     'error':x['original_score'].get('prediction_error')}
                    for x in records if x['original_score']['correct'] is not True]
            (args.output/f'{mode}_failures.json').write_text(json.dumps(failed,indent=2))
            print(mode,json.dumps(summary))
    finally:client.close()

if __name__=='__main__':main()
