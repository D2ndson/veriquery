"""Verify downloaded artifacts and produce the small public experiment summary."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, math


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    result=root/'results';out=root/'code/experiments/sql_qlora_7b';out.mkdir(parents=True,exist_ok=True)
    runs={name:[json.loads(line) for line in (result/name/'repair.jsonl').read_text().splitlines()] for name in ['base7b','tuned7b']}
    manifests={name:json.loads((result/name/'manifest.json').read_text()) for name in runs}
    for key in manifests['base7b']:
        if key not in {'adapter','adapter_sha256','output'}:
            assert manifests['base7b'][key]==manifests['tuned7b'][key],key
    assert sha(root/'data/mini_dev.json')==manifests['base7b']['dataset_sha256']
    for name,expected in manifests['tuned7b']['adapter_sha256'].items():
        assert sha(result/'qlora7b/adapter'/name)==expected,name
    by_id={name:{(r['db_id'],str(r['question_id'])):r for r in rows} for name,rows in runs.items()}
    assert len(by_id['base7b'])==len(by_id['tuned7b'])==100
    assert by_id['base7b'].keys()==by_id['tuned7b'].keys()
    summary={'scope':'SQL writer only, Qwen2.5-Coder-7B-Instruct; not the team 32B pipeline',
             'model':manifests['base7b']['model'],'model_revision':manifests['base7b']['revision'],'runs':{}}
    for name,rows in runs.items():
        counts=collections.Counter(r['original_score']['status'] for r in rows)
        n=sum(r['original_score']['correct'] is not None for r in rows)
        correct=sum(r['original_score']['correct'] is True for r in rows)
        summary['runs'][name]={'attempted':len(rows),'scorable':n,'correct':correct,'accuracy':correct/n,
            'correct_per_attempted':correct/len(rows),'status_counts':dict(counts),
            'model_calls':sum(r['writer']['model_calls'] for r in rows),
            'writer_seconds':sum(r['writer']['elapsed_seconds'] for r in rows)}
        original=json.loads((result/name/'summary.json').read_text())
        assert original['correct']==correct and original['scorable']==n
    transitions=collections.Counter();paired=[]
    for key,b in by_id['base7b'].items():
        t=by_id['tuned7b'][key];before=b['original_score']['correct'];after=t['original_score']['correct']
        assert b['question']==t['question']
        state=('unscorable' if before is None or after is None else
               'both_correct' if before and after else 'fixed' if after else 'regressed' if before else 'both_wrong')
        transitions[state]+=1
        paired.append({'db_id':key[0],'question_id':key[1],'base_correct':before,'tuned_correct':after,
                       'transition':state,'base_status':b['original_score']['status'],'tuned_status':t['original_score']['status']})
    summary['paired']=dict(transitions)
    discordant=transitions['fixed']+transitions['regressed'];lower=min(transitions['fixed'],transitions['regressed'])
    summary['exact_mcnemar_two_sided_p']=min(1,2*sum(math.comb(discordant,k) for k in range(lower+1))/2**discordant)
    summary['accuracy_change_percentage_points']=100*(transitions['fixed']-transitions['regressed'])/(100-transitions['unscorable'])
    summary['excluded']=[{'db_id':r['db_id'],'question_id':r['question_id'],'reason':r['original_score']['status']} for r in runs['base7b'] if r['original_score']['correct'] is None]
    summary['verification']={'same_questions_and_settings':True,'dataset_hash_matched':True,'all_adapter_hashes_matched':True}
    summary['limitations']=['One seed and one 100-question development subset','One reference result exceeds the 10000-row limit',
        'No clarifier, verifier, schema-memory retrieval or high-confidence labels in this component experiment',
        'No evidence that the 32B baseline improves','Disagreement-directed repair is a proposal, not an evaluated method']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'paired_outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    for name,manifest in manifests.items():(out/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
