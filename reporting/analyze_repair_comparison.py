"""Validate and summarize a complete three-arm repair comparison (stdlib only)."""
from pathlib import Path
import argparse, collections, copy, csv, hashlib, itertools, json, math, statistics

MODES=('generic','structure','evidence')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def normalized(m):
    x=copy.deepcopy(m)
    for k in ('mode','output'): x['settings'].pop(k,None)
    for k in ('disagreement_mode','notebook_dir'): x['config'].pop(k,None)
    return x

def key(r): return r['db_id'],str(r['question_id'])

def wilson(hits,n):
    if not n:return [None,None]
    z=1.959963984540054;p=hits/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]

def paired_p(fixed,regressed):
    n=fixed+regressed
    return min(1.,2*sum(math.comb(n,k) for k in range(min(fixed,regressed)+1))/2**n) if n else 1.

def check_status(score):
    if score['correct'] is not None and type(score['correct']) is not bool:
        raise ValueError('Correctness must be boolean or null')

def trigger(r):
    return any(round_.get('decision')=='repair' and any(c.get('name')=='dual_route' and c.get('severity')=='fail' for c in round_.get('checks',[])) for round_ in r.get('verification_rounds',[]))

def summary(rows):
    scored=[r for r in rows if r['score']['correct'] is not None]
    hits=sum(r['score']['correct'] for r in scored);n=len(scored)
    high=[r for r in scored if r.get('report_confidence')=='high'];hh=sum(r['score']['correct'] for r in high)
    ds=[d for r in rows for d in r.get('diagnostics',[])]
    probes=[p for d in ds for p in d.get('probes',[])]
    costs=[float(r.get('seconds',0)) for r in rows]
    usage=[r.get('usage',{}).get('total',{}) for r in rows]
    note_counts=collections.Counter(note for d in ds for note in d.get('notes',[]))
    return dict(attempted=len(rows),scorable=n,correct=hits,accuracy=hits/n if n else None,
        accuracy_wilson95=wilson(hits,n),status_counts=dict(collections.Counter(r['score']['status'] for r in rows)),
        execution_errors=sum(r['score']['status']=='prediction_error' for r in rows),
        executable_wrong=sum(r['score']['status']=='incorrect' for r in rows),
        high_confidence_count=len(high),high_confidence_accuracy=hh/len(high) if high else None,
        high_confidence_wilson95=wilson(hh,len(high)),high_confidence_coverage=len(high)/n if n else None,
        confidently_wrong_rate=(len(high)-hh)/n if n else None,
        dual_route_repair_trigger_questions=sum(trigger(r) for r in rows),
        diagnostic_trigger_questions=sum(bool(r.get('diagnostics')) for r in rows),diagnostic_rounds=len(ds),
        questions_with_any_probe=sum(any(d.get('probes') for d in r.get('diagnostics',[])) for r in rows),
        questions_with_successful_probe=sum(any(p.get('error') is None and not p.get('truncated') and 'rows_total' in p for d in r.get('diagnostics',[]) for p in d.get('probes',[])) for r in rows),
        unsupported_or_failed_diagnostic_rounds=sum(any(note.startswith('No further probes:') for note in d.get('notes',[])) for d in ds),
        differences=dict(collections.Counter(x for d in ds for x in d.get('differences',[]))),
        diagnostic_notes=dict(note_counts),probe_calls=len(probes),probe_errors=sum(bool(p.get('error')) for p in probes),
        probe_truncations=sum(bool(p.get('truncated')) for p in probes),
        schema_reads=sum(d.get('schema_reads',0) for d in ds),diagnostic_seconds=sum(d.get('elapsed_seconds',0) for d in ds),
        total_seconds=sum(costs),mean_seconds=statistics.mean(costs) if costs else None,median_seconds=statistics.median(costs) if costs else None,
        model_calls=sum(u.get('calls',0) for u in usage),prompt_tokens=sum(u.get('prompt_tokens',0) for u in usage),completion_tokens=sum(u.get('completion_tokens',0) for u in usage),
        pipeline_error_questions=sum(bool(r.get('error')) for r in rows))

def analyze(paths):
    manifests={m:json.loads((paths[m]/'manifest.json').read_text()) for m in MODES}
    rows={m:[json.loads(l) for l in (paths[m]/'traces.jsonl').read_text().splitlines() if l.strip()] for m in MODES}
    reference=normalized(manifests['generic']);expected=[key(r) for r in manifests['generic']['selected_ids']]
    if not expected or len(expected)!=len(set(expected)):raise ValueError('Invalid selected question IDs')
    for mode in MODES:
        m=manifests[mode]
        if m['settings']['mode']!=mode or m['config']['disagreement_mode']!=mode:raise ValueError(f'Arm mode mismatch: {mode}')
        if normalized(m)!=reference:raise ValueError(f'Non-allowed manifest differences: {mode}')
        if [key(r) for r in rows[mode]]!=expected:raise ValueError(f'Incomplete, duplicate, reordered or mismatched traces: {mode}')
        for r in rows[mode]:
            check_status(r['score'])
            if r.get('model_calls_attempted',0)>m['settings']['max_calls']:raise ValueError('Model call budget exceeded')
    for triplet in zip(*(rows[m] for m in MODES)):
        if len({r['question'] for r in triplet})!=1:raise ValueError('Question text mismatch')
        if len({r['score']['correct'] is None for r in triplet})!=1:raise ValueError('Scorability differs across arms; resolve gold execution discrepancy first')
        if triplet[0]['score']['correct'] is None and len({r['score']['status'] for r in triplet})!=1:raise ValueError('Unscorable reason mismatch')
    result={'scope':manifests['generic']['scope'],'model':manifests['generic']['settings']['model'],
            'partition':manifests['generic']['settings']['partition'],'runs':{m:summary(rows[m]) for m in MODES},
            'verification':{'complete_identical_questions':True,'only_mode_and_output_manifest_differences':True},
            'artifact_sha256':{m:{n:sha(paths[m]/n) for n in ('manifest.json','traces.jsonl')} for m in MODES},
            'comparisons':{},'confidence_notice':'Labels are existing heuristic labels, not calibrated probabilities.',
            'statistical_notice':'Three paired comparisons; raw exact McNemar p-values and Holm-adjusted p-values reported. Single seed, development subset; no held-out generalization claim.'}
    paired=[]
    for a,b in itertools.combinations(MODES,2):
        counts=collections.Counter()
        for ra,rb in zip(rows[a],rows[b]):
            x,y=ra['score']['correct'],rb['score']['correct']
            state='unscorable' if x is None else 'both_correct' if x and y else 'fixed' if y else 'regressed' if x else 'both_wrong'
            counts[state]+=1
            paired.append(dict(comparison=f'{b}_vs_{a}',db_id=ra['db_id'],question_id=ra['question_id'],before_correct=x,after_correct=y,transition=state,before_status=ra['score']['status'],after_status=rb['score']['status']))
        n=len(expected)-counts['unscorable'];f=counts['fixed'];r=counts['regressed']
        result['comparisons'][f'{b}_vs_{a}']={**dict(counts),'scorable_pairs':n,'gain_percentage_points':100*(f-r)/n if n else None,'exact_p':paired_p(f,r)}
    ordered=sorted(result['comparisons'],key=lambda k:result['comparisons'][k]['exact_p']);prev=0
    for i,k in enumerate(ordered):
        prev=max(prev,min(1,result['comparisons'][k]['exact_p']*(len(ordered)-i)))
        result['comparisons'][k]['holm_adjusted_p']=prev
    return result,paired

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for mode in MODES:p.add_argument('--'+mode,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new analysis output directory')
    result,paired=analyze({m:getattr(a,m) for m in MODES});a.output.mkdir(parents=True)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'paired_outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n');w.writeheader();w.writerows(paired)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
