"""Freeze a confidence threshold on calibration records; apply unchanged elsewhere."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def wilson(correct, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z*z/total
    middle = (p + z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [max(0, middle-half), min(1, middle+half)]


def metrics(records, threshold):
    scorable = [r for r in records if r['score']['correct'] is not None]
    accepted = [r for r in scorable if r.get('confidence') is not None and r['confidence'] >= threshold]
    hits = sum(bool(r['score']['correct']) for r in accepted)
    return {'attempted':len(records), 'scorable':len(scorable), 'accepted':len(accepted),
            'coverage':len(accepted)/len(scorable) if scorable else 0,
            'accuracy':hits/len(accepted) if accepted else None,
            'accuracy_wilson95':wilson(hits,len(accepted)),
            'confidently_wrong_rate':(len(accepted)-hits)/len(scorable) if scorable else None,
            'overall_accuracy':sum(bool(r['score']['correct']) for r in scorable)/len(scorable) if scorable else None}


def fit(records, target=.85, minimum_coverage=.30):
    thresholds=sorted({float(r['confidence']) for r in records if r.get('confidence') is not None})
    curve=[{'threshold':t,**metrics(records,t)} for t in thresholds]
    eligible=[x for x in curve if x['accuracy'] is not None and x['accuracy']>=target and x['coverage']>=minimum_coverage]
    chosen=max(eligible,key=lambda x:(x['coverage'],x['accuracy'],x['threshold'])) if eligible else None
    return {'target':target,'minimum_coverage':minimum_coverage,'threshold':chosen['threshold'] if chosen else None,
            'calibration_target_met':chosen is not None,'curve':curve,
            'notice':'Calibration accuracy is not held-out accuracy. Threshold selection uses point estimates; report final uncertainty and coverage.'}


def load(path):
    records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ids=[(r['db_id'],str(r['question_id'])) for r in records]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate question IDs')
    return records, ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation',type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error('Use a new output file')
    rows,ids=load(a.calibration);result=fit(rows)
    result['calibration_sha256']=hashlib.sha256(a.calibration.read_bytes()).hexdigest()
    result['calibration_ids']=ids
    if a.evaluation:
        test,test_ids=load(a.evaluation)
        if set(ids)&set(test_ids): raise ValueError('Calibration and evaluation overlap')
        result['evaluation_sha256']=hashlib.sha256(a.evaluation.read_bytes()).hexdigest()
        result['evaluation']=metrics(test,result['threshold']) if result['threshold'] is not None else None
        result['evaluation_notice']='No acceptable calibration policy' if result['threshold'] is None else 'Calibration threshold applied unchanged'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['curve','calibration_ids']},indent=2))

if __name__=='__main__': main()
