"""Freeze disjoint MiniDev project partitions without choosing by SQL outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bird import load_bird, fixed_subset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    db, qid = row['db_id'], row['question_id']
    if not isinstance(db, str) or not db or isinstance(qid, bool) or not isinstance(qid, (str, int)) or not str(qid).strip():
        raise ValueError('Invalid selected question identity')
    return db, str(qid)


def resolve_selection(examples, manifest, partition, source_sha256):
    if manifest.get('source_sha256') != source_sha256:
        raise ValueError('Selection source hash mismatch')
    partitions = manifest.get('partitions', {})
    if partition not in partitions:
        raise ValueError(f'Unknown selection partition: {partition}')
    by_key = {(e.db_id, str(e.question_id)): e for e in examples}
    if len(by_key) != len(examples):
        raise ValueError('Duplicate source question identity')
    seen = set()
    for rows in partitions.values():
        for row in rows:
            identity = key(row)
            if identity in seen:
                raise ValueError('Duplicate or overlapping selection identity')
            if identity not in by_key:
                raise ValueError(f'Selected identity missing from source: {identity}')
            seen.add(identity)
    if not partitions[partition]:
        raise ValueError('Selection partition is empty')
    return [by_key[key(row)] for row in partitions[partition]]


def freeze(source, historical_manifest, output, development_size=100, final_size=200, seed=6130):
    source, historical_manifest, output = map(Path, (source, historical_manifest, output))
    if output.exists():
        raise ValueError('Use a new split output directory; frozen splits are not overwritten')
    examples = load_bird(source)
    prior = json.loads(historical_manifest.read_text())
    if prior.get('dataset_sha256') != digest(source):
        raise ValueError('Historical evaluation used a different source hash')
    historic = prior['selected_ids']
    identities = [key(row) for row in historic]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate historical selection identity')
    by_key = {(e.db_id, str(e.question_id)): e for e in examples}
    if any(k not in by_key for k in identities):
        raise ValueError('Historical identity missing from source')
    if development_size < 1 or final_size < 1:
        raise ValueError('Development and final sizes must be positive')
    prior_keys = set(identities)
    remaining = [e for e in examples if (e.db_id, str(e.question_id)) not in prior_keys]
    if len(remaining) < development_size + final_size:
        raise ValueError('Insufficient unseen questions for requested splits')
    ranked = fixed_subset(remaining, size=len(remaining), seed=seed)
    split_examples = {'historical': [by_key[k] for k in identities],
                      'development': ranked[:development_size],
                      'final': ranked[development_size:development_size+final_size],
                      'calibration': ranked[development_size+final_size:]}
    parts = {name: [{'db_id': e.db_id, 'question_id': e.question_id} for e in rows]
             for name, rows in split_examples.items()}
    manifest = {'version': 1, 'seed': seed, 'source_sha256': digest(source),
                'historical_manifest_sha256': digest(historical_manifest), 'partitions': parts,
                'notice': 'Question-disjoint project splits, not official test splits. Historical questions were inspected previously. Freeze method on development, confidence thresholds on calibration, then evaluate final once.'}
    records = {key(row): row for row in json.loads(source.read_text())}
    output.mkdir(parents=True)
    file_hashes = {}
    for name, rows in parts.items():
        path = output/f'{name}.json'
        path.write_text(json.dumps([records[key(row)] for row in rows], indent=2, ensure_ascii=False)+'\n')
        file_hashes[name] = digest(path)
    manifest['split_file_sha256'] = file_hashes
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--historical-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--development-size', type=int, default=100)
    p.add_argument('--final-size', type=int, default=200)
    p.add_argument('--seed', type=int, default=6130)
    a=p.parse_args()
    m=freeze(a.data,a.historical_manifest,a.output,a.development_size,a.final_size,a.seed)
    print(json.dumps({'counts': {k:len(v) for k,v in m['partitions'].items()}, 'source_sha256':m['source_sha256']},indent=2))


if __name__ == '__main__':
    main()
