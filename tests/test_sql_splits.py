import json
import pytest
from training.freeze_sql_splits import freeze, digest, resolve_selection
from benchmarks.bird import load_bird


def inputs(tmp_path, n=12):
    data=tmp_path/'data.json'
    data.write_text(json.dumps([{'db_id':f'db{i%2}', 'question_id':i, 'question':f'Question {i}', 'SQL':'SELECT 1'} for i in range(n)]))
    historic=tmp_path/'historical.json'
    historic.write_text(json.dumps({'dataset_sha256':digest(data), 'selected_ids':[{'db_id':'db0','question_id':0},{'db_id':'db1','question_id':1}]}))
    return data,historic


def test_frozen_splits_are_disjoint_complete_and_deterministic(tmp_path):
    data,historic=inputs(tmp_path)
    first=freeze(data,historic,tmp_path/'first',3,4)
    second=freeze(data,historic,tmp_path/'second',3,4)
    assert first == second
    assert {k:len(v) for k,v in first['partitions'].items()} == {'historical':2,'development':3,'final':4,'calibration':3}
    ids=[r['question_id'] for rows in first['partitions'].values() for r in rows]
    assert len(ids)==len(set(ids))==12
    for name in first['partitions']:
        selected=resolve_selection(load_bird(data),first,name,digest(data))
        saved=load_bird(tmp_path/'first'/f'{name}.json')
        assert selected == saved
    with pytest.raises(ValueError,match='not overwritten'):
        freeze(data,historic,tmp_path/'first',3,4)


def test_selection_rejects_hash_changes_overlap_and_unknown_identity(tmp_path):
    data,historic=inputs(tmp_path)
    manifest=freeze(data,historic,tmp_path/'out',3,4)
    with pytest.raises(ValueError,match='hash mismatch'):
        resolve_selection(load_bird(data),manifest,'final','bad')
    manifest['partitions']['final'][0]=manifest['partitions']['development'][0]
    with pytest.raises(ValueError,match='overlapping'):
        resolve_selection(load_bird(data),manifest,'final',digest(data))
    manifest['partitions']['final'][0]={'db_id':'missing','question_id':99}
    with pytest.raises(ValueError,match='missing from source'):
        resolve_selection(load_bird(data),manifest,'final',digest(data))


def test_rejects_duplicate_source_and_historical_id(tmp_path):
    data,historic=inputs(tmp_path)
    prior=json.loads(historic.read_text());prior['selected_ids'].append(prior['selected_ids'][0]);historic.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='Duplicate historical'):
        freeze(data,historic,tmp_path/'out',3,4)
    rows=json.loads(data.read_text());rows.append(rows[0]);data.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Duplicate question_id'):
        freeze(data,historic,tmp_path/'out',3,4)


def test_rejects_historical_source_mismatch_and_too_large_splits(tmp_path):
    data,historic=inputs(tmp_path)
    with pytest.raises(ValueError,match='Insufficient'):
        freeze(data,historic,tmp_path/'out',10,4)
    prior=json.loads(historic.read_text());prior['dataset_sha256']='different';historic.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='different source hash'):
        freeze(data,historic,tmp_path/'out',3,4)
