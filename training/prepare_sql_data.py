"""Prepare schema-grounded, train-only SQL supervision; split by database."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veriquery.agents.sql_writer import SQLWriter
from veriquery.sql_execution import SQLiteExecutor, _validate_sql
from benchmarks.run_sql import digest
from sqlglot.errors import ParseError


def metadata_cards(schema):
    tables = schema['table_names_original']
    columns = schema['column_names_original']
    primary = {index for key in schema.get('primary_keys', [])
               for index in (key if isinstance(key, list) else [key])}
    cards = [{'table': name, 'columns': [], 'foreign_keys': []} for name in tables]
    for i, (table, column) in enumerate(columns):
        if table < 0:
            continue
        cards[table]['columns'].append({'name': column, 'type': schema['column_types'][i],
                                      'primary_key': i in primary})
    for source, target in schema.get('foreign_keys', []):
        table, column = columns[source]
        dest_table, dest_column = columns[target]
        cards[table]['foreign_keys'].append({'from': column, 'to_table': tables[dest_table], 'to_column': dest_column})
    return cards


def prepare(rows, db_root, held_out_db_ids, *, seed=6130, schemas=None):
    db_ids = sorted({r['db_id'] for r in rows})
    overlap = set(db_ids) & set(held_out_db_ids)
    if overlap:
        raise ValueError(f'Training databases overlap evaluation databases: {sorted(overlap)}')
    if len(db_ids) < 2:
        raise ValueError('At least two training databases required for a database-disjoint validation split')
    ranked = sorted(db_ids, key=lambda d: hashlib.sha256(f'{seed}:{d}'.encode()).hexdigest())
    val_ids = set(ranked[:max(1, round(len(ranked)*.1))])
    output = {'train': [], 'validation': []}
    skipped, fingerprints, seen = [], {}, set()
    for index, row in enumerate(rows):
        db_id = row['db_id']
        cards = None
        if schemas is None:
            root = Path(db_root).resolve()
            path = (root / db_id / f'{db_id}.sqlite').resolve()
            if not path.is_relative_to(root):
                raise ValueError('Database path escapes root')
            executor = SQLiteExecutor(path, timeout_seconds=10, max_rows=10000)
            if db_id not in fingerprints:
                fingerprints[db_id] = digest(path)
        else:
            cards = metadata_cards(schemas[db_id])
            executor = SQLiteExecutor('unused-metadata-only.sqlite')
        sql = row['SQL']
        if schemas is None:
            result = executor.execute(sql)
            if not result.ok:
                skipped.append({'row': index, 'db_id': db_id, 'reason': result.error})
                continue
        else:
            try:
                _validate_sql(sql)
            except (ValueError, ParseError, RecursionError) as exc:
                skipped.append({'row': index, 'db_id': db_id, 'reason': str(exc)})
                continue
        question, evidence = row['question'], row.get('evidence', '')
        key = (db_id, question, evidence)
        if key in seen:
            skipped.append({'row': index, 'db_id': db_id, 'reason': 'duplicate question/evidence'})
            continue
        seen.add(key)
        try:
            prompt = SQLWriter(lambda _: '', executor)._prompt(question, cards, evidence, db_id=db_id)
        except ValueError as exc:
            skipped.append({'row': index, 'db_id': db_id, 'reason': str(exc)})
            continue
        output['validation' if db_id in val_ids else 'train'].append(
            {'db_id': db_id, 'prompt': prompt, 'completion': sql, 'source_row': index})
    if not all(output.values()):
        raise ValueError('No usable train or validation examples')
    return output, {'validation_db_ids': sorted(val_ids), 'database_sha256': fingerprints, 'skipped': skipped,
                    'reference_validation': 'execution' if schemas is None else 'syntax_and_safety_only'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-json', type=Path, required=True, help='Official BIRD train or filtered-train JSON array')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--db-root', type=Path)
    group.add_argument('--tables-json', type=Path, help='Official train_tables.json; syntax validation only, no execution claim')
    p.add_argument('--evaluation-json', type=Path, required=True, help='Held-out Mini-Dev JSON; database IDs only are read')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.train_json.read_text())
    evaluation = json.loads(a.evaluation_json.read_text())
    schemas = {s['db_id']:s for s in json.loads(a.tables_json.read_text())} if a.tables_json else None
    out, manifest = prepare(rows, a.db_root, {r['db_id'] for r in evaluation}, schemas=schemas)
    a.output.mkdir(parents=True, exist_ok=False)
    for split, records in out.items():
        (a.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest.update(train_source_sha256=digest(a.train_json), evaluation_source_sha256=digest(a.evaluation_json),
                    counts={k: len(v) for k,v in out.items()}, seed=6130,
                    tables_sha256=digest(a.tables_json) if a.tables_json else None,
                    note='No evaluation labels used for training. See reference_validation for the checks actually performed.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
