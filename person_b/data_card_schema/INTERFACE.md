# Component B — interface for teammates

Everything lives in `schema_memory.py` (standard library only). Import it; don't copy it.

```python
from schema_memory import SchemaNotebook

nb = SchemaNotebook()                         # paths come from env vars / defaults (see README)
card = nb.get("california_schools")           # DatabaseCard (cached on disk, rebuilt if the DB changes)

hits = nb.retrieve(question, db_id, evidence="", k_tables=5)
hits.table_names        # ['satscores', 'schools']
hits.to_prompt()        # compact schema text for an LLM prompt  <-- what C's SQL writer should use
hits.join_keys          # list[JoinKey] between the selected tables
hits.column_hits        # {'satscores': ['AvgScrMath', 'cname', ...]}  columns that matched the question
hits.matched_values     # {'races.name': ['San Marino Grand Prix']}  exact DB spelling of values in the question
hits.to_dict()          # JSON-friendly summary (for shared state / logs)
```

## Who uses what

| Teammate | Use |
|---|---|
| **C – SQL writer** | `nb.retrieve(...).to_prompt()` as the schema part of the prompt. Column names are already quoted. |
| **D – Clarifier** | `hits.matched_values`: a value matching several columns (e.g. a name that is both a city and a district) is an ambiguity signal. Also `card.table(t).column(c).top_values` / `examples` to see if a term matches several columns or values (ambiguity). |
| **E – Verifier** | `join_keys[*].cardinality` (`many-to-many` ⇒ possible double counting), `row_count`, `null_rate`, `min_value`/`max_value` (range checks), column `notes` (TEXT numbers, TEXT dates). Record lessons with `nb.add_note(db_id, "amount is in cents", table="trans", column="amount")`. Undo with `nb.remove_note(db_id, note)`. |
| **A – Orchestrator** | Put `hits.to_dict()` into the shared state under `schema_context`. `SchemaNotebook(use_memory=False)` = ablation *without schema memory*. `nb.stats` gives hits/misses/seconds for cost logging. |
| **F – Analyst** | `card.table(t).columns[*].semantic_type` to choose aggregations / chart types. |

## Data card format
Formal JSON Schema: [`data_card.schema.json`](data_card.schema.json). Example: [`example_card_california_schools.json`](example_card_california_schools.json). Example retrieval prompt: [`example_prompt.txt`](example_prompt.txt).

```
DatabaseCard: db_id, schema_version, db_path, fingerprint, created_at, profile_seconds, notes[], tables[], join_keys[]
  TableCard:  name, row_count, stats_rows, primary_key[], sample_rows[], notes[], columns[]
    ColumnCard: name, declared_type, semantic_type(id|numeric|categorical|date|boolean|text), is_primary_key,
                null_rate, distinct_count, is_unique, min_value, max_value, examples[], top_values[[v,n]],
                description, value_description, notes[]
  JoinKey:    from_table, from_column, to_table, to_column, source(declared|inferred), cardinality, overlap
```

## Stability promise
- Field names above are frozen for `schema_version` 1.0. Adding fields bumps to 1.1 (backward compatible); renaming/removing bumps to 2.0 and will be announced.
- Cards are rebuilt automatically when the schema version or the DB file changes.
- Database access is **read-only** (`mode=ro`) with a 20 s per-query timeout.
