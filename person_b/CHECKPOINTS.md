# Component B — checkpoints

## Must do (from the project plan)
- [x] Schema scout producing data cards for **all 11** BIRD Mini-Dev databases (and full BIRD Dev) — 20 Oct 2026
- [x] Persistent notebook reused across questions (fingerprint-based rebuild, atomic saves, learned notes, value index)
- [x] Relevant-card retrieval feeding the SQL writer (`nb.retrieve(...).to_prompt()`, k = 5)
- [x] Ablation, B side: with vs without schema memory → **speed-up** measured (`python schema_memory.py ablation`)
- [ ] Ablation, end-to-end: accuracy with vs without schema memory — needs C's SQL writer (switch: `SchemaNotebook(use_memory=False)`)

## Good to have
- [x] Cards record past mistakes (`add_note` / `remove_note`, stored in `<db>.notes.json`, survive rebuilds)
- [x] Exact data values from the question returned to the SQL writer (`matched_values`)
- [ ] Embedding retrieval (sentence-transformers on the GPU) if lexical recall becomes the bottleneck
- [ ] Decide on BIRD Train (8.9 GB zip, ~30 GB unzipped, 69 more DBs; D: has ~44 GB free) to test unseen schemas
- [ ] LoRA fine-tune of a 7B coder model on verified trajectories (needs pipeline + trajectories)

## Integration (later, with the team)
- [x] Interface + JSON Schema: `data_card_schema/INTERFACE.md`, `data_card.schema.json`
- [x] Dataset-free self-test for teammates: `python schema_memory.py selftest` (19 checks, < 1 s)
- [ ] Agree card fields with A's shared state format (`RetrievalResult.to_dict()` → `schema_context`)
- [ ] GitHub branch + PR (postponed: everything stays local for now)

## Results — 20 Oct 2026

| Check | Mini-Dev (500 q) | Dev (1,534 q) |
|---|---|---|
| Cards built + valid | 11 / 11, 0 errors | 11 / 11, 0 errors |
| Cold profiling, all 11 DBs | 156 s | 55 s (OS file cache warm) |
| Warm reload from notebook | 0.01 s, 11 hits, 0 rebuilds | 0.01 s, 11 hits, 0 rebuilds |
| **Retrieval: all gold-SQL tables retrieved** | **469 / 500 = 93.8 %** | **1448 / 1534 = 94.4 %** |
| Avg tables selected | 3.5 | 3.4 |
| Retrieval latency | ~9 ms / question | ~9 ms / question |
| Note save / reload / remove | pass | pass |
| Self-test (synthetic DB) | 19 / 19 pass | |

Results are deterministic (identical on every run).

**Schema-memory ablation** (Dev, 3 questions per DB = 33 questions):

| | without memory | with memory |
|---|---|---|
| Scout runs | 33 (every question) | 11 (first question per DB) |
| Total time | 194.6 s | 64.7 s → **3.0× faster** |
| Median per question | 1.23 s | **0.015 s** |
| Retrieved tables | identical | identical |

The speed-up grows with more questions per database: after the first question, only the ~10 ms retrieval remains.

### Fixes found by testing
- Fake inferred joins (`alignment.id = colour.id` …) from generic `id` columns → generic names excluded (superhero: 39 → 11 joins, all declared).
- Short category codes (`LAM`, `KAM`) ignored by value matching → 3-char threshold, value hits weighted higher.
- Run-to-run variation (461–464) from Python set ordering → scoring and join-path search made order-stable.
- Hub tables (`races`, `drivers`) drowned out because every table has `raceId` / `driverId` → foreign-key columns no longer count as topic words.
- Names stored in data ("San Marino Grand Prix") invisible to retrieval → per-DB value index; one-word matches only for names/codes.
- k = 4 → 5 tables: +1.7 pts recall for +0.13 tables per question.
- Table/column notes lost on rebuild (found by the self-test) → learned notes kept in their own file and re-applied.

Remaining misses (`outputs/retrieval_misses_<dataset>.json`) are mostly in formula_1 and debit_card_specializing, where the question names an entity only indirectly ("where is he from" → `drivers.nationality`).
