# Member B: status report

**Component:** schema scout and persistent schema memory (data cards)
**Code:** [`schema_memory.py`](schema_memory.py), one file, Python standard library only
**Last updated:** 23 Oct 2026

The plan describes B's role as making "the agent understand a database before any SQL is written". Every item below comes from the VeriQuery project plan, and each one is marked with where the plan lists it. Results come from running the code on BIRD Mini-Dev (500 questions) and BIRD Dev (1,534 questions).

---

## Summary

| # | Plan item | Plan category | Status | Waiting on |
|---|---|---|---|---|
| 1 | Schema scout: list tables, sample rows, profile columns, find join keys | What B does | ✅ Done | — |
| 2 | Design the data card format | What B does | ✅ Done | Final check against A's shared state format |
| 3 | Store data cards in a schema notebook that persists across questions | What B does / Must do | ✅ Done | — |
| 4 | Retrieval: give the SQL writer only the relevant cards | What B does / Must do | ✅ Built and tested | **C** to plug it into the SQL writer |
| 5 | Schema scout producing data cards for all BIRD Mini-Dev databases | Must do | ✅ Done | — |
| 6 | Measure the schema-memory speed-up on repeated questions | What B does / Evaluation owned by B | ✅ Done | — |
| 7 | Ablation: with vs without schema memory | Must do | 🟡 Half done (speed). Accuracy is still open. | **C**'s SQL writer, plus the integrated pipeline (E runs the ablations) |
| 8 | Cards that record past mistakes ("this column is in cents") | Good to have | 🟡 Mechanism done; no real mistakes recorded yet | **E**'s verifier finding real mistakes |
| 9 | LoRA fine-tune of a small 7B coder model on verified trajectories | Good to have | ❌ Not started | **Verified trajectories** from the integrated pipeline (A + C + E), a GPU, and a model choice agreed with A and C |
| 10 | "Own trajectories" dataset (verified runs as training and verifier test data) | Datasets table, owner B | ❌ Not started | The integrated pipeline producing verified runs |
| 11 | Own section of the final report | Team roles | ⏳ Later | Final results (items 7 and 9) |

---

## 1. Done

### 1.1 Schema scout (plan: "lists tables, samples rows, profiles columns (types, null rates, value ranges, likely join keys)")

`profile_database(db_path)` opens a database **read-only**, with a 20-second limit on every query, and records the following.

**For every table**
- exact row count
- primary key
- 3 sample rows

**For every column**
- **type**: the declared type plus a semantic type (`id`, `numeric`, `categorical`, `date`, `boolean`, `text`)
- **null rate**
- **value range**: min and max
- distinct count and whether the column is unique
- example values, plus the full value list for columns with 50 or fewer distinct values
- BIRD's column description, when the dataset provides one
- warnings: numbers stored as text, dates stored as text, mostly-NULL columns, names that need quoting

**Likely join keys**
- foreign keys declared in the database
- inferred keys: same column name, unique on one side, and at least 80% of values overlap
- each join records its cardinality (`many-to-one`, `many-to-many`, …), which tells E where double counting can happen

**Result:** all 11 Mini-Dev databases and all 11 Dev databases profiled, with 0 validation errors. Profiling everything from scratch takes about 156 s; the two largest databases account for most of it.

### 1.2 Data card format (plan: "a short, structured summary per table and column")

- **Structure:** each database has one `DatabaseCard`, made of `TableCard`s, made of `ColumnCard`s, plus a list of `JoinKey`s.
- **Formal definition:** [`data_card_schema/data_card.schema.json`](data_card_schema/data_card.schema.json)
- **How teammates use it:** [`data_card_schema/INTERFACE.md`](data_card_schema/INTERFACE.md)
- **Real example:** [`data_card_schema/example_card_california_schools.json`](data_card_schema/example_card_california_schools.json)
- **Versioning:** the format is versioned (`schema_version: 1.0`), so any later change is visible to everyone.

### 1.3 Persistent schema notebook (plan: "stores data cards in a schema notebook that persists across questions on the same database")

- `SchemaNotebook` saves one JSON card per database in `outputs/schema_notebook/`.
- Later questions on the same database **reuse the saved card instead of profiling again**.
- The card is rebuilt automatically if the database file changes.
- Files are written atomically, so a crash can't leave a half-written card.
- **Result:** a fresh notebook reloads all 11 cards in about 0.01 s, with 0 rebuilds.

### 1.4 Relevant-card retrieval (plan: "builds retrieval so the SQL writer only gets the cards relevant to the question")

`nb.retrieve(question, db_id, evidence)` returns:
- the relevant tables, with their columns
- the joins between those tables
- the exact database values mentioned in the question (e.g. `'Fresno'` → `schools.City`)
- `to_prompt()`: compact schema text ready to paste into the SQL writer's prompt (see [`data_card_schema/example_prompt.txt`](data_card_schema/example_prompt.txt))

**How it was measured:** for every question, I check whether the retrieved tables include **all** tables used by BIRD's correct ("gold") SQL query.

| Dataset | Questions | All needed tables retrieved | Avg tables sent | Time per question |
|---|---|---|---|---|
| Mini-Dev | 500 | **469 (93.8%)** | 3.5 | ~9 ms |
| Dev | 1,534 | **1,448 (94.4%)** | 3.4 | ~9 ms |

**What this means:** in about 94 out of 100 questions, the SQL writer gets every table it needs, while seeing only about 3–4 tables instead of the whole schema (up to 13 tables). Questions where a table was missed are listed in `outputs/retrieval_misses_<dataset>.json` for error analysis.

### 1.5 Schema-memory speed-up on repeated questions (plan: B's evaluation, "Schema-memory speed-up on repeated questions")

`python schema_memory.py ablation --dataset dev` takes 3 questions per database (33 in total) and runs them two ways.

| | Without memory | With memory |
|---|---|---|
| Times the scout ran | 33 (every question) | 11 (first question per database only) |
| Total time | 194.6 s | 64.7 s (**3.0× faster**) |
| Median time per question | 1.23 s | **0.015 s** |
| Tables retrieved | identical | identical |

**What this means:** memory makes repeated questions about 80× faster, and it never changes which tables are retrieved. The overall 3× figure grows with more questions per database, because after the first question only the ~10 ms retrieval step remains.

### 1.6 Cards that record past mistakes: the mechanism (plan, good to have: "cards that record past mistakes on that database ('this column is in cents')")

- `nb.add_note(db_id, "amount is in cents", table="trans", column="amount")` attaches a lesson to the database, a table or a column.
- Notes are stored in `<db_id>.notes.json` and **survive rebuilds**.
- Notes appear in the prompt the SQL writer sees, so the same mistake isn't repeated.
- `nb.remove_note(db_id, note)` deletes a note that turns out to be wrong.
- Tested: saving, reloading, surviving a rebuild, and removing all pass.

---

## 2. Not done yet, and what each item is waiting for

### 2.1 Ablation: accuracy with vs without schema memory (plan: B must do, "Ablation result: with vs without schema memory")

- **Done:** the speed half (section 1.5), and the on/off switch. `SchemaNotebook(use_memory=False)` turns memory off without changing anything else.
- **Missing:** whether schema memory improves *answer accuracy*. Measuring that means generating SQL, running it, and scoring it against BIRD's answers.
- **Waiting for:**
  - **C:** the SQL writer, which turns B's retrieved cards into SQL, and C's BIRD Mini-Dev scoring.
  - **A:** the orchestrator and evaluation harness that run the whole system.
  - **E:** the plan gives E the job of running all three ablations ("Runs the three ablations (no clarifier, no verifier, no schema memory)"). B provides the switch; E runs the full-system comparison.
- **B's part once those exist:** run the system with memory on and off, and report the accuracy difference next to the speed-up.

### 2.2 Cards that record past mistakes: real content (plan, good to have)

- **Done:** the mechanism (section 1.6).
- **Missing:** actual mistakes from real runs.
- **Waiting for:**
  - **E:** the verifier, which detects mistakes such as wrong units, double counting after joins, or wrong date ranges and can call `nb.add_note(...)`.
  - **A:** the orchestrator, which connects the verifier to the notebook.

### 2.3 LoRA fine-tune of a small 7B coder model (plan, good to have: "LoRA fine-tune of a small 7B coder model on verified trajectories (needs GPU access)")

- **What it is:** take the runs where the full pipeline produced a *verified correct* answer, and use them to train a small open-source model. The input is the question plus B's schema prompt; the output is the verified SQL. The model is downloaded and trained locally on a GPU, so no API is involved. The goal is a free, local model that can do the SQL writer's job. The plan's contributions list calls this "distil verified trajectories into a small 7B coder model with LoRA".
- **Status:** not started.
- **Waiting for:**
  1. **Verified trajectories.** These only exist once the pipeline is integrated: A's orchestrator runs questions, C's SQL writer writes SQL, and E's verifier confirms which answers are correct.
  2. **Model choice.** The plan only says "a small 7B coder model". A chooses the team's LLM, and C owns the SQL-writer prompt the small model has to fit, so the model should be agreed with both.
  3. **GPU.** The plan says "needs GPU access". This laptop has an RTX 4060 with 8 GB, which is enough for 4-bit LoRA on a 7B model, but only just. The plan's fallback: "No GPU for LoRA → Drop distillation; it is stretch only".
- **B's part once those exist:** export the verified trajectories as training data, fine-tune with LoRA, and compare the fine-tuned model with the base model as the SQL writer.

### 2.4 "Own trajectories" dataset (plan, datasets table: "Own trajectories — Verified runs as training and verifier test data — Grows over project — Owner B")

- **What it is:** a growing record of pipeline runs (question, B's schema context, SQL, result, verifier verdict). It's used for two things: training data for LoRA (2.3), and test data for E's verifier.
- **Status:** not started.
- **Waiting for:** the integrated pipeline (A + C + E) producing runs, and A's shared state format, which decides what one run record contains.

### 2.5 Final check of the data card format against the shared state (plan practices: "Lock the shared state format on day 1 (question, plan, data cards, SQL, results, verification status)")

- **Status:** B's card format is finished and documented, but A hasn't published the shared state format yet.
- **B's proposal:** store `RetrievalResult.to_dict()` under a `schema_context` field.
- **Waiting for:** **A** to publish or confirm the shared state format.

### 2.6 B's section of the final report (plan: "Everyone writes their own section of the final report")

- **Waiting for:** the final results of 2.1, and of 2.3 if it's attempted.
- **Already available for it:** all results in sections 1.1–1.5, and the error analysis files.

---

## 3. Who B is waiting on

| Teammate | What B needs from them | Unblocks |
|---|---|---|
| **A** (orchestrator, lead) | Shared state format; orchestrator and evaluation harness; agreement on the LoRA model | 2.1, 2.2, 2.4, 2.5, 2.3 |
| **C** (SQL writer) | SQL writer that uses B's retrieval; Mini-Dev scoring; prompt format | 2.1, 2.3 |
| **E** (verifier) | Verifier verdicts (verified trajectories); mistake detection; running the ablations | 2.1, 2.2, 2.3, 2.4 |

**Nothing else in B's list can move forward without these teammates.** All B work that can be done independently is finished.

---

## 4. Additional work done (not required by the plan)

These were added to test or support the plan items above. None of them changes the plan.

- **BIRD Dev (1,534 questions) tested as well as Mini-Dev.** This confirmed that the retrieval results hold beyond the 500 Mini-Dev questions. Mini-Dev and Dev share database names, but 5 of the 11 database files differ, so each dataset keeps its own notebook.
- **Self-test:** `python schema_memory.py selftest` runs 19 checks on a small synthetic database in under a second, with no download needed. Teammates can use it to check that B works on their machine.
- **Smoke test:** `python schema_memory.py smoke --dataset minidev|dev` rebuilds and validates every card, checks memory reuse, measures retrieval recall, and checks that notes save and reload.
- **Bugs found and fixed by testing:**
  - fake joins on generic `id` columns
  - short category codes like `LAM` being ignored
  - results changing slightly between runs
  - "hub" tables like `races` and `drivers` being drowned out by foreign-key columns
  - names stored in the data ("San Marino Grand Prix") being invisible to retrieval
  - notes being lost when a card was rebuilt

## 5. Known limitations

- Retrieval recall checks **tables**, not columns or joins, so it is an upper bound on B's contribution, not SQL accuracy.
- About 6% of questions miss at least one needed table, mostly in `formula_1` and `debit_card_specializing`, where the question names something only indirectly (for example, "where is he from?" → `drivers.nationality`).
- The speed-up was measured on 33 questions. It's enough to show the effect, but a larger run would give firmer numbers.

## 6. How to reproduce

```bash
python schema_memory.py selftest                     # 19 checks, no dataset needed
python schema_memory.py smoke --dataset minidev      # cards + memory + retrieval recall on Mini-Dev
python schema_memory.py smoke --dataset dev          # same on full Dev
python schema_memory.py ablation --dataset dev       # schema-memory speed-up
python schema_memory.py retrieve --db california_schools "Which school in Fresno has the highest average SAT math score?"
```

Data download and setup steps are in [`README.md`](README.md). The results history is in [`CHECKPOINTS.md`](CHECKPOINTS.md).
