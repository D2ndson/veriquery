# BIRD Mini-Dev results: comparing `Latest-Combined` and `experiment/sql-writer-qlora-results`

Every configuration below ran under the same conditions:
- **Same questions and order:** the frozen split and fixed order described in Setup.
- **Same harness and model server:** `veriquery.pipeline.evaluate` with vLLM serving Qwen2.5-Coder-7B-Instruct plus each adapter.
- **Same models for the other agents:** the clarifier, verifier and reporter are all the local 7B base model (`--profile local_all`).
- **Same run settings:** no repairer unless stated, `summarise=false`, 16 workers.

The run files in `cluster/` (`runs_dev_a1..a3`, `runs_c1`, `runs_test`) reproduce every row.

## Setup

- **Test split: `minidev_test`.** These are E's frozen 99 questions (`data/subsets/minidev_dev100.json`). 2 have gold SQL that won't execute, so n = 97. Each configuration ran on it once, after the choices were made on dev.
- **Dev split: `minidev_dev`.** This is the rest of Mini-Dev, shuffled with seed 0, first 150 questions. Every comparison and choice of settings was made here.
- **How comparisons are reported:** "+a / −b" is the paired count of questions that flip from wrong to right and from right to wrong between two systems. p is a two-sided exact sign test on those flips.

## Test split (n = 97)

| Writer | System | Accuracy | Confidently wrong |
|---|---|---|---|
| base | writer only (JSON prompt) | 43.3% | 57% |
| base | original pipeline (as delivered) | 52.6% | 28% |
| writer-qlora | writer only | 58.8% | 41% |
| writer-dpo | writer only (text prompt) | 60.8% | 39% |
| writer-sft | full | 58.8% | 25% |
| writer-sft | full + adjudication | 59.8% | 24% |
| writer-dpo | full | 56.7% | 26% |
| **writer-qlora** | **full** | **63.9%** | **20%** |
| writer-qlora | full + adjudication | 63.9% | 21% |

| Comparison | Flips | p |
|---|---|---|
| original pipeline → writer-qlora full | +16 / −5 | 0.027 |
| base writer → qlora writer, alone | +21 / −6 | 0.006 |
| qlora writer alone → qlora full system | +6 / −1 | 0.13 |
| sft full → qlora full | +13 / −8 | 0.38 |
| sft full → dpo full | +0 / −2 | 0.50 |
| qlora full → qlora full + adjudication | +0 / −0 | 1.0 |

## Final check: the committed code, re-run fresh (job 52679)

This re-ran the committed code in a fresh checkout (`cluster/runs_final_check.txt`), with an empty LLM cache and an empty schema notebook. Every answer was generated anew. In this run 98 test questions had scorable gold SQL.

| System | BIRD test | Confidently wrong | ShopTrap (40, clarifier on) |
|---|---|---|---|
| Original pipeline, base writer | 52.0% | 29% | 62.5% |
| **Default as shipped** (writer-qlora, full, repairer) | **64.3%** | 23% | 60.0% |
| Default without the repairer | 62.2% | 20% | — |

- **BIRD:** the default beats the original pipeline by +17 / −5 (p = 0.017). Removing the repairer gives +16 / −6 (p = 0.052), and the repairer itself is +6 / −4.
- **Reproducibility:**
  - The fresh no-repairer run matches the cached evaluation of the same system: +1 / −2, 62.2% vs 63.9%.
  - The fresh baseline reproduces exactly: +0 / −0.
- **ShopTrap:** no gain, +3 / −4 (p = 1.0). With every agent on the local 7B, the traps (ambiguity, fan-out, units) are caught at the same rate by both systems; the catch rate is 67% for each.

## Dev split (n = 150)

| Writer | System | Accuracy | Confidently wrong |
|---|---|---|---|
| base | writer only (JSON / text) | 40.0% / 38.7% | 60% / 61% |
| base | original / full | 45.3% / 45.3% | 35% / 33% |
| writer-sft / dpo / dpo-lf | writer only | 44.0% / 47.3% / 44.0% | 56% / 53% / 56% |
| writer-qlora | writer only | 49.3% | 51% |
| writer-sft | full / + adjudication | 50.7% / 53.3% | 31% / 31% |
| writer-dpo | full / + repairer | 52.0% / 52.0% | 30% / 33% |
| writer-qlora | full / + adjudication | 53.3% / 55.3% | 27% / 30% |

Paired flips on dev:
- **Full pipeline around the same SFT writer:** +10 / −0 (p = 0.002).
- **Adjudication:** +4 / −0 with SFT, +5 / −2 with QLoRA.
- **Repairer:** +4 / −4.
- **Adaptive fast path:** turning it off gained one question (+1 / −0). It skips sampling on about a third of the questions.

## What changed as a result

- **Default writer:** the `hybrid` and `local_all` profiles now use `writer-qlora`. Its adapter files go in `artifacts/adapters/writer-qlora/`; they are not tracked in git.
- **Prompt format follows the writer:** a writer always gets the prompt format it was trained on (`writer_formats` in `configs/models.yaml`):
  - QLoRA: C's JSON prompt.
  - SFT and DPO: the text plan-then-SQL prompt with few-shot examples.

  As a result, `final` with `writer-qlora` is exactly the `final_json` system that was evaluated.
- **Adjudication stays opt-in** (`--set adjudicate=true`). It was neutral to slightly positive on dev and changed no answers on test with the default writer.
- **Speed:** drafts, ambiguous-question readings and the adjudication judge calls run concurrently (`Config.parallel`, on by default). The parallel and sequential runs sent identical model requests and produced identical SQL on all 300 dev answers. The full system with the QLoRA writer averaged 21.8 s per question on test, against 26.4 s for the original pipeline.
- **Bugs found by these runs, now fixed:**
  - **Lost answers on a cold cache:** concurrent workers shared one temporary file when writing schema-cache files, and 10 questions were lost on the first run. `schema_memory.atomic_write` now gives each writer its own temporary file.
  - **Adjudicator crash:** the result comparison crashed when a column mixed text and missing values. It no longer crashes, and any adjudication failure now keeps the majority vote.
