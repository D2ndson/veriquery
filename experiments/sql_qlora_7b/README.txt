Verified SQL writer experiment — 3 November 2026

Qwen2.5-Coder-7B-Instruct base versus its QLoRA adapter, evaluated with the same prompts, budgets and 100 unique MiniDev questions. One reference result exceeded the row cap, leaving 99 scorable questions.

Base: 36/99 correct (36.36%), 19 execution errors, 44 executable wrong answers.
QLoRA: 40/99 correct (40.40%), 10 execution errors, 49 executable wrong answers.
Paired outcomes: 13 fixed, 9 regressed, 27 both correct, 50 both wrong; 1 excluded.
Gain: 4.04 percentage points. Exact paired two-sided p = 0.523.

The improvement is modest and uncertain. This is not a reproduction of the team's 32B run or a full-pipeline/high-confidence evaluation. Disagreement-directed repair remains a proposal.

See VeriQuery_SQL_Writer_Report.docx for methods, coloured figures and examples. summary.json records recomputed results; paired_outcomes.csv contains per-question outcomes; the two manifests record the matching evaluation configuration. Raw database result rows and model weights are deliberately not committed. Local full traces and the verified adapter are retained in the VeriQuery results folder.

Recompute from the locally downloaded run artifacts:
python reporting/summarize_experiment.py /path/to/VeriQuery
python reporting/plot_results.py /path/to/VeriQuery
python reporting/build_report.py /path/to/VeriQuery

Training: NTU Slurm 50403, 343 optimizer steps, 5,500 retained training examples and 562 validation examples. Evaluation: NTU Slurm 50407. Training used schema metadata and syntax checks, not execution validation of every training reference. Actual environment versions are recorded in the run artifacts.
