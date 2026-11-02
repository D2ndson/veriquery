SQL writer fine-tuning experiment

This is supervised QLoRA on official BIRD training questions and reference SQL. It is not verified-trajectory distillation and is not a novel algorithm by itself. No GPU result is implied by the unit tests.

The model defaults to Qwen/Qwen2.5-Coder-7B-Instruct, pinned to commit c03e6d358207e414f1eca0bb1891e29f1db0e242. Compare this base model with its own adapter. The team's Qwen2.5-Coder-32B run is a separate reference, not a matched fine-tuning baseline.

Environment
Install the appropriate CUDA PyTorch wheel, the project dependencies, training/requirements.txt and remotezip. Record the actual pip freeze with the run. Do not replace another assignment's PEFT installation; use an isolated environment. GPU scripts must run on a Slurm compute node.

Data
python training/download_data.py
python training/prepare_sql_data.py --train-json data/train_filtered.json --tables-json data/train_tables.json --evaluation-json data/mini_dev.json --output data/prepared

The downloader pins current Hugging Face source commits in data/download_manifest.json and downloads the archived July 2025 Mini-Dev source. It removes only byte-equivalent duplicate records and rejects conflicting IDs. Preparation rejects overlap between training and evaluation databases; validation is held out by training database. Schema metadata mode checks SQL syntax and safety, not execution or semantic correctness. Use --db-root instead of --tables-json to execute training references against downloaded databases. Overlength prompts/targets are reported and dropped, never silently truncated. Dataset license: CC BY-SA 4.0; preserve BIRD attribution when sharing derived data. The dataset and model weights are not committed.

GPU smoke
python training/train_sql_lora.py --data data/prepared --output results/smoke --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --smoke

Full training
python training/train_sql_lora.py --data data/prepared --output results/qlora7b --revision c03e6d358207e414f1eca0bb1891e29f1db0e242

Resume after an interrupted job by using the same command plus --resume-from-checkpoint results/qlora7b/checkpoint-N. The data, revision and experiment settings must match. Checkpoints are saved every 100 optimizer steps; progress since the last checkpoint may be lost on interruption.

Matched 100-question evaluations
python training/evaluate_sql_adapter.py --data data/mini_dev.json --db-root data/dev_databases --output results/base7b --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --context-tokens 16384
python training/evaluate_sql_adapter.py --data data/mini_dev.json --db-root data/dev_databases --output results/tuned7b --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --context-tokens 16384 --adapter results/qlora7b/adapter
python -m benchmarks.analyze_sql_failures results/base7b/repair.jsonl --compare results/tuned7b/repair.jsonl

Both runs use the same deterministic 100 unique questions, model revision, 4-bit loading, prompt, SQL execution limits and context/output budgets. Their full manifests are required to interpret a paired comparison. BIRD execution accuracy uses sets of rows; the bag diagnostic checks duplicate multiplicity separately. Gold errors or truncation are unscorable, not model failures. The existing team subset contains a duplicate ID and is not silently changed by these commands.

Novelty experiment to discuss
Use disagreement between existing SQL candidates to target the next check (join cardinality, filter values, aggregation grain, dates or units), then repair only from that evidence. Compare against current majority voting with the same model-call budget. Measure high-confidence accuracy, coverage, confidently-wrong rate, fixed cases, regressions and latency. This is a proposed experiment, not implemented or validated here. Candidate voting and abstention are prior work; avoid a first-ever-method claim.

Sources
https://huggingface.co/datasets/birdsql/bird23-train-filtered
https://github.com/bird-bench/mini_dev/tree/main/finetuning
https://huggingface.co/docs/peft/developer_guides/quantization
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
