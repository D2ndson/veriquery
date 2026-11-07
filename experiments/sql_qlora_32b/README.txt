32B SQL writer and targeted evidence experiment

Status: prepared and software-tested; GPU memory test, training and model evaluations are not yet complete. No 32B accuracy gain is claimed.

Model: Qwen/Qwen2.5-Coder-32B-Instruct, revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7.
Plan: training/EXPERIMENT_32B.txt.
Frozen IDs: selection_manifest.json (100 historical,100 development,100 calibration,200 final). These are disjoint questions within MiniDev, not an official held-out test release.

The generic, structure-only and targeted-evidence modes share candidate and repair limits. Evidence mode diagnoses only supported flat SELECT join/aggregation disagreements using bounded key-cardinality probes. Filter differences are classified but not probed. Unsupported shapes are explicitly skipped. The method has unit tests; its benefit requires the forthcoming GPU ablations.

The meeting's confidence target must be reported with coverage and uncertainty. training/calibrate_sql_confidence.py freezes a threshold on a separate calibration partition and applies it unchanged to the final evaluation. A target not reached must be reported as such.
