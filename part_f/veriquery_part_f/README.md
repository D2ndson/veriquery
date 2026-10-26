# VeriQuery — Part F

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `sample_sales.csv` and ask `total sales by region`.

## Local test

```bash
python evaluate.py --csv sample_sales.csv --cases sample_cases.jsonl
```

## Team integration

```python
from integration_example import run_part_f
report = run_part_f(question, sql_rows, sql_text, {'status': 'passed'})
```

The SQL writer must supply query results and SQL; the verifier must supply status. The reporter does not independently verify claims. The local evaluator is not the official InfiAgent-DABench evaluator.

## Current limitations

- Rule-based analytics, not general-purpose LLM reasoning or code generation.
- No arbitrary filters, time series, joins, or multi-step analytical planning.
- Ambiguous columns use documented assumptions; they are not automatically clarified.
- `unverified` means verification has not been performed; not a probability estimate.
- Official InfiAgent-DABench integration, benchmark results and demo video remain to be completed.
