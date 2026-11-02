# Clarifier (D)

Finds the unclear parts of a question and either asks the user about them one at a time, or (when no user is available)
answers each reading and labels it. It never writes SQL; it hands a clear question to the SQL writer.

## Use it in the pipeline

```python
from veriquery.clarifier import Clarifier, clarifier_node

clar = Clarifier(llm)          # llm = veriquery.clarifier.llm.LLM(...)  or any callable prompt -> str (e.g. E's OllamaLLM)
state = clarifier_node(state, clar, context_messages, ask_user=fn_or_None, max_questions=3)
```

- **Reads** `state["question"]`. `context_messages` = `[{"role": "system", "content": <schema / B's data cards + knowledge>}]`
- **Writes** `state["clarification"]`:
  - `decision`: `proceed` (clear) | `asked` (talked to the user) | `answer_each` (no user, several readings)
  - `resolved_question`: what the SQL writer (C) should answer
  - `interpretations`: `[{label, restated_question, assumptions}]` - with `answer_each`, C writes one SQL per item
  - `assumptions`, `dialogue` (`[{q, a}]`), `ambiguities` - for the reporter (F)
- `ask_user`: any function `question -> answer` (Streamlit chat input, the BIRD-INTERACT simulator, a human). `None` = no user.

## Files

| File | What |
|---|---|
| `clarifier.py` | ambiguity types, analyze -> decide -> converse, `clarifier_node()` |
| `llm.py` | OpenAI-compatible client (local vLLM or any API) with disk cache + token logging |

Evaluation on BIRD-INTERACT: `benchmarks/bird_interact/` (README there has the results).
Tests (no GPU): `python -m pytest tests/test_clarifier.py -q`
