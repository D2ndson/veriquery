"""Sample use case: the full VeriQuery pipeline on a small retail database.

    python examples/retail_demo.py                       # local Ollama, qwen2.5-coder:7b
    python examples/retail_demo.py --model qwen2.5-coder:14b
    VERIQUERY_LLM_PROVIDER=openai VERIQUERY_LLM_BASE_URL=http://host:8000/v1 python examples/retail_demo.py --model ...

The database is generated (deterministically) with traps the agents should handle: prices stored
in cents, cancelled and returned orders, and a one-to-many orders -> order_items join. In Q3 2026
the Sports category sells less, and Electronics orders are cancelled far more often, so "which
category lost the most revenue" has a different answer depending on whether cancelled orders count.

Each question's answer is printed next to a hand-written ground-truth query, and the full reports,
traces and charts are written to runs/demo-<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

from veriquery.agents.orchestrator import Config, VeriQuery
from veriquery.cli import render_text
from veriquery.llm import DEFAULT_MODEL, LLM

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "retail" / "retail.sqlite"

# Stable facts about the database, kept in the schema notebook (B) rather than passed as per-question
# evidence: the verifier reads evidence for its checks, so e.g. a year in it would look like part of the question.
NOTES = [
    ("Today is 2026-10-01, so 'last quarter' is calendar Q3 2026 unless the user says otherwise.", None, None),
    ("unit_price_cents is in cents; divide by 100 for dollars. "
     "Order line revenue = quantity * unit_price_cents.", "order_items", "unit_price_cents"),
    ("list_price_cents is in cents.", "products", "list_price_cents"),
    ("Report money in dollars (cents / 100.0) unless the user asks for cents.", None, None),
    ("Order value is the total of one order's lines: SUM(quantity * unit_price_cents) grouped by order_id. "
     "Average order value averages that per-order total over orders, not over customers or lines.", "orders", None),
]

CATEGORIES = ["Electronics", "Home & Kitchen", "Sports", "Books", "Toys"]
PRODUCTS = {
    "Electronics": [("Headphones", 8999), ("Smartwatch", 19999), ("Tablet", 32999), ("Charger", 1999)],
    "Home & Kitchen": [("Blender", 5999), ("Knife Set", 7499), ("Kettle", 2999)],
    "Sports": [("Yoga Mat", 2499), ("Dumbbells", 6999), ("Running Shoes", 11999), ("Bike Helmet", 4999)],
    "Books": [("Cookbook", 2499), ("Novel", 1499), ("Atlas", 3999)],
    "Toys": [("Puzzle", 1999), ("Building Blocks", 4499), ("Board Game", 3499)],
}
REGIONS = ["North", "South", "East", "West"]


def build_database(path: Path = DB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(7)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
            CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                category_id INTEGER REFERENCES categories(category_id), list_price_cents INTEGER);
            CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT, signup_date TEXT);
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
                order_date TEXT, status TEXT);
            CREATE TABLE order_items (order_id INTEGER REFERENCES orders(order_id),
                product_id INTEGER REFERENCES products(product_id), quantity INTEGER, unit_price_cents INTEGER);
        """)
        conn.executemany("INSERT INTO categories VALUES (?, ?)", list(enumerate(CATEGORIES, 1)))
        products, pid = [], 0
        for cid, cat in enumerate(CATEGORIES, 1):
            for name, price in PRODUCTS[cat]:
                pid += 1
                products.append((pid, name, cid, price))
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
        customers = [(i, f"Customer {i}", rng.choice(REGIONS),
                      (date(2024, 1, 1) + timedelta(days=rng.randrange(600))).isoformat()) for i in range(1, 401)]
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

        orders, items, oid = [], [], 0
        day = date(2025, 1, 1)
        while day <= date(2026, 9, 30):
            q3_2026 = day >= date(2026, 7, 1)
            for _ in range(rng.randint(3, 7)):
                oid += 1
                weights = [1.0, 1.0, 0.45 if q3_2026 else 1.0, 1.0, 1.0]  # Sports demand drops in Q3 2026
                cid = rng.choices(range(1, 6), weights)[0]
                lines = [p for p in products if p[2] == cid]
                has_electronics = cid == 1
                cancel_p = 0.40 if (has_electronics and q3_2026) else 0.05  # Electronics cancellations spike
                r = rng.random()
                status = "cancelled" if r < cancel_p else "returned" if r < cancel_p + 0.03 else "completed"
                orders.append((oid, rng.randint(1, 400), day.isoformat(), status))
                for p in rng.sample(lines, rng.randint(1, min(2, len(lines)))):
                    discount = rng.choice([1.0, 1.0, 1.0, 0.9])
                    items.append((oid, p[0], rng.randint(1, 3), int(p[3] * discount)))
            day += timedelta(days=1)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", items)
        conn.commit()
    return path


# Ground truth, written by hand, to check the agent's answers against.
_QUARTER_DROP = """
WITH q AS (
  SELECT c.name AS category,
         SUM(CASE WHEN o.order_date BETWEEN '2026-04-01' AND '2026-06-30' THEN i.quantity * i.unit_price_cents END) AS q2,
         SUM(CASE WHEN o.order_date BETWEEN '2026-07-01' AND '2026-09-30' THEN i.quantity * i.unit_price_cents END) AS q3
  FROM order_items i JOIN orders o ON o.order_id = i.order_id
  JOIN products p ON p.product_id = i.product_id JOIN categories c ON c.category_id = p.category_id
  {where}
  GROUP BY c.name)
SELECT category, ROUND((q2 - q3) / 100.0, 2) AS revenue_drop_dollars FROM q ORDER BY q2 - q3 DESC LIMIT 1"""

QUESTIONS = [
    {
        "id": "q1_clear",
        "question": "What was the total revenue in dollars from completed orders in 2025?",
        "truth": """SELECT ROUND(SUM(i.quantity * i.unit_price_cents) / 100.0, 2) FROM order_items i
                    JOIN orders o ON o.order_id = i.order_id
                    WHERE o.status = 'completed' AND o.order_date BETWEEN '2025-01-01' AND '2025-12-31'""",
    },
    {
        "id": "q2_ambiguous_no_user",
        "question": "Which product category lost the most revenue last quarter?",
        "truth": {"all orders": _QUARTER_DROP.format(where=""),
                  "completed orders only": _QUARTER_DROP.format(where="WHERE o.status = 'completed'")},
    },
    {
        "id": "q3_ambiguous_with_user",
        "question": "Which product category lost the most revenue last quarter?",
        "user": True,
        # Which applies depends on whether the clarifier asks about cancelled orders.
        "truth": {"all orders": _QUARTER_DROP.format(where=""),
                  "completed orders only": _QUARTER_DROP.format(where="WHERE o.status = 'completed'")},
    },
    {
        "id": "q4_analysis_chart",
        "question": "Compare the average order value in dollars by region for completed orders in 2026.",
        "truth": """SELECT c.region, ROUND(AVG(t.total) / 100.0, 2) FROM
                    (SELECT o.order_id, o.customer_id, SUM(i.quantity * i.unit_price_cents) AS total
                     FROM orders o JOIN order_items i ON i.order_id = o.order_id
                     WHERE o.status = 'completed' AND o.order_date >= '2026-01-01' GROUP BY o.order_id) t
                    JOIN customers c ON c.customer_id = t.customer_id GROUP BY c.region ORDER BY c.region""",
    },
]


def simulated_user(question: str) -> str:
    """Stands in for a business user answering the clarifier (like BIRD-INTERACT's simulator)."""
    # Answers every topic the question touches, as a person would: "lost to cancellations, or the
    # lowest total?" is about both what "lost" means and which orders count.
    q = question.lower()
    rules = [
        (r"lost|lose|loss|drop|decrease|decline|lowest|negative|previous|compared",
         "'Lost the most revenue' means the largest drop in revenue from Q2 2026 to Q3 2026."),
        (r"cancel|status|returned|completed|refund", "Only completed orders count as revenue."),
        (r"\bquarter\b|\bq[1-4]\b|period|time range|when", "Last quarter means calendar Q3 2026 (July to September)."),
    ]
    answers = [answer for pattern, answer in rules if re.search(pattern, q)]
    if not answers and re.search(r"revenue|metric|measure|gross|net", q):
        answers = ["Revenue is quantity times unit price, in dollars."]
    return " ".join(dict.fromkeys(answers)) or "Use your best judgement."


def ground_truth(sql) -> object:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if isinstance(sql, dict):
            return {k: conn.execute(v).fetchall() for k, v in sql.items()}
        return conn.execute(sql).fetchall()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--clarifier-model", help="a stronger model for the clarifier only (default: --model)")
    ap.add_argument("--candidates", type=int, default=3, help="SQL drafts per reading (1 = no voting)")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM replies")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    build_database()
    out_dir = Path("runs") / f"demo-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)
    cache = {"cache_path": None} if a.no_cache else {}
    llm = LLM.from_env(model=a.model, **cache)
    # think=False: Qwen3 models would otherwise spend the token budget reasoning before the JSON
    clarifier_llm = LLM.from_env(model=a.clarifier_model, think=False, **cache) if a.clarifier_model else None
    vq = VeriQuery(llm, Config(notebook_dir=str(out_dir.parent / "schema_notebook"), n_candidates=a.candidates),
                   clarifier_llm=clarifier_llm)
    for note, table, column in NOTES:
        vq.remember(DB_PATH, note, table=table, column=column)
    print(f"Database: {DB_PATH}\nModel: {llm.provider}/{llm.model}"
          f"{f'  (clarifier: {a.clarifier_model})' if a.clarifier_model else ''}\nOutput: {out_dir}\n")

    summary = []
    for item in QUESTIONS:
        if a.only and item["id"] not in a.only.split(","):
            continue
        print("=" * 100 + f"\n[{item['id']}]")
        state = vq.ask(item["question"], DB_PATH, ask_user=simulated_user if item.get("user") else None)
        print(render_text(state))
        truth = ground_truth(item["truth"])
        print(f"\nGround truth: {truth}")
        (out_dir / f"{item['id']}.json").write_text(json.dumps(
            {"report": state["report"], "trace": state["trace"], "schema_context": state["schema_context"],
             "clarification": state.get("clarification"), "ground_truth": truth,
             "answers": [{"sql_trace": x.get("sql_trace"), "verification_rounds": x["verification_rounds"]}
                         for x in state["answers"]]},
            indent=2, default=str), encoding="utf-8")
        for x in state["answers"]:
            fig = x.get("analysis") and x["analysis"].figure
            if fig is not None:
                fig.write_html(out_dir / f"{item['id']}{'_' + x['label'] if x['label'] else ''}_chart.html")
        summary.append({"id": item["id"], "clarification": state["report"]["clarification"],
                        "answers": [(x["label"], x["confidence"], x["answer"]) for x in state["report"]["answers"]],
                        "truth": truth, "seconds": state["report"]["seconds"]})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    usage = vq.usage()
    (out_dir / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    print("=" * 100 + f"\nLLM usage: {json.dumps(usage['total'])}\nWrote {out_dir}")


if __name__ == "__main__":
    main()
