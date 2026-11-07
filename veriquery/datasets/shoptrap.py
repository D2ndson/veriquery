"""ShopTrap: a deterministic e-commerce SQLite database with planted pitfalls, plus gold-SQL test questions.

It runs offline (no downloads), so the UI, the tests and the demo video never depend on BIRD being present.
Every pitfall is something a text-to-SQL system gets silently wrong:

  units         prices are stored in cents (unit_price_cents, amount_cents, refund_cents)
  fanout        orders.shipping_fee is one value per order; joining order_items repeats it per item
  date_format   products.launched is 'DD/MM/YYYY'; monthly_targets.month is 'YYYYMM'
  literal_case  cities / segments are stored as 'Singapore', 'Home Office' (users type 'singapore')
  null_order    products.rating is NULL for unrated products (SQLite sorts NULLs first)
  ratio         percentages need REAL division
  ambiguous     "revenue" can mean gross or net of discounts; "last quarter" needs a reference date
  why           a planted drop in Electronics revenue in 2025 Q2 (volume down, one product's price cut)
  anomaly       a planted flash-sale month (2024-03)

All names, emails and phone numbers are synthetic.
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

TODAY = "2025-07-01"
CATEGORIES = {
    "Electronics": [("Laptop Pro", 129900), ("Phone X", 89900), ("Tablet Air", 49900), ("Headphones", 19900),
                    ("Smartwatch", 24900), ("Monitor 27", 31900)],
    "Home": [("Air Fryer", 12900), ("Vacuum Robot", 39900), ("Desk Lamp", 4500), ("Blender", 8900),
             ("Coffee Maker", 15900), ("Bedding Set", 7900), ("Storage Box", 1900)],
    "Fashion": [("Running Shoes", 9900), ("Rain Jacket", 7900), ("Denim Jeans", 6900), ("Backpack", 5900),
                ("Sunglasses", 4900), ("Cotton Tee", 1900), ("Wool Scarf", 2900)],
    "Grocery": [("Olive Oil", 1500), ("Green Tea", 1200), ("Granola", 800), ("Dark Chocolate", 600),
                ("Honey", 1400), ("Coffee Beans", 1800)],
    "Accessories": [("Phone Case", 1500), ("USB-C Cable", 900), ("Power Bank", 3500), ("Screen Guard", 700),
                    ("Laptop Sleeve", 2500), ("Wireless Charger", 3900)],
    "Sports": [("Yoga Mat", 3500), ("Dumbbell Set", 8900), ("Water Bottle", 1500), ("Tennis Racket", 12900),
               ("Cycling Gloves", 2200), ("Jump Rope", 1100), ("Foam Roller", 2600)],
}
FIRST = ["Aiden", "Mei", "Rahul", "Siti", "Jun", "Priya", "Noah", "Hana", "Arjun", "Lina", "Wei", "Farah", "Ethan",
         "Zara", "Kenji", "Amira", "Lucas", "Nadia", "Ravi", "Chloe"]
LAST = ["Tan", "Lim", "Wong", "Singh", "Ng", "Rahman", "Lee", "Chua", "Kumar", "Teo", "Ong", "Nair", "Goh", "Ali"]
CITIES = ["Singapore"] * 6 + ["Johor Bahru", "Kuala Lumpur", "Jakarta", "Bangkok", "Manila"]
SEGMENTS = ["Consumer", "Consumer", "Corporate", "Home Office"]

SCHEMA = """
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT, phone TEXT, city TEXT,
                        signup_date TEXT, segment TEXT);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price_cents INTEGER,
                       launched TEXT, rating REAL);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT,
                     channel TEXT, shipping_fee REAL,
                     FOREIGN KEY (customer_id) REFERENCES customers(customer_id));
CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER,
                          unit_price_cents INTEGER, discount_pct REAL,
                          FOREIGN KEY (order_id) REFERENCES orders(order_id),
                          FOREIGN KEY (product_id) REFERENCES products(product_id));
CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, order_id INTEGER, amount_cents INTEGER, method TEXT,
                       FOREIGN KEY (order_id) REFERENCES orders(order_id));
CREATE TABLE returns (return_id INTEGER PRIMARY KEY, item_id INTEGER, reason TEXT, refund_cents INTEGER,
                      FOREIGN KEY (item_id) REFERENCES order_items(item_id));
CREATE TABLE monthly_targets (month TEXT, category TEXT, target_revenue REAL);
"""

DEFINITIONS = [
    {"term": "net revenue", "definition": "SUM(quantity * unit_price_cents * (1 - discount_pct/100.0)) / 100.0 over "
     "order_items of orders with status = 'delivered'"},
    {"term": "gross revenue", "definition": "SUM(quantity * unit_price_cents) / 100.0 over order_items of orders "
     "whose status is not 'cancelled'"},
]


def _months():
    y, m = 2023, 1
    while (y, m) <= (2025, 6):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def build_db(path: str | Path, seed: int = 7) -> Path:
    """Create the database file (overwrites). Deterministic for a given seed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rnd = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    custs = []
    for i in range(1, 301):
        first, last = rnd.choice(FIRST), rnd.choice(LAST)
        signup = date(2022, 6, 1) + timedelta(days=rnd.randint(0, 900))
        custs.append((i, f"{first} {last}", f"{first.lower()}.{last.lower()}{i}@example.com",
                      f"+65 {rnd.choice('89')}{rnd.randint(100, 999)} {rnd.randint(1000, 9999)}",
                      rnd.choice(CITIES), signup.isoformat(), rnd.choice(SEGMENTS)))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", custs)

    prods, pid = [], 0
    for cat, items in CATEGORIES.items():
        for name, price in items:
            pid += 1
            launched = date(2022, 1, 1) + timedelta(days=rnd.randint(0, 1100))
            rating = None if rnd.random() < 0.2 else round(rnd.uniform(3.0, 5.0), 1)
            prods.append((pid, name, cat, price, launched.strftime("%d/%m/%Y"), rating))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", prods)
    by_cat = {c: [p for p in prods if p[2] == c] for c in CATEGORIES}
    laptop_id = next(p[0] for p in prods if p[1] == "Laptop Pro")
    cat_weight = {"Electronics": 1.0, "Home": 1.2, "Fashion": 1.4, "Grocery": 2.0, "Accessories": 1.6, "Sports": 0.9}

    oid = iid = payid = retid = 0
    orders, items, pays, rets = [], [], [], []
    for idx, (y, m) in enumerate(_months()):
        n = int(110 + 3 * idx + (45 if m in (11, 12) else 0))
        if (y, m) == (2024, 3):
            n *= 3                                    # planted flash-sale anomaly
        for _ in range(n):
            oid += 1
            day = rnd.randint(1, 28)
            od = date(y, m, day)
            recent = (y, m) == (2025, 6)
            r = rnd.random()
            status = ("pending" if recent and r < 0.25 else "cancelled" if r < 0.08 else
                      "returned" if r < 0.13 else "delivered")
            fee = rnd.choice([0.0, 3.5, 5.0, 7.5])
            orders.append((oid, rnd.randint(1, 300), od.isoformat(), status, rnd.choice(["web", "web", "app", "store"]), fee))
            net_total = 0.0
            for _ in range(rnd.randint(1, 4)):
                cat = rnd.choices(list(CATEGORIES), weights=[cat_weight[c] for c in CATEGORIES])[0]
                if cat == "Electronics" and (y, m) >= (2025, 4) and rnd.random() < 0.30:
                    cat = "Accessories"               # planted: Electronics volume down in 2025 Q2
                p = rnd.choice(by_cat[cat])
                price = p[3]
                if p[0] == laptop_id and (y, m) >= (2025, 4):
                    price = int(price * 0.90)         # planted: Laptop Pro price cut
                iid += 1
                q = rnd.randint(1, 3)
                disc = rnd.choice([0, 0, 0, 5, 10, 15, 20])
                items.append((iid, oid, p[0], q, price, float(disc)))
                net_total += q * price * (1 - disc / 100.0)
                if status == "returned" and rnd.random() < 0.5:
                    retid += 1
                    rets.append((retid, iid, rnd.choice(["damaged", "wrong item", "changed mind"]), int(q * price * (1 - disc / 100.0))))
            payid += 1
            pays.append((payid, oid, int(net_total + fee * 100), rnd.choice(["card", "paynow", "cash"])))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
    conn.executemany("INSERT INTO payments VALUES (?,?,?,?)", pays)
    conn.executemany("INSERT INTO returns VALUES (?,?,?,?)", rets)
    targets = [(f"{y}{m:02d}", cat, round(rnd.uniform(20000, 60000), 2)) for y, m in _months() for cat in CATEGORIES]
    conn.executemany("INSERT INTO monthly_targets VALUES (?,?,?)", targets)
    conn.commit()
    conn.close()
    return path


NET = ("SUM(oi.quantity * oi.unit_price_cents * (1 - oi.discount_pct / 100.0)) / 100.0")
GROSS = "SUM(oi.quantity * oi.unit_price_cents) / 100.0"


def _q(id, question, gold, trap, difficulty=1, evidence="", readings=None, expect=""):
    return {"id": id, "question": question, "evidence": evidence, "gold_sql": gold, "trap": trap,
            "difficulty": difficulty, "readings": readings or [], "expect": expect}


def questions() -> list[dict]:
    """Gold-SQL test questions. `readings` lists the gold SQL of every sensible reading of an ambiguous question."""
    ev = f"Today is {TODAY}."
    net_def = "Net revenue means revenue after discounts, from delivered orders only."
    Q = [
        # ---- plain
        _q("p01", "How many customers are there?", "SELECT COUNT(*) FROM customers", "plain"),
        _q("p02", "How many orders were placed in 2024?", "SELECT COUNT(*) FROM orders WHERE order_date LIKE '2024%'", "plain"),
        _q("p03", "How many products are in each category?", "SELECT category, COUNT(*) FROM products GROUP BY category", "plain"),
        _q("p04", "Which channel has the most orders?", "SELECT channel FROM orders GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 1", "plain"),
        _q("p05", "What are the different payment methods?", "SELECT DISTINCT method FROM payments", "plain"),
        _q("p06", "How many orders were cancelled?", "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'", "plain"),
        _q("p07", "List the products in the Sports category.", "SELECT name FROM products WHERE category = 'Sports'", "plain"),
        _q("p08", "How many customers signed up in 2023?", "SELECT COUNT(*) FROM customers WHERE signup_date LIKE '2023%'", "plain"),
        # ---- joins
        _q("j01", "Which city has the most customers who placed an order in 2024?",
           "SELECT c.city FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_date LIKE '2024%' "
           "GROUP BY c.city ORDER BY COUNT(DISTINCT c.customer_id) DESC LIMIT 1", "plain", 2),
        _q("j02", "How many units of Laptop Pro were sold in 2024?",
           "SELECT SUM(oi.quantity) FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
           "JOIN orders o ON o.order_id = oi.order_id WHERE p.name = 'Laptop Pro' AND o.order_date LIKE '2024%'", "plain", 2),
        _q("j03", "Which product category has the most returned items?",
           "SELECT p.category FROM returns r JOIN order_items oi ON oi.item_id = r.item_id "
           "JOIN products p ON p.product_id = oi.product_id GROUP BY p.category ORDER BY COUNT(*) DESC LIMIT 1", "plain", 2),
        _q("j04", "How many distinct customers bought a Phone X?",
           "SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
           "JOIN products p ON p.product_id = oi.product_id WHERE p.name = 'Phone X'", "plain", 2),
        _q("j05", "What is the total refund amount for damaged items?",
           "SELECT SUM(refund_cents) / 100.0 FROM returns WHERE reason = 'damaged'", "units", 2,
           expect="units: refund_cents is in cents; the answer should be in dollars"),
        _q("j06", "Which customer segment placed the most orders?",
           "SELECT c.segment FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.segment "
           "ORDER BY COUNT(*) DESC LIMIT 1", "plain", 2),
        # ---- fan-out
        _q("f01", "What is the total shipping fee charged on web orders in 2024?",
           "SELECT SUM(shipping_fee) FROM orders WHERE channel = 'web' AND order_date LIKE '2024%'", "fanout", 2,
           expect="verifier: joining order_items would repeat each order's shipping_fee once per item"),
        _q("f02", "What is the total shipping fee on orders that contain at least one Electronics item?",
           "SELECT SUM(shipping_fee) FROM orders WHERE order_id IN (SELECT oi.order_id FROM order_items oi "
           "JOIN products p ON p.product_id = oi.product_id WHERE p.category = 'Electronics')", "fanout", 3,
           expect="verifier: a join + SUM(shipping_fee) multiplies by items per order"),
        _q("f03", "What is the average shipping fee per order for app orders?",
           "SELECT AVG(shipping_fee) FROM orders WHERE channel = 'app'", "fanout", 2),
        _q("f04", "How many orders contain more than one item?",
           "SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(*) > 1)", "fanout", 3),
        # ---- units
        _q("u01", "What is the average list price of a product in dollars?",
           "SELECT AVG(unit_price_cents) / 100.0 FROM products", "units", 1,
           expect="units: unit_price_cents is in cents"),
        _q("u02", "What is the most expensive product's price in dollars?",
           "SELECT MAX(unit_price_cents) / 100.0 FROM products", "units"),
        _q("u03", "What is the total amount paid by card, in dollars?",
           "SELECT SUM(amount_cents) / 100.0 FROM payments WHERE method = 'card'", "units"),
        # ---- date formats
        _q("d01", "How many products were launched in 2024?",
           "SELECT COUNT(*) FROM products WHERE launched LIKE '%/2024'", "date_format", 2,
           expect="date_format: launched is DD/MM/YYYY, not ISO"),
        _q("d02", "What was the total revenue target for Electronics in 2024?",
           "SELECT SUM(target_revenue) FROM monthly_targets WHERE category = 'Electronics' AND month LIKE '2024%'", "date_format", 2,
           expect="date_format: month is stored as YYYYMM"),
        _q("d03", "Which product category was launched most recently on average?",
           "SELECT category FROM products GROUP BY category ORDER BY AVG(substr(launched,7,4) || substr(launched,4,2) || substr(launched,1,2)) DESC LIMIT 1",
           "date_format", 3),
        # ---- literal case
        _q("l01", "How many customers live in singapore?", "SELECT COUNT(*) FROM customers WHERE city = 'Singapore'", "literal_case", 1,
           expect="literals: the stored value is 'Singapore'"),
        _q("l02", "How many home office customers are there?", "SELECT COUNT(*) FROM customers WHERE segment = 'Home Office'", "literal_case"),
        _q("l03", "How many orders came through the WEB channel in 2023?",
           "SELECT COUNT(*) FROM orders WHERE channel = 'web' AND order_date LIKE '2023%'", "literal_case"),
        # ---- ratios
        _q("r01", "What percentage of orders were cancelled?",
           "SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) FROM orders", "ratio", 2,
           expect="int_div: integer division would return 0"),
        _q("r02", "What is the return rate, as a percentage of delivered and returned orders that were returned?",
           "SELECT 100.0 * SUM(CASE WHEN status = 'returned' THEN 1 ELSE 0 END) / COUNT(*) FROM orders "
           "WHERE status IN ('delivered', 'returned')", "ratio", 3),
        _q("r03", "What share of customers are in the Corporate segment?",
           "SELECT 1.0 * SUM(CASE WHEN segment = 'Corporate' THEN 1 ELSE 0 END) / COUNT(*) FROM customers", "ratio", 2),
        # ---- NULL ordering
        _q("n01", "Which product has the lowest rating?",
           "SELECT name FROM products WHERE rating IS NOT NULL ORDER BY rating ASC LIMIT 1", "null_order", 2,
           expect="null_order: unrated products would sort first"),
        _q("n02", "What is the average product rating?", "SELECT AVG(rating) FROM products", "null_order"),
        # ---- ambiguous (clarifier)
        _q("a01", "What was our revenue last quarter?",
           f"SELECT {NET} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
           "WHERE o.status = 'delivered' AND o.order_date >= '2025-04-01' AND o.order_date < '2025-07-01'",
           "ambiguous", 3, ev,
           readings=[f"SELECT {NET} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
                     "WHERE o.status = 'delivered' AND o.order_date >= '2025-04-01' AND o.order_date < '2025-07-01'",
                     f"SELECT {GROSS} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
                     "WHERE o.status <> 'cancelled' AND o.order_date >= '2025-04-01' AND o.order_date < '2025-07-01'"],
           expect="clarifier: 'revenue' is gross or net"),
        _q("a02", "Which category made the most revenue in 2024?",
           f"SELECT p.category FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
           f"WHERE o.status = 'delivered' AND o.order_date LIKE '2024%' GROUP BY p.category ORDER BY {NET} DESC LIMIT 1",
           "ambiguous", 3, ev,
           readings=[f"SELECT p.category FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
                     f"WHERE o.status = 'delivered' AND o.order_date LIKE '2024%' GROUP BY p.category ORDER BY {NET} DESC LIMIT 1",
                     f"SELECT p.category FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
                     f"WHERE o.status <> 'cancelled' AND o.order_date LIKE '2024%' GROUP BY p.category ORDER BY {GROSS} DESC LIMIT 1"]),
        _q("a03", "How many big orders were there in 2024?",
           "SELECT COUNT(*) FROM (SELECT oi.order_id FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
           "WHERE o.order_date LIKE '2024%' GROUP BY oi.order_id HAVING SUM(oi.quantity * oi.unit_price_cents) >= 50000)",
           "ambiguous", 3, ev,
           readings=["SELECT COUNT(*) FROM (SELECT oi.order_id FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
                     "WHERE o.order_date LIKE '2024%' GROUP BY oi.order_id HAVING SUM(oi.quantity * oi.unit_price_cents) >= 50000)",
                     "SELECT COUNT(*) FROM (SELECT oi.order_id FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
                     "WHERE o.order_date LIKE '2024%' GROUP BY oi.order_id HAVING COUNT(*) >= 3)"],
           expect="clarifier: 'big' is undefined (value vs item count)"),
        _q("a04", "What was net revenue in 2024?",
           f"SELECT {NET} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'delivered' AND o.order_date LIKE '2024%'",
           "plain", 2, net_def,
           expect="defined by the evidence, so the clarifier should NOT ask"),
        # ---- why / trend / forecast / anomaly (python analyst tools)
        _q("w01", "Why did Electronics revenue change from Q1 2025 to Q2 2025?",
           f"SELECT CASE WHEN o.order_date < '2025-04-01' THEN '2025Q1' ELSE '2025Q2' END AS quarter, p.name AS product, "
           f"SUM(oi.quantity) AS units, SUM(oi.quantity * oi.unit_price_cents) / 100.0 AS revenue "
           f"FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
           f"WHERE p.category = 'Electronics' AND o.status = 'delivered' AND o.order_date >= '2025-01-01' AND o.order_date < '2025-07-01' "
           f"GROUP BY quarter, p.name", "why", 4, net_def + " " + ev,
           expect="analyst: price / volume / mix decomposition; Laptop Pro price cut and lower volume"),
        _q("w02", "What drove the change in total delivered order count between 2024 and 2025 (first half of each year)?",
           "SELECT substr(order_date,1,4) AS year, channel, COUNT(*) AS orders FROM orders WHERE status = 'delivered' "
           "AND substr(order_date,6,2) <= '06' AND substr(order_date,1,4) IN ('2024','2025') GROUP BY year, channel", "why", 4),
        _q("t01", "Show monthly delivered revenue for 2024 and flag any anomalous months.",
           "SELECT substr(o.order_date,1,7) AS month, SUM(oi.quantity * oi.unit_price_cents) / 100.0 AS revenue "
           "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'delivered' AND o.order_date LIKE '2024%' "
           "GROUP BY month ORDER BY month", "anomaly", 3, "",
           expect="analyst: 2024-03 flash-sale month is an anomaly"),
        _q("t02", "Forecast total monthly order counts for the next 3 months.",
           "SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders FROM orders GROUP BY month ORDER BY month",
           "forecast", 3),
    ]
    return Q


def write_all(out_dir: str | Path, seed: int = 7) -> dict:
    """Build the DB, definitions and question set under out_dir; verify every gold query runs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = build_db(out / "shoptrap.sqlite", seed)
    qs = questions()
    conn = sqlite3.connect(db)
    bad = []
    for q in qs:
        for sql in [q["gold_sql"], *q["readings"]]:
            try:
                rows = conn.execute(sql).fetchall()
                if not rows:
                    bad.append((q["id"], "empty result"))
            except sqlite3.Error as e:
                bad.append((q["id"], str(e)))
    conn.close()
    with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (out / "definitions.json").write_text(json.dumps(DEFINITIONS, indent=1), encoding="utf-8")
    (out / "README.md").write_text(__doc__.strip() + f"\n\nToday's date inside the dataset is {TODAY}. "
                                   f"{len(qs)} questions in questions.jsonl.\n", encoding="utf-8")
    return {"db": str(db), "questions": len(qs), "problems": bad}


if __name__ == "__main__":
    import sys
    print(write_all(sys.argv[1] if len(sys.argv) > 1 else "data/shoptrap"))
