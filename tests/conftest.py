import sqlite3

import pytest


@pytest.fixture
def db(tmp_path):
    """Tiny shop database with the traps the verifier should catch.

    customers 1:N orders (customer 1 has 3 orders), yearmonth stores dates as 'YYYYMM',
    and products has prices in cents, US-style dates and a NULL rating.
    """
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT, credit INTEGER);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, order_date TEXT);
        CREATE TABLE yearmonth (customer_id INTEGER, date TEXT, consumption REAL);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price_cents INTEGER, launched TEXT, rating REAL);

        INSERT INTO customers VALUES
            (1, 'Ann', 'East', 100), (2, 'Bob', 'East', 200), (3, 'Cy', 'East', 300),
            (4, 'Di', 'West', 400), (5, 'Ed', 'West', 500);
        INSERT INTO orders VALUES
            (1, 1, 10.0, '2012-01-05'), (2, 1, 20.0, '2012-06-10'), (3, 1, 30.0, '2012-12-20'),
            (4, 2, 15.0, '2013-02-01'), (5, 2, 25.0, '2012-03-03'),
            (6, 3, 5.0, '2013-07-07'), (7, 5, 50.0, '2012-11-11');
        INSERT INTO yearmonth VALUES
            (1, '201201', 1.5), (1, '201202', 2.5), (2, '201301', 3.0), (4, '201212', 4.0);
        INSERT INTO products VALUES
            (1, 'Lamp', 1999, '1/5/2012', 4.5), (2, 'Desk', 15000, '3/9/2012', NULL),
            (3, 'Chair', 4500, '7/1/2013', 3.5), (4, 'Rug', 2500, '2/2/2013', 4.0);
        """
    )
    conn.commit()
    conn.close()
    return str(path)
