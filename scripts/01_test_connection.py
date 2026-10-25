"""Step 1: confirm the AACT connection works and print basic counts.

Run:  .venv/bin/python scripts/01_test_connection.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.db import get_engine  # noqa: E402

engine = get_engine()

with engine.connect() as conn:
    total = pd.read_sql("SELECT COUNT(*) AS n FROM ctgov.studies", conn)
    print(f"Connected. Total studies in AACT: {total.n[0]:,}\n")

    status = pd.read_sql(
        """
        SELECT overall_status, COUNT(*) AS n
        FROM ctgov.studies
        WHERE UPPER(study_type) = 'INTERVENTIONAL'
        GROUP BY overall_status
        ORDER BY n DESC
        """,
        conn,
    )
    print("Interventional trials by status:")
    print(status.to_string(index=False))

    tables = pd.read_sql(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'ctgov' ORDER BY table_name
        """,
        conn,
    )
    print(f"\n{len(tables)} tables available in schema ctgov:")
    print(", ".join(tables.table_name))
