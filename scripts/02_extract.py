"""Step 2: pull the cohort's tables from AACT into local parquet files.

Cohort: interventional trials that started between START and END (inclusive).
Each table is saved to data/raw/<table>.parquet so the team can work offline
from one frozen copy. Re-running overwrites the files.

Run:  .venv/bin/python scripts/02_extract.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.db import get_engine  # noqa: E402

START, END = "2008-01-01", "2020-12-31"
RAW = ROOT / "data" / "raw"

COHORT = f"""
    SELECT nct_id FROM ctgov.studies
    WHERE UPPER(study_type) = 'INTERVENTIONAL'
      AND start_date BETWEEN '{START}' AND '{END}'
"""

# table -> column list ("*" = all columns). facilities is very large, so only
# the location columns are pulled.
TABLES = {
    "studies": "*",
    "eligibilities": "*",
    "designs": "*",
    "sponsors": "*",
    "countries": "*",
    "conditions": "*",
    "browse_conditions": "*",
    "interventions": "*",
    "calculated_values": "*",
    "facilities": "nct_id, city, state, country",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    with engine.connect() as conn:
        for table, cols in TABLES.items():
            sql = f"SELECT {cols} FROM ctgov.{table} WHERE nct_id IN ({COHORT})"
            chunks = pd.read_sql(sql, conn, chunksize=100_000)
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(RAW / f"{table}.parquet", index=False)
            print(f"{table:<18} {len(df):>10,} rows  {df.shape[1]:>3} cols")

    (RAW / "SNAPSHOT.txt").write_text(
        f"Extracted from AACT cloud on {date.today().isoformat()}\n"
        f"Cohort: interventional trials with start_date {START} to {END}\n"
    )
    print(f"\nSaved to {RAW}")


if __name__ == "__main__":
    main()
