"""Step 4 (representation component): pull the extra AACT tables Role Y needs.

02_extract.py does not pull enrolled demographics or site ZIP codes, so this
script adds them without touching the shared files:

  data/raw/baseline_demographics.parquet   enrolled sex, race, ethnicity, age (standard NIH/OMB measures)
  data/raw/result_groups_baseline.parquet  baseline groups, used to find each trial's "Total" column
  data/raw/facilities_geo.parquet          every site with zip, latitude, longitude

Same cohort as 02_extract.py (interventional, start_date START..END).
Run after 02_extract.py:  .venv/bin/python scripts/04_extract_representation.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.db import get_engine  # noqa: E402

START, END = "2008-01-01", "2020-12-31"  # keep in sync with scripts/02_extract.py
RAW = ROOT / "data" / "raw"

COHORT = f"""
    SELECT nct_id FROM ctgov.studies
    WHERE UPPER(study_type) = 'INTERVENTIONAL'
      AND start_date BETWEEN '{START}' AND '{END}'
"""

# Standard ClinicalTrials.gov baseline measures. "Customized" versions are free
# text with inconsistent categories, so they are left out (see notebook 06).
DEMO_TITLES = [
    "Sex: Female, Male",
    "Gender",
    "Race (NIH/OMB)",
    "Ethnicity (NIH/OMB)",
    "Age, Categorical",
    "Race and Ethnicity Not Collected",
]

QUERIES = {
    "baseline_demographics": f"""
        SELECT nct_id, result_group_id, ctgov_group_code, title, classification,
               category, param_type, param_value, param_value_num, number_analyzed
        FROM ctgov.baseline_measurements
        WHERE nct_id IN ({COHORT})
          AND title IN ({", ".join("'" + t + "'" for t in DEMO_TITLES)})
    """,
    "result_groups_baseline": f"""
        SELECT id, nct_id, ctgov_group_code, result_type, title
        FROM ctgov.result_groups
        WHERE nct_id IN ({COHORT}) AND UPPER(result_type) = 'BASELINE'
    """,
    "facilities_geo": f"""
        SELECT nct_id, status, city, state, zip, country, latitude, longitude
        FROM ctgov.facilities
        WHERE nct_id IN ({COHORT})
    """,
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    with engine.connect() as conn:
        for name, sql in QUERIES.items():
            chunks = pd.read_sql(text(sql), conn, chunksize=200_000)
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(RAW / f"{name}.parquet", index=False)
            print(f"{name:<24} {len(df):>10,} rows  {df['nct_id'].nunique():>8,} trials")

    with open(RAW / "SNAPSHOT.txt", "a") as f:
        f.write(f"Representation tables (04_extract_representation.py) extracted {date.today().isoformat()}\n")
    print(f"\nSaved to {RAW}")


if __name__ == "__main__":
    main()
