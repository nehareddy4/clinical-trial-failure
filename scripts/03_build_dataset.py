"""Step 3: build the one-row-per-trial analysis dataset with outcome labels.

Reads data/raw/*.parquet (from step 2) and writes
data/processed/trials.parquet and a small CSV sample for quick inspection.

Outcome (label):
  1 = TERMINATED and why_stopped is about enrollment/accrual
  0 = COMPLETED
  NaN = everything else (other stop reasons, still running, withdrawn...)
        -- kept in the file for the statistical / competing-risks analysis.

The why_stopped categories here are a first-pass keyword version (v0);
the labeling workstream will replace them with a validated classifier.

Run:  .venv/bin/python scripts/03_build_dataset.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Checked in order; first match wins.
STOP_REASONS = [
    ("enrollment", r"accru|enrol|recruit|slow|insufficient (?:number|patients|subjects|participants)"
                   r"|lack of (?:patients|subjects|participants|eligible)|low (?:number|participation)"
                   r"|inclusion|not enough (?:patients|subjects|participants)|participation"),
    ("covid", r"covid|sars-cov|pandemic|coronavirus"),
    ("safety", r"safety|adverse|toxicit|side effect"),
    ("efficacy", r"efficacy|futility|futile|lack of (?:benefit|effect)|interim analysis"),
    ("funding", r"fund|financ|budget|grant|money|cost"),
    ("business", r"business|sponsor'?s? decision|by (?:the )?sponsor|management decision|administrative"
                 r"|development (?:program|plan)|strateg|commercial|company|portfolio|priorit"),
    ("pi_or_site", r"\bpi\b|principal investigator|investigator (?:left|leaving|moved)|staff"),
]


def classify_stop(text):
    if not isinstance(text, str) or not text.strip():
        return "not_reported"
    t = text.lower()
    for name, pattern in STOP_REASONS:
        if re.search(pattern, t):
            return name
    return "other"


def read(name):
    return pd.read_parquet(RAW / f"{name}.parquet")


def keep(df, cols):
    """Select only columns that exist (AACT columns vary slightly by release)."""
    return df[[c for c in cols if c in df.columns]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    studies = read("studies")
    df = keep(studies, [
        "nct_id", "overall_status", "why_stopped", "phase", "start_date",
        "completion_date", "enrollment", "enrollment_type", "number_of_arms",
        "has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device", "source_class",
    ]).copy()
    df["overall_status"] = df["overall_status"].str.upper()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["start_year"] = df["start_date"].dt.year

    # Outcome
    df["stop_reason"] = np.where(
        df["overall_status"] == "TERMINATED", df["why_stopped"].map(classify_stop), None
    )
    df["label"] = np.select(
        [
            (df["overall_status"] == "TERMINATED") & (df["stop_reason"] == "enrollment"),
            df["overall_status"] == "COMPLETED",
        ],
        [1, 0],
        default=np.nan,
    )

    # Lead sponsor type
    sponsors = read("sponsors")
    lead = sponsors[sponsors["lead_or_collaborator"].str.lower() == "lead"]
    df = df.merge(
        lead.drop_duplicates("nct_id")[["nct_id", "agency_class"]]
            .rename(columns={"agency_class": "lead_sponsor_class"}),
        on="nct_id", how="left",
    )
    collab = sponsors[sponsors["lead_or_collaborator"].str.lower() == "collaborator"]
    df = df.merge(collab.groupby("nct_id").size().rename("n_collaborators"),
                  on="nct_id", how="left")

    # Design
    designs = keep(read("designs"), [
        "nct_id", "allocation", "intervention_model", "primary_purpose", "masking",
    ])
    df = df.merge(designs.drop_duplicates("nct_id"), on="nct_id", how="left")

    # Eligibility (raw text kept for the NLP workstream)
    elig = keep(read("eligibilities"), [
        "nct_id", "gender", "minimum_age", "maximum_age", "healthy_volunteers",
        "adult", "child", "older_adult", "criteria",
    ])
    df = df.merge(elig.drop_duplicates("nct_id"), on="nct_id", how="left")
    df["criteria_n_chars"] = df["criteria"].str.len()

    # Counts from child tables
    for table, col in [("conditions", "n_conditions"), ("interventions", "n_interventions"),
                       ("countries", "n_countries"), ("facilities", "n_facilities")]:
        t = read(table)
        if table == "countries" and "removed" in t.columns:
            t = t[t["removed"] != True]  # noqa: E712
        df = df.merge(t.groupby("nct_id").size().rename(col), on="nct_id", how="left")
    df["us_site"] = df["nct_id"].isin(read("countries").query("name == 'United States'")["nct_id"])

    # LEAKAGE WARNING: these columns reflect the trial's final state, not its start.
    # Do not use them as model features without the version-history fix.
    df.attrs["leaky_columns"] = ["enrollment", "enrollment_type", "completion_date", "n_facilities", "n_countries"]

    df.to_parquet(OUT / "trials.parquet", index=False)
    df.drop(columns=["criteria"]).sample(min(500, len(df)), random_state=0) \
      .to_csv(OUT / "trials_sample.csv", index=False)

    print(f"Trials in cohort: {len(df):,}")
    print("\nOverall status:")
    print(df["overall_status"].value_counts().to_string())
    print("\nWhy terminated (v0 keyword labels):")
    print(df["stop_reason"].value_counts().to_string())
    print("\nModeling label (1 = terminated for low enrollment, 0 = completed):")
    print(df["label"].value_counts(dropna=False).to_string())
    pos = (df["label"] == 1).sum()
    print(f"\nPositive rate among labeled trials: {pos / df['label'].notna().sum():.1%}")
    print(f"Saved {OUT / 'trials.parquet'}")


if __name__ == "__main__":
    main()
