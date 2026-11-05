"""Draw the 500-text stop-reason sample and write one labeling sheet per person.

Distinct why_stopped texts from terminated trials, stratified by v0 category.
Each text is weighted by the number of trials it stands for.
Each sheet has 200 rows: 20 calibration and 80 overlap texts that everyone labels,
plus 100 texts for that person only.

Run once:  .venv/bin/python scripts/stop_reason_sheets.py
Existing sheets are not overwritten unless --force is given.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stop_reasons import (ANNOTATORS, CATEGORIES, LABEL_DIR, YN, guide_examples,  # noqa: E402
                              n_v0_matches, normalize, sheet_path, v0_patterns)

SEED = 20261104
N_FREQUENT = 20  # most common texts, always included
ALLOC = {        # texts per stratum; 480 + 20 frequent = 500
    "enrollment_single": 70, "enrollment_multi": 80, "other": 120, "safety": 40,
    "covid": 35, "efficacy": 35, "funding": 35, "business": 35, "pi_or_site": 30,
}
N_CALIBRATION, N_OVERLAP, N_DEV_PER_PERSON = 20, 80, 20

CODE_HELP = {
    "enrollment": "not enough participants (slow/low accrual, no eligible patients, patients declined)",
    "safety": "safety concern (adverse events, toxicity, deaths, safety hold)",
    "efficacy": "results so far (futility, no benefit, or clear benefit at interim)",
    "funding": "money (funding lost/ended/not obtained, budget, grant expired)",
    "business": "sponsor/company decision, strategy, portfolio; 'sponsor decision' alone",
    "operational": "PI/staff left, site closed, drug supply, non-safety regulatory/admin, logistics",
    "covid": "COVID-19 pandemic is the main cause",
    "other": "clear reason that fits none of the above",
    "unclear": "no usable reason ('terminated', 'see description', 'investigator decision')",
}


def build_frame():
    t = pd.read_parquet(ROOT / "data/processed/trials.parquet",
                        columns=["nct_id", "overall_status", "why_stopped", "stop_reason"])
    t = t[(t["overall_status"] == "TERMINATED") & t["why_stopped"].fillna("").str.strip().ne("")].copy()
    t["norm"] = t["why_stopped"].map(normalize)

    patterns = v0_patterns()
    g = t.groupby("norm").agg(
        n_trials=("nct_id", "size"),
        text=("why_stopped", lambda s: s.value_counts().index[0]),
        v0=("stop_reason", lambda s: s.value_counts().index[0]),
    ).reset_index()
    g["n_v0_matches"] = g["text"].map(lambda x: n_v0_matches(x, patterns))

    # guide examples stay off the sheets but are kept, with their guide label, so their trials still count
    ex = guide_examples().assign(norm=lambda d: d["text"].map(normalize)).drop(columns="text")
    guide = g.merge(ex, on="norm")
    guide = guide.assign(stratum="guide", N_h=len(guide), n_h=len(guide), weight=guide["n_trials"],
                         role="guide", split="guide", item_id=[f"GX{i:02d}" for i in range(1, len(guide) + 1)])
    g = g[~g["norm"].isin(ex["norm"])].copy()

    g["stratum"] = g["v0"]
    multi = (g["v0"] == "enrollment") & (g["n_v0_matches"] >= 2)
    g.loc[g["v0"] == "enrollment", "stratum"] = np.where(multi[g["v0"] == "enrollment"],
                                                         "enrollment_multi", "enrollment_single")
    top = g.nlargest(N_FREQUENT, "n_trials").index
    g.loc[top, "stratum"] = "frequent"
    return g, guide


def draw_sample(g, rng):
    parts = [g[g["stratum"] == "frequent"].assign(N_h=N_FREQUENT, n_h=N_FREQUENT)]
    for stratum, n in ALLOC.items():
        pool = g[g["stratum"] == stratum]
        pick = pool.sample(n, random_state=rng.integers(2**31))
        parts.append(pick.assign(N_h=len(pool), n_h=n))
    s = pd.concat(parts, ignore_index=True)
    s["weight"] = s["N_h"] / s["n_h"] * s["n_trials"]
    s = s.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)
    s.insert(0, "item_id", [f"SR{i:03d}" for i in range(1, len(s) + 1)])
    return s


def assign(s, rng):
    s["role"], s["assigned_to"], s["split"] = "unique", "", "test"
    # calibration: 2 texts per stratum, topped up from "other"
    calib = []
    for stratum in ALLOC:
        calib += list(s[s["stratum"] == stratum].index[:2])
    calib += [i for i in s[s["stratum"] == "other"].index if i not in calib][:N_CALIBRATION - len(calib)]
    s.loc[calib, ["role", "assigned_to", "split"]] = ["calibration", "all", "dev"]

    rest = s[(s["role"] == "unique") & (s["stratum"] != "frequent")]
    overlap = rest.sample(N_OVERLAP, random_state=rng.integers(2**31)).index
    s.loc[overlap, ["role", "assigned_to"]] = ["overlap", "all"]

    # deal unique texts out by stratum so everyone gets a similar mix
    uniq = s[s["role"] == "unique"].sample(frac=1, random_state=rng.integers(2**31))
    uniq = uniq.sort_values("stratum", kind="stable")
    for k, idx in enumerate(uniq.index):
        s.loc[idx, "assigned_to"] = ANNOTATORS[k % len(ANNOTATORS)]
    for a in ANNOTATORS:
        mine = s[(s["role"] == "unique") & (s["assigned_to"] == a)]
        dev = mine.sample(N_DEV_PER_PERSON, random_state=rng.integers(2**31)).index
        s.loc[dev, "split"] = "dev"
    return s


def write_sheet(s, annotator, rng, path):
    calib = s[s["role"] == "calibration"].sort_values("item_id")
    main = s[(s["role"] == "overlap") | ((s["role"] == "unique") & (s["assigned_to"] == annotator))]
    main = main.sample(frac=1, random_state=rng.integers(2**31))
    rows = pd.concat([calib.assign(batch="1 - calibration"), main.assign(batch="2 - main")])
    sheet = pd.DataFrame({
        "row": range(1, len(rows) + 1),
        "item_id": rows["item_id"].to_numpy(),
        "batch": rows["batch"].to_numpy(),
        "why_stopped": rows["text"].to_numpy(),
        "primary_reason": "",
        "enrollment_problem": "",
        "notes": "",
    })
    codes = pd.DataFrame({"code": list(CODE_HELP), "use when the trial stopped because of...": list(CODE_HELP.values())})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheet.to_excel(xw, sheet_name="labels", index=False)
        codes.to_excel(xw, sheet_name="codes", index=False)

    wb = load_workbook(path)
    ws = wb["labels"]
    n = len(sheet) + 1
    for col, width in zip("ABCDEFG", [6, 9, 15, 80, 17, 19, 40]):
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2, max_row=n, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    fill = PatternFill("solid", fgColor="FFF2CC")
    for r in range(2, N_CALIBRATION + 2):
        ws[f"C{r}"].fill = fill
    ws.freeze_panes = "E2"
    dv_code = DataValidation(type="list", formula1='"' + ",".join(CATEGORIES) + '"', allow_blank=True,
                             showErrorMessage=True, error="Pick a code from the list (see the 'codes' tab).")
    dv_yn = DataValidation(type="list", formula1='"' + ",".join(YN) + '"', allow_blank=True,
                           showErrorMessage=True, error="Y or N")
    ws.add_data_validation(dv_code)
    ws.add_data_validation(dv_yn)
    dv_code.add(f"E2:E{n}")
    dv_yn.add(f"F2:F{n}")
    wc = wb["codes"]
    wc.column_dimensions["A"].width, wc.column_dimensions["B"].width = 14, 90
    wb.save(path)
    return len(sheet)


def main():
    force = "--force" in sys.argv
    existing = [sheet_path(a) for a in ANNOTATORS if sheet_path(a).exists()]
    if existing and not force:
        raise SystemExit(f"Sheets already exist ({', '.join(p.name for p in existing)}). They may hold "
                         "labels, so nothing was written. Use --force only to start over.")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    g, guide = build_frame()
    s = assign(draw_sample(g, rng), rng)

    s.drop(columns=["norm"]).to_csv(LABEL_DIR / "sample.csv", index=False)
    guide.drop(columns=["norm"]).to_csv(LABEL_DIR / "guide_items.csv", index=False)
    frame = pd.concat([g, guide])
    strata = frame.groupby("stratum").agg(distinct_texts=("norm", "size"), trials=("n_trials", "sum"))
    strata["sampled"] = pd.concat([s, guide]).groupby("stratum").size()
    strata.to_csv(LABEL_DIR / "strata.csv")

    for a in ANNOTATORS:
        n = write_sheet(s, a, rng, sheet_path(a))
        print(f"wrote {sheet_path(a).relative_to(ROOT)}  ({n} rows)")
    print(f"\nFrame: {len(frame):,} distinct texts from {frame['n_trials'].sum():,} terminated trials; "
          f"{len(guide)} of them are guide examples ({guide['n_trials'].sum():,} trials, labeled by the guide)")
    print(strata.to_string())
    print("\nRoles:", s["role"].value_counts().to_dict(), " Splits:", s["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
