"""Generates notebooks/07_stop_reason_validation.ipynb (run once; then execute the notebook)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# 07 — Stop-reason validation

**Question:** How accurate is the v0 keyword classifier that defines our outcome (`label = 1` = terminated for
low enrollment), and can a free-tier LLM label stop reasons as well as we do?

**Inputs**
- `data/labels/stop_reasons/sample.csv`: 500 distinct `why_stopped` texts, stratified by v0 category, with weights
  (`scripts/stop_reason_sheets.py`)
- `data/labels/stop_reasons/sheet_<name>.xlsx`: our hand labels, following `docs/stop_reason_labeling_guide.md`
- `data/labels/stop_reasons/llm/sample_<model>.csv`: LLM labels (`scripts/stop_reason_llm.py sample`)

**Weights.** Each sampled text is weighted by the number of terminated trials that share it, so weighted precision
and recall describe all 18,303 terminated trials with a stop reason. Some strata were oversampled, so unweighted
rates would be misleading."""),

code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.stop_reasons import (CATEGORIES, LABEL_DIR, LLM_DIR, V0_TO_GUIDE, agreement_table, binary_report,
                              cohen_kappa, gold_labels, invalid_entries, read_sheets, to_v0_scale, valid)
pd.set_option("display.max_colwidth", 120)

sample = pd.read_csv(LABEL_DIR / "sample.csv")
sample["v0_code"] = sample["v0"].map(V0_TO_GUIDE)
labels = read_sheets()
print(f"{len(sample)} sampled texts; label rows read: {len(labels)} from "
      f"{labels['annotator'].nunique()} sheets")"""),

md("""## 1. The sample

The frame is every distinct `why_stopped` text among TERMINATED trials. The guide's examples are not on the sheets;
they form the `guide` stratum and count with their guide label (`guide_items.csv`). The 20 most frequent remaining
texts are always included (`frequent`). `other` and multi-keyword enrollment texts are oversampled, since that is
where v0 is most likely to be wrong."""),

code("""strata = pd.read_csv(LABEL_DIR / "strata.csv", index_col=0)
strata["trials_per_sampled_text"] = (strata["trials"] / strata["sampled"]).round(1)
display(strata.sort_values("trials", ascending=False))
print(sample.groupby(["role", "split"]).size().unstack(fill_value=0))"""),

md("""## 2. v0 problems visible before any labeling

1. **Negations.** v0 takes the first keyword match, so "Business decision (not due to safety)" becomes `safety`.
2. **Missed enrollment failures.** Texts like "No patient enter this study" match none of v0's keywords and end up
   in `other`."""),

code("""t = pd.read_parquet(ROOT / "data/processed/trials.parquet", columns=["overall_status", "why_stopped", "stop_reason"])
term = t[t["overall_status"] == "TERMINATED"]
neg = term["why_stopped"].fillna("").str.contains(
    r"\\b(?:not|no|unrelated|without)\\b[^.;]{0,40}(?:safety|adverse|toxicit|side effect)", case=False)
print(f"Terminated trials whose text rules out a safety reason ('not due to safety', 'no safety concerns'): {neg.sum():,}")
print(term.loc[neg, "stop_reason"].value_counts().to_string())
print(f"\\n-> {(neg & term['stop_reason'].eq('safety')).sum():,} of {term['stop_reason'].eq('safety').sum():,} "
      "v0 'safety' labels sit on texts that rule safety out.")
print(term.loc[neg & term["stop_reason"].eq("safety"), "why_stopped"].sample(6, random_state=0).to_string())"""),

md("""## 3. Labeling progress"""),

code("""if len(labels):
    prog = labels.assign(done=labels["primary_reason"].isin(CATEGORIES) & labels["enrollment_problem"].isin(["Y", "N"]))
    display(prog.groupby(["annotator", "batch"])["done"].agg(["sum", "size"]).unstack())
    bad = invalid_entries(labels)
    if len(bad):
        print("Values to fix in the sheets:")
        display(bad)
else:
    print("No sheets found yet.")"""),

md("""## 4. Calibration round (batch 1): what to discuss

All four of us label the same 20 texts first. Below is the first-pass agreement and every text where we differed.
These are in the `dev` split, so discussing them doesn't affect the agreement reported in section 5."""),

code("""calib_ids = sample.loc[sample["role"] == "calibration", "item_id"]
v = valid(labels)
summary, wide = agreement_table(v, calib_ids, "primary_reason")
if summary is None:
    print("Waiting for at least two people to finish batch 1.")
else:
    display(summary.to_frame("primary_reason"))
    diff = wide[wide.nunique(axis=1) > 1].join(sample.set_index("item_id")["text"])
    print(f"{len(diff)} of {len(wide)} calibration texts had disagreements:")
    display(diff)"""),

md("""## 5. Inter-annotator agreement (80 overlap texts, test split)

We can't expect the LLM or v0 to agree with our gold labels more than we agree with each other.
Kappa: 0 = chance agreement, 1 = perfect agreement."""),

code("""overlap_ids = sample.loc[sample["role"] == "overlap", "item_id"]
v_bin = v.assign(is_enrollment=np.where(v["primary_reason"] == "enrollment", "Y", "N"))
rows = {}
for col, name in [("primary_reason", "primary_reason (9 codes)"),
                  ("is_enrollment", "enrollment vs not (main label)"),
                  ("enrollment_problem", "enrollment_problem (Y/N)")]:
    s, _ = agreement_table(v_bin, overlap_ids, col)
    if s is not None:
        rows[name] = s
human = pd.DataFrame(rows)
display(human if len(human) else "Waiting for overlap items from at least two people.")"""),

md("""## 6. Gold labels

Each text gets one gold label. For texts several people labeled, that's the majority vote. Ties are settled together
and written to `data/labels/stop_reasons/adjudicated.csv` (columns: `item_id, primary_reason, enrollment_problem, note`)."""),

code("""adj_path = LABEL_DIR / "adjudicated.csv"
adjudicated = pd.read_csv(adj_path, dtype=str) if adj_path.exists() else None
gold = gold_labels(labels, adjudicated) if len(v) else pd.DataFrame(columns=["item_id", "primary_reason", "enrollment_problem"])
ev = sample.merge(gold, on="item_id", how="inner")
ties = ev[(ev["primary_reason"] == "TIE") | (ev["enrollment_problem"] == "TIE")]
print(f"Gold labels: {len(ev)} of {len(sample)} texts  |  ties to adjudicate: {len(ties)}")
if len(ties):
    display(ties[["item_id", "text"]].merge(v.pivot(index="item_id", columns="annotator", values="primary_reason"),
                                            left_on="item_id", right_index=True))
ev = ev[(ev["primary_reason"] != "TIE") & (ev["enrollment_problem"] != "TIE")]
if len(ev):
    ev[["item_id", "primary_reason", "enrollment_problem"]].to_csv(LABEL_DIR / "gold.csv", index=False)
    # guide examples found in the data (e.g. "Slow accrual", 282 trials) count with their guide label
    guide_items = pd.read_csv(LABEL_DIR / "guide_items.csv").assign(v0_code=lambda d: d["v0"].map(V0_TO_GUIDE))
    ev = pd.concat([ev, guide_items], ignore_index=True)
    print(f"+ {len(guide_items)} guide-example texts ({guide_items['n_trials'].sum():,} trials) with guide labels")
    print("Weighted share of terminated trials by gold primary reason:")
    print((ev.groupby("primary_reason")["weight"].sum() / ev["weight"].sum() * 100).round(1)
          .sort_values(ascending=False).to_string())"""),

md("""## 7. How good is the v0 keyword classifier?

This uses all gold-labeled texts, since v0 was never tuned on them. It compares v0's output with our gold
`primary_reason`. Gold `unclear` is merged into `other`, because v0 doesn't have an unclear category."""),

code("""if len(ev) >= 30:
    ev["gold_v0scale"] = ev["primary_reason"].map(to_v0_scale)
    print(f"Cohen's kappa, v0 vs gold (8 categories): {cohen_kappa(ev['v0_code'], ev['gold_v0scale']):.3f}")
    print(f"Unweighted exact agreement: {(ev['v0_code'] == ev['gold_v0scale']).mean():.1%}")
    display(pd.crosstab(ev["v0_code"].rename("v0"), ev["gold_v0scale"].rename("gold")))

    print("\\nPositive class = enrollment  (weighted to all terminated trials, 95% stratified-bootstrap CI)")
    rep, counts = binary_report(ev, "v0_code", "primary_reason")
    display(rep); display(counts)

    est_true = ev.loc[ev["primary_reason"] == "enrollment", "weight"].sum()
    est_v0 = ev.loc[ev["v0_code"] == "enrollment", "weight"].sum()
    print(f"Estimated terminated-for-enrollment trials: gold-based {est_true:,.0f} vs v0 {est_v0:,.0f} "
          f"(v0 count in data: {(term['stop_reason'] == 'enrollment').sum():,})")

    ev["gold_broad"] = np.where(ev["enrollment_problem"] == "Y", "enrollment", "no")
    rep_b, _ = binary_report(ev, "v0_code", "gold_broad")
    print("\\nSensitivity: positive = any enrollment problem mentioned (enrollment_problem = Y)")
    display(rep_b)
else:
    print("Needs at least 30 gold-labeled texts.")"""),

code("""if len(ev) >= 30:
    cols = ["item_id", "stratum", "text", "v0_code", "primary_reason", "enrollment_problem"]
    print("v0 says enrollment, gold disagrees (false positives):")
    display(ev[(ev["v0_code"] == "enrollment") & (ev["primary_reason"] != "enrollment")][cols].head(15))
    print("Gold says enrollment, v0 missed it (false negatives):")
    display(ev[(ev["v0_code"] != "enrollment") & (ev["primary_reason"] == "enrollment")][cols].head(15))"""),

md("""## 8. LLM labels vs gold (test split only)

The prompt may be tuned on `dev` texts, so LLM results use `test` texts only, with v0 on the same texts for
comparison. Each file in `data/labels/stop_reasons/llm/` is one model."""),

code("""llm_files = sorted(LLM_DIR.glob("sample_*.csv")) if LLM_DIR.exists() else []
test = ev[ev["split"] == "test"] if len(ev) else ev
results = {}
for f in llm_files:
    m = pd.read_csv(f, dtype=str).rename(columns={"id": "item_id", "primary_reason": "llm_reason",
                                                   "enrollment_problem": "llm_problem"})
    d = test.merge(m[["item_id", "llm_reason", "llm_problem"]], on="item_id", how="inner").dropna(subset=["llm_reason"])
    if len(d) < 30:
        continue
    rep, _ = binary_report(d, "llm_reason", "primary_reason")
    rep_v0, _ = binary_report(d, "v0_code", "primary_reason")
    results[f.stem.removeprefix("sample_")] = {
        "texts": len(d),
        "kappa_9codes": round(cohen_kappa(d["llm_reason"], d["primary_reason"]), 3),
        "kappa_enrollment_problem": round(cohen_kappa(d["llm_problem"], d["enrollment_problem"]), 3),
        "enrollment_precision": rep.loc["precision", "estimate"],
        "enrollment_recall": rep.loc["recall", "estimate"],
        "enrollment_f1": rep.loc["f1", "estimate"],
        "v0_f1_same_texts": rep_v0.loc["f1", "estimate"],
    }
if results:
    display(pd.DataFrame(results).T)
else:
    print("No LLM label file with >= 30 gold test texts yet. Run: python scripts/stop_reason_llm.py sample")"""),

md("""## 9. Summary

*To fill in after labeling:* human agreement (section 5), v0 precision and recall (section 7), and how close the
LLM gets to human agreement (section 8).

**Before labeling:** about half of v0's `safety` labels are on texts that rule safety out, and some enrollment
failures with no matching keyword fall into `other`."""),
]

nb["cells"] = cells
out = ROOT / "notebooks" / "07_stop_reason_validation.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
