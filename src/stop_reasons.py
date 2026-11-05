"""Helpers for the stop-reason labeling (notebook 07): the labeling guide, label sheets,
agreement, gold labels and weighted precision/recall."""
import importlib.util
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "stop_reason_labeling_guide.md"
LABEL_DIR = ROOT / "data" / "labels" / "stop_reasons"
LLM_DIR = LABEL_DIR / "llm"

ANNOTATORS = ["manish", "neha", "shweta", "sushma"]
CATEGORIES = ["enrollment", "safety", "efficacy", "funding", "business",
              "operational", "covid", "other", "unclear"]
YN = ["Y", "N"]

# v0 has no "unclear" category, so gold "unclear" is compared against v0 as "other"
V0_TO_GUIDE = {"enrollment": "enrollment", "safety": "safety", "efficacy": "efficacy",
               "funding": "funding", "business": "business", "pi_or_site": "operational",
               "covid": "covid", "other": "other"}


def normalize(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip().strip(" .;,:!")


def to_v0_scale(code):
    return "other" if code == "unclear" else code


def guide_for_llm():
    text = GUIDE.read_text(encoding="utf-8")
    m = re.search(r"<!-- llm-guide:start -->(.*?)<!-- llm-guide:end -->", text, re.S)
    if not m:
        raise ValueError(f"llm-guide markers not found in {GUIDE}")
    return m.group(1).strip()


def guide_examples():
    section = guide_for_llm().split("## Examples", 1)[1]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[1] in CATEGORIES and cells[2] in YN:
            rows.append(dict(zip(["text", "primary_reason", "enrollment_problem"], cells)))
    return pd.DataFrame(rows)


def v0_patterns():
    spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "scripts" / "03_build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STOP_REASONS


def n_v0_matches(text, patterns):
    t = text.lower() if isinstance(text, str) else ""
    return sum(bool(re.search(p, t)) for _, p in patterns)


def sheet_path(annotator):
    return LABEL_DIR / f"sheet_{annotator}.xlsx"


def _clean_code(x):
    x = str(x).strip().lower() if pd.notna(x) else ""
    return x if x in CATEGORIES else (np.nan if x == "" else f"INVALID:{x}")


def _clean_yn(x):
    x = str(x).strip().upper()[:1] if pd.notna(x) else ""
    return x if x in YN else (np.nan if x == "" else f"INVALID:{x}")


def read_sheets():
    """One row per (annotator, item_id). Unrecognised values are kept as "INVALID:<value>"."""
    frames = []
    for a in ANNOTATORS:
        if sheet_path(a).exists():
            s = pd.read_excel(sheet_path(a), sheet_name="labels", dtype=str)
            frames.append(s.assign(annotator=a))
    cols = ["annotator", "item_id", "batch", "primary_reason", "enrollment_problem", "notes"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["primary_reason"] = df["primary_reason"].map(_clean_code)
    df["enrollment_problem"] = df["enrollment_problem"].map(_clean_yn)
    return df[cols]


def invalid_entries(labels):
    bad = labels["primary_reason"].astype(str).str.startswith("INVALID") | \
          labels["enrollment_problem"].astype(str).str.startswith("INVALID")
    return labels[bad]


def valid(labels):
    return labels[labels["primary_reason"].isin(CATEGORIES) & labels["enrollment_problem"].isin(YN)]


def cohen_kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a.values == b.values).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(c, 0) * pb.get(c, 0) for c in set(a) | set(b))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(wide):
    """wide: items x raters, no missing values."""
    counts = wide.apply(lambda r: r.value_counts(), axis=1).fillna(0).to_numpy()
    n = counts.sum(axis=1)[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    pbar, pe = p_i.mean(), (p_j ** 2).sum()
    return np.nan if pe == 1 else (pbar - pe) / (1 - pe)


def agreement_table(labels, item_ids, column):
    """Agreement on the items in item_ids, using only items every annotator has labeled."""
    w = labels[labels["item_id"].isin(item_ids)].pivot(index="item_id", columns="annotator", values=column)
    w = w.dropna()
    if len(w) < 2 or w.shape[1] < 2:
        return None, w
    pairs = {f"{x}-{y}": cohen_kappa(w[x], w[y]) for x, y in combinations(w.columns, 2)}
    summary = pd.Series({
        "items": len(w),
        "raters": w.shape[1],
        "all_agree_%": round(100 * (w.nunique(axis=1) == 1).mean(), 1),
        "fleiss_kappa": round(fleiss_kappa(w), 3),
        "mean_pairwise_cohen": round(np.nanmean(list(pairs.values())), 3),
        **{f"cohen_{k}": round(v, 3) for k, v in pairs.items()},
    })
    return summary, w


def _majority(values):
    vc = pd.Series(values).value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return "TIE"
    return vc.index[0]


def gold_labels(labels, adjudicated=None):
    """Majority label per item ("TIE" when tied); rows in `adjudicated` override."""
    g = valid(labels).groupby("item_id").agg(
        n_raters=("annotator", "size"),
        primary_reason=("primary_reason", _majority),
        enrollment_problem=("enrollment_problem", _majority),
    )
    g["source"] = np.where(g["n_raters"] > 1, "majority", "single")
    if adjudicated is not None and len(adjudicated):
        adj = adjudicated.set_index("item_id")
        for col in ["primary_reason", "enrollment_problem"]:
            override = adj[col].dropna()
            override = override[override.index.isin(g.index)]
            g.loc[override.index, col] = override
        g.loc[g.index.isin(adj.index), "source"] = "adjudicated"
    return g.reset_index()


def weighted_binary(pred, gold, w):
    pred, gold, w = np.asarray(pred, bool), np.asarray(gold, bool), np.asarray(w, float)
    tp = (w * (pred & gold)).sum()
    prec = tp / (w * pred).sum() if (w * pred).sum() else np.nan
    rec = tp / (w * gold).sum() if (w * gold).sum() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return prec, rec, f1


def binary_report(df, pred_col, gold_col, positive="enrollment", weight="weight",
                  strata="stratum", n_boot=1000, seed=0):
    """Weighted precision, recall and F1 with stratified bootstrap 95% CIs, plus raw counts."""
    pred, gold = df[pred_col].eq(positive), df[gold_col].eq(positive)
    point = weighted_binary(pred, gold, df[weight])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(df[strata].to_numpy() == s) for s in df[strata].unique()]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boots.append(weighted_binary(pred.to_numpy()[idx], gold.to_numpy()[idx], df[weight].to_numpy()[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"estimate": point, "ci_low": lo, "ci_high": hi},
                       index=["precision", "recall", "f1"]).round(3)
    counts = pd.crosstab(pred.rename(f"{pred_col}={positive}"), gold.rename(f"{gold_col}={positive}"))
    return out, counts
