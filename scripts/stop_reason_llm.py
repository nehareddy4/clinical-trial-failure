"""Label why_stopped texts with an LLM through an OpenAI-compatible API (e.g. Groq, Gemini, Ollama).

The prompt is the llm-guide section of docs/stop_reason_labeling_guide.md.
Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env (see .env.example).

Run:
  .venv/bin/python scripts/stop_reason_llm.py sample           # the 500 sampled texts
  .venv/bin/python scripts/stop_reason_llm.py all              # all distinct terminated texts
  .venv/bin/python scripts/stop_reason_llm.py all --shard 2/4  # second quarter of "all"

Results are saved to data/labels/stop_reasons/llm/ after every request; re-running continues
where the last run stopped.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stop_reasons import CATEGORIES, LABEL_DIR, LLM_DIR, YN, guide_for_llm, normalize  # noqa: E402

load_dotenv(ROOT / ".env")

SYSTEM = (
    "You label why clinical trials were terminated, using the team's labeling guide below. "
    "Follow its definitions, rules and examples exactly. Label only from the text given.\n\n"
    + guide_for_llm()
    + "\n\nAnswer with JSON only, in exactly this form, one entry per input item, same order:\n"
    '{"labels": [{"id": "<item id>", "primary_reason": "<one code>", "enrollment_problem": "Y or N"}]}\n'
    f"Allowed primary_reason codes: {', '.join(CATEGORIES)}."
)


class LLM:
    def __init__(self):
        self.base = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")
        self.interval = float(os.getenv("LLM_MIN_INTERVAL", "3"))
        self.json_mode = True
        self.last = 0.0
        if not self.base or not self.model:
            raise SystemExit("Set LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY) in .env; see .env.example.")

    def chat(self, user, attempts=6):
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(attempts):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            req = urllib.request.Request(
                f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 400 and self.json_mode and "response_format" in detail.lower():
                    self.json_mode = False  # provider has no JSON mode
                    body.pop("response_format")
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    retry = e.headers.get("Retry-After")
                    delay = float(retry) if retry and retry.replace(".", "").isdigit() else 10 * 2 ** attempt
                    print(f"  HTTP {e.code}; waiting {delay:.0f}s ({detail[:80]!r})")
                    time.sleep(min(delay, 600))
                    continue
                raise SystemExit(f"HTTP {e.code} from {self.base}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  connection problem ({e}); retrying")
                time.sleep(10 * 2 ** attempt)
        raise SystemExit("Giving up after repeated errors. Re-run later; finished batches are saved.")


def parse(content):
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = obj.get("labels", []) if isinstance(obj, dict) else obj
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        code = str(it.get("primary_reason", "")).strip().lower()
        yn = str(it.get("enrollment_problem", "")).strip().upper()[:1]
        out.append({"id": str(it.get("id", "")).strip(), "primary_reason": code if code in CATEGORIES else None,
                    "enrollment_problem": yn if yn in YN else None})
    return out


def load_items(target):
    if target == "sample":
        s = pd.read_csv(LABEL_DIR / "sample.csv")
        return pd.DataFrame({"id": s["item_id"], "text": s["text"]})
    t = pd.read_parquet(ROOT / "data/processed/trials.parquet", columns=["overall_status", "why_stopped"])
    t = t[(t["overall_status"] == "TERMINATED") & t["why_stopped"].fillna("").str.strip().ne("")]
    t = t.assign(norm=t["why_stopped"].map(normalize))
    variants = t.groupby(["norm", "why_stopped"]).size().rename("n").reset_index()
    g = variants.sort_values("n", ascending=False).drop_duplicates("norm")  # most common wording
    g["id"] = g["norm"].map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:12])
    return g.rename(columns={"why_stopped": "text"})[["id", "text", "norm"]].sort_values("id")


def label_batch(llm, batch):
    payload = json.dumps([{"id": i, "text": t} for i, t in zip(batch["id"], batch["text"])], ensure_ascii=False)
    got = {r["id"]: r for r in parse(llm.chat("Label these items:\n" + payload)) if r["id"]}
    rows = []
    for i, t in zip(batch["id"], batch["text"]):
        r = got.get(i)
        if (r is None or not r["primary_reason"] or not r["enrollment_problem"]) and len(batch) > 1:
            r = (parse(llm.chat("Label these items:\n" + json.dumps([{"id": i, "text": t}], ensure_ascii=False)))
                 or [{}])[0]  # retry missing answers one at a time
        rows.append({"id": i, "primary_reason": (r or {}).get("primary_reason"),
                     "enrollment_problem": (r or {}).get("enrollment_problem")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["sample", "all"])
    ap.add_argument("--shard", help="k/n: label only the k-th of n equal parts (e.g. 2/4)")
    ap.add_argument("--limit", type=int, help="stop after this many new texts (trial runs)")
    args = ap.parse_args()

    llm = LLM()
    items = load_items(args.target)
    suffix = ""
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        items = items.iloc[k - 1::n]
        suffix = f"_shard-{k}-of-{n}"
    slug = re.sub(r"[^a-z0-9]+", "-", llm.model.lower()).strip("-")
    out = LLM_DIR / f"{args.target}_{slug}{suffix}.csv"
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    done = pd.read_csv(out, dtype=str) if out.exists() else pd.DataFrame(columns=["id"])
    done = done[done.get("primary_reason", pd.Series(dtype=str)).notna()] if len(done) else done
    todo = items[~items["id"].isin(set(done["id"]))]
    print(f"{llm.model}: {len(items) - len(todo):,} of {len(items):,} already labeled -> {out.relative_to(ROOT)}")
    if args.limit:
        todo = todo.head(args.limit)
    size = int(os.getenv("LLM_BATCH_SIZE", "20"))

    for start in range(0, len(todo), size):
        batch = todo.iloc[start:start + size]
        res = label_batch(llm, batch).merge(batch, on="id")
        res["model"] = llm.model
        res["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        done = pd.concat([done[~done["id"].isin(res["id"])], res], ignore_index=True)
        done.to_csv(out, index=False)
        bad = res["primary_reason"].isna().sum()
        print(f"  {min(start + size, len(todo)):,}/{len(todo):,}" + (f"  ({bad} unparsed)" if bad else ""))
    print("done")


if __name__ == "__main__":
    main()
