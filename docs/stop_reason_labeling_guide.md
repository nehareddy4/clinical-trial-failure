# Stop-reason labeling guide

We are hand-labeling 500 `why_stopped` texts from terminated trials. The labels are used for two things:

1. Measuring how accurate the keyword rules in `scripts/03_build_dataset.py` (v0) are.
2. Checking an LLM labeler against our labels before we use it on all ~13,000 distinct texts.

These labels decide which trials count as *terminated for low enrollment* (`label = 1`), so we all need to label the same way. The LLM gets the definitions, rules and examples below, word for word. If you change them, the LLM's instructions change too.

## What to do

1. Pull the repo and open `data/labels/stop_reasons/sheet_<yourname>.xlsx` (Excel, or Google Sheets and then download it again as `.xlsx` under the same name).
2. **Batch 1: calibration (rows 1–20).** Everyone gets the same 20 texts. Label them, then save, commit and push. We compare answers in notebook 07, settle any disagreements, and update this guide if needed. Then move on to batch 2.
3. **Batch 2: main (rows 21–200).** This takes about an hour, since most texts are under 80 characters.
4. Fill in `primary_reason` and `enrollment_problem` for every row (both are dropdowns). Use `notes` if you are unsure.
5. Label from the text alone. Don't look up the trial, don't edit the `why_stopped` or `item_id` columns, and don't sort the rows.
6. Commit and push **only your own sheet**. That way nobody gets merge conflicts.

<!-- llm-guide:start -->
## The two columns

**`primary_reason`**: the main reason the trial stopped. Choose one code:

| Code | Use when the text says the trial stopped because of... |
|---|---|
| `enrollment` | not enough participants: slow, low or poor accrual, recruitment or enrollment; no eligible patients found; patients declined to take part; target sample size could not be reached |
| `safety` | a safety concern: adverse events, toxicity, deaths, risk to participants, a safety-related regulatory hold |
| `efficacy` | the results so far: futility, lack of efficacy, an interim analysis showing no benefit, or showing clear benefit (stopped early because it worked) |
| `funding` | money: funding lost, ended or never obtained; budget cuts; grant expired |
| `business` | a sponsor or company decision: strategy, portfolio, development program dropped, company acquired or closed, or "sponsor decision" with no other reason |
| `operational` | how the study was run: the PI or staff left, a site closed, drug or device supply problems, a regulatory, ethics or administrative issue that is not about safety, protocol or logistics problems |
| `covid` | the COVID-19 pandemic, given as the main cause |
| `other` | a clear reason that fits none of the above (for example, a new standard of care made the study obsolete, the study was replaced by another one, or the design changed) |
| `unclear` | no usable reason: "terminated", "see detailed description", "investigator decision", or text that cannot be interpreted |

**`enrollment_problem`**: `Y` if the text says too few participants were enrolled or available (slow, low or insufficient accrual, recruitment or enrollment; no eligible patients), **whatever the primary reason is**. Otherwise `N`.

## Rules

1. If the text gives one reason, use its code.
2. If the text gives a cause ("A because of B"), use the cause B. For example, "Funding withdrawn due to slow accrual" is `enrollment`, and "Low enrollment due to COVID-19" is `covid` with `enrollment_problem = Y`.
3. Recruitment trouble explained by the patients is still `enrollment`. This covers competing trials, strict criteria, patients refusing randomization, a rare condition, and patients no longer available.
4. If the text lists several independent reasons ("low accrual and loss of funding"), use the one listed first, unless the text says another one is the main reason. Still set `enrollment_problem = Y` if any listed reason is an enrollment problem.
5. Use `covid` only when the pandemic is the main cause. "Slow inclusion (partly due to Covid-19)" is `enrollment`.
6. Ignore reasons the text rules out. "Business decision, not due to safety" is `business`.
7. "Sponsor decision" on its own is `business`. "Investigator decision" or "PI decision" on its own is `unclear`.
8. "Recruitment" problems caused by staff (for example "no research assistant to recruit") are `operational`, not `enrollment`, unless the text also says there were too few participants.
9. Use only what the text says. If two codes fit, choose the better one and write a note.

## Examples

| why_stopped | primary_reason | enrollment_problem |
|---|---|---|
| Slow accrual | enrollment | Y |
| No patient enter this study. | enrollment | Y |
| Unable to find enough patients who accept randomization | enrollment | Y |
| We no longer had an appropriate patient population. Study closed | enrollment | Y |
| Lack of accrual and funding expires in June, 2014. | enrollment | Y |
| Slow enrollment and PI departure | enrollment | Y |
| Funding was withdrawn because of slow accrual | enrollment | Y |
| Slow inclusion rate (partly due to Covid-19) | enrollment | Y |
| Recruitment stopped due to lack of research assistant | operational | N |
| Study halted due to COVID-19 pandemic; enrollment could not resume | covid | Y |
| COVID-19 | covid | N |
| Terminated due to liver findings | safety | N |
| Safety concerns; enrollment was also slower than planned | safety | Y |
| Futility | efficacy | N |
| Interim analysis showed clear benefit of the study drug | efficacy | N |
| Lack of funding | funding | N |
| Business decision (not due to safety reason). | business | N |
| Sponsor decision | business | N |
| PI left institution | operational | N |
| Drug availability | operational | N |
| IRB approval lapsed and was not renewed | operational | N |
| Approval of new agents for this disease made the study obsolete | other | N |
| Investigator decision | unclear | N |
| See termination reason in detailed description | unclear | N |
<!-- llm-guide:end -->

## From labels to the model's outcome (team decision)

- **Proposed main definition:** `label = 1` when `primary_reason == enrollment`.
- **Sensitivity check:** `label = 1` when `enrollment_problem == Y`. This also counts COVID-caused shortfalls and texts where low enrollment is one of several reasons.
- **Matching Report 1's six categories:** `operational`, `covid` and `unclear` merge into `other`.

When two people disagree on an overlap item, we decide by majority vote. Ties are settled together and recorded in `data/labels/stop_reasons/adjudicated.csv`.
