# readmission-capacity

A care team that can phone one in ten diabetic patients after discharge has to pick which ones. Ranked by LightGBM, that call list of 1,974 discharges holds 569 of the 2,279 test-split discharges that ended in a readmission within 30 days: 25.0% of them (95% CI 23.2 to 26.7). Calling whoever had the most inpatient stays in the prior year, the obvious rule without a model, holds 468 (20.5%), and the LACE index 361 (15.8%).

The data is the UCI Diabetes 130-US hospitals extract, 101,766 encounters from 1999 to 2008. Everything below comes from one command, `readmit run`, which takes about two minutes on one core of a 2.1 GHz Xeon, and [results/RESULTS.md](results/RESULTS.md) has every table it writes.

| Ranking of the test split | Readmissions on the 10% list (95% CI) | Precision of the list | AUROC (95% CI) | Brier |
|---|---|---|---|---|
| Prior inpatient stays | 20.5% (18.8 to 22.3) | 23.7% | 0.605 (0.591 to 0.618) | |
| LACE | 15.8% (13.8 to 16.7) | 18.3% | 0.573 (0.559 to 0.585) | |
| Logistic regression | 23.0% (21.1 to 24.7) | 26.5% | 0.662 (0.649 to 0.674) | 0.0979 |
| LightGBM, shipped | 25.0% (23.2 to 26.7) | 28.8% | 0.673 (0.659 to 0.685) | 0.0969 |
| XGBoost | 24.8% (23.1 to 26.6) | 28.7% | 0.672 (0.659 to 0.684) | 0.0968 |

Intervals come from 2,000 bootstrap resamples over patients. Paired on the same resamples, LightGBM puts 4.4 points more of the readmissions on the list than the prior-stays rule (2.7 to 6.0), which is 101 more patients, and 2.1 points more than logistic regression (0.8 to 3.3). LightGBM and XGBoost tie, 0.2 points apart (-0.8 to 1.1). LightGBM ships because it was ahead on validation; the test split played no part in choosing it.

## What goes in

- The label is the extract's own: readmitted within 30 days.
- The 2,423 encounters that ended in death or hospice are dropped. Those patients can't come back, and keeping them teaches the model that a hospice discharge means low risk. Kept in, they lift test AUROC to 0.679, because a discharge code that says the patient died predicts no readmission almost perfectly.
- The split is 60/20/20 by a hash of the patient number, so nobody appears in both train and test.
- Features: prior-year inpatient, emergency and outpatient visits; length of stay; lab, procedure and medication counts; the three diagnoses grouped as in Strack et al. (2014); a Charlson score from those ICD-9 codes; LACE; admission type and source; discharge disposition; payer; admitting specialty; A1C and glucose results; insulin, metformin and medication changes; age band.
- Race and gender aren't features. They stay in the data for the subgroup checks below. Retrained with them as features, LightGBM scores 0.673 AUROC and 24.7% on the list, no better.
- LightGBM and XGBoost get plain hyperparameters and early stopping on validation AUC (287 and 175 trees). Isotonic calibration, also fit on validation, turns scores into probabilities.

LACE here is an approximation: the extract has three diagnosis codes per encounter and counts emergency visits over a year rather than six months, so it probably understates the published index.

## What a care team would check before using it

Calibration. The probabilities said the list would hold 539 readmissions, and it held 569. By decile of predicted risk, predicted and observed rates stay within 1.6 points of each other, the widest gap being the top decile (27.3% predicted, 28.9% observed).

Who it misses. 943 of the 2,279 readmissions had no inpatient, emergency or outpatient visit in the prior year. The list holds 110 of them, and they make up 833 of the 1,710 it misses. 78% of the list had a prior inpatient stay, against 33% of all discharges. The model finds people with a history; most of what it misses are people the record says little about.

Subgroups, with the list cut at 10% of all discharges:

- Age: 31.1% of readmissions under 50 make the list, 26.7% at 50 to 69, 21.5% at 70 and over.
- Gender: 26.0% for women, 23.7% for men.
- Race: 23.9% for African American and 25.3% for Caucasian patients, the two groups with more than a few hundred encounters. For the 417 encounters with no race recorded, 4.1% make the list and 2 of their 31 readmissions do. That's too few to conclude much from, and it's the group I'd watch.

## Reason codes

Each encounter on the list comes with the three features that pushed its score up most, from exact TreeSHAP, written the way a care coordinator would read them: "14 inpatient stays in the prior year; transferred to another inpatient care facility; primary diagnosis: diabetes". Prior inpatient stays are the first reason for 61% of the list and the discharge disposition for 37%.

The plain-language descriptions of admission and discharge codes are checked in CI against IDs_mapping.csv, the code table that comes with the UCI download, and the job prints the official wording on its summary page. A reason explains the model's score, not the patient: "no payer recorded" can be a reason, and it describes the record, not the person's care.

## Simulated shifts

The extract has no admission dates, so drift over time can't be studied with it. Instead, three shifts are applied to the test split on purpose, and the shipped model scores each one as if it were next month's discharges:

| What was shifted | Readmissions on the list | Expected on list | Observed on list | Alerts |
|---|---|---|---|---|
| nothing | 25.0% | 539 | 569 | none |
| patients 70 and over made 70% of discharges (45% in test) | 23.9% | 523 | 585 | PSI on age |
| patients with a prior inpatient stay made 60% (33% in test) | 23.4% | 600 | 628 | PSI on prior inpatient stays |
| second and third diagnoses blanked on a random 30% | 25.1% | 533 | 571 | missing-value rate on both |

The first two resample real encounters with their real outcomes, so their numbers are measured rather than modeled. The older population is where calibration slips most: 523 readmissions expected on the list, 585 observed. Blanking diagnoses barely moves the list, and no PSI alert fires (the largest is 0.028, for the Charlson score, against an alert threshold of 0.2); only the missing-value check catches it, which is why monitoring needs both.

## The scoring service

`POST /score` takes encounters with the extract's field names and returns calibrated risk, the percentile among validation scores, whether the encounter makes the call list at the configured capacity (`READMIT_CAPACITY`, 10% by default) and the three reasons.

```bash
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"encounters": [{"age": "[70-80)", "time_in_hospital": 9, "number_inpatient": 4}]}'
```

In a closed-loop load test with one uvicorn worker on one core and the load generator on the other, the service scores one encounter per request at 71.7 requests a second for a single client (p50 13.3 ms, p95 18.7 ms) and tops out at 78.5 a second with four clients. In batches of 100 it scores about 515 encounters a second, most of that time going to exact TreeSHAP for the reasons. No request failed, and the server held 179 to 185 MB. Priced at half an AWS c7i.large on demand, that's $0.17 per million encounters scored one at a time and $0.024 per million in batches, assuming the throughput carries over to that vCPU, which I haven't measured. One encounter used to take 35 ms; most of it went to building a pandas frame for LightGBM twice per request, and a matrix of category codes gives it identical input at a fraction of the cost.

```bash
docker build -t readmit .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/models:ro" readmit
```

The image is `python:3.12-slim` pinned by digest, with dependencies installed from a hashed lock.

## Running it

```bash
make install    # the package and dev tools
make run        # download (checked against its SHA-256), train, evaluate: results/ and artifacts/
make test       # lint, tests, and the browser scorer against a fresh export
make loadtest   # the service on one core, the load generator on the other
make export     # trees, calibration, capacity curves and the call list for the browser demo
```

On every push, CI downloads the extract from UCI, retrains with the locked libraries, and fails if AUROC, AUPRC or Brier score move by more than 0.002 or recall at 10% by more than 0.005 from the committed results. The same job checks the reason-code descriptions against IDs_mapping.csv and the browser's JavaScript scorer against the freshly trained model.

## Limits

- The data is from 1999 to 2008, from 130 US hospitals, and covers diabetic patients only.
- Without dates, there's no out-of-time test; the split separates patients, not years.
- A readmission to a hospital outside the 130 in the database doesn't show up in the label.
- Being on the list isn't being helped. This measures who a fixed number of calls would reach, not whether calling them prevents a readmission.
- One in ten is an assumption about capacity. The browser demo lets you move it and watch the list change.

## License

MIT, see [LICENSE](LICENSE).
