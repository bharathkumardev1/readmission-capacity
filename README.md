# readmission-capacity

Which discharged diabetic patients should a care team call first, if it can only call some
of them? On the UCI extract of 101,766 inpatient encounters from 130 US hospitals
(1999–2008), a LightGBM model gives a call list the size of 10% of discharges that reaches
**24.9% of the patients who are back within 30 days** (95% CI 23.2–26.7%), 2.5 times what
calling at random would. Ranking by prior inpatient visits, the usual rule of thumb, reaches
20.5%; a LACE score reaches 15.8%. The model's probabilities are calibrated (Brier 0.097),
and the evaluation shows where the list goes blind: 41% of readmissions are patients with
no hospital visit in the prior year, and the list reaches only 12% of them.

I built readmission models at work on data I can't share. This is the same problem on
public data, set up so every number below can be checked.

```bash
pip install -e ".[dev]"
readmit run          # downloads the data, trains, evaluates: about 2 minutes on one core
```

## Results

Test split: 19,745 encounters from 13,908 patients who appear nowhere in training, 11.5%
readmitted within 30 days. Intervals are 95%, from 2,000 bootstrap resamples over
patients. Full tables are in [`results/RESULTS.md`](results/RESULTS.md), generated from
[`results/metrics.json`](results/metrics.json).

| Model | AUROC | Recall at 5% | Recall at 10% | Precision at 10% | Recall at 20% | Brier |
|---|---|---|---|---|---|---|
| Prior inpatient visits (rule) | 0.605 (0.591–0.618) | 0.125 | 0.205 (0.188–0.223) | 0.237 | 0.345 | |
| LACE index | 0.573 (0.559–0.585) | 0.078 | 0.158 (0.138–0.167) | 0.183 | 0.281 | |
| Logistic regression | 0.662 (0.649–0.675) | 0.132 | 0.229 (0.211–0.245) | 0.264 | 0.383 | 0.0980 |
| **LightGBM** | **0.674 (0.660–0.687)** | **0.143** | **0.249 (0.232–0.267)** | **0.288** | **0.397** | **0.0970** |

"Recall at 10%" is the share of all test readmissions that land on a call list holding the
top-scored 10% of encounters. "Precision at 10%" is the share of calls that reach someone who
would have been readmitted; calling at random gets 11.5%.

Paired over the same bootstrap resamples, LightGBM beats logistic regression by 0.012 AUROC
(0.006 to 0.017) and 2.2 points of recall at 10% (1.0 to 3.3), and beats the prior-visits
rule by 4.4 points of recall at 10% (2.9 to 6.0).

An AUROC of 0.67 is low next to numbers you see quoted for readmission models. This
dataset has 50 coarse columns, three diagnosis codes per stay and no labs beyond two flags,
and the outcome is noisy. For context, a JAMA systematic review of 26 readmission models
(Kansagara et al., 2011) found c-statistics from 0.55 to 0.83, and LACE scored 0.684 in its
original validation (on a different outcome: death or readmission). The interesting part
here is the gap between models at a fixed capacity, which is what a care team spends.

## What I decided, and what it's worth

| Decision | Why | What undoing it does |
|---|---|---|
| Split by patient | 16,773 patients have more than one encounter. A random split by encounter puts the same person in train and test. | On this data it didn't inflate anything (0.671 AUROC, 0.231 recall at 10%). I'd still never ship the leaky version; the check is cheap and being wrong is expensive. |
| Drop death and hospice discharges (2,423 encounters) | These patients can't be readmitted. Keeping them teaches the model that "discharged to hospice" means low risk. | AUROC 0.680 instead of 0.674, a gain that comes entirely from patients nobody would call. |
| Score at a fixed outreach capacity | A care team has a call budget, not a threshold. Recall at K% is the number that maps to patients reached. | |
| Calibrate on the validation split (isotonic) | Planning needs probabilities, not ranks: "this list will reach about 570 of 2,280 readmissions". | Nothing here: raw LightGBM probabilities were already calibrated (Brier 0.0969 raw, 0.0970 after). It stays as a guard for retrained models. |
| Early stopping on validation AUC, plain hyperparameters | 188 trees. I didn't tune much; the comparison to logistic regression is the point. | |

Things I tried or considered and didn't keep:

- **LACE as the model.** It's the standard clinical score, so it's the baseline. On this
  data it ranks worse than the prior-visits rule, partly because the extract only has
  three diagnosis codes for the Charlson part and counts emergency visits over a year
  instead of six months.
- **A time-based split.** The right split for a deployed model, and the one I'd use with
  real data. The extract has no dates, so it isn't possible here. The PSI checks below are
  a sanity check, not drift detection.
- **Class weighting.** LightGBM with `is_unbalance` got 0.671 AUROC and 0.242 recall at
  10%, slightly worse than unweighted, and it shifts the probabilities so they'd need
  recalibrating. A call list is a ranking, so reweighting has little to offer it.

## Where it fails

The list follows prior utilization. Patients with at least one inpatient visit in the
prior year are 33% of test encounters but 79% of the call list (23.8% of them are flagged,
against 3.1% of everyone else). Of the 943 readmitted patients with no inpatient,
emergency or outpatient visit in the prior year, the list reaches 109 (12%), and they make
up 834 of the 1,711 readmissions the list misses. Nothing in this extract describes those
patients well enough to find them; fixing that needs data the extract doesn't have
(social factors, labs, post-discharge follow-up).

By subgroup, AUROC is lowest for patients 70 and over (0.649, against 0.705 under 50).
Encounters with no recorded race are flagged half as often as others (4.6% against about
10%) and their recall is 0.097, against about 0.25 elsewhere, at a lower base rate (7.4%).
Recall by gender is 0.259 for women and 0.238 for men. These are the numbers I'd watch
first if this were deployed; the full table is in RESULTS.md.

## Serving and monitoring

`readmit serve` (or the Docker image) exposes:

- `POST /score`: up to 1,000 encounters per call, fields as in the UCI extract. Returns the
  calibrated 30-day risk, the encounter's percentile among validation scores, and whether
  it makes the call list at `READMIT_CAPACITY` (default 10%).
- `GET /healthz` and `GET /metrics` (Prometheus: encounters scored, latency histogram).

```bash
curl -s -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"encounters":[{"age":"[70-80)","time_in_hospital":9,"number_inpatient":4}]}'
```

For monitoring, `readmit run` reports the population stability index of key inputs and of
the model score between train and test. They're all under 0.001 here because both come
from the same period. In production I'd compute the same PSI weekly against the training
distribution, alert at 0.2, and track recall at capacity once 30-day outcomes arrive.

## Browser demo

[`web/gbdt.mjs`](web/gbdt.mjs) scores the LightGBM trees in the browser, following
LightGBM's own decision rules for missing and categorical values, and turns each model's
test ranking into a capacity curve from 2.5 KB of packed labels. The test in
[`web/test`](web/test) checks it against `booster.predict` on 2,000 test rows (largest
difference under 1e-9). The demo runs on
[bharathkumardev1.github.io](https://bharathkumardev1.github.io).

## Layout

```
src/readmit/
  data.py        download (UCI, SHA-256 pinned), exclusions, patient-level split
  features.py    ICD-9 groups, Charlson, LACE, model features
  models.py      logistic regression, LightGBM, isotonic calibration
  evaluate.py    AUROC/AUPRC/Brier, recall and precision at capacity, patient bootstrap, PSI
  experiment.py  the whole run; writes results/metrics.json
  report.py      results/RESULTS.md
  service.py     FastAPI scoring service
  export.py      trees, calibration and capacity curves for the browser
web/             JS scorer and its parity test
```

## Tests and CI

`pytest` covers the exclusions, the patient split, ICD-9 grouping, Charlson and LACE
points, the capacity metrics, the patient-level bootstrap, PSI, model save/load, and the
service, on a small synthetic frame. CI also downloads the data from UCI, checks it against
the pinned hash, retrains, fails if the headline numbers move, re-runs the JS parity test
on the fresh model, and smoke-tests the Docker image.

## What's still rough

- No external validation. One extract, one period, one random split by patient.
- LACE is approximated (three diagnosis codes, one-year emergency count).
- The capacity framing assumes every flagged call has the same cost and the same chance
  of preventing a readmission. Real outreach programs don't.
- No fairness constraint. The subgroup table measures; nothing here corrects.

## Data

Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records. BioMed Research International.
Distributed by the UCI Machine Learning Repository (dataset 296). The download is
checked against SHA-256 `da21942c…a20c4359` from UCI first and a copy in the
Fairlearn project's repository second.

## License

MIT
