# Results

`readmit run` at commit `7dea816` on 2026-10-20: Python 3.13.16, LightGBM 4.7.0, scikit-learn 1.9.1. 101,766 encounters, 2,423 dropped for death or hospice discharge. Test split: 19,745 encounters from 13,908 patients, 11.5% readmitted within 30 days. Intervals: 2,000 bootstrap resamples over patients.

## Test split

| Model | AUROC (95% CI) | AUPRC | Recall at 5% | Recall at 10% (95% CI) | Precision at 10% | Recall at 20% | Brier |
|---|---|---|---|---|---|---|---|
| prior inpatient visits | 0.605 (0.591–0.618) | 0.175 | 0.125 | 0.205 (0.188–0.223) | 0.237 | 0.345 |  |
| LACE | 0.573 (0.559–0.585) | 0.144 | 0.078 | 0.158 (0.138–0.167) | 0.183 | 0.281 |  |
| logistic regression | 0.662 (0.649–0.675) | 0.219 | 0.132 | 0.229 (0.211–0.245) | 0.264 | 0.383 | 0.0980 |
| LightGBM | 0.674 (0.660–0.687) | 0.235 | 0.143 | 0.249 (0.232–0.267) | 0.288 | 0.397 | 0.0970 |

## Paired differences (bootstrap over patients)

| Comparison | Mean | 95% CI |
|---|---|---|
| prior inpatient visits - LACE auroc | +0.032 | +0.015 to +0.048 |
| prior inpatient visits - LACE recall@10 | +0.053 | +0.033 to +0.072 |
| prior inpatient visits - logistic regression auroc | -0.058 | -0.069 to -0.047 |
| prior inpatient visits - logistic regression recall@10 | -0.022 | -0.037 to -0.008 |
| prior inpatient visits - LightGBM auroc | -0.069 | -0.081 to -0.058 |
| prior inpatient visits - LightGBM recall@10 | -0.044 | -0.060 to -0.029 |
| LACE - logistic regression auroc | -0.090 | -0.104 to -0.076 |
| LACE - logistic regression recall@10 | -0.075 | -0.093 to -0.055 |
| LACE - LightGBM auroc | -0.102 | -0.116 to -0.087 |
| LACE - LightGBM recall@10 | -0.097 | -0.117 to -0.076 |
| logistic regression - LightGBM auroc | -0.012 | -0.017 to -0.006 |
| logistic regression - LightGBM recall@10 | -0.022 | -0.033 to -0.010 |

## Calibration of LightGBM on test (deciles of predicted risk)

| Decile | n | Predicted | Observed |
|---|---|---|---|
| 1 | 1,975 | 0.037 | 0.042 |
| 2 | 1,975 | 0.054 | 0.063 |
| 3 | 1,975 | 0.066 | 0.063 |
| 4 | 1,975 | 0.080 | 0.079 |
| 5 | 1,975 | 0.093 | 0.092 |
| 6 | 1,974 | 0.106 | 0.107 |
| 7 | 1,974 | 0.116 | 0.108 |
| 8 | 1,974 | 0.137 | 0.144 |
| 9 | 1,974 | 0.169 | 0.168 |
| 10 | 1,974 | 0.277 | 0.288 |

## Subgroups (LightGBM; call list is the top 10% of all test encounters)

| Split by | Group | n | Readmit rate | AUROC | Share on call list | Recall |
|---|---|---|---|---|---|---|
| age | 50 to 69 | 7,687 | 0.106 | 0.686 | 0.099 | 0.258 |
| age | 70 and over | 8,867 | 0.122 | 0.649 | 0.094 | 0.219 |
| age | under 50 | 3,191 | 0.120 | 0.705 | 0.120 | 0.316 |
| gender | Female | 10,609 | 0.116 | 0.670 | 0.104 | 0.259 |
| gender | Male | 9,136 | 0.115 | 0.679 | 0.096 | 0.238 |
| race | AfricanAmerican | 3,692 | 0.117 | 0.664 | 0.098 | 0.232 |
| race | Caucasian | 14,806 | 0.117 | 0.673 | 0.103 | 0.255 |
| race | Hispanic | 400 | 0.107 | 0.696 | 0.080 | 0.233 |
| race | Other | 288 | 0.097 | 0.726 | 0.094 | 0.321 |
| race | missing | 417 | 0.074 | 0.690 | 0.046 | 0.097 |
| admission | elective | 3,731 | 0.111 | 0.687 | 0.091 | 0.278 |
| admission | emergency | 10,421 | 0.119 | 0.668 | 0.105 | 0.248 |
| admission | other | 5,593 | 0.111 | 0.673 | 0.097 | 0.232 |
| prior inpatient visits | 1 or more | 6,582 | 0.171 | 0.634 | 0.238 | 0.393 |
| prior inpatient visits | none | 13,163 | 0.088 | 0.643 | 0.031 | 0.110 |

## Who the call list misses

Of 2,279 test readmissions, the 10% call list reaches 568. 41% of all readmissions had no inpatient, emergency or outpatient visit in the prior year; the list reaches 109 of them, and they are 834 of the 1,711 misses.

## Stability (PSI, train vs test)

| Feature | PSI |
|---|---|
| number_inpatient | 0.0000 |
| time_in_hospital | 0.0005 |
| num_medications | 0.0006 |
| lace | 0.0004 |
| model score | 0.0006 |

## Design checks (LightGBM retrained with one decision changed)

| Variant | Test AUROC | Recall at 10% | Test encounters |
|---|---|---|---|
| as built: exclusions, split by patient | 0.674 | 0.249 | 19,745 |
| split by encounter instead of patient | 0.671 | 0.231 | 19,910 |
| keep death and hospice discharges | 0.680 | 0.248 | 20,225 |
| class-weighted training (is_unbalance) | 0.671 | 0.242 | 19,745 |

LightGBM: 188 trees (early stopping on validation AUC), fit in 4.3 s on one thread.

