# Results

`readmit run` at commit `d71f68f` on 2026-10-21: Intel(R) Xeon(R) Processor @ 2.10GHz, Python 3.13.16, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, pandas 3.0.6. 101,766 encounters, 2,423 dropped for death or hospice discharge. Test split: 19,745 encounters from 13,908 patients, 11.5% back within 30 days. Intervals: 2,000 bootstrap resamples over patients.

Shipped model, picked on validation by recall at 10% and then AUROC: **LightGBM**.

## Test split

| Ranking | AUROC (95% CI) | AUPRC | Recall at 5% | Recall at 10% (95% CI) | Precision at 10% | Recall at 20% | Brier |
|---|---|---|---|---|---|---|---|
| prior inpatient stays | 0.605 (0.591 to 0.618) | 0.175 | 0.125 | 0.205 (0.188 to 0.223) | 0.237 | 0.345 |  |
| LACE | 0.573 (0.559 to 0.585) | 0.144 | 0.078 | 0.158 (0.138 to 0.167) | 0.183 | 0.281 |  |
| logistic regression | 0.662 (0.649 to 0.674) | 0.218 | 0.135 | 0.230 (0.211 to 0.247) | 0.265 | 0.381 | 0.0979 |
| LightGBM | 0.673 (0.659 to 0.685) | 0.237 | 0.147 | 0.250 (0.232 to 0.267) | 0.288 | 0.402 | 0.0969 |
| XGBoost | 0.672 (0.659 to 0.684) | 0.239 | 0.144 | 0.248 (0.231 to 0.266) | 0.287 | 0.398 | 0.0968 |

## Paired differences (bootstrap over patients)

| Comparison | Mean | 95% CI |
|---|---|---|
| prior inpatient stays - LACE auroc | +0.032 | +0.015 to +0.048 |
| prior inpatient stays - LACE recall@10 | +0.053 | +0.033 to +0.072 |
| prior inpatient stays - logistic regression auroc | -0.057 | -0.069 to -0.046 |
| prior inpatient stays - logistic regression recall@10 | -0.023 | -0.038 to -0.008 |
| prior inpatient stays - LightGBM auroc | -0.068 | -0.079 to -0.057 |
| prior inpatient stays - LightGBM recall@10 | -0.044 | -0.060 to -0.027 |
| prior inpatient stays - XGBoost auroc | -0.068 | -0.078 to -0.057 |
| prior inpatient stays - XGBoost recall@10 | -0.042 | -0.059 to -0.027 |
| LACE - logistic regression auroc | -0.090 | -0.104 to -0.075 |
| LACE - logistic regression recall@10 | -0.076 | -0.095 to -0.056 |
| LACE - LightGBM auroc | -0.100 | -0.116 to -0.086 |
| LACE - LightGBM recall@10 | -0.097 | -0.117 to -0.077 |
| LACE - XGBoost auroc | -0.100 | -0.115 to -0.086 |
| LACE - XGBoost recall@10 | -0.095 | -0.115 to -0.075 |
| logistic regression - LightGBM auroc | -0.011 | -0.016 to -0.005 |
| logistic regression - LightGBM recall@10 | -0.021 | -0.033 to -0.008 |
| logistic regression - XGBoost auroc | -0.010 | -0.015 to -0.005 |
| logistic regression - XGBoost recall@10 | -0.019 | -0.031 to -0.007 |
| LightGBM - XGBoost auroc | +0.001 | -0.002 to +0.003 |
| LightGBM - XGBoost recall@10 | +0.002 | -0.008 to +0.011 |

## Calibration of LightGBM on test (deciles of predicted risk)

| Decile | n | Predicted | Observed |
|---|---|---|---|
| 1 | 1,975 | 0.038 | 0.041 |
| 2 | 1,975 | 0.050 | 0.062 |
| 3 | 1,975 | 0.068 | 0.064 |
| 4 | 1,975 | 0.085 | 0.084 |
| 5 | 1,975 | 0.093 | 0.093 |
| 6 | 1,974 | 0.105 | 0.105 |
| 7 | 1,974 | 0.116 | 0.108 |
| 8 | 1,974 | 0.130 | 0.133 |
| 9 | 1,974 | 0.180 | 0.175 |
| 10 | 1,974 | 0.273 | 0.289 |

## Subgroups (LightGBM; the call list is the top 10% of all test encounters)

| Split by | Group | n | Readmit rate | AUROC | Share on call list | Recall |
|---|---|---|---|---|---|---|
| age | 50 to 69 | 7,687 | 0.106 | 0.686 | 0.097 | 0.267 |
| age | 70 and over | 8,867 | 0.122 | 0.648 | 0.095 | 0.215 |
| age | under 50 | 3,191 | 0.120 | 0.700 | 0.120 | 0.311 |
| gender | Female | 10,609 | 0.116 | 0.669 | 0.103 | 0.260 |
| gender | Male | 9,136 | 0.115 | 0.677 | 0.096 | 0.237 |
| race | AfricanAmerican | 3,692 | 0.117 | 0.662 | 0.105 | 0.239 |
| race | Caucasian | 14,806 | 0.117 | 0.673 | 0.101 | 0.253 |
| race | Hispanic | 400 | 0.107 | 0.687 | 0.080 | 0.233 |
| race | Other | 288 | 0.097 | 0.715 | 0.101 | 0.393 |
| race | missing | 417 | 0.074 | 0.686 | 0.041 | 0.065 |
| admission | elective | 3,731 | 0.111 | 0.681 | 0.088 | 0.271 |
| admission | emergency | 10,421 | 0.119 | 0.665 | 0.102 | 0.240 |
| admission | other | 5,593 | 0.111 | 0.677 | 0.104 | 0.256 |
| prior inpatient stays | 1 or more | 6,582 | 0.171 | 0.635 | 0.234 | 0.389 |
| prior inpatient stays | none | 13,163 | 0.088 | 0.641 | 0.033 | 0.114 |

## Who the call list misses

Of 2,279 test readmissions, the 10% call list (1,974 encounters) reaches 569; the calibrated probabilities expected 539. 943 readmissions had no inpatient, emergency or outpatient visit in the prior year; the list reaches 110 of them, and they are 833 of the 1,710 misses. Encounters with a prior inpatient stay are 33% of test and 78% of the list.

## Reason codes on the call list

Over the 1,974 encounters on the list: how often each feature is the first reason, and how often it is among the top three.

| Feature | First reason | In top three |
|---|---|---|
| number_inpatient | 61.3% | 76.0% |
| discharge_disposition_id | 36.8% | 56.1% |
| charlson | 1.1% | 36.0% |
| payer_code | 0.1% | 29.9% |
| number_emergency | 0.7% | 16.5% |
| diag_1_group | 0.0% | 13.0% |
| medical_specialty | 0.0% | 11.9% |
| diag_2_group | 0.0% | 10.2% |
| insulin | 0.0% | 9.6% |
| lace | 0.0% | 9.3% |
| num_medications | 0.1% | 8.6% |
| number_diagnoses | 0.0% | 4.9% |
| diag_3_group | 0.0% | 4.7% |
| number_outpatient | 0.0% | 3.7% |
| admission_type_id | 0.0% | 2.6% |
| age_years | 0.0% | 2.2% |
| admission_source_id | 0.0% | 1.7% |
| num_lab_procedures | 0.0% | 1.4% |
| time_in_hospital | 0.0% | 0.9% |
| num_procedures | 0.0% | 0.4% |
| max_glu_serum | 0.0% | 0.3% |
| diabetesMed | 0.0% | 0.1% |

The five highest-risk encounters:

| Risk | Readmitted | Reasons |
|---|---|---|
| 0.67 | yes | 10 inpatient stays in the prior year; Charlson comorbidity score 10; third diagnosis: neoplasms |
| 0.67 | yes | 14 inpatient stays in the prior year; discharge disposition code 5; primary diagnosis: diabetes |
| 0.65 | no | 18 inpatient stays in the prior year; 9 emergency visits in the prior year; LACE score 13 |
| 0.64 | yes | 21 inpatient stays in the prior year; 1 emergency visit in the prior year; second diagnosis: digestive |
| 0.64 | yes | 9 inpatient stays in the prior year; Charlson comorbidity score 3; age 20 to 30 |

## Simulated shifts (see src/readmit/shift.py for what each one changes)

| Shift | Readmit rate | AUROC | Recall at 10% | Precision at 10% | Expected on list | Observed on list | Alerts |
|---|---|---|---|---|---|---|---|
| none (test as is) | 0.115 | 0.673 | 0.250 | 0.288 | 539 | 569 | none |
| older | 0.124 | 0.670 | 0.239 | 0.296 | 523 | 585 | PSI age_years |
| more prior inpatient stays | 0.136 | 0.677 | 0.234 | 0.318 | 600 | 628 | PSI number_inpatient |
| secondary diagnoses missing | 0.115 | 0.672 | 0.251 | 0.289 | 533 | 571 | missing diag_2_group, missing diag_3_group |

PSI by input for each shift:

| Input | none (test as is) | older | more prior inpatient stays | secondary diagnoses missing |
|---|---|---|---|---|
| age_years | 0.001 | 0.268 | 0.001 | 0.001 |
| number_inpatient | 0.000 | 0.001 | 0.284 | 0.000 |
| number_emergency | 0.000 | 0.000 | 0.014 | 0.000 |
| time_in_hospital | 0.000 | 0.004 | 0.007 | 0.000 |
| num_medications | 0.001 | 0.001 | 0.005 | 0.001 |
| charlson | 0.000 | 0.001 | 0.003 | 0.028 |
| lace | 0.000 | 0.003 | 0.010 | 0.004 |
| model score | 0.001 | 0.025 | 0.116 | 0.001 |

## Design checks (LightGBM retrained with one decision changed)

| Variant | Test AUROC | Recall at 10% | Test encounters |
|---|---|---|---|
| as built | 0.673 | 0.250 | 19,745 |
| split by encounter instead of patient | 0.672 | 0.226 | 19,910 |
| keep death and hospice discharges | 0.679 | 0.252 | 20,225 |
| class-weighted training (is_unbalance) | 0.672 | 0.245 | 19,745 |
| race and gender as features | 0.673 | 0.247 | 19,745 |

Early stopping on validation AUC: LightGBM 287 trees, XGBoost 175 trees. Fit time on one thread: logistic regression 0.7 s, LightGBM 4.3 s, XGBoost 4.2 s. The whole run took 119 s.

