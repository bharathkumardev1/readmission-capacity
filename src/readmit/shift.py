"""Simulated population shifts: what happens to the call list, and does monitoring see it?

The extract has no admission dates, so drift over time can't be studied here. Instead each
shift below is applied to the test split on purpose, and the shipped model is scored on the
result as if it were next month's discharges:

* ``older``: encounters resampled so patients 70 and over are 70% of discharges (45% in test).
* ``more prior inpatient stays``: resampled so 60% of encounters had an inpatient stay in the
  prior year (33% in test).
* ``secondary diagnoses missing``: the second and third diagnosis codes blanked on a random
  30% of encounters, as if one hospital's feed stopped sending them. Outcomes don't change;
  the inputs get worse.

The resampled shifts keep each encounter's real outcome, so recall and calibration on them
are measured, not modeled. For each shift: PSI against train for the inputs and the score
(an alert at 0.2 or more), the share of missing values in the diagnosis, payer and specialty
columns (an alert when it rises 5 points over train), the call list's recall and precision
at 10% of the shifted population, and the number of readmissions the calibrated
probabilities expected on the list against the number that happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from .features import add_features

PSI_ALERT = 0.2
MISSING_ALERT = 0.05  # a column's share of missing values up 5 points on train
WATCHED = ("age_years", "number_inpatient", "number_emergency", "time_in_hospital", "num_medications",
           "charlson", "lace")
WATCHED_MISSING = ("diag_1_group", "diag_2_group", "diag_3_group", "payer_code", "medical_specialty")


def resample(df: pd.DataFrame, mask: np.ndarray, share: float, rng: np.random.Generator) -> pd.DataFrame:
    """Same number of encounters, drawn with replacement so ``mask`` rows make up ``share``."""
    p = np.where(mask, share / mask.mean(), (1 - share) / (1 - mask.mean()))
    idx = rng.choice(len(df), size=len(df), replace=True, p=p / p.sum())
    return df.iloc[idx].reset_index(drop=True)


def drop_secondary_diagnoses(raw: pd.DataFrame, share: float, rng: np.random.Generator,
                             specialties: list[str]) -> pd.DataFrame:
    out = raw.copy()
    hit = rng.random(len(out)) < share
    out.loc[hit, ["diag_2", "diag_3"]] = np.nan
    return add_features(out, specialties)


def shifted(test: pd.DataFrame, raw_test: pd.DataFrame, specialties: list[str], seed: int = 0
            ) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return {
        "older": resample(test, (test["age_years"] >= 70).to_numpy(), 0.70, rng),
        "more prior inpatient stays": resample(test, (test["number_inpatient"] > 0).to_numpy(), 0.60, rng),
        "secondary diagnoses missing": drop_secondary_diagnoses(raw_test, 0.30, rng, specialties),
    }


def measure(name: str, df: pd.DataFrame, train: pd.DataFrame, train_score: np.ndarray, score_fn,
            prob_fn) -> dict:
    y = df["y"].to_numpy()
    s = score_fn(df)
    prob = prob_fn(s)
    at10 = E.at_capacity(y, s, 0.10)
    order = np.lexsort((np.arange(len(y)), -s))[: at10["flagged"]]
    psi = {c: E.psi(train[c].to_numpy(float), df[c].to_numpy(float)) for c in WATCHED}
    psi["model score"] = E.psi(train_score, s)
    missing = {c: {"train": float((train[c] == "missing").mean()), "now": float((df[c] == "missing").mean())}
               for c in WATCHED_MISSING}
    alerts = [f"PSI {k}" for k, v in psi.items() if v >= PSI_ALERT]
    alerts += [f"missing {k}" for k, v in missing.items() if v["now"] - v["train"] >= MISSING_ALERT]
    return {
        "shift": name, "n": int(len(y)), "readmit_rate": float(y.mean()),
        "auroc": float(E.roc_auc_score(y, s)), "recall_at_10": at10["recall"],
        "precision_at_10": at10["precision"],
        "expected_on_list": float(prob[order].sum()), "observed_on_list": int(y[order].sum()),
        "mean_predicted": float(prob.mean()), "psi": psi, "missing": missing, "alerts": sorted(alerts),
    }
