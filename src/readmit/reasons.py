"""Reason codes for the call list: why the model put an encounter near the top.

For each encounter, the three features with the largest positive TreeSHAP contribution to
its log-odds, written the way a care coordinator would read them ("3 inpatient stays in the
prior year"). They explain the model's score, not the patient's risk factors: a reason is
what moved this model, which can include a proxy (no payer recorded) that says nothing
about care.

The three ID columns get a description for their common codes and show the code otherwise.
The descriptions are checked against the dataset's own IDs_mapping.csv, which comes in the
UCI download, by tests/test_reasons.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Booster

TOP = 3

ADMISSION_TYPE = {"1": "emergency admission", "2": "urgent admission", "3": "elective admission",
                  "7": "trauma center admission"}
ADMISSION_SOURCE = {"1": "admitted on physician referral", "2": "admitted on clinic referral",
                    "3": "admitted on HMO referral", "7": "admitted through the emergency room"}
DISCHARGE = {"1": "discharged home", "2": "transferred to another short-term hospital",
             "3": "discharged to a skilled nursing facility",
             "6": "discharged home with home health service",
             "22": "discharged to a rehab facility"}


def _plural(n: float, word: str, plural: str | None = None) -> str:
    n = int(n)
    return f"{n} {word if n == 1 else (plural or word + 's')}"


TEXT = {
    "number_inpatient": lambda v: f"{_plural(v, 'inpatient stay')} in the prior year",
    "number_emergency": lambda v: f"{_plural(v, 'emergency visit')} in the prior year",
    "number_outpatient": lambda v: f"{_plural(v, 'outpatient visit')} in the prior year",
    "time_in_hospital": lambda v: f"{int(v)}-day stay",
    "num_medications": lambda v: f"{_plural(v, 'medication')} during the stay",
    "num_lab_procedures": lambda v: f"{_plural(v, 'lab test')} during the stay",
    "num_procedures": lambda v: f"{_plural(v, 'procedure')} during the stay",
    "number_diagnoses": lambda v: f"{_plural(v, 'diagnosis', 'diagnoses')} on record",
    "age_years": lambda v: f"age {int(v) - 5} to {int(v) + 5}",
    "charlson": lambda v: f"Charlson comorbidity score {int(v)}",
    "lace": lambda v: f"LACE score {int(v)}",
    "admission_type_id": lambda v: ADMISSION_TYPE.get(v, f"admission type code {v}"),
    "admission_source_id": lambda v: ADMISSION_SOURCE.get(v, f"admission source code {v}"),
    "discharge_disposition_id": lambda v: DISCHARGE.get(v, f"discharge disposition code {v}"),
    "payer_code": lambda v: "no payer recorded" if v == "missing" else f"payer code {v}",
    "medical_specialty": lambda v: ("no admitting specialty recorded" if v == "missing"
                                    else f"admitting specialty {v}"),
    "diag_1_group": lambda v: f"primary diagnosis: {v}",
    "diag_2_group": lambda v: f"second diagnosis: {v}",
    "diag_3_group": lambda v: f"third diagnosis: {v}",
    "A1Cresult": lambda v: "no A1C result" if v in ("missing", "None") else f"A1C {v}",
    "max_glu_serum": lambda v: ("no serum glucose result" if v in ("missing", "None")
                                else f"serum glucose {v}"),
    "insulin": lambda v: f"insulin: {v.lower()}",
    "metformin": lambda v: f"metformin: {v.lower()}",
    "change": lambda v: "diabetes medication changed" if v == "Ch" else "no diabetes medication change",
    "diabetesMed": lambda v: "on diabetes medication" if v == "Yes" else "no diabetes medication",
}


def describe(feature: str, value) -> str:
    fn = TEXT.get(feature)
    return fn(value) if fn else f"{feature} = {value}"


def reason_codes(booster: Booster, df: pd.DataFrame, top: int = TOP) -> list[list[dict]]:
    """Per row, up to ``top`` features that pushed the score up, largest first."""
    contrib = booster.contributions(df)[:, :-1]
    values = df[booster.features].to_numpy(dtype=object)
    out = []
    for i in range(len(df)):
        order = np.argsort(-contrib[i], kind="stable")[:top]
        out.append([{"feature": booster.features[j], "text": describe(booster.features[j], values[i, j]),
                     "log_odds": round(float(contrib[i, j]), 4)}
                    for j in order if contrib[i, j] > 0])
    return out


def summarize(reasons: list[list[dict]]) -> dict:
    """How often each feature is the first reason, and how often it's among the top three."""
    first: dict[str, int] = {}
    any_top: dict[str, int] = {}
    for rs in reasons:
        if rs:
            first[rs[0]["feature"]] = first.get(rs[0]["feature"], 0) + 1
        for r in rs:
            any_top[r["feature"]] = any_top.get(r["feature"], 0) + 1
    n = len(reasons)
    return {"encounters": n,
            "first_reason": {k: v / n for k, v in sorted(first.items(), key=lambda kv: -kv[1])},
            "in_top3": {k: v / n for k, v in sorted(any_top.items(), key=lambda kv: -kv[1])}}
