"""Features, plus the two clinical baselines: prior inpatient visits and the LACE index.

LACE (van Walraven et al., CMAJ 2010) scores Length of stay, Acuity of admission,
Comorbidity (Charlson) and Emergency visits; its published c-statistic for death or
readmission within 30 days is 0.684. This dataset only has three diagnosis codes per
encounter and counts emergency visits over the prior year rather than six months, so the
LACE here is an approximation and probably understates the real thing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Diagnosis groups used by Strack et al. (2014) for this dataset.
def icd9_group(code) -> str:
    if not isinstance(code, str) or not code:
        return "missing"
    if code[0] in "VE":
        return "other"
    try:
        x = float(code)
    except ValueError:
        return "other"
    if 390 <= x < 460 or int(x) == 785:
        return "circulatory"
    if 460 <= x < 520 or int(x) == 786:
        return "respiratory"
    if 520 <= x < 580 or int(x) == 787:
        return "digestive"
    if int(x) == 250:
        return "diabetes"
    if 800 <= x < 1000:
        return "injury"
    if 710 <= x < 740:
        return "musculoskeletal"
    if 580 <= x < 630 or int(x) == 788:
        return "genitourinary"
    if 140 <= x < 240:
        return "neoplasms"
    return "other"


# Charlson comorbidities from ICD-9 (Deyo adaptation), as prefix patterns with weights.
_CHARLSON = [
    (r"^41[02]", 1),  # myocardial infarction
    (r"^428", 1),  # congestive heart failure
    (r"^(44[0-3]|441|7854|V434)", 1),  # peripheral vascular disease
    (r"^43[0-8]", 1),  # cerebrovascular disease
    (r"^290", 1),  # dementia
    (r"^(49[0-6]|50[0-5]|5064)", 1),  # chronic pulmonary disease
    (r"^(710[014]|714[0-2]|71481|725)", 1),  # rheumatic disease
    (r"^53[1-4]", 1),  # peptic ulcer
    (r"^(5712|571[4-6])", 1),  # mild liver disease
    (r"^250(\.?[0-37])", 1),  # diabetes without complications
    (r"^250(\.?[4-6])", 2),  # diabetes with complications
    (r"^(342|3441)", 2),  # hemiplegia
    (r"^(582|583[0-7]|58[568])", 2),  # renal disease
    (r"^(1[4-6]\d|17[0-24-9]|18\d|19[0-5]|20[0-8])", 2),  # malignancy
    (r"^(572[2-8]|456[0-2])", 3),  # severe liver disease
    (r"^19[6-9]", 6),  # metastatic tumor
    (r"^04[2-4]", 6),  # AIDS
]
_CHARLSON_RX = [(re.compile(p), w) for p, w in _CHARLSON]


def charlson(codes) -> int:
    """Charlson score from the encounter's diagnosis codes; each condition counts once."""
    total = 0
    clean = [c.replace(".", "") for c in codes if isinstance(c, str)]
    for rx, weight in _CHARLSON_RX:
        if any(rx.match(c) for c in clean):
            total += weight
    return total


def lace(df: pd.DataFrame) -> pd.Series:
    los = df["time_in_hospital"]
    l_pts = np.select([los < 1, los == 1, los == 2, los == 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
    a_pts = np.where(df["admission_type_id"].isin([1, 7]), 3, 0)  # emergency, trauma
    c = df["charlson"]
    c_pts = np.where(c >= 4, 5, c)
    e_pts = np.minimum(df["number_emergency"], 4)
    return pd.Series(l_pts + a_pts + c_pts + e_pts, index=df.index, name="lace")


AGE_MID = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}
NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age_years",
    "charlson",
    "lace",
]
CATEGORICAL = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "metformin",
    "change",
    "diabetesMed",
]
TOP_SPECIALTIES = 15


def add_features(df: pd.DataFrame, specialties: list[str] | None = None) -> pd.DataFrame:
    """Derived columns. ``specialties`` (the most common in train) keeps test from adding new ones."""
    out = df.copy()
    out["age_years"] = out["age"].map(AGE_MID)
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = out[f"diag_{i}"].map(icd9_group)
    out["charlson"] = [charlson(row) for row in out[["diag_1", "diag_2", "diag_3"]].itertuples(index=False)]
    out["lace"] = lace(out)
    if specialties is not None:
        known = out["medical_specialty"].isin(specialties)
        out["medical_specialty"] = out["medical_specialty"].where(
            known | out["medical_specialty"].isna(), "other"
        )
    for col in CATEGORICAL:
        out[col] = out[col].astype("string").fillna("missing").astype(str)
    return out


def top_specialties(train: pd.DataFrame, n: int = TOP_SPECIALTIES) -> list[str]:
    return train["medical_specialty"].value_counts().head(n).index.tolist()
