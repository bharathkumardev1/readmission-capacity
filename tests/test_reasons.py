import csv
from pathlib import Path

import numpy as np
import pytest
from synth import frame

from readmit import data as D
from readmit.features import add_features, top_specialties
from readmit.models import Booster
from readmit.reasons import ADMISSION_SOURCE, ADMISSION_TYPE, DISCHARGE, describe, reason_codes, summarize

MAPPING = Path(__file__).resolve().parents[1] / "data" / D.MAPPING_NAME
# A word each description must share with the dataset's own wording in IDs_mapping.csv.
KEYWORDS = {
    ("admission_type_id", "1"): "emergency", ("admission_type_id", "2"): "urgent",
    ("admission_type_id", "3"): "elective", ("admission_type_id", "7"): "trauma",
    ("admission_source_id", "1"): "physician", ("admission_source_id", "2"): "clinic",
    ("admission_source_id", "3"): "hmo", ("admission_source_id", "7"): "emergency",
    ("discharge_disposition_id", "1"): "home", ("discharge_disposition_id", "2"): "short term",
    ("discharge_disposition_id", "3"): "snf", ("discharge_disposition_id", "6"): "home health",
    ("discharge_disposition_id", "22"): "rehab",
}
OURS = {"admission_type_id": ADMISSION_TYPE, "admission_source_id": ADMISSION_SOURCE,
        "discharge_disposition_id": DISCHARGE}


def read_mapping(path: Path) -> dict[tuple[str, str], str]:
    """IDs_mapping.csv is three tables in one file, each starting with its ``*_id`` header."""
    official: dict[tuple[str, str], str] = {}
    column = None
    for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()):
        row = [c.strip() for c in row]
        if not row or not any(row):
            continue
        if row[0].endswith("_id"):
            column = row[0]
            continue
        official[(column, row[0])] = row[1].lower() if len(row) > 1 else ""
    return official


def test_descriptions_read_like_a_note():
    assert describe("number_inpatient", 3.0) == "3 inpatient stays in the prior year"
    assert describe("number_inpatient", 1.0) == "1 inpatient stay in the prior year"
    assert describe("number_diagnoses", 9.0) == "9 diagnoses on record"
    assert describe("age_years", 75.0) == "age 70 to 80"
    assert describe("discharge_disposition_id", "99") == "discharge disposition code 99"
    assert describe("change", "Ch") == "diabetes medication changed"


def test_reason_codes_are_the_largest_positive_contributions():
    parts = D.split(D.clean(frame(4000)))
    spec = top_specialties(parts["train"])
    tr, va, te = (add_features(parts[k], spec) for k in ("train", "valid", "test"))
    model = Booster.train("lightgbm", tr, va, rounds=200)
    rows = te.head(50)
    reasons = reason_codes(model, rows)
    contrib = model.contributions(rows)[:, :-1]
    for i, rs in enumerate(reasons):
        weights = [r["log_odds"] for r in rs]
        assert weights == sorted(weights, reverse=True) and all(w > 0 for w in weights)
        if rs:
            assert rs[0]["log_odds"] == pytest.approx(contrib[i].max(), abs=1e-4)
    s = summarize(reasons)
    assert s["encounters"] == 50 and sum(s["first_reason"].values()) <= 1 + 1e-9
    assert np.isfinite(list(s["in_top3"].values())).all()


@pytest.mark.skipif(not MAPPING.exists(), reason="IDs_mapping.csv only comes with the UCI download")
def test_descriptions_match_the_datasets_own_mapping():
    official = read_mapping(MAPPING)
    for (col, code), word in KEYWORDS.items():
        assert code in OURS[col]
        assert word in official[(col, code)], (col, code, official[(col, code)])
