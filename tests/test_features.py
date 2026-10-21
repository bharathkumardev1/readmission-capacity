import pandas as pd
from synth import frame

from readmit import data as D
from readmit.features import CATEGORICAL, NUMERIC, add_features, charlson, icd9_group, lace, top_specialties


def test_icd9_groups_and_charlson():
    assert icd9_group("428") == "circulatory" and icd9_group("786") == "respiratory"
    assert icd9_group("250.83") == "diabetes" and icd9_group("V57") == "other"
    assert icd9_group(None) == "missing" and icd9_group("996") == "injury"
    assert charlson(["428", "250.01", None]) == 2  # heart failure + diabetes
    assert charlson(["428", "428.0", "585"]) == 3  # each condition counts once; renal is 2
    assert charlson(["197", "250.6", "414"]) == 8  # metastatic 6 + diabetes with complications 2


def test_lace_points():
    df = pd.DataFrame({"time_in_hospital": [1, 5, 14], "admission_type_id": [3, 1, 7],
                       "charlson": [0, 2, 6], "number_emergency": [0, 1, 9]})
    assert lace(df).tolist() == [1, 4 + 3 + 2 + 1, 7 + 3 + 5 + 4]


def test_features_are_complete_and_unseen_specialties_become_other():
    parts = D.split(D.clean(frame()))
    spec = top_specialties(parts["train"], n=1)
    out = add_features(parts["test"], spec)
    assert not out[NUMERIC].isna().any().any()
    assert set(out["medical_specialty"]) <= {*spec, "other", "missing"}
    assert all(out[c].map(type).eq(str).all() for c in CATEGORICAL)
