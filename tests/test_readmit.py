import numpy as np
import pandas as pd
import pytest
from synth import frame

from readmit import data as D
from readmit import evaluate as E
from readmit.features import add_features, charlson, icd9_group, lace, top_specialties
from readmit.models import GBDT, Calibrator, logistic


def test_clean_drops_death_and_hospice_and_labels_30_days():
    df = frame()
    out = D.clean(df)
    assert not out["discharge_disposition_id"].isin(D.DEATH_OR_HOSPICE).any()
    assert len(out) == (~df["discharge_disposition_id"].isin(D.DEATH_OR_HOSPICE)).sum()
    assert out["y"].sum() == (out["readmitted"] == "<30").sum()


def test_split_keeps_each_patient_in_one_split():
    parts = D.split(D.clean(frame()))
    seen = [set(p["patient_nbr"]) for p in parts.values()]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert D.bucket(12345) == D.bucket(12345) and 0 <= D.bucket(7) < 100


def test_content_hash_ignores_line_endings():
    assert D.content_sha256(b"a,b\r\n1,2\r\n") == D.content_sha256(b"a,b\n1,2\n")


def test_icd9_groups_and_charlson():
    assert icd9_group("428") == "circulatory" and icd9_group("786") == "respiratory"
    assert icd9_group("250.83") == "diabetes" and icd9_group("V57") == "other"
    assert icd9_group(None) == "missing" and icd9_group("996") == "injury"
    assert charlson(["428", "250.01", None]) == 2  # heart failure + diabetes
    assert charlson(["428", "428.0", "585"]) == 3  # each condition counts once; renal is 2
    assert charlson(["197", "250.6", "414"]) == 8  # metastatic 6 + diabetes with complications 2


def test_lace_points():
    df = pd.DataFrame(
        {
            "time_in_hospital": [1, 5, 14],
            "admission_type_id": [3, 1, 7],
            "charlson": [0, 2, 6],
            "number_emergency": [0, 1, 9],
        }
    )
    assert lace(df).tolist() == [1, 4 + 3 + 2 + 1, 7 + 3 + 5 + 4]


def test_capacity_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    r = E.at_capacity(y, s, 0.2)
    assert r["flagged"] == 2 and r["precision"] == 0.5 and r["recall"] == pytest.approx(1 / 3)
    assert r["lift"] == pytest.approx(0.5 / 0.3)


def test_bootstrap_resamples_whole_patients():
    rng = np.random.default_rng(0)
    groups = np.array([1, 1, 2, 3, 3, 3])
    idx = E.cluster_indices(groups, rng)
    counts = pd.Series(groups[idx]).value_counts()
    sizes = {1: 2, 2: 1, 3: 3}
    assert all(counts[g] % sizes[g] == 0 for g in counts.index)


def test_psi_is_zero_for_identical_and_large_for_shifted():
    x = np.random.default_rng(0).normal(size=5000)
    assert E.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert E.psi(x, x + 1.0) > 0.25


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    parts = D.split(D.clean(frame(4000)))
    spec = top_specialties(parts["train"])
    tr, va, te = (add_features(parts[k], spec) for k in ("train", "valid", "test"))
    gb = GBDT.train(tr, va, rounds=200)
    d = tmp_path_factory.mktemp("model")
    gb.save(d)
    cal = Calibrator(gb.score(va), va["y"].to_numpy())
    import json

    (d / "calibration.json").write_text(json.dumps(cal.table()))
    np.save(d / "reference_scores.npy", gb.score(va))
    return {"dir": d, "gb": gb, "test": te, "train": tr}


def test_models_learn_the_planted_signal(trained):
    te = trained["test"]
    auc = E.roc_auc_score(te["y"], trained["gb"].score(te))
    lr = logistic(trained["train"])
    lr_auc = E.roc_auc_score(te["y"], lr.predict_proba(te[lr.feature_names_in_])[:, 1])
    assert auc > 0.65 and lr_auc > 0.65


def test_saved_model_scores_identically(trained):
    loaded = GBDT.load(trained["dir"])
    np.testing.assert_allclose(loaded.score(trained["test"]), trained["gb"].score(trained["test"]))


def test_service_scores_encounters(trained, monkeypatch):
    from fastapi.testclient import TestClient

    from readmit.service import Scorer, app

    app.state.scorer = Scorer(trained["dir"])
    with TestClient(app) as c:
        assert c.get("/healthz").json()["status"] == "ok"
        body = {
            "encounters": [
                {"age": "[70-80)", "time_in_hospital": 9, "number_inpatient": 4},
                {"age": "[40-50)", "time_in_hospital": 1},
            ]
        }
        r = c.post("/score", json=body)
        assert r.status_code == 200
        out = r.json()["results"]
        assert out[0]["risk"] > out[1]["risk"] and 0 <= out[1]["risk"] <= 1
        assert c.post("/score", json={"encounters": []}).status_code == 422
        assert "readmit_scored_total" in c.get("/metrics").text
