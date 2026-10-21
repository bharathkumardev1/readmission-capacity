import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from synth import frame

from readmit import data as D
from readmit.features import add_features, top_specialties
from readmit.models import Booster, Calibrator


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    parts = D.split(D.clean(frame(4000)))
    spec = top_specialties(parts["train"])
    tr, va = (add_features(parts[k], spec) for k in ("train", "valid"))
    model = Booster.train("lightgbm", tr, va, rounds=200)
    d = tmp_path_factory.mktemp("model")
    model.save(d)
    (d / "calibration.json").write_text(json.dumps(Calibrator(model.score(va), va["y"].to_numpy()).table()))
    np.save(d / "reference_scores.npy", model.score(va))
    return d


@pytest.fixture
def client(model_dir):
    from readmit.service import Scorer, app

    app.state.scorer = Scorer(model_dir)
    with TestClient(app) as c:
        yield c
    del app.state.scorer


def test_scores_rank_risk_and_explain_it(client):
    assert client.get("/healthz").json()["status"] == "ok"
    body = {"encounters": [{"age": "[70-80)", "time_in_hospital": 9, "number_inpatient": 4},
                           {"age": "[40-50)", "time_in_hospital": 1}]}
    r = client.post("/score", json=body)
    assert r.status_code == 200
    high, low = r.json()["results"]
    assert high["risk"] > low["risk"] and 0 <= low["risk"] <= 1
    assert high["reasons"] and "4 inpatient stays in the prior year" in high["reasons"]


def test_bad_requests_and_metrics(client):
    assert client.post("/score", json={"encounters": []}).status_code == 422
    zero_days = {"encounters": [{"age": "[70-80)", "time_in_hospital": 0}]}
    assert client.post("/score", json=zero_days).status_code == 422
    client.post("/score", json={"encounters": [{"age": "[70-80)", "time_in_hospital": 3}]})
    text = client.get("/metrics").text
    assert "readmit_scored_total" in text and "readmit_on_call_list_total" in text
