import numpy as np
import pytest
from synth import frame

from readmit import data as D
from readmit.evaluate import roc_auc_score
from readmit.features import add_features, top_specialties
from readmit.models import FEATURES, Booster, Calibrator, logistic


@pytest.fixture(scope="module")
def parts():
    p = D.split(D.clean(frame(4000)))
    spec = top_specialties(p["train"])
    return {k: add_features(v, spec) for k, v in p.items()}


@pytest.fixture(scope="module", params=["lightgbm", "xgboost"])
def booster(request, parts):
    return Booster.train(request.param, parts["train"], parts["valid"], rounds=300)


def test_models_learn_the_planted_signal(parts, booster):
    te = parts["test"]
    assert roc_auc_score(te["y"], booster.score(te)) > 0.65
    lr = logistic(parts["train"])
    assert roc_auc_score(te["y"], lr.predict_proba(te[FEATURES])[:, 1]) > 0.65


def test_contributions_add_up_to_the_score(parts, booster):
    te = parts["test"].head(200)
    contrib = booster.contributions(te)
    assert contrib.shape == (200, len(FEATURES) + 1)
    np.testing.assert_allclose(contrib.sum(axis=1), booster.score(te), rtol=1e-5, atol=1e-5)
    inpatient = FEATURES.index("number_inpatient")
    high = te["number_inpatient"].to_numpy() >= 2
    assert contrib[high, inpatient].mean() > contrib[~high, inpatient].mean()


def test_saved_model_scores_identically(parts, booster, tmp_path):
    booster.save(tmp_path)
    loaded = Booster.load(tmp_path)
    assert (loaded.kind, loaded.trees) == (booster.kind, booster.trees)
    np.testing.assert_allclose(loaded.score(parts["test"]), booster.score(parts["test"]), rtol=1e-6)


def test_unseen_categories_score_as_missing(parts, booster):
    row = parts["test"].head(1).copy()
    row["medical_specialty"] = "a specialty nobody has seen"
    assert np.isfinite(booster.score(row)).all()


def test_calibrator_round_trips_through_its_table():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-x))).astype(int)
    cal = Calibrator(x, y)
    again = Calibrator.from_table(cal.table())
    probe = np.linspace(-3, 3, 50)
    np.testing.assert_allclose(again(probe), cal(probe))
