import numpy as np
import pandas as pd
from synth import frame

from readmit import data as D
from readmit import shift
from readmit.features import add_features, top_specialties
from readmit.models import Booster, Calibrator


def test_resample_hits_the_target_share_and_keeps_rows_whole():
    df = pd.DataFrame({"g": [1] * 300 + [0] * 700, "y": np.arange(1000) % 2})
    out = shift.resample(df, (df["g"] == 1).to_numpy(), 0.6, np.random.default_rng(0))
    assert len(out) == 1000 and abs(out["g"].mean() - 0.6) < 0.05
    assert set(map(tuple, out[["g", "y"]].to_numpy())) <= set(map(tuple, df[["g", "y"]].to_numpy()))


def test_each_shift_moves_what_it_says_and_monitoring_sees_it():
    parts = D.split(D.clean(frame(6000)))
    spec = top_specialties(parts["train"])
    tr, va, te = (add_features(parts[k], spec) for k in ("train", "valid", "test"))
    model = Booster.train("lightgbm", tr, va, rounds=200)
    cal = Calibrator(model.score(va), va["y"].to_numpy())
    shifted = shift.shifted(te, parts["test"], spec)
    assert (shifted["older"]["age_years"] >= 70).mean() > (te["age_years"] >= 70).mean() + 0.1
    assert (shifted["more prior inpatient stays"]["number_inpatient"] > 0).mean() > 0.55
    before = (te["diag_2_group"] == "missing").mean()
    after = (shifted["secondary diagnoses missing"]["diag_2_group"] == "missing").mean()
    assert abs(after - (before + 0.3 * (1 - before))) < 0.05
    res = shift.measure("secondary diagnoses missing", shifted["secondary diagnoses missing"], tr,
                        model.score(tr), model.score, cal)
    assert "missing diag_2_group" in res["alerts"] and 0 <= res["recall_at_10"] <= 1
