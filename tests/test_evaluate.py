import numpy as np
import pandas as pd
import pytest

from readmit import evaluate as E


def test_capacity_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    r = E.at_capacity(y, s, 0.2)
    assert r["flagged"] == 2 and r["precision"] == 0.5 and r["recall"] == pytest.approx(1 / 3)
    assert r["lift"] == pytest.approx(0.5 / 0.3)


def test_capacity_ties_get_equal_partial_credit():
    # Four encounters tie at the cutoff and two of the four list places are left for them:
    # each gets half credit, whatever order the rows are in.
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([5, 4, 4, 4, 4, 1, 1, 1, 1, 1])
    r = E.at_capacity(y, s, 0.3)
    assert r["flagged"] == 3
    assert r["recall"] == pytest.approx((1 + 2 * 2 / 4) / 3)
    perm = np.array([0, 3, 2, 1, 4, 5, 6, 7, 8, 9])
    assert E.at_capacity(y[perm], s[perm], 0.3) == r


def test_bootstrap_resamples_whole_patients():
    groups = np.array([1, 1, 2, 3, 3, 3])
    idx = E.cluster_indices(groups, np.random.default_rng(0))
    counts = pd.Series(groups[idx]).value_counts()
    sizes = {1: 2, 2: 1, 3: 3}
    assert all(counts[g] % sizes[g] == 0 for g in counts.index)


def test_bootstrap_intervals_and_paired_differences():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    noise = rng.normal(0, 1, 400)
    out = E.bootstrap(y, np.arange(400) // 2, {"good": good, "noise": noise}, n=200)
    lo, hi = out["models"]["good"]["auroc"]
    assert 0.8 < lo < hi
    d = out["paired"]["good - noise auroc"]
    assert d["ci95"][0] > 0


def test_calibration_table_and_psi():
    rng = np.random.default_rng(0)
    p = rng.random(10_000)
    y = (rng.random(10_000) < p).astype(int)
    rows = E.calibration_table(y, p)
    assert len(rows) == 10 and all(abs(r["predicted"] - r["observed"]) < 0.05 for r in rows)
    x = rng.normal(size=5000)
    assert E.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert E.psi(x, x + 1.0) > 0.25
