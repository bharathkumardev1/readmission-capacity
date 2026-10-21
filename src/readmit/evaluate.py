"""Metrics that match how the scores get used: a care team can call K% of discharges.

AUROC says how well a model ranks in general. The question a care-management team asks is
narrower: if we can call only the top 10% of discharged patients, how many of the people
who will be back within 30 days are on that list? So next to AUROC and calibration this
reports, at each outreach capacity, precision (share of calls that reach someone who would
have been readmitted), recall (share of all readmissions reached) and lift over calling at
random.

Intervals come from a bootstrap over patients, not encounters, since a patient's encounters
aren't independent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CAPACITIES = (0.05, 0.10, 0.20)
BOOTSTRAP = 2000


def at_capacity(y: np.ndarray, score: np.ndarray, capacity: float) -> dict[str, float]:
    """Flag the top ``capacity`` share of encounters by score (ties broken by position)."""
    n = len(y)
    k = max(1, int(round(capacity * n)))
    order = np.lexsort((np.arange(n), -score))[:k]
    caught = y[order].sum()
    base = y.mean()
    precision = caught / k
    return {
        "flagged": int(k),
        "precision": float(precision),
        "recall": float(caught / max(y.sum(), 1)),
        "lift": float(precision / base) if base else float("nan"),
    }


def summary(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict:
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }
    if prob is not None:
        out["brier"] = float(brier_score_loss(y, prob))
        out["mean_predicted"] = float(prob.mean())
        out["observed_rate"] = float(y.mean())
    for c in CAPACITIES:
        out[f"top{int(c * 100)}"] = at_capacity(y, score, c)
    return out


def cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Encounter indices of a bootstrap resample drawn over patients."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    picked = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([members[i] for i in picked])


def bootstrap(y, groups, scores: dict[str, np.ndarray], n: int = BOOTSTRAP, seed: int = 0) -> dict:
    """95% intervals for AUROC and recall at 10% for each model, and paired differences."""
    rng = np.random.default_rng(seed)
    names = list(scores)
    draws = {m: {"auroc": [], "recall@10": []} for m in names}
    for _ in range(n):
        idx = cluster_indices(groups, rng)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for m in names:
            s = scores[m][idx]
            draws[m]["auroc"].append(roc_auc_score(yy, s))
            draws[m]["recall@10"].append(at_capacity(yy, s, 0.10)["recall"])
    out = {"resamples": n, "models": {}, "paired": {}}
    for m in names:
        out["models"][m] = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws[m].items()
        }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for k in ("auroc", "recall@10"):
                d = np.asarray(draws[a][k]) - np.asarray(draws[b][k])
                out["paired"][f"{a} - {b} {k}"] = {
                    "mean": float(d.mean()),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                }
    return out


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    """Observed vs predicted rate by decile of predicted probability."""
    order = np.argsort(prob, kind="stable")
    rows = []
    for part in np.array_split(order, bins):
        rows.append(
            {"n": int(len(part)), "predicted": float(prob[part].mean()), "observed": float(y[part].mean())}
        )
    return rows


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected`` on expected's deciles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())
