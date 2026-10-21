"""Train, evaluate and write results/metrics.json. ``readmit run`` calls this.

Everything a number in the README depends on happens here, in this order: the split, the
five rankings (two rules, three models), the choice of the model to ship (on validation;
test plays no part), test metrics with patient-bootstrap intervals, calibration, subgroups,
the misses, reason codes for the call list, the simulated shifts, and the design checks
that retrain with one decision undone.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E
from . import shift
from .features import DESCRIPTIVE, add_features, top_specialties
from .models import FEATURES, Booster, Calibrator, logistic
from .reasons import reason_codes, summarize

BOOSTERS = {"LightGBM": "lightgbm", "XGBoost": "xgboost"}
SUBGROUPS = {
    "age": lambda df: np.where(df["age_years"] < 50, "under 50",
                               np.where(df["age_years"] < 70, "50 to 69", "70 and over")),
    "gender": lambda df: df["gender"],
    "race": lambda df: df["race"],
    "admission": lambda df: np.where(df["admission_type_id"].isin(["1", "7"]), "emergency",
                                     np.where(df["admission_type_id"] == "3", "elective", "other")),
    "prior inpatient stays": lambda df: np.where(df["number_inpatient"] > 0, "1 or more", "none"),
}


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _data_source(data_dir: Path = Path("data")) -> str | None:
    path = data_dir / "source.json"
    return json.loads(path.read_text())["url"] if path.exists() else None


def meta(command: str) -> dict:
    import lightgbm
    import sklearn
    import xgboost

    return {"git_sha": git("rev-parse", "HEAD"),
            "data_source": _data_source(),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "command": command, "date": time.strftime("%Y-%m-%d"), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "cpu": cpu_name(),
            "cpus": os.cpu_count()}


def prepare(data_dir: Path) -> dict:
    raw = D.load(D.download(data_dir))
    cleaned = D.clean(raw)
    parts = D.split(cleaned)
    specialties = top_specialties(parts["train"])
    return {"raw": raw, "cleaned": cleaned, "raw_parts": parts, "specialties": specialties,
            **{k: add_features(v, specialties) for k, v in parts.items()}}


def flagged_mask(score: np.ndarray, capacity: float = 0.10) -> np.ndarray:
    k = max(1, int(round(capacity * len(score))))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.lexsort((np.arange(len(score)), -score))[:k]] = True
    return mask


def subgroups(test: pd.DataFrame, y: np.ndarray, score: np.ndarray) -> dict:
    flagged = flagged_mask(score)  # capacity is set on the whole population, as it would be in practice
    out = {}
    for name, fn in SUBGROUPS.items():
        labels = np.asarray(fn(test))
        rows = []
        for g in sorted(set(labels)):
            m = labels == g
            if m.sum() < 200 or y[m].min() == y[m].max():
                continue
            rows.append({"group": str(g), "n": int(m.sum()), "readmit_rate": float(y[m].mean()),
                         "auroc": float(E.roc_auc_score(y[m], score[m])),
                         "flagged_share": float(flagged[m].mean()),
                         "recall_at_10": float((flagged[m] & (y[m] == 1)).sum() / max(y[m].sum(), 1))})
        out[name] = rows
    return out


def misses(test: pd.DataFrame, y: np.ndarray, score: np.ndarray, prob: np.ndarray) -> dict:
    flagged = flagged_mask(score)
    pos = y == 1
    visits = test["number_inpatient"] + test["number_emergency"] + test["number_outpatient"]
    no_history = (visits == 0).to_numpy()
    prior_inpatient = (test["number_inpatient"] > 0).to_numpy()
    return {
        "readmissions": int(pos.sum()), "flagged": int(flagged.sum()), "caught": int((pos & flagged).sum()),
        # What the calibrated probabilities said the list would reach, before looking at outcomes.
        "expected_on_list": float(prob[flagged].sum()),
        "missed": int((pos & ~flagged).sum()),
        "missed_with_no_prior_visits": int((pos & ~flagged & no_history).sum()),
        "caught_with_no_prior_visits": int((pos & flagged & no_history).sum()),
        "readmissions_with_no_prior_visits": int((pos & no_history).sum()),
        "share_with_prior_inpatient": {"all": float(prior_inpatient.mean()),
                                       "on_list": float(prior_inpatient[flagged].mean())},
    }


def design_checks(prep: dict) -> dict:
    """What each data and modelling decision is worth: LightGBM retrained with one change."""
    raw = prep["raw"]
    with_dead = raw.assign(y=(raw["readmitted"] == "<30").astype(np.int8))
    variants = (
        ("as built", D.clean(raw), "patient_nbr", None, FEATURES),
        ("split by encounter instead of patient", D.clean(raw), "encounter_id", None, FEATURES),
        ("keep death and hospice discharges", with_dead, "patient_nbr", None, FEATURES),
        ("class-weighted training (is_unbalance)", D.clean(raw), "patient_nbr", {"is_unbalance": True},
         FEATURES),
        ("race and gender as features", D.clean(raw), "patient_nbr", None, FEATURES + DESCRIPTIVE),
    )
    out = {}
    for name, frame, key, params, features in variants:
        parts = D.split(frame, key)
        spec = top_specialties(parts["train"])
        tr, va, te = (add_features(parts[k], spec) for k in ("train", "valid", "test"))
        model = Booster.train("lightgbm", tr, va, params, features=features)
        s, y = model.score(te), te["y"].to_numpy()
        out[name] = {"test_auroc": float(E.roc_auc_score(y, s)),
                     "recall_at_10": E.at_capacity(y, s, 0.10)["recall"], "test_n": int(len(te))}
    return out


def run(data_dir: Path, model_dir: Path, command: str = "readmit run",
        resamples: int = E.BOOTSTRAP) -> tuple[dict, dict]:
    """Returns (results for metrics.json, arrays for the browser export)."""
    t_start = time.perf_counter()
    prep = prepare(data_dir)
    train, valid, test = prep["train"], prep["valid"], prep["test"]
    y_valid, y_test = valid["y"].to_numpy(), test["y"].to_numpy()

    seconds = {}
    t = time.perf_counter()
    lr = logistic(train)
    seconds["logistic regression"] = time.perf_counter() - t
    boosters = {}
    for name, kind in BOOSTERS.items():
        t = time.perf_counter()
        boosters[name] = Booster.train(kind, train, valid)
        seconds[name] = time.perf_counter() - t

    valid_scores = {"logistic regression": lr.predict_proba(valid[FEATURES])[:, 1],
                    **{n: b.score(valid) for n, b in boosters.items()}}
    test_scores = {"prior inpatient stays": test["number_inpatient"].to_numpy(dtype=float),
                   "LACE": test["lace"].to_numpy(dtype=float),
                   "logistic regression": lr.predict_proba(test[FEATURES])[:, 1],
                   **{n: b.score(test) for n, b in boosters.items()}}
    calibrators = {n: Calibrator(valid_scores[n], y_valid) for n in valid_scores}
    probs = {n: calibrators[n](test_scores[n]) for n in calibrators}

    # The model to ship is picked on validation: recall at 10%, then AUROC.
    on_valid = {n: {"recall_at_10": E.at_capacity(y_valid, valid_scores[n], 0.10)["recall"],
                    "auroc": float(E.roc_auc_score(y_valid, valid_scores[n]))} for n in boosters}
    shipped = max(boosters, key=lambda n: (on_valid[n]["recall_at_10"], on_valid[n]["auroc"]))
    model, cal = boosters[shipped], calibrators[shipped]
    s_test, p_test = test_scores[shipped], probs[shipped]

    flagged = flagged_mask(s_test)
    list_rows = test[flagged].copy()
    list_rows["_score"] = s_test[flagged]
    list_rows = list_rows.sort_values("_score", ascending=False, kind="stable")
    reasons = reason_codes(model, list_rows)
    examples = [{"risk": float(cal(np.array([r["_score"]]))[0]), "readmitted": int(r["y"]),
                 "reasons": [x["text"] for x in rs]}
                for (_, r), rs in zip(list_rows.head(5).iterrows(), reasons[:5], strict=True)]

    train_score = model.score(train)
    shifts = [shift.measure(name, df, train, train_score, model.score, cal)
              for name, df in shift.shifted(test, prep["raw_parts"]["test"], prep["specialties"]).items()]
    shifts.insert(0, shift.measure("none (test as is)", test, train, train_score, model.score, cal))

    results = {
        "meta": {**meta(command),
                 "rows": {"raw": int(len(prep["raw"])), "after_exclusions": int(len(prep["cleaned"])),
                          "excluded_death_or_hospice": int(len(prep["raw"]) - len(prep["cleaned"]))},
                 "splits": {k: {"encounters": int(len(v)), "patients": int(v["patient_nbr"].nunique()),
                                "readmit_rate": float(v["y"].mean())}
                            for k, v in (("train", train), ("valid", valid), ("test", test))},
                 "features": FEATURES, "trees": {n: b.trees for n, b in boosters.items()},
                 "fit_seconds": {k: round(v, 2) for k, v in seconds.items()}},
        "shipped": {"model": shipped, "picked_on": "validation recall at 10%, then AUROC",
                    "validation": on_valid},
        "test": {n: E.summary(y_test, s, probs.get(n)) for n, s in test_scores.items()},
        "bootstrap": E.bootstrap(y_test, test["patient_nbr"].to_numpy(), test_scores, n=resamples),
        "calibration": {n: E.calibration_table(y_test, p) for n, p in probs.items()},
        "subgroups": subgroups(test, y_test, s_test),
        "misses_at_10": misses(test, y_test, s_test, p_test),
        "reasons_at_10": {**summarize(reasons), "examples": examples},
        "psi_train_vs_test": {**{c: E.psi(train[c].to_numpy(float), test[c].to_numpy(float))
                                 for c in shift.WATCHED},
                              "model score": E.psi(train_score, s_test)},
        "shifts": shifts,
        "design_checks": design_checks(prep),
    }
    results["meta"]["seconds"] = round(time.perf_counter() - t_start, 1)

    model_dir = Path(model_dir)
    model.save(model_dir)
    (model_dir / "calibration.json").write_text(json.dumps(cal.table()))
    np.save(model_dir / "reference_scores.npy", valid_scores[shipped])  # percentiles and the list cut
    arrays = {"y": y_test, "patient": test["patient_nbr"].to_numpy(),
              **{n.replace(" ", "_"): s for n, s in test_scores.items()}}
    return results, arrays
