"""Train, evaluate, and write results/metrics.json. ``readmit run`` calls this."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E
from .features import add_features, top_specialties
from .models import GBDT, Calibrator, logistic

SUBGROUPS = {
    "age": lambda df: np.where(
        df["age_years"] < 50, "under 50", np.where(df["age_years"] < 70, "50 to 69", "70 and over")
    ),
    "gender": lambda df: df["gender"],
    "race": lambda df: df["race"],
    "admission": lambda df: np.where(
        df["admission_type_id"].isin(["1", "7"]),
        "emergency",
        np.where(df["admission_type_id"] == "3", "elective", "other"),
    ),
    "prior inpatient visits": lambda df: np.where(df["number_inpatient"] > 0, "1 or more", "none"),
}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def prepare(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw = D.load(D.download(data_dir))
    cleaned = D.clean(raw)
    parts = D.split(cleaned)
    specialties = top_specialties(parts["train"])
    out = {k: add_features(v, specialties) for k, v in parts.items()}
    out["_counts"] = pd.DataFrame(
        [
            {
                "raw": len(raw),
                "after_exclusions": len(cleaned),
                "excluded_death_or_hospice": len(raw) - len(cleaned),
            }
        ]
    )
    return out


def design_checks(data_dir: Path) -> dict:
    """What the two data decisions in readmit.data are worth: test AUROC with each one undone."""
    raw = D.load(D.download(data_dir))
    with_dead = raw.assign(y=(raw["readmitted"] == "<30").astype(np.int8))
    variants = (
        ("as built: exclusions, split by patient", D.clean(raw), "patient_nbr", None),
        ("split by encounter instead of patient", D.clean(raw), "encounter_id", None),
        ("keep death and hospice discharges", with_dead, "patient_nbr", None),
        ("class-weighted training (is_unbalance)", D.clean(raw), "patient_nbr", {"is_unbalance": True}),
    )
    out = {}
    for name, frame, key, params in variants:
        b = frame[key].map(D.bucket)  # encounter_id here is the leaky version
        parts = {k: frame[(b >= lo) & (b < hi)].reset_index(drop=True) for k, (lo, hi) in D.SPLITS.items()}
        spec = top_specialties(parts["train"])
        tr, va, te = (add_features(parts[k], spec) for k in ("train", "valid", "test"))
        gb = GBDT.train(tr, va, params)
        s, y = gb.score(te), te["y"].to_numpy()
        out[name] = {
            "test_auroc": float(E.roc_auc_score(y, s)),
            "recall_at_10": E.at_capacity(y, s, 0.10)["recall"],
            "test_n": int(len(te)),
        }
    return out


def run(data_dir: Path, out_dir: Path, model_dir: Path) -> dict:
    t0 = time.perf_counter()
    parts = prepare(data_dir)
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    y_test = test["y"].to_numpy()
    groups = test["patient_nbr"].to_numpy()

    timings = {}
    t = time.perf_counter()
    lr = logistic(train)
    timings["logistic_fit_s"] = time.perf_counter() - t
    t = time.perf_counter()
    gb = GBDT.train(train, valid)
    timings["lightgbm_fit_s"] = time.perf_counter() - t

    lr_valid = lr.predict_proba(valid[lr.feature_names_in_])[:, 1]
    lr_test = lr.predict_proba(test[lr.feature_names_in_])[:, 1]
    gb_valid, gb_test = gb.score(valid), gb.score(test)
    cal = Calibrator(gb_valid, valid["y"].to_numpy())
    lr_cal = Calibrator(lr_valid, valid["y"].to_numpy())
    gb_prob, lr_prob = cal(gb_test), lr_cal(lr_test)

    scores = {
        "prior inpatient visits": test["number_inpatient"].to_numpy(dtype=float),
        "LACE": test["lace"].to_numpy(dtype=float),
        "logistic regression": lr_test,
        "LightGBM": gb_test,
    }
    probs = {"logistic regression": lr_prob, "LightGBM": gb_prob}
    models = {m: E.summary(y_test, s, probs.get(m)) for m, s in scores.items()}
    models["LightGBM"]["brier_uncalibrated_sigmoid"] = float(
        np.mean((1 / (1 + np.exp(-gb_test)) - y_test) ** 2)
    )

    subgroups = {}
    for name, fn in SUBGROUPS.items():
        labels = np.asarray(fn(test))
        rows = []
        for g in sorted(set(labels)):
            m = labels == g
            if m.sum() < 200 or y_test[m].min() == y_test[m].max():
                continue
            # Capacity is set on the whole population, as it would be in practice.
            k = int(round(0.10 * len(y_test)))
            flagged = np.zeros(len(y_test), dtype=bool)
            flagged[np.lexsort((np.arange(len(y_test)), -gb_test))[:k]] = True
            rows.append(
                {
                    "group": str(g),
                    "n": int(m.sum()),
                    "readmit_rate": float(y_test[m].mean()),
                    "auroc": float(E.roc_auc_score(y_test[m], gb_test[m])),
                    "flagged_share": float(flagged[m].mean()),
                    "recall_at_10": float((flagged[m] & (y_test[m] == 1)).sum() / max(y_test[m].sum(), 1)),
                }
            )
        subgroups[name] = rows

    k = int(round(0.10 * len(y_test)))
    flagged = np.zeros(len(y_test), dtype=bool)
    flagged[np.lexsort((np.arange(len(y_test)), -gb_test))[:k]] = True
    pos = y_test == 1
    no_history = (
        test["number_inpatient"] + test["number_emergency"] + test["number_outpatient"] == 0
    ).to_numpy()
    misses = {
        "readmissions": int(pos.sum()),
        "caught_at_10": int((pos & flagged).sum()),
        "missed_with_no_prior_visits": int((pos & ~flagged & no_history).sum()),
        "missed_total": int((pos & ~flagged).sum()),
        "caught_with_no_prior_visits": int((pos & flagged & no_history).sum()),
        "share_of_all_readmissions_with_no_prior_visits": float((pos & no_history).sum() / pos.sum()),
    }

    drift = {
        col: E.psi(train[col].to_numpy(float), test[col].to_numpy(float))
        for col in ("number_inpatient", "time_in_hospital", "num_medications", "lace")
    }
    drift["model score"] = E.psi(gb.score(train), gb_test)

    importance = sorted(
        zip(gb.booster.feature_name(), gb.booster.feature_importance("gain").tolist(), strict=True),
        key=lambda kv: -kv[1],
    )

    model_dir = Path(model_dir)
    gb.save(model_dir)
    (model_dir / "calibration.json").write_text(json.dumps(cal.table()))
    np.save(model_dir / "reference_scores.npy", gb_valid)  # percentiles and the call-list cut

    counts = parts["_counts"].iloc[0].to_dict()
    result = {
        "meta": {
            "git_sha": _git_sha(),
            "date": time.strftime("%Y-%m-%d"),
            "python": platform.python_version(),
            "lightgbm": __import__("lightgbm").__version__,
            "sklearn": __import__("sklearn").__version__,
            "rows": {k: int(v) for k, v in counts.items()},
            "splits": {
                k: {
                    "encounters": int(len(v)),
                    "patients": int(v["patient_nbr"].nunique()),
                    "readmit_rate": float(v["y"].mean()),
                }
                for k, v in (("train", train), ("valid", valid), ("test", test))
            },
            "lightgbm_trees": int(gb.booster.best_iteration),
            **timings,
            "seconds": time.perf_counter() - t0,
        },
        "test": models,
        "bootstrap": E.bootstrap(y_test, groups, scores),
        "calibration": {
            "LightGBM": E.calibration_table(y_test, gb_prob),
            "logistic regression": E.calibration_table(y_test, lr_prob),
        },
        "subgroups": subgroups,
        "misses_at_10": misses,
        "psi_train_vs_test": drift,
        "importance_gain_top15": importance[:15],
        "design_checks": design_checks(data_dir),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    np.savez_compressed(
        out_dir / "test_scores.npz",
        y=y_test,
        lightgbm=gb_test,
        prob=gb_prob,
        lace=scores["LACE"],
        logistic=lr_test,
        patient=groups,
    )
    return result
