"""Export what the browser demo needs: the shipped model's trees, its calibration, each
ranking's test order packed as bits for the capacity slider, and the call list with its
reason codes.

The capacity slider needs one thing per ranking: the test labels in score order. Packed one
bit per encounter that's 2.5 KB a ranking, and recall at any capacity is a prefix sum.
The call list's reasons are computed here with LightGBM's own TreeSHAP, so the demo shows
exactly the reasons the service returns without a second implementation in JavaScript.
``parity.json`` holds Python's raw scores for test rows, so web/test can check the JS tree
walker against LightGBM itself.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import AGE_MID
from .models import Booster, Calibrator
from .reasons import reason_codes

LIST_ROWS = 200


def compact(node: dict):
    """LightGBM's dump_model() tree, keeping only what scoring needs. A leaf becomes its value."""
    if "leaf_value" in node:
        return node["leaf_value"]
    out = {"f": node["split_feature"], "t": node["threshold"],
           "l": compact(node["left_child"]), "r": compact(node["right_child"])}
    if node["decision_type"] == "==":
        out["c"] = 1
    if node["default_left"]:
        out["dl"] = 1
    if node["missing_type"] != "None":
        out["m"] = node["missing_type"]
    return out


def pack_ranked_labels(y: np.ndarray, score: np.ndarray) -> str:
    order = np.lexsort((np.arange(len(y)), -score))  # same order and ties as evaluate.at_capacity
    return base64.b64encode(np.packbits(y[order].astype(np.uint8)).tobytes()).decode()


def export(model_dir: Path, test: pd.DataFrame, out: Path, parity_rows: int = 2000,
           results_dir: Path | None = None) -> dict:
    """``results_dir``: copy its metrics.json and load.json next to the demo files, so a page
    that prints this project's numbers reads them from the results files instead of retyping
    them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = Booster.load(model_dir)
    if model.kind != "lightgbm":
        raise NotImplementedError("the browser scorer reads LightGBM trees")
    z = np.load(Path(model_dir) / "test_scores.npz")
    y = z["y"]
    if len(test) != len(y) or not (test["y"].to_numpy() == y).all():
        raise ValueError("test split doesn't line up with artifacts/test_scores.npz; rerun readmit run")
    names = {"prior_inpatient_stays": "prior inpatient stays", "LACE": "LACE",
             "logistic_regression": "logistic regression", "LightGBM": "LightGBM", "XGBoost": "XGBoost"}
    ranked = {label: pack_ranked_labels(y, z[key]) for key, label in names.items() if key in z}

    dump = model.model.dump_model(num_iteration=model.trees)
    calibration = json.loads((Path(model_dir) / "calibration.json").read_text())
    reference = np.load(Path(model_dir) / "reference_scores.npy")
    defaults = {c: (float(test[c].median()) if c not in model.categories else test[c].mode().iloc[0])
                for c in model.features}
    spec = {"features": dump["feature_names"], "categories": model.categories,
            "trees": [compact(t["tree_structure"]) for t in dump["tree_info"]],
            "calibration": calibration, "defaults": defaults,
            "call_threshold_raw": float(np.quantile(reference, 0.9))}  # top 10% of validation scores
    (out / "model.json").write_text(json.dumps(spec, separators=(",", ":")))
    (out / "capacity.json").write_text(json.dumps({"n": int(len(y)), "positives": int(y.sum()),
                                                   "ranked_labels": ranked}))

    score = z["LightGBM"]
    order = np.lexsort((np.arange(len(y)), -score))[:LIST_ROWS]
    rows = test.iloc[order]
    reasons = reason_codes(model, rows)
    prob = Calibrator.from_table(calibration)(score[order])
    age = {v: k for k, v in AGE_MID.items()}
    call_list = [{"rank": i + 1, "risk": round(float(p), 3), "readmitted": int(r["y"]),
                  "age": age.get(r["age_years"], ""), "prior_inpatient": int(r["number_inpatient"]),
                  "days": int(r["time_in_hospital"]), "reasons": [x["text"] for x in rs]}
                 for i, ((_, r), rs, p) in enumerate(zip(rows.iterrows(), reasons, prob, strict=True))]
    (out / "call_list.json").write_text(json.dumps(call_list, separators=(",", ":")))

    sample = test.iloc[:parity_rows]
    records = sample[model.features].to_dict(orient="records")
    (out / "parity.json").write_text(json.dumps({"rows": records, "raw": model.score(sample).tolist()},
                                                default=float))
    copied = []
    for name in ("metrics.json", "load.json"):
        if results_dir is not None and (Path(results_dir) / name).exists():
            shutil.copyfile(Path(results_dir) / name, out / name)
            copied.append(name)
    return {"trees": len(spec["trees"]), "model_kb": round((out / "model.json").stat().st_size / 1024, 1),
            "call_list_rows": len(call_list), "parity_rows": len(records), "results": copied}
