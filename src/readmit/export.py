"""Export what the browser demo needs: the LightGBM trees, the calibration map, and each
model's test-set ranking packed as bits for the outreach-capacity slider.

The capacity slider needs only one thing per model: the test labels in score order. Packed
one bit per encounter that's 2.5 KB a model, and recall at any capacity is a prefix sum.
``parity.json`` holds Python's raw scores for test rows so web/test can check the JS tree
walker against LightGBM itself.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL, NUMERIC
from .models import GBDT


def compact(node: dict):
    """LightGBM's dump_model() tree, keeping only what scoring needs. A leaf becomes its value."""
    if "leaf_value" in node:
        return node["leaf_value"]
    out = {
        "f": node["split_feature"],
        "t": node["threshold"],
        "l": compact(node["left_child"]),
        "r": compact(node["right_child"]),
    }
    if node["decision_type"] == "==":
        out["c"] = 1
    if node["default_left"]:
        out["dl"] = 1
    if node["missing_type"] != "None":
        out["m"] = node["missing_type"]
    return out


def pack_ranked_labels(y: np.ndarray, score: np.ndarray) -> str:
    order = np.lexsort((np.arange(len(y)), -score))
    return base64.b64encode(np.packbits(y[order].astype(np.uint8)).tobytes()).decode()


def export(
    model_dir: Path, results_dir: Path, test: pd.DataFrame, out: Path, parity_rows: int = 2000
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gb = GBDT.load(model_dir)
    z = np.load(Path(results_dir) / "test_scores.npz")
    y = z["y"]
    ranked = {
        "LightGBM": pack_ranked_labels(y, z["lightgbm"]),
        "logistic regression": pack_ranked_labels(y, z["logistic"]),
        "LACE": pack_ranked_labels(y, z["lace"]),
    }
    dump = gb.booster.dump_model(num_iteration=gb.booster.best_iteration)
    trees = [compact(t["tree_structure"]) for t in dump["tree_info"]]
    defaults = {
        c: (float(test[c].median()) if c in NUMERIC else test[c].mode().iloc[0])
        for c in NUMERIC + CATEGORICAL
    }
    model = {
        "features": dump["feature_names"],
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "categories": gb.categories,
        "trees": trees,
        "calibration": json.loads((Path(model_dir) / "calibration.json").read_text()),
        "defaults": defaults,
        # The call list cut: the top 10% of validation scores, same as the service uses.
        "call_threshold_raw": float(np.quantile(np.load(Path(model_dir) / "reference_scores.npy"), 0.9)),
    }
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "capacity.json").write_text(
        json.dumps({"n": int(len(y)), "positives": int(y.sum()), "ranked_labels": ranked})
    )
    rows = test.iloc[:parity_rows]
    expected = gb.score(rows)
    records = rows[NUMERIC + CATEGORICAL].to_dict(orient="records")
    (out / "parity.json").write_text(json.dumps({"rows": records, "raw": expected.tolist()}, default=float))
    return {
        "trees": len(trees),
        "model_kb": round((out / "model.json").stat().st_size / 1024, 1),
        "parity_rows": len(records),
    }
