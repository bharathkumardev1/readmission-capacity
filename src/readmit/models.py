"""Logistic regression, LightGBM and XGBoost, each calibrated on the validation split.

Logistic regression shows what the trees buy. LightGBM and XGBoost are the two gradient
boosting libraries a team would choose between; both get plain hyperparameters (below), the
same features with the same fixed category vocabularies, and early stopping on validation
AUC, so the comparison is between libraries rather than between tuning budgets. Isotonic
calibration, also fit on validation, turns scores into probabilities a care team can plan
with.

Both boosters return exact TreeSHAP contributions for a row (``contributions``), which is
where the call list's reason codes come from.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC

FEATURES = NUMERIC + CATEGORICAL
LIGHTGBM_PARAMS = {
    "objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_cat_to_onehot": 8, "cat_smooth": 20, "seed": 0, "deterministic": True,
    "force_row_wise": True, "num_threads": 1, "verbose": -1,
}
XGBOOST_PARAMS = {
    "objective": "binary:logistic", "eta": 0.03, "max_depth": 6, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "max_cat_to_onehot": 8,
    "tree_method": "hist", "seed": 0, "nthread": 1, "eval_metric": "auc",
}
ROUNDS = 3000
EARLY_STOP = 200


def logistic(train: pd.DataFrame, features: list[str] = FEATURES) -> object:
    cats = [c for c in features if c not in NUMERIC]
    pre = ColumnTransformer([
        ("num", StandardScaler(), [c for c in features if c in NUMERIC]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    model.fit(train[features], train["y"])
    return model


def vocabularies(train: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    return {c: sorted(train[c].astype(str).unique()) for c in features if c not in NUMERIC}


def frame(df: pd.DataFrame, categories: dict[str, list[str]], features: list[str]) -> pd.DataFrame:
    """Model input with fixed category codes; a value train never saw becomes missing."""
    x = df[features].copy()
    for col in features:
        if col in categories:
            vals = x[col].astype(str)
            x[col] = pd.Categorical(vals.where(vals.isin(categories[col])), categories=categories[col])
        else:
            x[col] = x[col].astype(float)
    return x


class Booster:
    """One trained gradient-boosted model, LightGBM or XGBoost, behind the same calls."""

    def __init__(self, kind: str, model, categories: dict[str, list[str]], trees: int,
                 features: list[str] = FEATURES):
        self.kind, self.model, self.categories, self.trees = kind, model, categories, trees
        self.features = list(features)

    @classmethod
    def train(cls, kind: str, train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None,
              rounds: int = ROUNDS, features: list[str] = FEATURES) -> Booster:
        cats = vocabularies(train, features)
        xt, xv = frame(train, cats, features), frame(valid, cats, features)
        if kind == "lightgbm":
            dtrain = lgb.Dataset(xt, train["y"], categorical_feature=list(cats))
            dvalid = lgb.Dataset(xv, valid["y"], reference=dtrain)
            model = lgb.train({**LIGHTGBM_PARAMS, **(params or {}), "metric": "auc"}, dtrain, rounds,
                              valid_sets=[dvalid],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            return cls(kind, model, cats, model.best_iteration, features)
        if kind == "xgboost":
            dtrain = xgb.DMatrix(xt, train["y"], enable_categorical=True)
            dvalid = xgb.DMatrix(xv, valid["y"], enable_categorical=True)
            model = xgb.train({**XGBOOST_PARAMS, **(params or {})}, dtrain, rounds,
                              evals=[(dvalid, "valid")], early_stopping_rounds=EARLY_STOP,
                              verbose_eval=False)
            return cls(kind, model, cats, model.best_iteration + 1, features)
        raise ValueError(f"unknown booster {kind!r}")

    def _predict(self, df: pd.DataFrame, **kw) -> np.ndarray:
        x = frame(df, self.categories, self.features)
        if self.kind == "lightgbm":
            return self.model.predict(x, num_iteration=self.trees, **kw)
        return self.model.predict(xgb.DMatrix(x, enable_categorical=True),
                                  iteration_range=(0, self.trees), **kw)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Raw margin (log-odds); a ranking, not yet a probability."""
        if self.kind == "lightgbm":
            return self._predict(df, raw_score=True)
        return self._predict(df, output_margin=True)

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """TreeSHAP per feature in ``features`` order, plus the bias last; each row sums to the margin."""
        if self.kind == "lightgbm":
            return self._predict(df, pred_contrib=True)
        return self._predict(df, pred_contribs=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.kind == "lightgbm":
            self.model.save_model(str(path / "model.txt"), num_iteration=self.trees)
        else:
            self.model[: self.trees].save_model(str(path / "model.json"))
        (path / "booster.json").write_text(json.dumps({"kind": self.kind, "trees": self.trees,
                                                       "features": self.features,
                                                       "categories": self.categories}))

    @classmethod
    def load(cls, path: Path) -> Booster:
        path = Path(path)
        info = json.loads((path / "booster.json").read_text())
        if info["kind"] == "lightgbm":
            model = lgb.Booster(model_file=str(path / "model.txt"))
        else:
            model = xgb.Booster()
            model.load_model(str(path / "model.json"))
        return cls(info["kind"], model, info["categories"], info["trees"], info["features"])


class Calibrator:
    """Isotonic map from raw score to probability, fit on validation predictions."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return self.iso.predict(scores)

    def table(self) -> dict:
        return {"x": self.iso.X_thresholds_.tolist(), "y": self.iso.y_thresholds_.tolist()}

    @classmethod
    def from_table(cls, table: dict) -> Calibrator:
        cal = cls.__new__(cls)
        cal.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.iso.fit(table["x"], table["y"])
        return cal
