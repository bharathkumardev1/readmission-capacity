"""Logistic regression and LightGBM, both calibrated on the validation split.

LightGBM is the model I'd ship; logistic regression is there to show what the trees buy.
Hyperparameters are deliberately plain (and listed in configs/lightgbm.json); early stopping
on the validation split picks the number of trees. Isotonic calibration, also fit on the
validation split, turns scores into probabilities a care team can plan with.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_to_onehot": 8,
    "cat_smooth": 20,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}


def logistic(train: pd.DataFrame) -> object:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    model.fit(train[NUMERIC + CATEGORICAL], train["y"])
    return model


class GBDT:
    """LightGBM with fixed category vocabularies, so scoring a new row can't shift codes."""

    def __init__(self, booster: lgb.Booster, categories: dict[str, list[str]]):
        self.booster = booster
        self.categories = categories

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[NUMERIC + CATEGORICAL].copy()
        for col in CATEGORICAL:
            vals = x[col].astype(str)
            known = vals.where(vals.isin(self.categories[col]))  # unseen -> NaN -> code -1
            x[col] = pd.Categorical(known, categories=self.categories[col])
        for col in NUMERIC:
            x[col] = x[col].astype(float)
        return x

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.frame(df), raw_score=True)

    @classmethod
    def train(
        cls, train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None, rounds: int = 3000
    ) -> GBDT:
        cats = {c: sorted(train[c].astype(str).unique()) for c in CATEGORICAL}
        tmp = cls(None, cats)  # type: ignore[arg-type]
        dtrain = lgb.Dataset(tmp.frame(train), train["y"], categorical_feature=CATEGORICAL)
        dvalid = lgb.Dataset(tmp.frame(valid), valid["y"], reference=dtrain)
        booster = lgb.train(
            {**PARAMS, **(params or {}), "metric": "auc"},
            dtrain,
            rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        return cls(booster, cats)

    def save(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(Path(path) / "model.txt"), num_iteration=self.booster.best_iteration)
        (Path(path) / "categories.json").write_text(json.dumps(self.categories))

    @classmethod
    def load(cls, path: Path) -> GBDT:
        booster = lgb.Booster(model_file=str(Path(path) / "model.txt"))
        return cls(booster, json.loads((Path(path) / "categories.json").read_text()))


class Calibrator:
    """Isotonic map from raw score to probability, fit on validation predictions."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return self.iso.predict(scores)

    def table(self) -> dict:
        return {"x": self.iso.X_thresholds_.tolist(), "y": self.iso.y_thresholds_.tolist()}
