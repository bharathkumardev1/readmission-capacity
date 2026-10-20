"""Scoring service: ``POST /score`` takes encounters and returns calibrated 30-day risk.

Each response also says where the encounter falls in the test-set score distribution and
whether it would be on a call list at the configured outreach capacity, because that is the
decision the score feeds.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .features import CATEGORICAL, NUMERIC, add_features
from .models import GBDT

SCORED = Counter("readmit_scored_total", "Encounters scored")
LATENCY = Histogram(
    "readmit_score_seconds",
    "Scoring latency per request",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
RAW_COLUMNS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "metformin",
    "change",
    "diabetesMed",
]


class Encounter(BaseModel):
    """Fields as they appear in the UCI extract. Unknown values may be omitted."""

    age: str = Field(examples=["[70-80)"])
    time_in_hospital: int = Field(ge=1, le=60)
    number_inpatient: int = Field(0, ge=0)
    number_emergency: int = Field(0, ge=0)
    number_outpatient: int = Field(0, ge=0)
    num_medications: int = Field(15, ge=0)
    num_lab_procedures: int = Field(44, ge=0)
    num_procedures: int = Field(1, ge=0)
    number_diagnoses: int = Field(8, ge=0)
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    race: str | None = None
    gender: str | None = None
    payer_code: str | None = None
    medical_specialty: str | None = None
    A1Cresult: str | None = None
    max_glu_serum: str | None = None
    insulin: str = "No"
    metformin: str = "No"
    change: str = "No"
    diabetesMed: str = "Yes"


class ScoreRequest(BaseModel):
    encounters: list[Encounter] = Field(min_length=1, max_length=1000)


class Scorer:
    def __init__(self, model_dir: Path, capacity: float = 0.10):
        import json

        self.model = GBDT.load(model_dir)
        cal = json.loads((Path(model_dir) / "calibration.json").read_text())
        self.cal_x, self.cal_y = np.asarray(cal["x"]), np.asarray(cal["y"])
        ref = np.load(Path(model_dir) / "reference_scores.npy")
        self.reference = np.sort(ref)
        self.capacity = capacity
        self.threshold = float(np.quantile(self.reference, 1 - capacity))
        self.specialties = [
            c for c in self.model.categories["medical_specialty"] if c not in ("missing", "other")
        ]

    def score(self, rows: list[dict]) -> list[dict]:
        df = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
        feats = add_features(df, self.specialties)
        raw = self.model.score(feats[NUMERIC + CATEGORICAL])
        prob = np.interp(raw, self.cal_x, self.cal_y)
        pct = np.searchsorted(self.reference, raw, side="right") / len(self.reference)
        return [
            {
                "risk": round(float(p), 4),
                "percentile": round(float(q) * 100, 1),
                "on_call_list": bool(r >= self.threshold),
                "raw_score": round(float(r), 5),
            }
            for p, q, r in zip(prob, pct, raw, strict=True)
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "scorer"):
        app.state.scorer = Scorer(
            Path(os.environ.get("READMIT_MODEL_DIR", "artifacts")),
            float(os.environ.get("READMIT_CAPACITY", "0.10")),
        )
    yield


app = FastAPI(title="readmit", version="1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    s: Scorer = app.state.scorer
    return {"status": "ok", "trees": s.model.booster.num_trees(), "capacity": s.capacity}


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    t0 = time.perf_counter()
    try:
        out = app.state.scorer.score([e.model_dump() for e in req.encounters])
    except Exception as exc:
        raise HTTPException(500, "scoring failed") from exc
    LATENCY.observe(time.perf_counter() - t0)
    SCORED.inc(len(out))
    return {"capacity": app.state.scorer.capacity, "results": out}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
