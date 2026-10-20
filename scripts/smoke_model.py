"""Train a throwaway model on synthetic data so CI can smoke-test the Docker image without
downloading the dataset. Usage: python scripts/smoke_model.py OUT_DIR"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synth import frame  # noqa: E402

from readmit import data as D  # noqa: E402
from readmit.features import add_features, top_specialties  # noqa: E402
from readmit.models import GBDT, Calibrator  # noqa: E402

out = Path(sys.argv[1])
parts = D.split(D.clean(frame(4000)))
spec = top_specialties(parts["train"])
tr, va = add_features(parts["train"], spec), add_features(parts["valid"], spec)
gb = GBDT.train(tr, va, rounds=100)
gb.save(out)
(out / "calibration.json").write_text(json.dumps(Calibrator(gb.score(va), va["y"].to_numpy()).table()))
np.save(out / "reference_scores.npy", gb.score(va))
print(f"wrote {out}")
