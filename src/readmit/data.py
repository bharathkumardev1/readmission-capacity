"""The UCI "Diabetes 130-US hospitals for years 1999-2008" extract, cleaned and split.

101,766 inpatient encounters of diabetic patients at 130 US hospitals (Strack et al., 2014,
BioMed Research International, doi:10.1155/2014/781670). The label is whether the patient
came back within 30 days (``readmitted == "<30"``).

Two decisions here matter for every number downstream:

* Encounters that ended in death or hospice are dropped. Those patients can't be
  readmitted, so keeping them would teach the model that "discharged to hospice" means low
  risk and inflate every metric.
* The split is by patient, not by encounter. 16,773 patients have more than one encounter;
  splitting encounters at random would put the same person in train and test.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_ZIP = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
# Fairlearn's copy, used only if UCI is unreachable. The hash below must match either way.
MIRROR = "https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_data.csv"
CSV_NAME = "diabetic_data.csv"
CSV_SHA256 = "da21942c25a10a51a8842907ab01a8530b06232616b17d317441d592a20c4359"

# discharge_disposition_id codes from the dataset's IDs_mapping.csv
DEATH_OR_HOSPICE = {
    11: "Expired",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    19: "Expired at home (hospice)",
    20: "Expired in a medical facility (hospice)",
    21: "Expired, place unknown (hospice)",
}
SPLITS = {"train": (0, 60), "valid": (60, 80), "test": (80, 100)}


def content_sha256(data: bytes) -> str:
    """SHA-256 of the CSV with Windows line endings normalized, so either copy hashes alike."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def download(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    target = data_dir / CSV_NAME
    if target.exists() and content_sha256(target.read_bytes()) == CSV_SHA256:
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for source in (UCI_ZIP, MIRROR):
        try:
            with urllib.request.urlopen(source, timeout=120) as resp:
                blob = resp.read()
            if source.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    blob = z.read(next(n for n in z.namelist() if n.endswith(CSV_NAME)))
            got = content_sha256(blob)
            if got != CSV_SHA256:
                errors.append(f"{source}: sha256 {got} does not match the pinned {CSV_SHA256}")
                continue
            target.write_bytes(blob.replace(b"\r\n", b"\n"))
            return target
        except Exception as exc:  # noqa: BLE001 - report every source that failed
            errors.append(f"{source}: {exc}")
    raise RuntimeError("could not get the dataset:\n  " + "\n  ".join(errors))


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop death and hospice discharges and add the binary label ``y``."""
    out = df[~df["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)].copy()
    out["y"] = (out["readmitted"] == "<30").astype(np.int8)
    return out.reset_index(drop=True)


def bucket(patient: int) -> int:
    """0-99 from a hash of the patient id; stable across machines and library versions."""
    return int(hashlib.sha256(str(int(patient)).encode()).hexdigest()[:8], 16) % 100


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """60/20/20 by patient: every encounter of a patient lands in the same split."""
    b = df["patient_nbr"].map(bucket)
    return {name: df[(b >= lo) & (b < hi)].reset_index(drop=True) for name, (lo, hi) in SPLITS.items()}
