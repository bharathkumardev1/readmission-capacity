import io
import json
import zipfile

import pytest
from synth import frame

from readmit import data as D


def test_clean_drops_death_and_hospice_and_labels_30_days():
    df = frame()
    out = D.clean(df)
    assert not out["discharge_disposition_id"].isin(D.DEATH_OR_HOSPICE).any()
    assert len(out) == (~df["discharge_disposition_id"].isin(D.DEATH_OR_HOSPICE)).sum()
    assert out["y"].sum() == (out["readmitted"] == "<30").sum()


def test_split_keeps_each_patient_in_one_split():
    parts = D.split(D.clean(frame()))
    seen = [set(p["patient_nbr"]) for p in parts.values()]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert D.bucket(12345) == D.bucket(12345) and 0 <= D.bucket(7) < 100


def test_content_hash_ignores_line_endings():
    assert D.content_sha256(b"a,b\r\n1,2\r\n") == D.content_sha256(b"a,b\n1,2\n")


def test_download_checks_the_hash_and_falls_back(tmp_path, monkeypatch):
    good = b"a,b\r\n1,2\r\n"
    monkeypatch.setattr(D, "CSV_SHA256", D.content_sha256(good))
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr(f"dataset_diabetes/{D.CSV_NAME}", good)
        z.writestr("dataset_diabetes/IDs_mapping.csv", b"admission_type_id,description\n1,Emergency\n")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("dataset_diabetes.zip", inner.getvalue())  # the UCI download nests a zip
    (tmp_path / "bad.csv").write_bytes(b"tampered")
    (tmp_path / "good.zip").write_bytes(zipped.getvalue())
    sources = ((tmp_path / "bad.csv").as_uri(), (tmp_path / "missing.zip").as_uri(),
               (tmp_path / "good.zip").as_uri())
    path = D.download(tmp_path / "data", sources)
    assert path.read_bytes() == b"a,b\n1,2\n"
    assert (tmp_path / "data" / D.MAPPING_NAME).read_text().startswith("admission_type_id")
    source = json.loads((tmp_path / "data" / "source.json").read_text())
    assert source["url"] == sources[2] and "ids_mapping.csv" in source["files"]
    with pytest.raises(RuntimeError, match="does not match"):
        D.download(tmp_path / "other", sources[:1])
