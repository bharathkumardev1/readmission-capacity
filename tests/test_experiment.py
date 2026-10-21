import json

from synth import frame

from readmit import data as D
from readmit.cli import main
from readmit.experiment import run
from readmit.report import write_markdown


def test_full_run_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "download", lambda data_dir: tmp_path / "unused.csv")
    monkeypatch.setattr(D, "load", lambda path: frame(6000))
    results, arrays = run(tmp_path, tmp_path / "model", resamples=50)
    assert results["shipped"]["model"] in ("LightGBM", "XGBoost")
    assert set(results["test"]) == {"prior inpatient stays", "LACE", "logistic regression", "LightGBM",
                                    "XGBoost"}
    assert [s["shift"] for s in results["shifts"]][0] == "none (test as is)"
    assert set(results["design_checks"]) >= {"as built", "race and gender as features"}
    assert len(results["reasons_at_10"]["examples"]) == 5
    assert (tmp_path / "model" / "booster.json").exists() and len(arrays["y"]) == results["test"]["LACE"]["n"]
    write_markdown(json.loads(json.dumps(results)), tmp_path / "RESULTS.md")
    text = (tmp_path / "RESULTS.md").read_text()
    assert "## Simulated shifts" in text and "| XGBoost |" in text


def test_cli_records_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "download", lambda data_dir: tmp_path / "unused.csv")
    monkeypatch.setattr(D, "load", lambda path: frame(3000))
    import readmit.experiment as ex

    real = ex.run
    monkeypatch.setattr(ex, "run", lambda d, m, c: real(d, m, c, resamples=20))
    assert main(["run", "--data-dir", str(tmp_path), "--model-dir", str(tmp_path / "m"),
                 "--out", str(tmp_path / "out")]) == 0
    meta = json.loads((tmp_path / "out" / "metrics.json").read_text())["meta"]
    assert meta["command"].startswith("readmit run --data-dir")


def test_new_results_dont_make_the_checkout_dirty(tmp_path, monkeypatch):
    import subprocess

    from readmit.experiment import meta

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.json").write_text("{}")
    (tmp_path / "model.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "metrics.json").write_text('{"auroc": 0.67}')
    assert meta("readmit run")["git_dirty"] is False
    (tmp_path / "model.py").write_text("x = 2\n")
    assert meta("readmit run")["git_dirty"] is True
