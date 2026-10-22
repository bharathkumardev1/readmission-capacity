"""readmit: 30-day readmission risk evaluated the way a care team would use it.

readmit run        download the UCI data, train, evaluate; write results/ and artifacts/
readmit serve      scoring API on :8000
readmit loadtest   closed-loop load test against a running service
readmit report     rewrite results/RESULTS.md from metrics.json and load.json
readmit export     files for the browser demo (trees, calibration, capacity curves, call list)
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def _load_results(results_dir: Path) -> dict | None:
    path = results_dir / "load.json"
    return json.loads(path.read_text()) if path.exists() else None


def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np

    from .experiment import run
    from .report import write_markdown

    results, arrays = run(Path(args.data_dir), Path(args.model_dir), args.command_line)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    write_markdown(results, out / "RESULTS.md", _load_results(out))
    np.savez_compressed(Path(args.model_dir) / "test_scores.npz", **arrays)
    ship = results["shipped"]["model"]
    t = results["test"][ship]
    print(json.dumps({"shipped": ship, "auroc": round(t["auroc"], 3),
                      "recall_at_10": round(t["top10"]["recall"], 3), "brier": round(t["brier"], 4)}))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("readmit.service:app", host=args.host, port=args.port)
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import data as D
    from .loadtest import USD_PER_VCPU_HOUR, run
    from .service import RAW_COLUMNS

    test = D.split(D.clean(D.load(D.download(Path(args.data_dir)))))["test"]
    raw = test[RAW_COLUMNS].astype(object).where(pd.notna(test[RAW_COLUMNS]), None)
    rows = [{k: v for k, v in r.items() if v is not None} for r in raw.to_dict(orient="records")]
    levels = tuple(int(c) for c in args.levels.split(","))
    out = []
    for batch in (int(b) for b in args.batches.split(",")):
        out += run(args.base, rows, batch, levels, args.seconds)
    from .experiment import meta

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {**meta(args.command_line), "server": args.server_note,
                                         "usd_per_vcpu_hour": USD_PER_VCPU_HOUR}, "levels": out}, indent=1)
                    + "\n")
    if (path.parent / "metrics.json").exists():
        cmd_report(argparse.Namespace(results=str(path.parent)))
    return 0 if all(r["errors"] == 0 for r in out) else 1


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_markdown

    results = Path(args.results)
    write_markdown(json.loads((results / "metrics.json").read_text()), results / "RESULTS.md",
                   _load_results(results))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .experiment import prepare
    from .export import export

    print(json.dumps(export(Path(args.model_dir), prepare(Path(args.data_dir))["test"], Path(args.out),
                            results_dir=Path(args.results) if args.results else None)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="readmit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="download, train, evaluate")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="scoring API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("loadtest", help="closed-loop load test against a running service")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batches", default="1,100", help="encounters per request, comma-separated")
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--server-note", default="")
    p.add_argument("--out", default="results/load.json")
    p.set_defaults(func=cmd_loadtest)

    p = sub.add_parser("report", help="rewrite RESULTS.md from the results files")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="files for the browser demo")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--results", default="results",
                   help="copy metrics.json and load.json from here; '' to skip")
    p.add_argument("--out", default="build/web")
    p.set_defaults(func=cmd_export)

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["readmit", *argv])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
