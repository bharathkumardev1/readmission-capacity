"""readmit: 30-day readmission risk evaluated the way a care team would use it.

readmit run            download the UCI data, train, evaluate, write results/ and artifacts/
readmit export --out   files for the browser demo (trees, calibration, capacity curves)
readmit serve          scoring API on :8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="readmit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out", default="results")
    r.add_argument("--model-dir", default="artifacts")
    x = sub.add_parser("export")
    x.add_argument("--data-dir", default="data")
    x.add_argument("--results", default="results")
    x.add_argument("--model-dir", default="artifacts")
    x.add_argument("--out", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "run":
        from .experiment import run
        from .report import write_markdown

        res = run(Path(args.data_dir), Path(args.out), Path(args.model_dir))
        write_markdown(res, Path(args.out) / "RESULTS.md")
        t = res["test"]["LightGBM"]
        print(
            json.dumps(
                {
                    "auroc": round(t["auroc"], 3),
                    "recall_at_10": round(t["top10"]["recall"], 3),
                    "brier": round(t["brier"], 4),
                }
            )
        )
        return 0
    if args.cmd == "export":
        from .experiment import prepare
        from .export import export

        test = prepare(Path(args.data_dir))["test"]
        print(export(Path(args.model_dir), Path(args.results), test, Path(args.out)))
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("readmit.service:app", host="0.0.0.0", port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
