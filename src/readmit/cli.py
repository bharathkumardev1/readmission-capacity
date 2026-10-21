"""readmit: 30-day readmission risk evaluated the way a care team would use it.

readmit run      download the UCI data, train, evaluate; write results/ and artifacts/
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np

    from .experiment import run
    from .report import write_markdown

    results, arrays = run(Path(args.data_dir), Path(args.model_dir), args.command_line)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    write_markdown(results, out / "RESULTS.md")
    np.savez_compressed(Path(args.model_dir) / "test_scores.npz", **arrays)
    ship = results["shipped"]["model"]
    t = results["test"][ship]
    print(json.dumps({"shipped": ship, "auroc": round(t["auroc"], 3),
                      "recall_at_10": round(t["top10"]["recall"], 3), "brier": round(t["brier"], 4)}))
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

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["readmit", *argv])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
