"""results/RESULTS.md from results/metrics.json, so the markdown never drifts from the data."""

from __future__ import annotations

from pathlib import Path


def _ci(ci: list[float], fmt: str = ".3f") -> str:
    return f"{ci[0]:{fmt}} to {ci[1]:{fmt}}"


def load_section(load: dict) -> list[str]:
    m = load["meta"]
    lines = ["## Load test\n",
             f"`{m['command']}` at commit `{(m['git_sha'] or 'unknown')[:7]}`"
             f"{' with uncommitted changes' if m['git_dirty'] else ''} on {m['date']}: {m['server']}; "
             f"{m['cpu']}. Each client posts a batch of test encounters, waits for the scores and posts "
             f"the next. Dollars per million encounters price one vCPU at ${m['usd_per_vcpu_hour']:.4f} an "
             "hour (half an AWS c7i.large on demand in us-east-1) and assume this throughput carries over "
             "to it, which I haven't measured.\n",
             "| Encounters per request | Clients | Requests/s | Encounters/s | p50 ms | p95 ms | p99 ms "
             "| Errors | Server RSS MB | $ per million encounters |\n" + "|---" * 10 + "|"]
    lines += [f"| {x['batch']} | {x['concurrency']} | {x['requests_per_s']:,.1f} | "
              f"{x['encounters_per_s']:,.0f} | {x['p50_ms']:.1f} | {x['p95_ms']:.1f} | {x['p99_ms']:.1f} | "
              f"{x['errors']} | {x['server_rss_mb']:.0f} | {x['usd_per_million_encounters']:.4f} |"
              for x in load["levels"]]
    return lines


def write_markdown(r: dict, path: Path, load: dict | None = None) -> None:
    """``load`` is results/load.json when the load test has been run."""
    m, b, ship = r["meta"], r["bootstrap"], r["shipped"]["model"]
    te = m["splits"]["test"]
    lines = [
        "# Results\n",
        f"`{m['command']}` at commit `{(m['git_sha'] or 'unknown')[:7]}`"
        f"{' with uncommitted changes' if m['git_dirty'] else ''} on {m['date']}: {m['cpu']}, "
        f"Python {m['python']}, LightGBM {m['lightgbm']}, XGBoost {m['xgboost']}, scikit-learn "
        f"{m['sklearn']}, pandas {m['pandas']}. {m['rows']['raw']:,} encounters, "
        f"{m['rows']['excluded_death_or_hospice']:,} dropped for death or hospice discharge. Test split: "
        f"{te['encounters']:,} encounters from {te['patients']:,} patients, {te['readmit_rate']:.1%} back "
        f"within 30 days. Intervals: {b['resamples']:,} bootstrap resamples over patients.\n",
        f"Shipped model, picked on validation by recall at 10% and then AUROC: **{ship}**.\n",
        "## Test split\n",
        "| Ranking | AUROC (95% CI) | AUPRC | Recall at 5% | Recall at 10% (95% CI) | Precision at 10% "
        "| Recall at 20% | Brier |\n|---|---|---|---|---|---|---|---|",
    ]
    for name, t in r["test"].items():
        ci = b["models"][name]
        brier = f"{t['brier']:.4f}" if "brier" in t else ""
        lines.append(f"| {name} | {t['auroc']:.3f} ({_ci(ci['auroc'])}) | {t['auprc']:.3f} | "
                     f"{t['top5']['recall']:.3f} | {t['top10']['recall']:.3f} ({_ci(ci['recall@10'])}) | "
                     f"{t['top10']['precision']:.3f} | {t['top20']['recall']:.3f} | {brier} |")
    lines += ["", "## Paired differences (bootstrap over patients)\n",
              "| Comparison | Mean | 95% CI |\n|---|---|---|"]
    lines += [f"| {k} | {v['mean']:+.3f} | {_ci(v['ci95'], '+.3f')} |" for k, v in b["paired"].items()]
    lines += ["", f"## Calibration of {ship} on test (deciles of predicted risk)\n",
              "| Decile | n | Predicted | Observed |\n|---|---|---|---|"]
    lines += [f"| {i} | {row['n']:,} | {row['predicted']:.3f} | {row['observed']:.3f} |"
              for i, row in enumerate(r["calibration"][ship], 1)]
    lines += ["", f"## Subgroups ({ship}; the call list is the top 10% of all test encounters)\n",
              "| Split by | Group | n | Readmit rate | AUROC | Share on call list | Recall |\n"
              "|---|---|---|---|---|---|---|"]
    for name, rows in r["subgroups"].items():
        lines += [f"| {name} | {g['group']} | {g['n']:,} | {g['readmit_rate']:.3f} | {g['auroc']:.3f} | "
                  f"{g['flagged_share']:.3f} | {g['recall_at_10']:.3f} |" for g in rows]
    mi = r["misses_at_10"]
    lines += [
        "", "## Who the call list misses\n",
        f"Of {mi['readmissions']:,} test readmissions, the 10% call list ({mi['flagged']:,} encounters) "
        f"reaches "
        f"{mi['caught']:,}; the calibrated probabilities expected {mi['expected_on_list']:.0f}. "
        f"{mi['readmissions_with_no_prior_visits']:,} readmissions had no inpatient, emergency or outpatient "
        f"visit in the prior year; the list reaches {mi['caught_with_no_prior_visits']:,} of them, and they "
        f"are "
        f"{mi['missed_with_no_prior_visits']:,} of the {mi['missed']:,} misses. Encounters with a prior "
        f"inpatient stay are {mi['share_with_prior_inpatient']['all']:.0%} of test and "
        f"{mi['share_with_prior_inpatient']['on_list']:.0%} of the list.\n",
    ]
    rs = r["reasons_at_10"]
    lines += ["## Reason codes on the call list\n",
              f"Over the {rs['encounters']:,} encounters on the list: how often each feature is the first "
              "reason, and how often it is among the top three.\n",
              "| Feature | First reason | In top three |\n|---|---|---|"]
    lines += [f"| {k} | {rs['first_reason'].get(k, 0):.1%} | {v:.1%} |" for k, v in rs["in_top3"].items()]
    lines += ["", "The five highest-risk encounters:\n", "| Risk | Readmitted | Reasons |\n|---|---|---|"]
    lines += [f"| {e['risk']:.2f} | {'yes' if e['readmitted'] else 'no'} | {'; '.join(e['reasons'])} |"
              for e in rs["examples"]]
    lines += ["", "## Simulated shifts (see src/readmit/shift.py for what each one changes)\n",
              "| Shift | Readmit rate | AUROC | Recall at 10% | Precision at 10% | Expected on list | "
              "Observed on list | Alerts |\n|---|---|---|---|---|---|---|---|"]
    for sh in r["shifts"]:
        lines.append(f"| {sh['shift']} | {sh['readmit_rate']:.3f} | {sh['auroc']:.3f} | "
                     f"{sh['recall_at_10']:.3f} | {sh['precision_at_10']:.3f} | "
                     f"{sh['expected_on_list']:.0f} | {sh['observed_on_list']:,} | "
                     f"{', '.join(sh['alerts']) or 'none'} |")
    names = [sh["shift"] for sh in r["shifts"]]
    lines += ["", "PSI by input for each shift:\n",
              "| Input | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)]
    for k in r["shifts"][0]["psi"]:
        lines.append(f"| {k} | " + " | ".join(f"{sh['psi'][k]:.3f}" for sh in r["shifts"]) + " |")
    lines += ["", "## Design checks (LightGBM retrained with one decision changed)\n",
              "| Variant | Test AUROC | Recall at 10% | Test encounters |\n|---|---|---|---|"]
    lines += [f"| {k} | {v['test_auroc']:.3f} | {v['recall_at_10']:.3f} | {v['test_n']:,} |"
              for k, v in r["design_checks"].items()]
    trees = ", ".join(f"{k} {v} trees" for k, v in m["trees"].items())
    fit = ", ".join(f"{k} {v:.1f} s" for k, v in m["fit_seconds"].items())
    lines += ["", f"Early stopping on validation AUC: {trees}. Fit time on one thread: {fit}. "
              f"The whole run took {m['seconds']:.0f} s.\n"]
    if load:
        lines += load_section(load)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")
