"""results/RESULTS.md from results/metrics.json."""

from __future__ import annotations

from pathlib import Path


def write_markdown(r: dict, path: Path) -> None:
    m = r["meta"]
    b = r["bootstrap"]["models"]
    lines = [
        "# Results\n",
        f"`readmit run` at commit `{m['git_sha']}` on {m['date']}: Python {m['python']}, "
        f"LightGBM {m['lightgbm']}, scikit-learn {m['sklearn']}. {m['rows']['raw']:,} encounters, "
        f"{m['rows']['excluded_death_or_hospice']:,} dropped for death or hospice discharge. "
        f"Test split: {m['splits']['test']['encounters']:,} encounters from "
        f"{m['splits']['test']['patients']:,} patients, {m['splits']['test']['readmit_rate']:.1%} readmitted "
        f"within 30 days. Intervals: {r['bootstrap']['resamples']:,} bootstrap resamples over patients.\n",
        "## Test split\n",
        "| Model | AUROC (95% CI) | AUPRC | Recall at 5% | Recall at 10% (95% CI) | Precision at 10% "
        "| Recall at 20% | Brier |\n|---|---|---|---|---|---|---|---|",
    ]
    for name, t in r["test"].items():
        ci_a, ci_r = b[name]["auroc"], b[name]["recall@10"]
        brier = f"{t['brier']:.4f}" if "brier" in t else ""
        lines.append(
            f"| {name} | {t['auroc']:.3f} ({ci_a[0]:.3f}–{ci_a[1]:.3f}) | {t['auprc']:.3f} | "
            f"{t['top5']['recall']:.3f} | {t['top10']['recall']:.3f} ({ci_r[0]:.3f}–{ci_r[1]:.3f}) | "
            f"{t['top10']['precision']:.3f} | {t['top20']['recall']:.3f} | {brier} |"
        )
    lines += [
        "",
        "## Paired differences (bootstrap over patients)\n",
        "| Comparison | Mean | 95% CI |\n|---|---|---|",
    ]
    for k, v in r["bootstrap"]["paired"].items():
        lines.append(f"| {k} | {v['mean']:+.3f} | {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} |")
    lines += [
        "",
        "## Calibration of LightGBM on test (deciles of predicted risk)\n",
        "| Decile | n | Predicted | Observed |\n|---|---|---|---|",
    ]
    for i, row in enumerate(r["calibration"]["LightGBM"], 1):
        lines.append(f"| {i} | {row['n']:,} | {row['predicted']:.3f} | {row['observed']:.3f} |")
    lines += [
        "",
        "## Subgroups (LightGBM; call list is the top 10% of all test encounters)\n",
        "| Split by | Group | n | Readmit rate | AUROC | Share on call list | Recall |\n"
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in r["subgroups"].items():
        for row in rows:
            lines.append(
                f"| {name} | {row['group']} | {row['n']:,} | {row['readmit_rate']:.3f} | "
                f"{row['auroc']:.3f} | {row['flagged_share']:.3f} | {row['recall_at_10']:.3f} |"
            )
    mi = r["misses_at_10"]
    lines += [
        "",
        "## Who the call list misses\n",
        f"Of {mi['readmissions']:,} test readmissions, the 10% call list ({mi['flagged']:,} encounters) "
        f"reaches {mi['caught_at_10']:,}; the calibrated probabilities predicted "
        f"{mi['expected_on_list']:.0f}. "
        f"{mi['share_of_all_readmissions_with_no_prior_visits']:.0%} of all readmissions had no "
        f"inpatient, emergency or outpatient visit in the prior year; the list reaches "
        f"{mi['caught_with_no_prior_visits']:,} of them, and they are "
        f"{mi['missed_with_no_prior_visits']:,} of the {mi['missed_total']:,} misses.\n",
        "## Stability (PSI, train vs test)\n",
        "| Feature | PSI |\n|---|---|",
    ]
    for k, v in r["psi_train_vs_test"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += [
        "",
        "## Design checks (LightGBM retrained with one decision changed)\n",
        "| Variant | Test AUROC | Recall at 10% | Test encounters |\n|---|---|---|---|",
    ]
    for k, v in r["design_checks"].items():
        lines.append(f"| {k} | {v['test_auroc']:.3f} | {v['recall_at_10']:.3f} | {v['test_n']:,} |")
    lines += [
        "",
        f"LightGBM: {m['lightgbm_trees']} trees (early stopping on validation AUC), "
        f"fit in {m['lightgbm_fit_s']:.1f} s on one thread.\n",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")
