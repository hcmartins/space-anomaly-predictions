"""Step 17: the final report -- a single markdown document summarizing the
whole run: data/QC/labeling/split statistics, leakage-check status, and
every task's per-horizon, per-model headline metrics with confidence
intervals and baseline comparison. Every number in this report is read
directly from the artifacts the run itself produced, never re-derived or
hand-typed.
"""
from __future__ import annotations

from pathlib import Path


def _format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _task_section(task_name: str, task_result: dict) -> str:
    lines = [f"### Task: {task_name}\n"]
    for h, horizon_result in task_result["horizons"].items():
        lines.append(f"**Horizon {h} sol** -- status: `{horizon_result['status']}`, "
                     f"n_train={horizon_result['n_train']}, n_val={horizon_result['n_val']}, "
                     f"n_test={horizon_result['n_test']}\n")
        if horizon_result["status"] != "trained":
            continue

        for model_name, model_result in horizon_result["models"].items():
            eval_result = model_result.get("evaluation", {})
            test_split = eval_result.get("splits", {}).get("test", {})
            metrics = test_split.get("metrics") or {}
            ci = test_split.get("ci") or {}
            headline_keys = [k for k in metrics if k in (
                "mae", "rmse", "r2", "correlation", "tss", "roc_auc", "pr_auc", "precision", "recall",
            )]
            metric_strs = []
            for key in headline_keys:
                value_str = _format_metric(metrics[key])
                if key in ci:
                    lo, hi = ci[key]
                    metric_strs.append(f"{key}={value_str} (90% CI [{_format_metric(lo)}, {_format_metric(hi)}])")
                else:
                    metric_strs.append(f"{key}={value_str}")
            lines.append(f"- `[{model_result['kind']}]` **{model_name}**: " + ", ".join(metric_strs))

        if "sequence" in horizon_result:
            for window, seq_result in horizon_result["sequence"].items():
                if seq_result["status"] == "infeasible":
                    lines.append(f"- `[sequence]` window={window} sol: INFEASIBLE "
                                  f"(n_train_windows={seq_result['n_train_windows']})")
                else:
                    for rnn_type, rnn_result in seq_result["models"].items():
                        lines.append(f"- `[sequence]` {rnn_type} window={window} sol "
                                      f"(n_train_windows={rnn_result['n_train_windows']}): {rnn_result['metrics']}")
        lines.append("")
    return "\n".join(lines)


def generate_final_report(
    *,
    run_paths: dict,
    manifest_dict: dict,
    pipeline_summary: dict,
    task_results: dict,
) -> Path:
    # Derived from the run's own manifest, never a separately-maintained
    # constant -- a hardcoded module-level label here previously drifted out
    # of sync with the real-only pivot (always said "SIMULATED" even for
    # real runs, contradicting the "Data source: real" line right below it;
    # found and fixed 2026-08-05).
    data_source_label = f"{manifest_dict['data_source'].upper()} (Martian Chapter 4 mars.run_all pipeline)"
    lines = [
        f"# Final Report -- run `{manifest_dict['run_id']}`",
        "",
        f"[DATA SOURCE: {data_source_label}]",
        "",
        f"- Data source: `{manifest_dict['data_source']}`",
        f"- Started (UTC): {manifest_dict['started_at_utc']}",
        f"- Finished (UTC): {manifest_dict.get('finished_at_utc', 'n/a')}",
        f"- Status: `{manifest_dict['status']}`",
        f"- Seed strategy: {manifest_dict['seed_strategy']}",
        "",
        "## Pipeline summary",
        "",
    ]
    for key, value in pipeline_summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Step timings (seconds)")
    lines.append("")
    for step, seconds in manifest_dict.get("step_timings_seconds", {}).items():
        lines.append(f"- {step}: {seconds:.2f}s")
    lines.append("")

    lines.append("## Leakage checks")
    lines.append("")
    lines.append(f"```\n{manifest_dict.get('leakage_checks')}\n```")
    lines.append("")

    lines.append("## Task results")
    lines.append("")
    for task_name, task_result in task_results.items():
        lines.append(_task_section(task_name, task_result))

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Manifests: `{run_paths['manifests']}`")
    lines.append(f"- Data: `{run_paths['data']}`")
    lines.append(f"- Labels: `{run_paths['labels']}`")
    lines.append(f"- Features: `{run_paths['features']}`")
    lines.append(f"- Splits: `{run_paths['splits']}`")
    lines.append(f"- Models (checkpoints): `{run_paths['models']}`")
    lines.append(f"- Metrics: `{run_paths['metrics']}`")
    lines.append(f"- Figures: `{run_paths['figures']}`")
    lines.append(f"- Tables: `{run_paths['tables']}`")
    lines.append(f"- Checksums: `{run_paths['reports']}/checksums.json`")
    lines.append("")
    lines.append(f"[DATA SOURCE: {data_source_label}]")

    report_path = Path(run_paths["reports"]) / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
