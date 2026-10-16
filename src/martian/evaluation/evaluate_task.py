"""Orchestrates full per-model, per-horizon, per-split evaluation: metrics +
confidence intervals for every split, the full required figure suite (test
split), failure analysis across the applicable dimensions (test split), and
comparison against the task's appropriate baseline. Called from
modeling/train.py once per model, immediately after that model is fit, so
the already-fitted model/predictions never need to be reconstructed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.metrics import classification_metrics, regression_metrics

from .baseline_comparison import compare_to_baseline
from .classification_figures import (
    confusion_matrix_figure,
    pr_curve_figure,
    reliability_diagram,
    roc_curve_figure,
)
from .confidence_intervals import bootstrap_ci_classification, bootstrap_ci_regression, load_evaluation_config
from .failure_analysis import run_all_breakdowns
from .figure_provenance import FigureProvenance, assign_figure_number, save_figure_with_provenance
from .regression_figures import observed_vs_predicted_figure, residual_analysis_figure


def evaluate_and_report(
    *,
    task_name: str,
    task_type: str,
    model_kind: str,
    model_name: str,
    horizon: int,
    split_frames: dict,   # split_name -> dataframe with target_col, y_pred/y_prob columns, and bookkeeping columns
    target_col: str,
    prediction_col: str,
    threshold: float,
    baseline_metrics: dict | None,
    episode_catalogue: pd.DataFrame | None,
    figures_dir: Path,
    results_dir: Path,
    config_paths: list,
    full_report: bool = True,
    eval_config: dict | None = None,
    tables_dir: Path | None = None,
    data_source_label: str = "REAL (Martian Chapter 4 evaluation)",
    threshold_source: str | None = None,
) -> dict:
    """`full_report=False` skips figures/failure-analysis (used for sequence
    models, whose sample sizes are typically too small for either to be
    meaningful) but still computes metrics + CIs for every split.

    `tables_dir` (failure-analysis breakdowns, baseline-comparison table)
    defaults to `results_dir` when omitted, preserving the original
    single-directory behaviour for callers that don't need the
    metrics/tables split -- only mars/run_all.py's required output
    structure passes a distinct `tables_dir`."""
    eval_config = eval_config or load_evaluation_config()
    tables_dir = Path(tables_dir) if tables_dir is not None else results_dir
    metric_fn = regression_metrics if task_type == "regression" else classification_metrics

    result: dict = {"splits": {}}
    if task_type != "regression":
        # recorded so every reported metric can be traced to its operating point
        result["decision_threshold"] = float(threshold)
        result["decision_threshold_source"] = threshold_source or "fixed (caller did not record a source)"

    for split_name, df in split_frames.items():
        if len(df) == 0:
            result["splits"][split_name] = {"n": 0, "metrics": None, "ci": None}
            continue
        y_true = df[target_col].to_numpy()
        y_pred = df[prediction_col].to_numpy()

        if task_type == "regression":
            metrics = regression_metrics(y_true, y_pred)
            ci = bootstrap_ci_regression(
                y_true, y_pred, block_ids=df["cell_id"].to_numpy() if "cell_id" in df.columns else None,
                config=eval_config,
            )
        else:
            metrics = classification_metrics(y_true, y_pred, threshold=threshold)
            ci = bootstrap_ci_classification(
                y_true, y_pred, threshold=threshold,
                block_ids=df["cell_id"].to_numpy() if "cell_id" in df.columns else None,
                config=eval_config,
            )
            metrics["prevalence"] = float(np.mean(y_true))
            ci["prevalence"] = (float("nan"), float("nan"))  # not bootstrapped separately; see metrics["prevalence"]

        result["splits"][split_name] = {"n": len(df), "metrics": metrics, "ci": ci}

    if baseline_metrics is not None and "test" in result["splits"] and result["splits"]["test"]["metrics"]:
        comparison_table = compare_to_baseline(result["splits"]["test"]["metrics"], baseline_metrics)
        result["baseline_comparison"] = comparison_table.to_dict(orient="records")
        tables_dir.mkdir(parents=True, exist_ok=True)
        comparison_table.to_csv(tables_dir / f"{model_name}_h{horizon}_baseline_comparison.csv", index=False)

    if full_report and "test" in split_frames and len(split_frames["test"]) > 0:
        test_df = split_frames["test"]
        y_true = test_df[target_col].to_numpy()
        y_pred = test_df[prediction_col].to_numpy()
        figure_key_base = f"{task_name}_{model_name}_h{horizon}"

        figures_written = {}
        if task_type == "regression":
            figures_written.update(
                _write_regression_figures(
                    y_true, y_pred, task_name=task_name, model_name=model_name, horizon=horizon,
                    n=len(test_df), figures_dir=figures_dir, config_paths=config_paths,
                    figure_key_base=figure_key_base, data_source_label=data_source_label,
                )
            )
        else:
            y_pred_binary = (y_pred >= threshold).astype(int)
            figures_written.update(
                _write_classification_figures(
                    y_true, y_pred, y_pred_binary, task_name=task_name, model_name=model_name, horizon=horizon,
                    n=len(test_df), figures_dir=figures_dir, config_paths=config_paths,
                    figure_key_base=figure_key_base, n_bins=eval_config["reliability_diagram"]["n_bins"],
                    data_source_label=data_source_label,
                )
            )
        result["figures"] = {k: str(v["png"]) for k, v in figures_written.items()}

        breakdowns = run_all_breakdowns(
            test_df, y_true_col=target_col, y_pred_col=prediction_col, metric_fn=metric_fn,
            episode_catalogue=episode_catalogue, config=eval_config,
        )
        breakdown_paths = {}
        tables_dir.mkdir(parents=True, exist_ok=True)
        for dimension, table in breakdowns.items():
            path = tables_dir / f"failure_analysis_{model_name}_h{horizon}_{dimension}.csv"
            table.to_csv(path, index=False)
            breakdown_paths[dimension] = str(path)
        result["failure_analysis"] = breakdown_paths

    summary_path = results_dir / f"{model_name}_h{horizon}_evaluation.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _write_regression_figures(y_true, y_pred, *, task_name, model_name, horizon, n, figures_dir, config_paths, figure_key_base, data_source_label):
    written = {}
    fig1, src1 = observed_vs_predicted_figure(y_true, y_pred)
    key1 = f"{figure_key_base}_obs_vs_pred"
    prov1 = FigureProvenance(
        figure_number=assign_figure_number(key1),
        caption=f"Observed vs. predicted, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="task-native units (see task config target_description)",
        n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["observed_vs_predicted"] = save_figure_with_provenance(
        fig1, figures_dir / f"{model_name}_h{horizon}_obs_vs_pred", prov1, source_df=src1
    )

    fig2, src2 = residual_analysis_figure(y_true, y_pred)
    key2 = f"{figure_key_base}_residuals"
    prov2 = FigureProvenance(
        figure_number=assign_figure_number(key2),
        caption=f"Residual analysis, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="task-native units (residual = predicted - observed)",
        n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["residual_analysis"] = save_figure_with_provenance(
        fig2, figures_dir / f"{model_name}_h{horizon}_residuals", prov2, source_df=src2
    )
    return written


def _write_classification_figures(y_true, y_prob, y_pred_binary, *, task_name, model_name, horizon, n, figures_dir, config_paths, figure_key_base, n_bins, data_source_label):
    written = {}

    fig1, src1 = confusion_matrix_figure(y_true, y_pred_binary)
    key1 = f"{figure_key_base}_confusion_matrix"
    prov1 = FigureProvenance(
        figure_number=assign_figure_number(key1), caption=f"Confusion matrix, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="count", n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["confusion_matrix"] = save_figure_with_provenance(
        fig1, figures_dir / f"{model_name}_h{horizon}_confusion_matrix", prov1, source_df=src1
    )

    fig2, src2 = reliability_diagram(y_true, y_prob, n_bins=n_bins)
    key2 = f"{figure_key_base}_reliability"
    prov2 = FigureProvenance(
        figure_number=assign_figure_number(key2), caption=f"Reliability diagram with bin counts, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="probability", n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["reliability_diagram"] = save_figure_with_provenance(
        fig2, figures_dir / f"{model_name}_h{horizon}_reliability", prov2, source_df=src2
    )

    fig3, src3 = pr_curve_figure(y_true, y_prob)
    key3 = f"{figure_key_base}_pr_curve"
    prov3 = FigureProvenance(
        figure_number=assign_figure_number(key3), caption=f"Precision-recall curve, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="probability", n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["pr_curve"] = save_figure_with_provenance(
        fig3, figures_dir / f"{model_name}_h{horizon}_pr_curve", prov3, source_df=src3
    )

    fig4, src4 = roc_curve_figure(y_true, y_prob)
    key4 = f"{figure_key_base}_roc_curve"
    prov4 = FigureProvenance(
        figure_number=assign_figure_number(key4), caption=f"ROC curve, {task_name} ({model_name}, horizon={horizon} sol, test split).",
        units="probability", n=n, split="test", horizon_sols=horizon, data_source=data_source_label,
        script="modeling/train.py + evaluation/evaluate_task.py", config_paths=config_paths,
    )
    written["roc_curve"] = save_figure_with_provenance(
        fig4, figures_dir / f"{model_name}_h{horizon}_roc_curve", prov4, source_df=src4
    )
    return written
