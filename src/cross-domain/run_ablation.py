"""Real A1-A7 ablation training (Section 5.6.1, `configs/ablation.yaml`) --
the README's own described "seven full ablation training runs", now
authorized. `evaluation/ablation.py` remains the structural registry this
script reads; nothing there changes.

    cd src/cross-domain
    python run_ablation.py

Trains/loads all 7 configurations for real:
  A1  pre-existing checkpoint heads, unchanged (zero new training)
  A2  a new, capacity-matched unimodal model (fusion/unimodal.py),
      trained fresh per domain (training/train_unimodal.py)
  A3-A7  the real CrossDomainFusionModel/train_fusion, varying
      use_fusion_pathway/other_domain_embedding/lambda_align per
      configs/ablation.yaml's own real entries

Evaluates every configuration against the SAME real A1 baseline (via
evaluation/comparison.py, the same machinery run_fusion.py uses), computes
the real `decisive_comparisons` deltas, and writes
outputs/ablation_run_<timestamp>/ablation_report.md (the presentation
layer, evaluation/presentation.py) alongside every trained model's own
state dict/history and the full negative-transfer/failure-case CSVs.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config, load_solar_encoder
from evaluation.ablation import decisive_comparisons, get_configuration, load_ablation_config
from evaluation.comparison import evaluate_predictions
from evaluation.presentation import (
    ablation_comparison_figure,
    failure_case_attention_figure,
    negative_transfer_figure,
    render_presentation_report,
    save_figure,
)
from evaluation.unimodal_baseline import load_unimodal_head_for_domain
from fusion.model import CrossDomainFusionModel
from fusion.unimodal import CapacityMatchedUnimodalModel, domain_share_of_trainable_params
from fusion_data import FUSION_SEED, MARTIAN_EXPORT_DIR, SOLAR_EXPORT_DIR, ensure_both_exports, load_domain_arrays, load_fusion_config
from training.train_fusion import train_fusion
from training.train_unimodal import train_unimodal

DOMAINS = ("solar", "martian")
_TRAINING_CONFIG_KEYS = (
    "learning_rate", "batch_size_solar", "batch_size_martian", "epochs",
    "early_stopping_patience", "lambda_align", "lambda_bal", "lambda_wd",
)


def _base_training_config(fusion_config: dict) -> dict:
    return {k: fusion_config[k] for k in _TRAINING_CONFIG_KEYS}


def _tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def _fmt(value: float) -> str:
    return "NaN" if isinstance(value, float) and np.isnan(value) else f"{value:.4f}" if isinstance(value, float) else str(value)


def _predict_joint(model: CrossDomainFusionModel, x_solar: torch.Tensor, x_martian: torch.Tensor, *, cfg: dict, seed: int) -> dict:
    """One forward call gives both domains' predictions -- cfg supplies
    use_fusion_pathway/other_domain_embedding, matching what the model was
    TRAINED with (evaluating an ablation under its own real regime, not a
    mismatched one)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.forward(
            x_solar, x_martian,
            use_fusion_pathway=cfg["use_fusion_pathway"], other_domain_embedding=cfg["other_domain_embedding"],
        )
    result = {}
    for domain, y_key, alpha_key in (("solar", "y_hat_solar", "alpha_solar"), ("martian", "y_hat_martian", "alpha_martian")):
        result[domain] = {
            "y_prob": out[y_key].numpy(),
            "alpha": out[alpha_key].numpy() if out[alpha_key] is not None else None,
        }
    return result


def _train_a1(encoders_config: dict) -> dict:
    print("--- A1: unimodal_baseline (pre-existing checkpoints, zero new training) ---", flush=True)
    solar_encoder = load_solar_encoder(encoders_config)
    martian_encoder, martian_provenance = load_martian_encoder(encoders_config)
    heads = {"solar": load_unimodal_head_for_domain("solar", encoders_config), "martian": load_unimodal_head_for_domain("martian", encoders_config)}
    encoders = {"solar": solar_encoder, "martian": martian_encoder}
    return {"heads": heads, "encoders": encoders, "martian_provenance": martian_provenance}


def _predict_a1(a1: dict, domain: str, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return a1["heads"][domain](a1["encoders"][domain](x)).numpy()


def _train_a7(arrays: dict, encoders_config: dict, base_training_config: dict, seed: int) -> tuple[CrossDomainFusionModel, dict]:
    print("--- A7: full_model (the real fusion model) ---", flush=True)
    solar_encoder = load_solar_encoder(encoders_config)
    martian_encoder, _provenance = load_martian_encoder(encoders_config)
    fusion_config = load_fusion_config()
    torch.manual_seed(seed)
    model = CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=fusion_config["shared_embedding_dim"], attention_hidden_size=fusion_config["attention_scorer_hidden_size"],
        alignment_variance_weight=fusion_config["alignment_variance_weight"],
    )
    cfg = get_configuration("A7")
    model, history, convergence_status, runtime_seconds = train_fusion(
        model,
        arrays["solar"]["X_train"], arrays["solar"]["y_train"], arrays["martian"]["X_train"], arrays["martian"]["y_train"],
        arrays["solar"]["X_val"], arrays["solar"]["y_val"], arrays["martian"]["X_val"], arrays["martian"]["y_val"],
        training_config=base_training_config, seed=seed,
        use_fusion_pathway=cfg["use_fusion_pathway"], other_domain_embedding=cfg["other_domain_embedding"],
    )
    print(f"    convergence_status={convergence_status}, runtime={runtime_seconds:.1f}s", flush=True)
    return model, {"history": history, "convergence_status": convergence_status, "runtime_seconds": runtime_seconds}


def _train_fusion_variant(config_id: str, arrays: dict, encoders_config: dict, base_training_config: dict, seed: int) -> tuple[CrossDomainFusionModel, dict]:
    cfg = get_configuration(config_id)
    print(f"--- {config_id}: {cfg['name']} ---", flush=True)
    solar_encoder = load_solar_encoder(encoders_config)
    martian_encoder, _provenance = load_martian_encoder(encoders_config)
    fusion_config = load_fusion_config()
    torch.manual_seed(seed)
    model = CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=fusion_config["shared_embedding_dim"], attention_hidden_size=fusion_config["attention_scorer_hidden_size"],
        alignment_variance_weight=fusion_config["alignment_variance_weight"],
    )
    training_config = dict(base_training_config)
    training_config["lambda_align"] = base_training_config["lambda_align"] if cfg["use_alignment_loss"] else 0.0

    model, history, convergence_status, runtime_seconds = train_fusion(
        model,
        arrays["solar"]["X_train"], arrays["solar"]["y_train"], arrays["martian"]["X_train"], arrays["martian"]["y_train"],
        arrays["solar"]["X_val"], arrays["solar"]["y_val"], arrays["martian"]["X_val"], arrays["martian"]["y_val"],
        training_config=training_config, seed=seed,
        use_fusion_pathway=cfg["use_fusion_pathway"], other_domain_embedding=cfg["other_domain_embedding"],
    )
    print(f"    convergence_status={convergence_status}, runtime={runtime_seconds:.1f}s", flush=True)
    return model, {"history": history, "convergence_status": convergence_status, "runtime_seconds": runtime_seconds}


def _train_a2(arrays: dict, encoders_config: dict, base_training_config: dict, a7_model: CrossDomainFusionModel, seed: int) -> dict:
    print("--- A2: capacity_matched_unimodal (new model, trained fresh per domain) ---", flush=True)
    fusion_config = load_fusion_config()
    shared_dim = fusion_config["shared_embedding_dim"]

    solar_encoder = load_solar_encoder(encoders_config)
    martian_encoder, _provenance = load_martian_encoder(encoders_config)
    domain_encoders = {"solar": solar_encoder, "martian": martian_encoder}
    domain_targets = {
        "solar": domain_share_of_trainable_params(
            projection=a7_model.projection_solar, scorer=a7_model.scorer_solar_task, head=a7_model.head_solar
        ),
        "martian": domain_share_of_trainable_params(
            projection=a7_model.projection_martian, scorer=a7_model.scorer_martian_task, head=a7_model.head_martian
        ),
    }

    models: dict[str, CapacityMatchedUnimodalModel] = {}
    histories: dict[str, dict] = {}
    for domain in DOMAINS:
        model = CapacityMatchedUnimodalModel(domain_encoders[domain], shared_dim=shared_dim, target_param_count=domain_targets[domain])
        model, history, convergence_status, runtime_seconds = train_unimodal(
            model, arrays[domain]["X_train"], arrays[domain]["y_train"], arrays[domain]["X_val"], arrays[domain]["y_val"],
            domain=domain, training_config=base_training_config, seed=seed,
        )
        print(f"    [{domain}] target_params={domain_targets[domain]}, actual={model.trainable_parameter_count()}, "
              f"convergence_status={convergence_status}, runtime={runtime_seconds:.1f}s", flush=True)
        models[domain] = model
        histories[domain] = {"history": history, "convergence_status": convergence_status, "runtime_seconds": runtime_seconds,
                              "target_param_count": domain_targets[domain], "actual_param_count": model.trainable_parameter_count()}
    return {"models": models, "histories": histories}


def run_ablation(*, seed: int = FUSION_SEED, predictions_dir: Path | None = None, light: bool = False) -> Path:
    """`seed` overrides `FUSION_SEED` for stage-2 fusion/unimodal training and
    prediction, so this same entry point can be reused for the Section 5.6.1
    seed-variation sweep (`run_uncertainty_sweep.py`) without duplicating the
    training orchestration. A1 is unaffected by `seed`: it loads the existing,
    already-frozen Chapter 3/4 checkpoints and trains nothing.

    `predictions_dir`: when given, every config/domain's fused test-set
    predictions are also written there as
    `predictions_<config>_<domain>_seed<seed>.csv` (evaluation/comparison.py),
    the real input `uncertainty_analysis.py` consumes.

    `light`: when True, skips this run's own model checkpoints, figures, and
    presentation report (kept for a normal single-seed run, wasteful when
    repeated across many seeds purely to harvest predictions CSVs). Negative-
    transfer/failure-case CSVs and `ablation_report.md` are still skipped in
    this mode; only `history_*.json`, `decisive_comparisons.csv`, and (via
    `predictions_dir`) the per-instance predictions are written.
    """
    ensure_both_exports()
    arrays = {"solar": load_domain_arrays(SOLAR_EXPORT_DIR), "martian": load_domain_arrays(MARTIAN_EXPORT_DIR)}

    encoders_config = load_encoders_config()
    fusion_config = load_fusion_config()
    base_training_config = _base_training_config(fusion_config)

    run_id = f"ablation_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}Z"
    out_dir = CROSS_DOMAIN_ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    # ---- Train/load every configuration ----
    a1 = _train_a1(encoders_config)
    a7_model, a7_meta = _train_a7(arrays, encoders_config, base_training_config, seed)
    a2 = _train_a2(arrays, encoders_config, base_training_config, a7_model, seed)
    a3_model, a3_meta = _train_fusion_variant("A3", arrays, encoders_config, base_training_config, seed)
    a4_model, a4_meta = _train_fusion_variant("A4", arrays, encoders_config, base_training_config, seed)
    a5_model, a5_meta = _train_fusion_variant("A5", arrays, encoders_config, base_training_config, seed)
    a6_model, a6_meta = _train_fusion_variant("A6", arrays, encoders_config, base_training_config, seed)

    fusion_variants = {"A3": (a3_model, a3_meta), "A4": (a4_model, a4_meta), "A5": (a5_model, a5_meta), "A6": (a6_model, a6_meta), "A7": (a7_model, a7_meta)}

    # ---- Predict every configuration on val (threshold selection) + test ----
    print("--- Evaluation ---", flush=True)
    predictions: dict[str, dict[str, dict]] = {}  # config_id -> domain -> {"y_prob_val", "y_prob_test", "alpha_test"}

    predictions["A1"] = {
        domain: {
            "y_prob_val": _predict_a1(a1, domain, _tensor(arrays[domain]["X_val"])),
            "y_prob_test": _predict_a1(a1, domain, _tensor(arrays[domain]["X_test"])),
            "alpha_test": None,
        }
        for domain in DOMAINS
    }

    predictions["A2"] = {
        domain: {
            "y_prob_val": a2["models"][domain](_tensor(arrays[domain]["X_val"])).detach().numpy(),
            "y_prob_test": a2["models"][domain](_tensor(arrays[domain]["X_test"])).detach().numpy(),
            "alpha_test": None,
        }
        for domain in DOMAINS
    }

    for config_id, (model, _meta) in fusion_variants.items():
        cfg = get_configuration(config_id)
        model.eval()
        val_out = _predict_joint(model, _tensor(arrays["solar"]["X_val"]), _tensor(arrays["martian"]["X_val"]), cfg=cfg, seed=seed)
        test_out = _predict_joint(model, _tensor(arrays["solar"]["X_test"]), _tensor(arrays["martian"]["X_test"]), cfg=cfg, seed=seed + 1)
        predictions[config_id] = {
            domain: {"y_prob_val": val_out[domain]["y_prob"], "y_prob_test": test_out[domain]["y_prob"], "alpha_test": test_out[domain]["alpha"]}
            for domain in DOMAINS
        }

    # ---- Evaluate every config against the real A1 baseline (same machinery as run_fusion.py) ----
    eval_results: dict[str, dict[str, dict]] = {}
    for config_id in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        eval_results[config_id] = {}
        for domain in DOMAINS:
            y_train, y_val, y_test = arrays[domain]["y_train"], arrays[domain]["y_val"], arrays[domain]["y_test"]
            eval_results[config_id][domain] = evaluate_predictions(
                y_train, y_val, y_test,
                predictions[config_id][domain]["y_prob_val"], predictions[config_id][domain]["y_prob_test"],
                predictions["A1"][domain]["y_prob_val"], predictions["A1"][domain]["y_prob_test"],
                arrays[domain]["regime_test"],
                alpha_fused_test=predictions[config_id][domain]["alpha_test"],
                predictions_dir=predictions_dir, config_id=config_id, domain=domain, seed=seed,
            )

    # ---- decisive_comparisons deltas (real, from evaluation/ablation.py's own registry) ----
    ablation_config = load_ablation_config()
    comparisons = decisive_comparisons(ablation_config)
    comparison_summary = []
    for pair in comparisons:
        for domain in DOMAINS:
            tss_values = {cfg_id: eval_results[cfg_id][domain]["tss_fused"] for cfg_id in pair}
            comparison_summary.append({"comparison": " vs ".join(pair), "domain": domain, **{f"tss_{cfg_id}": tss_values[cfg_id] for cfg_id in pair}})

    # ---- Save artifacts ----
    for config_id, (model, meta) in fusion_variants.items():
        (out_dir / f"history_{config_id}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if not light:
            torch.save(model.state_dict(), out_dir / f"model_{config_id}.pt")
    if not light:
        for domain in DOMAINS:
            torch.save(a2["models"][domain].state_dict(), out_dir / f"model_A2_{domain}.pt")
    (out_dir / "history_A2.json").write_text(json.dumps(a2["histories"], indent=2, default=str), encoding="utf-8")

    tss_summary = {config_id: {domain: eval_results[config_id][domain]["tss_fused"] for domain in DOMAINS} for config_id in eval_results}
    (out_dir / "tss_summary.json").write_text(json.dumps({"seed": seed, "tss": tss_summary}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(comparison_summary).to_csv(out_dir / "decisive_comparisons.csv", index=False)

    if light:
        total_elapsed = time.perf_counter() - start
        print(f"\nAblation run complete (light mode, seed={seed}) in {total_elapsed:.1f}s: {out_dir}", flush=True)
        return out_dir

    for config_id in eval_results:
        for domain in DOMAINS:
            eval_results[config_id][domain]["negative_transfer_table"].to_csv(out_dir / f"negative_transfer_{config_id}_{domain}.csv", index=False)
            eval_results[config_id][domain]["failure_cases"].to_csv(out_dir / f"failure_cases_{config_id}_{domain}.csv", index=False)

    # ---- Presentation layer ----
    print("--- Presentation layer ---", flush=True)
    data_source_label = f"REAL (solar) / {a1['martian_provenance']}"
    figure_paths: dict[str, Path] = {}
    ablation_results_for_report: dict[str, dict[str, dict]] = {
        cfg_id: {domain: {"tss": eval_results[cfg_id][domain]["tss_fused"]} for domain in DOMAINS} for cfg_id in eval_results
    }
    for domain in DOMAINS:
        results_for_domain = {cfg_id: v[domain] for cfg_id, v in ablation_results_for_report.items()}
        fig, _ = ablation_comparison_figure(results_for_domain, domain=domain, data_source_label=data_source_label)
        figure_paths[f"ablation_comparison_{domain}"] = save_figure(fig, out_dir / "figures" / f"ablation_comparison_{domain}.png", data_source_label=data_source_label)

    # Genuine pairwise comparisons (e.g. A7 vs A2 specifically, not each
    # config's separately-computed vs-A1 comparison) -- evaluate_predictions
    # is called fresh per pair here rather than reusing eval_results, which
    # is keyed to the fixed A1 baseline.
    negative_transfer_tables = {}
    failure_case_tables = {}
    for pair in comparisons:
        if len(pair) != 2:
            continue
        fused_id, unimodal_id = pair[0], pair[1]
        for domain in DOMAINS:
            key = f"{fused_id}_vs_{unimodal_id}_{domain}"
            y_train, y_val, y_test = arrays[domain]["y_train"], arrays[domain]["y_val"], arrays[domain]["y_test"]
            pair_result = evaluate_predictions(
                y_train, y_val, y_test,
                predictions[fused_id][domain]["y_prob_val"], predictions[fused_id][domain]["y_prob_test"],
                predictions[unimodal_id][domain]["y_prob_val"], predictions[unimodal_id][domain]["y_prob_test"],
                arrays[domain]["regime_test"],
                alpha_fused_test=predictions[fused_id][domain]["alpha_test"],
            )
            negative_transfer_tables[key] = pair_result["negative_transfer_table"]
            failure_case_tables[key] = pair_result["failure_cases"]

    for key, table in negative_transfer_tables.items():
        fig, _ = negative_transfer_figure(table, domain=key, data_source_label=data_source_label)
        figure_paths[f"negative_transfer_{key}"] = save_figure(fig, out_dir / "figures" / f"negative_transfer_{key}.png", data_source_label=data_source_label)
    for key, table in failure_case_tables.items():
        fig, _ = failure_case_attention_figure(table, domain=key, data_source_label=data_source_label)
        figure_paths[f"failure_cases_{key}"] = save_figure(fig, out_dir / "figures" / f"failure_cases_{key}.png", data_source_label=data_source_label)

    presentation_report = render_presentation_report(
        run_id=run_id,
        negative_transfer_tables=negative_transfer_tables, failure_case_tables=failure_case_tables,
        figure_paths={k: v.relative_to(out_dir) for k, v in figure_paths.items()},
        data_source_label=data_source_label,
        ablation_results=ablation_results_for_report,
    )

    ablation_report = _render_ablation_report(
        run_id=run_id, eval_results=eval_results, comparison_summary=comparison_summary,
        martian_provenance=a1["martian_provenance"], a2_histories=a2["histories"],
        figure_paths={k: v.relative_to(out_dir) for k, v in figure_paths.items()},
    )
    (out_dir / "ablation_report.md").write_text(ablation_report + "\n\n---\n\n" + presentation_report, encoding="utf-8")

    total_elapsed = time.perf_counter() - start
    print(f"\nAblation run complete in {total_elapsed:.1f}s: {out_dir}", flush=True)
    print(f"Report: {out_dir / 'ablation_report.md'}", flush=True)
    return out_dir


def _render_ablation_report(*, run_id: str, eval_results: dict, comparison_summary: list[dict], martian_provenance: str,
                             a2_histories: dict, figure_paths: dict) -> str:
    lines = [
        f"# A1-A7 Ablation Comparison -- Real Run `{run_id}`",
        "",
        f"**Martian data provenance: {martian_provenance}.** Both domains' encoders are trained",
        "on real, not simulated, data as of this run. Where martian TSS is reported as NaN below,",
        "that split had zero positive examples in the sequence-feasible subset for that",
        "configuration -- a real class-balance finding, not a fabricated value or an artifact of",
        "the ablation axis being tested; check the per-domain n_train/prevalence figures in",
        "final_report.md before interpreting any NaN as \"broken.\"",
        "",
        "## TSS by configuration",
        "",
        "| config | solar TSS | martian TSS |",
        "|---|---|---|",
    ]
    for config_id in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        solar_tss = eval_results[config_id]["solar"]["tss_fused"]
        martian_tss = eval_results[config_id]["martian"]["tss_fused"]
        lines.append(f"| {config_id} | {_fmt(solar_tss)} | {_fmt(martian_tss)} |")
    lines.append("")

    lines.append("## A2 capacity-matching (real parameter counts)")
    lines.append("")
    lines.append("| domain | target_param_count | actual_param_count |")
    lines.append("|---|---|---|")
    for domain, meta in a2_histories.items():
        lines.append(f"| {domain} | {meta['target_param_count']} | {meta['actual_param_count']} |")
    lines.append("")

    lines.append("## Decisive comparisons (evaluation/ablation.py::decisive_comparisons)")
    lines.append("")
    for row in comparison_summary:
        other_keys = [k for k in row if k.startswith("tss_")]
        values = ", ".join(f"{k[4:]}={_fmt(row[k])}" for k in other_keys)
        lines.append(f"- **{row['comparison']}** ({row['domain']}): {values}")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for key in sorted(figure_paths):
        if key.startswith("ablation_comparison_"):
            lines.append(f"![{key}]({figure_paths[key].as_posix()})")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the A1-A7 ablation comparison.")
    parser.add_argument("--seed", type=int, default=FUSION_SEED, help="overrides FUSION_SEED for stage-2 training (default: FUSION_SEED)")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                         help="if given, also writes predictions_<config>_<domain>_seed<NN>.csv here (uncertainty_analysis.py's input format)")
    parser.add_argument("--light", action="store_true",
                         help="skip this run's own model checkpoints/figures/presentation report; only history/tss_summary/predictions are written")
    args = parser.parse_args(argv)

    try:
        run_ablation(seed=args.seed, predictions_dir=args.predictions_dir, light=args.light)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
