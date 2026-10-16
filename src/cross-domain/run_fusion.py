"""Chapter 5's first real, end-to-end cross-domain fusion training run --
the README's own described "deliberate next-step decision requiring an
explicit check-in", now authorized.

    cd src/cross-domain
    python run_fusion.py

Loads real solar/martian sequence-tensor exports (fusion_data.py --
produced by src/solar/export_fusion_sequences.py and
src/martian/export_fusion_sequences.py, invoked automatically here if
their outputs don't already exist), the two real frozen checkpoints, runs
Algorithm 5.1 Stage 2 fusion training for real (A7, "full_model" --
see run_ablation.py for the other 6 configurations), evaluates fused
predictions against each domain's own real unimodal baseline
(evaluation/unimodal_baseline.py) via evaluation/comparison.py, and writes
a full training report plus a negative-transfer/failure-case presentation
report (evaluation/presentation.py). Every real, honest limitation of this
run (martian's real but comparatively small sequence-feasible sample, and
any degenerate single-class split) is stated plainly in both reports --
never hidden.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config, load_solar_encoder
from evaluation.comparison import evaluate_predictions
from evaluation.presentation import failure_case_attention_figure, negative_transfer_figure, render_presentation_report, save_figure
from evaluation.unimodal_baseline import load_unimodal_head_for_domain
from fusion.model import CrossDomainFusionModel
from fusion_data import FUSION_SEED, MARTIAN_EXPORT_DIR, SOLAR_EXPORT_DIR, ensure_both_exports, load_domain_arrays, load_fusion_config
from training.train_fusion import train_fusion


def _evaluate_domain(domain: str, arrays: dict, other_arrays: dict, model: CrossDomainFusionModel, unimodal_head, encoder) -> dict:
    x_test = torch.tensor(arrays["X_test"], dtype=torch.float32)
    x_val = torch.tensor(arrays["X_val"], dtype=torch.float32)
    other_x_test = torch.tensor(other_arrays["X_test"], dtype=torch.float32)
    other_x_val = torch.tensor(other_arrays["X_val"], dtype=torch.float32)

    with torch.no_grad():
        if domain == "solar":
            out_test = model.forward(x_test, other_x_test)
            out_val = model.forward(x_val, other_x_val)
            y_prob_fused_test = out_test["y_hat_solar"].numpy()
            y_prob_fused_val = out_val["y_hat_solar"].numpy()
            alpha_fused_test = out_test["alpha_solar"].numpy()
        else:
            out_test = model.forward(other_x_test, x_test)
            out_val = model.forward(other_x_val, x_val)
            y_prob_fused_test = out_test["y_hat_martian"].numpy()
            y_prob_fused_val = out_val["y_hat_martian"].numpy()
            alpha_fused_test = out_test["alpha_martian"].numpy()

        y_prob_unimodal_test = unimodal_head(encoder(x_test)).numpy()
        y_prob_unimodal_val = unimodal_head(encoder(x_val)).numpy()

    return evaluate_predictions(
        arrays["y_train"], arrays["y_val"], arrays["y_test"],
        y_prob_fused_val, y_prob_fused_test,
        y_prob_unimodal_val, y_prob_unimodal_test,
        arrays["regime_test"],
        alpha_fused_test=alpha_fused_test,
    )


def _fmt(value: float) -> str:
    return "NaN (undefined -- see class-balance caveat below)" if np.isnan(value) else f"{value:.4f}"


def _render_report(*, run_id: str, convergence_status: str, runtime_seconds: float, steps_per_epoch: int,
                    martian_provenance: str, solar: dict, martian: dict) -> str:
    lines = [
        f"# Cross-Domain Fusion -- Real Run `{run_id}`",
        "",
        f"**Martian data provenance: {martian_provenance}.** Both domains' encoders are trained",
        "on real, not simulated, data as of this run.",
        "",
        f"- Convergence status: `{convergence_status}`",
        f"- Runtime: {runtime_seconds:.1f}s",
        f"- Steps per epoch: {steps_per_epoch} "
        + ("(**floored to 1** -- martian's real train sample is smaller than batch_size_martian, "
           "see configs/fusion.yaml; this is a real, expected consequence of the current sequence-"
           "feasibility-filtered sample size, not a bug)" if steps_per_epoch == 1 else ""),
        "",
        "## Per-domain results",
        "",
        "| | solar | martian |",
        "|---|---|---|",
        f"| n_train / n_val / n_test | {solar['n_train']} / {solar['n_val']} / {solar['n_test']} | "
        f"{martian['n_train']} / {martian['n_val']} / {martian['n_test']} |",
        f"| train / val / test prevalence | {_fmt(solar['train_prevalence'])} / {_fmt(solar['val_prevalence'])} / "
        f"{_fmt(solar['test_prevalence'])} | {_fmt(martian['train_prevalence'])} / {_fmt(martian['val_prevalence'])} / "
        f"{_fmt(martian['test_prevalence'])} |",
        f"| TSS (fused) | {_fmt(solar['tss_fused'])} | {_fmt(martian['tss_fused'])} |",
        f"| TSS (unimodal baseline) | {_fmt(solar['tss_unimodal'])} | {_fmt(martian['tss_unimodal'])} |",
        f"| threshold (fused, source) | {solar['threshold_fused']:.3f} ({solar['threshold_fused_source']}) | "
        f"{martian['threshold_fused']:.3f} ({martian['threshold_fused_source']}) |",
        f"| threshold (unimodal, source) | {solar['threshold_unimodal']:.3f} ({solar['threshold_unimodal_source']}) | "
        f"{martian['threshold_unimodal']:.3f} ({martian['threshold_unimodal_source']}) |",
        f"| negative-transfer regimes | {solar['n_negative_transfer_regimes']}/{solar['n_regimes_evaluated']} | "
        f"{martian['n_negative_transfer_regimes']}/{martian['n_regimes_evaluated']} |",
        f"| failure cases catalogued | {solar['n_failure_cases']} | {martian['n_failure_cases']} |",
        "",
    ]
    for domain, results in (("solar", solar), ("martian", martian)):
        if np.isnan(results["train_prevalence"]) or results["train_prevalence"] == 0.0 or results["test_prevalence"] == 0.0:
            lines.append(
                f"**{domain.upper()} CLASS-BALANCE CAVEAT**: at least one split has zero positive examples "
                f"(train prevalence={_fmt(results['train_prevalence'])}, test prevalence="
                f"{_fmt(results['test_prevalence'])}). TSS is mathematically undefined here (POD and/or FAR "
                "has no denominator) -- this is a real, honest data-availability finding, not a fabricated "
                "metric or a bug in this run.\n"
            )
    lines += [
        "## Files in this run directory",
        "",
        "- `training_history.json` -- per-epoch loss breakdown (Eq. 5.8's terms)",
        "- `model_state_dict.pt` -- the trained fusion model's full state dict",
        "- `negative_transfer_{solar,martian}.csv` -- per-regime TSS delta (Section 5.6.2)",
        "- `failure_cases_{solar,martian}.csv` -- missed-event/false-alarm catalogue (Section 5.6.3)",
        "- `negative_transfer_and_failure_report.md` -- the presentation layer (figures + stamped tables)",
    ]
    return "\n".join(lines)


def run_fusion() -> Path:
    ensure_both_exports()

    solar = load_domain_arrays(SOLAR_EXPORT_DIR)
    martian = load_domain_arrays(MARTIAN_EXPORT_DIR)

    encoders_config = load_encoders_config()
    fusion_config = load_fusion_config()

    solar_encoder = load_solar_encoder(encoders_config)
    martian_encoder, martian_provenance = load_martian_encoder(encoders_config)

    # Seed BEFORE model construction so DomainProjection/AttentionScorer/head
    # weight initialization is reproducible across process runs too, not
    # just the batch sampling inside train_fusion (which seeds separately).
    torch.manual_seed(FUSION_SEED)
    model = CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=fusion_config["shared_embedding_dim"],
        attention_hidden_size=fusion_config["attention_scorer_hidden_size"],
        alignment_variance_weight=fusion_config["alignment_variance_weight"],
    )

    training_config = {
        k: fusion_config[k]
        for k in (
            "learning_rate", "batch_size_solar", "batch_size_martian", "epochs",
            "early_stopping_patience", "lambda_align", "lambda_bal", "lambda_wd",
        )
    }

    print("--- Stage 2: real fusion training ---", flush=True)
    start = time.perf_counter()
    model, history, convergence_status, runtime_seconds = train_fusion(
        model,
        solar["X_train"], solar["y_train"], martian["X_train"], martian["y_train"],
        solar["X_val"], solar["y_val"], martian["X_val"], martian["y_val"],
        training_config=training_config, seed=FUSION_SEED,
    )
    print(f"    convergence_status={convergence_status}, runtime={runtime_seconds:.1f}s", flush=True)

    steps_per_epoch = max(
        1,
        min(
            len(solar["X_train"]) // training_config["batch_size_solar"],
            len(martian["X_train"]) // training_config["batch_size_martian"],
        ),
    )

    print("--- Evaluation ---", flush=True)
    model.eval()
    solar_head = load_unimodal_head_for_domain("solar", encoders_config)
    martian_head = load_unimodal_head_for_domain("martian", encoders_config)
    solar_results = _evaluate_domain("solar", solar, martian, model, solar_head, solar_encoder)
    martian_results = _evaluate_domain("martian", martian, solar, model, martian_head, martian_encoder)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}Z"
    out_dir = CROSS_DOMAIN_ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")
    solar_results["negative_transfer_table"].to_csv(out_dir / "negative_transfer_solar.csv", index=False)
    martian_results["negative_transfer_table"].to_csv(out_dir / "negative_transfer_martian.csv", index=False)
    solar_results["failure_cases"].to_csv(out_dir / "failure_cases_solar.csv", index=False)
    martian_results["failure_cases"].to_csv(out_dir / "failure_cases_martian.csv", index=False)

    report = _render_report(
        run_id=run_id, convergence_status=convergence_status, runtime_seconds=runtime_seconds,
        steps_per_epoch=steps_per_epoch, martian_provenance=martian_provenance,
        solar=solar_results, martian=martian_results,
    )
    (out_dir / "final_report.md").write_text(report, encoding="utf-8")

    print("--- Presentation layer ---", flush=True)
    data_source_label = f"REAL (solar) / {martian_provenance}"
    figure_paths: dict[str, Path] = {}
    for domain, results in (("solar", solar_results), ("martian", martian_results)):
        fig, _ = negative_transfer_figure(results["negative_transfer_table"], domain=domain, data_source_label=data_source_label)
        figure_paths[f"negative_transfer_{domain}"] = save_figure(
            fig, out_dir / "figures" / f"negative_transfer_{domain}.png", data_source_label=data_source_label
        )
        fig, _ = failure_case_attention_figure(results["failure_cases"], domain=domain, data_source_label=data_source_label)
        figure_paths[f"failure_cases_{domain}"] = save_figure(
            fig, out_dir / "figures" / f"failure_cases_{domain}.png", data_source_label=data_source_label
        )
    presentation_report = render_presentation_report(
        run_id=run_id,
        negative_transfer_tables={"solar": solar_results["negative_transfer_table"], "martian": martian_results["negative_transfer_table"]},
        failure_case_tables={"solar": solar_results["failure_cases"], "martian": martian_results["failure_cases"]},
        figure_paths={k: v.relative_to(out_dir) for k, v in figure_paths.items()},
        data_source_label=data_source_label,
    )
    (out_dir / "negative_transfer_and_failure_report.md").write_text(presentation_report, encoding="utf-8")

    total_elapsed = time.perf_counter() - start
    print(f"\nRun complete in {total_elapsed:.1f}s: {out_dir}", flush=True)
    print(f"Final report: {out_dir / 'final_report.md'}", flush=True)
    print(f"Presentation report: {out_dir / 'negative_transfer_and_failure_report.md'}", flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run_fusion()
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
