"""Section 5.6.1's uncertainty-quantification protocol, executed for real --
closes the "no paired-bootstrap confidence intervals exist yet" gap
CHAPTER_5_PARAMETER_VALUES.md's caveat 4 has flagged since this package's
first build.

    cd src/cross-domain
    python run_uncertainty_sweep.py

Retrains the A1-A7 ablation (run_ablation.py, in its `light=True` mode --
history/tss_summary/predictions only, no duplicate model checkpoints or
figures across seeds) at >= 5 independent seeds, collecting every
config/domain's fused test-set predictions into one shared
outputs/uncertainty_sweep_<id>/ directory via evaluation/comparison.py's
`predictions_dir` kwarg, then runs uncertainty_analysis.py's full protocol
(seed variation, block bootstrap, paired comparison) against them and
writes uncertainty_report.md alongside the raw predictions CSVs.

A1 does not vary with seed -- it is the unchanged, already-frozen Chapter
3/4 checkpoint, deterministic in eval mode -- so it is retrained (cheaply;
no gradient steps) and re-predicted once per seed anyway for uniformity,
and its resulting predictions_A1_<domain>_seed<NN>.csv files are expected
to be byte-identical across seeds. This is a real, honest property of A1,
not a bug: uncertainty_analysis.py's own seed-sd column will show it
directly as 0.0, which is the correct answer for a config with no
stochastic training.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import uncertainty_analysis
from encoders.solar_encoder import CROSS_DOMAIN_ROOT
from fusion_data import FUSION_SEED
from run_ablation import run_ablation

# FUSION_SEED (44) is kept first so it remains the "primary seed" whose
# point estimate matches the existing canonical run
# (outputs/ablation_run_20260814T064653396282Z) cited elsewhere in
# CHAPTER_5_PARAMETER_VALUES.md -- >= 5 seeds total per Section 5.6.1.
SEEDS = [FUSION_SEED, 45, 46, 47, 48]

RESAMPLES = 1000

COMPARISONS = [("A7", "A2"), ("A7", "A1"), ("A7", "A3"), ("A7", "A4"),
               ("A5", "A7"), ("A5", "A6"), ("A6", "A7")]


def run_uncertainty_sweep(seeds: list[int] = SEEDS, resamples: int = RESAMPLES) -> Path:
    sweep_id = f"uncertainty_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}Z"
    predictions_dir = CROSS_DOMAIN_ROOT / "outputs" / sweep_id
    predictions_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== seed {seed} ({i}/{len(seeds)}) ===", flush=True)
        seed_start = time.perf_counter()
        run_ablation(seed=seed, predictions_dir=predictions_dir, light=True)
        print(f"    seed {seed} complete in {time.perf_counter() - seed_start:.1f}s", flush=True)
    training_elapsed = time.perf_counter() - start
    print(f"\nTraining sweep complete ({len(seeds)} seeds) in {training_elapsed:.1f}s", flush=True)

    print("--- Uncertainty analysis (Section 5.6.1 protocol) ---", flush=True)
    frames = uncertainty_analysis.load(predictions_dir)
    if not frames:
        raise SystemExit(f"No prediction files found in {predictions_dir} -- the sweep produced nothing.")
    report = uncertainty_analysis.build_report(
        frames, uncertainty_analysis.DEFAULT_BLOCK_LENGTHS, resamples, COMPARISONS
    )
    report_path = predictions_dir / "uncertainty_report.md"
    report_path.write_text(report, encoding="utf-8")

    total_elapsed = time.perf_counter() - start
    print(f"\nUncertainty sweep complete in {total_elapsed:.1f}s: {predictions_dir}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return predictions_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run_uncertainty_sweep()
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
