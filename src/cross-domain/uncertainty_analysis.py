"""
Uncertainty quantification for the Chapter 5 cross-domain fusion results.

Implements the three-component protocol specified in Section 5.6.1:

  1. Seed variation      -- mean and standard deviation of TSS across >= 5 seeds.
  2. Block bootstrap     -- stratified block resampling of the test partition,
                            B = 1000 resamples, 95% percentile intervals.
                            Blocks are contiguous in time so that the temporal
                            clustering of flares / dust events is preserved.
  3. Paired comparison   -- differences between configurations (and per-regime
                            negative-transfer deltas) recomputed on identical
                            resamples; a difference is "established" only where
                            the 95% interval excludes zero.

This script does no training. It consumes per-instance predictions that the
fusion package already produces, so it can be run without touching the model
code.

--------------------------------------------------------------------------
INPUT FORMAT
--------------------------------------------------------------------------
One CSV per (configuration, domain, seed), named:

    predictions_<config>_<domain>_seed<NN>.csv        e.g. predictions_A7_solar_seed44.csv

with columns:

    y_true      int   0/1 ground-truth label, in chronological order
    y_prob      float predicted probability
    threshold   float operating point used for this model (repeated per row,
                      or supply --threshold-from-file to read a single value)
    regime      str   regime label for this instance (solar: calendar year;
                      martian: Ls quartile Q1-Q4)

Rows MUST be in chronological order -- the block bootstrap depends on it.

Emitted directly by this package's own evaluation/comparison.py::
evaluate_predictions (its `predictions_dir`/`config_id`/`domain`/`seed`
kwargs), called once per config/domain by run_ablation.py -- see
run_uncertainty_sweep.py, which drives the real seed sweep this script's
--input-dir is meant to point at. Real block lengths are domain-specific,
not the single illustrative "24" used as a fallback below: solar rows are
12-minute cadence (src/solar/configs/goes_preprocessing.yaml's
resample_cadence_minutes: 12) and the forecast horizon is 24h
(src/solar/configs/labeling.yaml's forecast_horizon_hours: 24), so the
real block length spanning one full horizon is 24*60/12 = 120 rows, not
24. Martian rows are already one row per sol with a 1-sol horizon, so 1 is
correct there. --block-length-solar/--block-length-martian below default
to these real values; --block-length overrides both uniformly for anyone
who deliberately wants a single value across domains.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python uncertainty_analysis.py --input-dir outputs/ablation_run_<id> \
                                   --output uncertainty_report.md \
                                   --resamples 1000

--block-length-solar/--block-length-martian are the forecast window in
instances for each domain (see above); --block-length overrides both.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(r"predictions_(?P<config>[^_]+)_(?P<domain>[^_]+)_seed(?P<seed>\d+)\.csv$")

# Real, domain-specific default block lengths -- see the module docstring's
# "INPUT FORMAT" section for the derivation. Never a single global default:
# solar and martian rows do not represent the same span of physical time.
DEFAULT_BLOCK_LENGTHS = {"solar": 120, "martian": 1}


# ---------------------------------------------------------------- metrics


def true_skill_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TSS = TP/(TP+FN) - FP/(FP+TN). Returns NaN where either rate is undefined."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    if (tp + fn) == 0 or (fp + tn) == 0:
        return float("nan")
    return tp / (tp + fn) - fp / (fp + tn)


def tss_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    return true_skill_statistic(y_true, (y_prob >= threshold).astype(int))


# ---------------------------------------------------------------- bootstrap


def block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a moving-block bootstrap index array of length ~n.

    Blocks of `block_length` consecutive positions are sampled with replacement
    from all possible start positions, preserving within-block temporal
    structure (event clustering, autocorrelation).
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    n_blocks = int(np.ceil(n / block_length))
    max_start = max(n - block_length, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def bootstrap_tss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    block_length: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the bootstrap distribution of TSS (length = resamples)."""
    n = len(y_true)
    out = np.empty(resamples, dtype=float)
    for b in range(resamples):
        idx = block_indices(n, block_length, rng)
        out[b] = tss_from_probs(y_true[idx], y_prob[idx], threshold)
    return out


def paired_bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    block_length: int,
    resamples: int,
    seed: int = 20260818,
) -> np.ndarray:
    """Bootstrap distribution of TSS(a) - TSS(b) on IDENTICAL resamples.

    Both frames must describe the same test instances in the same order.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison requires identical test partitions")
    rng = np.random.default_rng(seed)
    ya, pa, ta = a["y_true"].to_numpy(), a["y_prob"].to_numpy(), float(a["threshold"].iloc[0])
    yb, pb, tb = b["y_true"].to_numpy(), b["y_prob"].to_numpy(), float(b["threshold"].iloc[0])
    if not np.array_equal(ya, yb):
        raise ValueError("labels differ between configurations -- not the same partition")
    out = np.empty(resamples, dtype=float)
    n = len(ya)
    for i in range(resamples):
        idx = block_indices(n, block_length, rng)
        out[i] = tss_from_probs(ya[idx], pa[idx], ta) - tss_from_probs(yb[idx], pb[idx], tb)
    return out


def percentile_interval(dist: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    lo = (100.0 - level) / 2.0
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, lo)), float(np.percentile(finite, 100.0 - lo))


def established(interval: tuple[float, float]) -> bool:
    lo, hi = interval
    return np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)


# ---------------------------------------------------------------- loading


@dataclass(frozen=True)
class Key:
    config: str
    domain: str
    seed: int


def load(input_dir: Path) -> dict[Key, pd.DataFrame]:
    frames: dict[Key, pd.DataFrame] = {}
    for path in sorted(input_dir.glob("predictions_*.csv")):
        m = FILENAME_RE.search(path.name)
        if not m:
            print(f"  skipping unrecognised filename: {path.name}")
            continue
        df = pd.read_csv(path)
        missing = {"y_true", "y_prob", "threshold"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {sorted(missing)}")
        if "regime" not in df.columns:
            df["regime"] = "all"
        frames[Key(m["config"], m["domain"], int(m["seed"]))] = df
    return frames


# ---------------------------------------------------------------- report


def fmt(x: float, dp: int = 4) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{dp}f}"


def _block_length_for(domain: str, block_lengths: dict[str, int]) -> int:
    if domain not in block_lengths:
        raise KeyError(
            f"no block length specified for domain '{domain}' -- pass --block-length-{domain} "
            "or --block-length, since a wrong default here would silently under- or over-preserve "
            "that domain's real event-clustering timescale"
        )
    return block_lengths[domain]


def build_report(
    frames: dict[Key, pd.DataFrame],
    block_lengths: dict[str, int],
    resamples: int,
    comparisons: list[tuple[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("# Uncertainty quantification -- Chapter 5, Section 5.6.1 protocol\n")
    lines.append(
        "Block lengths (domain-specific -- see module docstring): "
        + ", ".join(f"{d}={n} instances" for d, n in sorted(block_lengths.items()))
        + f". Bootstrap resamples: B = {resamples}. "
    )
    lines.append("Intervals are 95% percentile intervals.\n")

    domains = sorted({k.domain for k in frames})
    configs = sorted({k.config for k in frames})

    # ---- component 1 + 2: per-configuration point estimate, seed spread, CI
    lines.append("\n## 1-2. Per-configuration TSS: seed variation and bootstrap intervals\n")
    for domain in domains:
        block_length = _block_length_for(domain, block_lengths)
        lines.append(f"\n**{domain}** (block length = {block_length} instances)\n")
        lines.append("| Config | Seeds | TSS mean | TSS sd | 95% CI (primary seed) |")
        lines.append("|---|---|---|---|---|")
        for cfg in configs:
            keys = sorted([k for k in frames if k.config == cfg and k.domain == domain],
                          key=lambda k: k.seed)
            if not keys:
                continue
            per_seed = []
            for k in keys:
                df = frames[k]
                per_seed.append(tss_from_probs(df["y_true"].to_numpy(),
                                               df["y_prob"].to_numpy(),
                                               float(df["threshold"].iloc[0])))
            per_seed_arr = np.array(per_seed, dtype=float)
            primary = frames[keys[0]]
            rng = np.random.default_rng(20260818)
            dist = bootstrap_tss(primary["y_true"].to_numpy(),
                                 primary["y_prob"].to_numpy(),
                                 float(primary["threshold"].iloc[0]),
                                 block_length, resamples, rng)
            lo, hi = percentile_interval(dist)
            sd = float(np.nanstd(per_seed_arr, ddof=1)) if per_seed_arr.size > 1 else float("nan")
            lines.append(
                f"| {cfg} | {len(keys)} | {fmt(float(np.nanmean(per_seed_arr)))} | "
                f"{fmt(sd)} | [{fmt(lo)}, {fmt(hi)}] |"
            )
        if any(len([k for k in frames if k.config == c and k.domain == domain]) < 5
               for c in configs):
            lines.append(
                "\n> Fewer than five seeds are present for at least one configuration. "
                "Section 5.6.1 specifies a minimum of five; the seed standard deviations "
                "above are not yet interpretable.\n"
            )

    # ---- component 3: paired differences
    lines.append("\n## 3. Paired comparisons (identical resamples)\n")
    for domain in domains:
        block_length = _block_length_for(domain, block_lengths)
        lines.append(f"\n**{domain}** (block length = {block_length} instances)\n")
        lines.append("| Comparison | Difference | 95% CI | Established? |")
        lines.append("|---|---|---|---|")
        for ca, cb in comparisons:
            ka = [k for k in frames if k.config == ca and k.domain == domain]
            kb = [k for k in frames if k.config == cb and k.domain == domain]
            if not ka or not kb:
                continue
            a, b = frames[sorted(ka, key=lambda k: k.seed)[0]], frames[sorted(kb, key=lambda k: k.seed)[0]]
            try:
                dist = paired_bootstrap_difference(a, b, block_length, resamples)
            except ValueError as exc:
                lines.append(f"| {ca} - {cb} | -- | -- | error: {exc} |")
                continue
            point = (tss_from_probs(a["y_true"].to_numpy(), a["y_prob"].to_numpy(),
                                    float(a["threshold"].iloc[0]))
                     - tss_from_probs(b["y_true"].to_numpy(), b["y_prob"].to_numpy(),
                                      float(b["threshold"].iloc[0])))
            lo, hi = percentile_interval(dist)
            lines.append(
                f"| {ca} - {cb} | {fmt(point)} | [{fmt(lo)}, {fmt(hi)}] | "
                f"{'yes' if established((lo, hi)) else 'no'} |"
            )

    # ---- per-regime negative transfer deltas, with intervals
    lines.append("\n## 4. Per-regime negative-transfer deltas (Eq. 5.10) with intervals\n")
    for domain in domains:
        block_length = _block_length_for(domain, block_lengths)
        fused = [k for k in frames if k.config == "A7" and k.domain == domain]
        uni = [k for k in frames if k.config == "A1" and k.domain == domain]
        if not fused or not uni:
            lines.append(f"\n**{domain}**: A1 and/or A7 predictions absent -- skipped.\n")
            continue
        a = frames[sorted(fused, key=lambda k: k.seed)[0]]
        b = frames[sorted(uni, key=lambda k: k.seed)[0]]
        lines.append(f"\n**{domain}** (A7 vs A1, block length = {block_length} instances)\n")
        lines.append("| Regime | n | delta | 95% CI | Negative transfer established? |")
        lines.append("|---|---|---|---|---|")
        for regime in sorted(a["regime"].astype(str).unique()):
            mask = a["regime"].astype(str) == regime
            sub_a, sub_b = a[mask].reset_index(drop=True), b[mask].reset_index(drop=True)
            if len(sub_a) < block_length:
                lines.append(f"| {regime} | {len(sub_a)} | -- | -- | regime shorter than block |")
                continue
            dist = paired_bootstrap_difference(sub_a, sub_b, block_length, resamples)
            point = (tss_from_probs(sub_a["y_true"].to_numpy(), sub_a["y_prob"].to_numpy(),
                                    float(sub_a["threshold"].iloc[0]))
                     - tss_from_probs(sub_b["y_true"].to_numpy(), sub_b["y_prob"].to_numpy(),
                                      float(sub_b["threshold"].iloc[0])))
            lo, hi = percentile_interval(dist)
            verdict = "yes" if (established((lo, hi)) and hi < 0) else "no"
            lines.append(f"| {regime} | {len(sub_a)} | {fmt(point)} | [{fmt(lo)}, {fmt(hi)}] | {verdict} |")

    lines.append("\n---\n")
    lines.append("Generated by uncertainty_analysis.py. Every value above is computed from the "
                 "supplied prediction files; nothing is imputed. Substitute these tables for the "
                 "point estimates in Tables 5.3 and 5.4 once at least five seeds are present.\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- entry point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="directory containing predictions_<config>_<domain>_seed<NN>.csv files")
    ap.add_argument("--output", type=Path, default=Path("uncertainty_report.md"))
    ap.add_argument("--block-length", type=int, default=None,
                    help="override block length for ALL domains (advanced; normally left unset "
                         "so --block-length-<domain> applies per domain)")
    ap.add_argument("--block-length-solar", type=int, default=DEFAULT_BLOCK_LENGTHS["solar"],
                    help=f"solar bootstrap block length in instances (default {DEFAULT_BLOCK_LENGTHS['solar']} "
                         "= 24h horizon / 12min cadence)")
    ap.add_argument("--block-length-martian", type=int, default=DEFAULT_BLOCK_LENGTHS["martian"],
                    help=f"martian bootstrap block length in instances (default {DEFAULT_BLOCK_LENGTHS['martian']} "
                         "= 1-sol horizon, 1 row per sol)")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()

    frames = load(args.input_dir)
    if not frames:
        raise SystemExit(f"No prediction files found in {args.input_dir}. See the module "
                         f"docstring for the expected filename pattern and columns.")
    print(f"Loaded {len(frames)} prediction file(s): "
          f"{sorted({k.config for k in frames})} x {sorted({k.domain for k in frames})}")

    block_lengths = (
        {"solar": args.block_length, "martian": args.block_length} if args.block_length is not None
        else {"solar": args.block_length_solar, "martian": args.block_length_martian}
    )

    comparisons = [("A7", "A2"), ("A7", "A1"), ("A7", "A3"), ("A7", "A4"),
                   ("A5", "A7"), ("A5", "A6"), ("A6", "A7")]
    report = build_report(frames, block_lengths, args.resamples, comparisons)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
