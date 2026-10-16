"""Chapter 6/7's figure/table/report presentation layer for negative-
transfer (Section 5.6.2) and failure-case (Section 5.6.3) results, plus
the A1-A7 ablation comparison (Section 5.6.1).

"Chapter 6/7" is this package's own invented label (confirmed by grep: no
real Chapter 6/7 numbering, citation, or figure-registry scheme exists
anywhere in this repo, unlike src/martian's real Chapter 4 figure registry
-- see evaluation/figure_provenance.py there). This module does not
attempt to fabricate one; it reports plainly and stamps every artifact
with its real data-source label instead.

stamp_figure/stamp_table_markdown/stamp_report are a small, deliberate
duplication of src/martian/data_sources/provenance.py's real, tested
convention -- same reasoning as encoders/frozen_encoder.py: cross-domain
never imports solar/martian modules directly (sys.path collision risk),
and these are each under 15 lines, cheap and safe to reproduce exactly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stamp_figure(fig, data_source_label: str) -> None:
    """Stamp a matplotlib Figure with a visible caption recording its data
    source. Must be called before fig.savefig()."""
    fig.text(0.01, 0.01, f"Data source: {data_source_label}", fontsize=7, color="dimgray", ha="left", va="bottom")


def stamp_table_markdown(df: pd.DataFrame, data_source_label: str, *, title: str | None = None) -> str:
    """Render df as a Markdown table with a mandatory data-source caption
    line. No `tabulate` dependency, by house convention."""
    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
    lines.append(f"*Data source: {data_source_label}*")
    lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def stamp_report(body_text: str, data_source_label: str) -> str:
    """Prefix/suffix a text report with a visible, hard-to-miss data-source
    declaration at both the top and the bottom."""
    banner = f"[DATA SOURCE: {data_source_label}]"
    return f"{banner}\n\n{body_text}\n\n{banner}"


def negative_transfer_figure(negative_transfer_df: pd.DataFrame, *, domain: str, data_source_label: str) -> tuple[plt.Figure, pd.DataFrame]:
    """Bar chart of Delta_t(r) per regime (Section 5.6.2), colored by
    whether that regime shows negative transfer. Returns (fig, source_df)
    -- the caller saves/stamps/closes the figure; this function never
    writes to disk itself, matching src/martian's evaluation figure
    convention."""
    df = negative_transfer_df.copy()
    colors = df["negative_transfer"].map({True: "firebrick", False: "steelblue"}).fillna("lightgray")

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(df)), 4))
    ax.bar(df["regime"].astype(str), df["delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Regime")
    ax.set_ylabel(r"$\Delta_t(r)$ = TSS(fused) - TSS(unimodal)")
    ax.set_title(f"Negative-transfer analysis by regime -- {domain}")
    fig.tight_layout()
    return fig, df


def failure_case_attention_figure(failure_cases_df: pd.DataFrame, *, domain: str, data_source_label: str) -> tuple[plt.Figure, pd.DataFrame]:
    """Scatter of attention weights at issuance for every catalogued
    failure case (Section 5.6.3), colored by failure type and marker
    style by whether the unimodal baseline shared the same failure."""
    df = failure_cases_df.copy()
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(df) == 0:
        ax.text(0.5, 0.5, "No failure cases catalogued", ha="center", va="center", transform=ax.transAxes)
    else:
        for failure_type, color in (("missed_event", "firebrick"), ("false_alarm", "darkorange")):
            subset = df[df["failure_type"] == failure_type]
            if len(subset) == 0:
                continue
            shared = subset["shared_with_unimodal"].astype(bool)
            ax.scatter(
                subset.loc[shared, "alpha_solar_at_issuance"], subset.loc[shared, "alpha_martian_at_issuance"],
                color=color, marker="x", label=f"{failure_type} (shared with unimodal)",
            )
            ax.scatter(
                subset.loc[~shared, "alpha_solar_at_issuance"], subset.loc[~shared, "alpha_martian_at_issuance"],
                color=color, marker="o", facecolors="none", label=f"{failure_type} (fusion-only)",
            )
        ax.legend(fontsize=7, loc="best")
    ax.set_xlabel(r"$\alpha_{solar}$ at issuance")
    ax.set_ylabel(r"$\alpha_{martian}$ at issuance")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Failure-case attention weights -- {domain}")
    fig.tight_layout()
    return fig, df


def ablation_comparison_figure(results_by_config: dict[str, dict], *, domain: str, data_source_label: str) -> tuple[plt.Figure, pd.DataFrame]:
    """Bar chart of TSS by ablation configuration id, for one domain.
    `results_by_config`: {config_id: {"tss": float, ...}} -- a NaN tss is
    plotted as an explicit gap (annotated "N/A"), never silently omitted
    or plotted as zero (that would misrepresent an undefined metric as a
    real zero-skill result)."""
    source_df = pd.DataFrame(
        [{"config_id": cfg_id, "tss": values.get("tss", float("nan"))} for cfg_id, values in results_by_config.items()]
    )
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(source_df)), 4))
    heights = source_df["tss"].fillna(0.0)
    bar_colors = ["lightgray" if pd.isna(v) else "steelblue" for v in source_df["tss"]]
    ax.bar(source_df["config_id"], heights, color=bar_colors)
    for i, (cfg_id, tss) in enumerate(zip(source_df["config_id"], source_df["tss"])):
        if pd.isna(tss):
            ax.text(i, 0.01, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel("TSS")
    ax.set_title(f"A1-A7 ablation comparison -- {domain}")
    fig.tight_layout()
    return fig, source_df


def save_figure(fig: plt.Figure, path: Path, *, data_source_label: str) -> Path:
    """Stamps, saves, and closes a figure -- the one place this module
    touches disk, so every figure this package produces goes through the
    same provenance-stamping step."""
    stamp_figure(fig, data_source_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_presentation_report(
    *,
    run_id: str,
    negative_transfer_tables: dict[str, pd.DataFrame],
    failure_case_tables: dict[str, pd.DataFrame],
    figure_paths: dict[str, Path],
    data_source_label: str,
    ablation_results: dict[str, dict[str, dict]] | None = None,
) -> str:
    """Markdown report combining stamped tables + figure links + real
    caveats. `negative_transfer_tables`/`failure_case_tables`: {domain:
    DataFrame}. `figure_paths`: {figure_key: Path}, used as relative links
    only (this function never re-derives a number that isn't already in
    one of the passed-in tables -- mars/final_report.py's own principle).
    `ablation_results` (optional): {config_id: {domain: {"tss": float}}}
    for run_ablation.py's report -- omitted entirely for run_fusion.py's
    single-model report."""
    lines: list[str] = [
        f"# Negative-Transfer and Failure-Case Report -- `{run_id}`",
        "",
        "\"Chapter 6/7\" is this package's own label for this presentation layer -- ",
        "no real thesis chapter/figure numbering scheme exists to inherit (see this module's docstring).",
        "",
    ]

    if ablation_results is not None:
        lines.append("## A1-A7 ablation comparison")
        lines.append("")
        domains = sorted({d for per_domain in ablation_results.values() for d in per_domain})
        for domain in domains:
            rows = pd.DataFrame(
                [
                    {"config_id": cfg_id, "tss": per_domain.get(domain, {}).get("tss", float("nan"))}
                    for cfg_id, per_domain in ablation_results.items()
                ]
            )
            lines.append(stamp_table_markdown(rows, data_source_label, title=f"TSS by configuration -- {domain}"))
            lines.append("")
            fig_key = f"ablation_comparison_{domain}"
            if fig_key in figure_paths:
                lines.append(f"![Ablation comparison -- {domain}]({figure_paths[fig_key].as_posix()})")
                lines.append("")

    lines.append("## Negative-transfer analysis (Section 5.6.2)")
    lines.append("")
    for domain, table in negative_transfer_tables.items():
        lines.append(stamp_table_markdown(table, data_source_label, title=f"Negative transfer by regime -- {domain}"))
        lines.append("")
        fig_key = f"negative_transfer_{domain}"
        if fig_key in figure_paths:
            lines.append(f"![Negative transfer -- {domain}]({figure_paths[fig_key].as_posix()})")
            lines.append("")

    lines.append("## Failure-case catalogue (Section 5.6.3)")
    lines.append("")
    for domain, table in failure_case_tables.items():
        if len(table) == 0:
            lines.append(f"*{domain}*: no failure cases catalogued (or not applicable for this configuration).")
            lines.append("")
            continue
        lines.append(stamp_table_markdown(table.head(20), data_source_label, title=f"Failure cases (first 20 of {len(table)}) -- {domain}"))
        lines.append("")
        fig_key = f"failure_cases_{domain}"
        if fig_key in figure_paths:
            lines.append(f"![Failure cases -- {domain}]({figure_paths[fig_key].as_posix()})")
            lines.append("")

    return stamp_report("\n".join(lines), data_source_label)
