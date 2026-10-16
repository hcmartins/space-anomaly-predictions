"""Provenance utilities shared by every Chapter 4 data provider.

Implements:
  - the per-record provenance contract (task requirement 1),
  - the `data_source` column requirement for processed datasets (requirement 5),
  - visible data-source stamping for figures / tables / reports (requirement 6).

Nothing in this module reads or writes real EMIRS/MCD/catalogue data. It only
defines the mechanism every provider (simulated or, later, real) must use.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

# The 8 provenance columns that are always constructed by attach_batch_provenance.
# `quality_flag` is the 9th required column but is supplied by the caller as a
# physical-variable-derived Series (it is also a documented EMIRS variable in its
# own right — see configs/emirs_variables.yaml — so one column satisfies both
# roles rather than duplicating it under two names).
_ALWAYS_ATTACHED_COLUMNS: tuple[str, ...] = (
    "data_source",
    "product_id",
    "processing_level",
    "version",
    "access_date",
    "original_filename",
    "retrieval_timestamp",
    "checksum",
)

# The full required-provenance contract enforced by BaseProvider (base.py).
# Uncertainty is deliberately excluded from this list: per task requirement 1 it
# is required "where available", so its presence is checked per-variable via the
# "<variable>_uncertainty" naming convention (see schema.py), not as one fixed
# column name that every provider must always emit.
REQUIRED_PROVENANCE_COLUMNS: tuple[str, ...] = _ALWAYS_ATTACHED_COLUMNS + ("quality_flag",)

UNCERTAINTY_SUFFIX = "_uncertainty"


@dataclass(frozen=True)
class ProviderIdentity:
    """Static identity shared by every record a single provider *instance* emits
    in one call. Distinct from the per-row fields (quality_flag, checksum,
    retrieval_timestamp, original_filename) which vary row to row."""

    data_source: str  # "SIMULATED" | "REAL"
    product_id: str
    processing_level: str
    version: str
    verification_status: str = "RESEARCHER VERIFICATION REQUIRED"

    @property
    def access_date(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["access_date"] = self.access_date
        return d


def compute_row_checksum(row_values: Iterable[Any]) -> str:
    """SHA-256 (truncated to 16 hex chars for column readability) over a row's own
    content. This is a REAL checksum of the (synthetic) record's content — it
    verifies internal consistency and reproducibility of the generated data; it
    does not, and cannot, certify the authenticity of a real external product."""
    payload = "|".join("" if v is None else str(v) for v in row_values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def attach_batch_provenance(
    df: pd.DataFrame,
    identity: ProviderIdentity,
    *,
    original_filenames: pd.Series,
    quality_flags: pd.Series,
    retrieval_timestamps: pd.Series,
    checksum_source_columns: list[str],
) -> pd.DataFrame:
    """Attach all required provenance columns to every row of `df`.

    `checksum_source_columns` must name columns already present in `df` at call
    time (the physical-variable columns) — the checksum is computed over exactly
    those, so it changes if and only if the underlying content changes.
    """
    out = df.copy()
    out["data_source"] = identity.data_source
    out["product_id"] = identity.product_id
    out["processing_level"] = identity.processing_level
    out["version"] = identity.version
    out["access_date"] = identity.access_date
    out["original_filename"] = original_filenames.reset_index(drop=True).values if len(out) else original_filenames.values
    out["retrieval_timestamp"] = retrieval_timestamps.reset_index(drop=True).values if len(out) else retrieval_timestamps.values
    out["quality_flag"] = quality_flags.reset_index(drop=True).values if len(out) else quality_flags.values

    if len(out) == 0:
        out["checksum"] = pd.Series([], dtype="object")
    else:
        missing_cols = [c for c in checksum_source_columns if c not in out.columns]
        if missing_cols:
            raise KeyError(
                f"checksum_source_columns references columns not present in the "
                f"dataframe being stamped: {missing_cols}"
            )
        out["checksum"] = [
            compute_row_checksum(row)
            for row in out[checksum_source_columns].itertuples(index=False, name=None)
        ]
    return out


def stamp_processed_dataset(df: pd.DataFrame, data_source_label: str) -> pd.DataFrame:
    """Requirement 5: every PROCESSED dataset (post-gridding, post-feature-
    engineering, post-labeling — Phases 2-4, not yet implemented) must carry a
    `data_source` column. This is distinct from the raw per-record provenance
    columns attached at load time: a processed row may aggregate several raw
    records, so its `data_source` label describes the *processing stage's*
    provenance, not any single raw record's."""
    out = df.copy()
    out["data_source"] = data_source_label
    return out


# --------------------------------------------------------------------------
# Requirement 6: visible data-source labels on every figure, table, and report
# --------------------------------------------------------------------------

def stamp_figure(fig, data_source_label: str) -> None:
    """Stamp a matplotlib Figure with a visible caption recording its data
    source. Must be called before fig.savefig()."""
    fig.text(
        0.01, 0.01, f"Data source: {data_source_label}",
        fontsize=7, color="dimgray", ha="left", va="bottom",
    )


def stamp_table_markdown(df: pd.DataFrame, data_source_label: str, *, title: str | None = None) -> str:
    """Render df as a Markdown table with a mandatory data-source caption line.
    Implemented by hand (no `tabulate` dependency) to keep the dependency
    footprint minimal."""
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
