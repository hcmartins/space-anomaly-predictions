"""Configurable, first-failing-rule quality-control pipeline shared by GOES
and SHARP. Rule definitions, priority order, enable flags, and thresholds
all live in configs/quality_control.yaml (one section per domain) -- see
that file's header for the verified-vs-researcher-decision distinction.
This module only implements the mechanics (mirrors
martian/preprocessing/quality_control.py's design exactly)..
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

QC_STATUS_ACCEPTED = "accepted"
QC_STATUS_REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class QCRuleResult:
    id: int
    name: str
    enabled: bool
    source: str
    n_flagged: int
    n_rejected_here: int


@dataclasses.dataclass
class QCReport:
    df: pd.DataFrame
    rule_results: list[QCRuleResult]
    domain: str

    @property
    def accepted(self) -> pd.DataFrame:
        mask = self.df["qc_status"] == QC_STATUS_ACCEPTED
        return self.df.loc[mask].drop(columns=["qc_status", "qc_rejection_rule"])

    @property
    def rejected(self) -> pd.DataFrame:
        mask = self.df["qc_status"] == QC_STATUS_REJECTED
        return self.df.loc[mask]

    def rule_counts(self) -> dict:
        return {r.name: r.n_rejected_here for r in self.rule_results}

    def summary(self) -> dict:
        n_input = len(self.df)
        n_accepted = int((self.df["qc_status"] == QC_STATUS_ACCEPTED).sum())
        n_rejected = n_input - n_accepted
        return {
            "domain": self.domain,
            "n_input": n_input,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "rejection_reasons": self.rule_counts(),
        }


def load_qc_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "quality_control.yaml"
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for domain in ("goes", "sharp"):
        ids = sorted(r["id"] for r in raw[domain]["rules"])
        expected = list(range(1, len(raw[domain]["rules"]) + 1))
        if ids != expected:
            raise ValueError(
                f"configs/quality_control.yaml [{domain}] rule ids must form a contiguous "
                f"1..N sequence with no gaps or duplicates; got {ids}."
            )
    return raw


def _check_missing_required_column(df: pd.DataFrame, params: dict) -> pd.Series:
    required = params["required_columns"]
    fails = pd.Series(False, index=df.index)
    for col in required:
        if col not in df.columns:
            fails |= True
            continue
        fails |= df[col].isna()
    return fails


def _check_invalid_quality_flag(df: pd.DataFrame, params: dict) -> pd.Series:
    col = params["column"]
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col] != params["pass_value"]


def _check_non_positive_value(df: pd.DataFrame, params: dict) -> pd.Series:
    fails = pd.Series(False, index=df.index)
    for col in params["columns"]:
        fails |= df[col] <= 0
    return fails


def _check_duplicate_row(df: pd.DataFrame, params: dict) -> pd.Series:
    key_columns = params["key_columns"]
    return df.duplicated(subset=key_columns, keep="first")


def _check_non_finite_value(df: pd.DataFrame, params: dict) -> pd.Series:
    """Rejects rows where any of `columns` is +/-inf -- a genuine JSOC/NOAA
    data artifact found in real ingested data (a handful of SHARP records
    with e.g. MEANJZH=5e+25 or TOTUSJZ=inf), not something either archive's
    own quality-flag column reliably catches. Checks isinf specifically, not
    ~isfinite: a NaN in an optional (non-required) physical keyword is a
    legitimate "this record didn't report that keyword" case handled
    elsewhere (aggregation skips NaN), and must not be rejected wholesale
    here -- only an actual infinite value is unambiguously invalid physics."""
    fails = pd.Series(False, index=df.index)
    for col in params["columns"]:
        if col in df.columns:
            fails |= np.isinf(pd.to_numeric(df[col], errors="coerce"))
    return fails


_RULE_FUNCTIONS = {
    "missing_required_column": _check_missing_required_column,
    "invalid_quality_flag": _check_invalid_quality_flag,
    "non_positive_value": _check_non_positive_value,
    "duplicate_row": _check_duplicate_row,
    "non_finite_value": _check_non_finite_value,
}


def run_quality_control(df: pd.DataFrame, domain: str, config: dict | None = None) -> QCReport:
    """Apply the configured `domain` ("goes" | "sharp") QC rules in strict
    priority (id-ascending) order. Each row is rejected under the FIRST
    enabled rule it fails; a rule disabled in config is evaluated for
    reporting purposes (n_flagged) but never rejects any row."""
    config = config or load_qc_config()
    rules = sorted(config[domain]["rules"], key=lambda r: r["id"])

    out = df.reset_index(drop=True).copy()
    already_rejected = pd.Series(False, index=out.index)
    rejection_rule = pd.Series(pd.NA, index=out.index, dtype="object")
    rule_results: list[QCRuleResult] = []

    for rule in rules:
        name = rule["name"]
        enabled = rule["enabled"]
        fn = _RULE_FUNCTIONS[name]
        fails = fn(out, rule.get("parameters", {}))
        n_flagged = int(fails.sum())

        if enabled:
            newly_rejected = fails & (~already_rejected)
            rejection_rule.loc[newly_rejected] = name
            n_rejected_here = int(newly_rejected.sum())
            already_rejected = already_rejected | fails
        else:
            n_rejected_here = 0

        rule_results.append(
            QCRuleResult(
                id=rule["id"], name=name, enabled=enabled, source=rule["source"],
                n_flagged=n_flagged, n_rejected_here=n_rejected_here,
            )
        )

    out["qc_status"] = np.where(already_rejected, QC_STATUS_REJECTED, QC_STATUS_ACCEPTED)
    out["qc_rejection_rule"] = rejection_rule
    return QCReport(df=out, rule_results=rule_results, domain=domain)
