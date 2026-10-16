"""Configurable, first-failing-rule quality-control pipeline for Martian
retrievals. Rule definitions, priority order, enable flags, and thresholds all
live in configs/quality_control.yaml — see that file's header comment for the
chapter-grounding discipline. This module only implements the mechanics.
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
    config: dict

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
            "n_input": n_input,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "rejection_reasons": self.rule_counts(),
        }


def load_qc_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "quality_control.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ids = sorted(r["id"] for r in raw["rules"])
    expected = list(range(1, len(raw["rules"]) + 1))
    if ids != expected:
        raise ValueError(
            f"configs/quality_control.yaml rule ids must form a contiguous "
            f"1..N sequence with no gaps or duplicates; got {ids}."
        )
    return raw


def _check_missing_required_variable(df: pd.DataFrame, params: dict) -> pd.Series:
    required = params["required_variables"]
    fails = pd.Series(False, index=df.index)
    for col in required:
        if col not in df.columns:
            fails |= True
            continue
        fails |= df[col].isna()
    return fails


def _check_invalid_quality_flag(df: pd.DataFrame, params: dict) -> pd.Series:
    return df["quality_flag"] != params["pass_value"]


def _check_impossible_coordinate_or_timestamp(df: pd.DataFrame, params: dict) -> pd.Series:
    lat_lo, lat_hi = params["latitude_range"]
    lon_lo, lon_hi = params["longitude_range"]
    my_lo, my_hi = params["mars_year_range"]
    sol_lo, sol_hi = params["sol_range"]
    fails = (
        df["latitude"].isna() | (df["latitude"] < lat_lo) | (df["latitude"] > lat_hi)
        | df["longitude"].isna() | (df["longitude"] < lon_lo) | (df["longitude"] > lon_hi)
        | df["mars_year"].isna() | (df["mars_year"] < my_lo) | (df["mars_year"] > my_hi)
        | df["sol"].isna() | (df["sol"] < sol_lo) | (df["sol"] > sol_hi)
        | df["utc_timestamp"].isna()
    )
    return fails


def _check_negative_optical_depth(df: pd.DataFrame, params: dict) -> pd.Series:
    fails = pd.Series(False, index=df.index)
    for col in params["variables"]:
        fails |= df[col] < 0
    return fails


def _check_physically_impossible_temperature(df: pd.DataFrame, params: dict) -> pd.Series:
    surf_lo, surf_hi = params["surface_temperature_k_range"]
    atm_lo, atm_hi = params["atmospheric_temperature_k_range"]
    fails = (df["surface_temperature_k"] < surf_lo) | (df["surface_temperature_k"] > surf_hi)
    atm_cols = [c for c in df.columns if c.startswith("atm_temp_k_level_") and not c.endswith("_uncertainty")]
    for col in atm_cols:
        vals = df[col]
        fails |= vals.notna() & ((vals < atm_lo) | (vals > atm_hi))
    return fails


def _check_excessive_retrieval_uncertainty(df: pd.DataFrame, params: dict) -> pd.Series:
    epsilon = params["epsilon"]
    fails = pd.Series(False, index=df.index)
    for var, max_rel in params["max_relative_uncertainty"].items():
        unc_col = f"{var}_uncertainty"
        denom = df[var].abs().clip(lower=epsilon)
        fails |= (df[unc_col] / denom) > max_rel
    return fails


def _check_unacceptable_viewing_geometry(df: pd.DataFrame, params: dict) -> pd.Series:
    if "emission_angle_deg" not in df.columns:
        # Real EMIRS has no documented viewing-geometry field (see
        # configs/quality_control.yaml rule 7, disabled for real data) --
        # not evaluable, not a failure, mirrors the same NaN-as-"not
        # applicable" treatment preprocessing/aggregation.py already uses
        # for pressure_level_coverage_frac.
        return pd.Series(False, index=df.index)
    return df["emission_angle_deg"] > params["max_emission_angle_deg"]


def _check_duplicate_retrieval(df: pd.DataFrame, params: dict) -> pd.Series:
    key_columns = params["key_columns"]
    return df.duplicated(subset=key_columns, keep="first")


_RULE_FUNCTIONS = {
    "missing_required_variable": _check_missing_required_variable,
    "invalid_quality_flag": _check_invalid_quality_flag,
    "impossible_coordinate_or_timestamp": _check_impossible_coordinate_or_timestamp,
    "negative_optical_depth": _check_negative_optical_depth,
    "physically_impossible_temperature": _check_physically_impossible_temperature,
    "excessive_retrieval_uncertainty": _check_excessive_retrieval_uncertainty,
    "unacceptable_viewing_geometry": _check_unacceptable_viewing_geometry,
    "duplicate_retrieval": _check_duplicate_retrieval,
}


def run_quality_control(df: pd.DataFrame, config: dict | None = None) -> QCReport:
    """Apply the configured QC rules in strict priority (id-ascending) order.
    Each retrieval is rejected under the FIRST enabled rule it fails; a rule
    disabled in config is evaluated for reporting purposes (n_flagged) but
    never rejects any row (n_rejected_here is always 0 for a disabled rule)."""
    config = config or load_qc_config()
    rules = sorted(config["rules"], key=lambda r: r["id"])

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
                id=rule["id"],
                name=name,
                enabled=enabled,
                source=rule["source"],
                n_flagged=n_flagged,
                n_rejected_here=n_rejected_here,
            )
        )

    out["qc_status"] = np.where(already_rejected, QC_STATUS_REJECTED, QC_STATUS_ACCEPTED)
    out["qc_rejection_rule"] = rejection_rule
    return QCReport(df=out, rule_results=rule_results, config=config)
