"""Schema definitions and validation for Martian Chapter 4 data providers.

`FieldSpec`/`validate_schema` are the generic, reusable validation mechanism:
each real provider (data_sources/real_emirs_provider.py,
data_sources/real_mcd_provider.py) is responsible for declaring its own
FieldSpec list from its own config (e.g. configs/real_emirs_schema.yaml) --
never hardcoded here, per the project rule that all scientific parameters
live in config files.

Validates, per task requirement 3 (schemas validated before processing) and the
four requested failure categories: missing columns, invalid units, duplicate
observations, and inconsistent versions.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class SchemaValidationError(ValueError):
    """Raised when a provider's output dataframe violates its declared schema."""


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    unit: str | None
    required: bool
    chapter_grounded: bool
    grounding_note: str
    verification_status: str
    valid_range: tuple[float, float] | None = None


def validate_schema(
    df: pd.DataFrame,
    field_specs: list[FieldSpec],
    provider_name: str,
    *,
    check_duplicates_on: list[str] | None = None,
    check_version_consistency: bool = True,
) -> None:
    """Validate `df` against `field_specs`. Raises SchemaValidationError listing
    EVERY violation found (not just the first), so a single failing run gives a
    complete diagnostic in one pass.

    Covers, per task requirement:
      - missing columns              (required-field presence check)
      - invalid units                (declared-unit-tag mismatch + numeric
                                       out-of-declared-range as a practical proxy
                                       for a full dimensional-unit check — see
                                       planning/SCHEMA_DOCUMENTATION.md for the
                                       explicit limitation of this proxy)
      - duplicate observations        (opt-in via check_duplicates_on)
      - inconsistent versions         (opt-in via check_version_consistency)
    """
    errors: list[str] = []

    # 1. Missing columns
    required_names = {s.name for s in field_specs if s.required}
    missing = required_names - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # 2a. Declared-unit-tag consistency (df.attrs["units"] vs schema's declared unit)
    declared_units = df.attrs.get("units", {})
    for spec in field_specs:
        if spec.unit is None or spec.name not in declared_units:
            continue
        if declared_units[spec.name] != spec.unit:
            errors.append(
                f"Column '{spec.name}' declares unit '{declared_units[spec.name]}' "
                f"but schema requires '{spec.unit}'."
            )

    # 2b. Numeric range sanity — a practical proxy for unit mismatches (e.g.
    # degrees vs radians typically manifests as an out-of-range value).
    for spec in field_specs:
        if spec.name not in df.columns or spec.valid_range is None:
            continue
        col = df[spec.name]
        out_of_range = col[(col.notna()) & ((col < spec.valid_range[0]) | (col > spec.valid_range[1]))]
        if len(out_of_range) > 0:
            errors.append(
                f"Column '{spec.name}' has {len(out_of_range)} value(s) outside "
                f"declared valid range {spec.valid_range} (unit={spec.unit!r}) — "
                f"possible unit mismatch or invalid-units bug."
            )

    # 3. Duplicate observations
    if check_duplicates_on:
        present_keys = [c for c in check_duplicates_on if c in df.columns]
        if present_keys:
            dup_mask = df.duplicated(subset=present_keys, keep=False)
            if dup_mask.any():
                errors.append(
                    f"{int(dup_mask.sum())} duplicate observation(s) found on key "
                    f"columns {present_keys}."
                )

    # 4. Inconsistent versions within a single run
    if check_version_consistency and "version" in df.columns:
        unique_versions = df["version"].dropna().unique()
        if len(unique_versions) > 1:
            errors.append(
                f"Inconsistent 'version' values within a single provider run: "
                f"{sorted(map(str, unique_versions))}. A single run must originate "
                f"from exactly one product version."
            )

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed for provider '{provider_name}' with "
            f"{len(errors)} violation(s):\n- " + "\n- ".join(errors)
        )
