"""Validates a solar ingestion run config BEFORE any network access starts.
Collects every problem found in one pass (same discipline as
martian/mars/validate_inputs.py)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

SUPPORTED_DATASETS = ("goes", "sharp")


class InputValidationError(ValueError):
    """Raised by `validate_inputs` listing every problem found."""


def validate_inputs(config: dict) -> None:
    errors: list[str] = []

    datasets = config.get("datasets")
    if not datasets:
        errors.append("Missing required config key: 'datasets' (e.g. ['goes', 'sharp'])")
    else:
        for name in datasets:
            if name not in SUPPORTED_DATASETS:
                errors.append(f"datasets contains {name!r} -- only {SUPPORTED_DATASETS} are implemented")

    if datasets and "goes" in datasets:
        goes_cfg = config.get("goes")
        if not goes_cfg or "start_date" not in goes_cfg or "end_date" not in goes_cfg:
            errors.append("goes.start_date and goes.end_date are required when 'goes' is in datasets")
        else:
            try:
                start = date.fromisoformat(goes_cfg["start_date"])
                end = date.fromisoformat(goes_cfg["end_date"])
                if end < start:
                    errors.append(f"goes.end_date {goes_cfg['end_date']!r} is before goes.start_date {goes_cfg['start_date']!r}")
            except ValueError as exc:
                errors.append(f"goes date range is not valid ISO dates (YYYY-MM-DD): {exc}")

    output_root = config.get("output_root")
    if output_root is not None:
        try:
            probe_dir = Path(output_root)
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_file = probe_dir / ".solar_write_probe"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
        except OSError as exc:
            errors.append(f"output_root {output_root!r} is not writable: {exc}")

    if errors:
        raise InputValidationError(
            f"Run config failed validation with {len(errors)} problem(s):\n- " + "\n- ".join(errors)
        )
