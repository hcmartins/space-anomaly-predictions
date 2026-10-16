"""Step 1: validate the run config BEFORE any expensive work starts.
Collects every problem found (not just the first), matching the
"validate_schema lists every violation in one pass" discipline already used
in data_sources/schema.py.
"""
from __future__ import annotations

from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = ("seed", "mars_year_range", "output_root")


class InputValidationError(ValueError):
    """Raised by `validate_inputs` listing every problem found."""


def validate_inputs(config: dict) -> None:
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            errors.append(f"Missing required config key: {key!r}")

    seed = config.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"seed must be an int, got {type(seed).__name__}")

    mars_year_range = config.get("mars_year_range")
    if mars_year_range is not None:
        if not (isinstance(mars_year_range, (list, tuple)) and len(mars_year_range) == 2):
            errors.append(f"mars_year_range must be a 2-element [start, end] list, got {mars_year_range!r}")
        else:
            start, end = mars_year_range
            if not (isinstance(start, int) and isinstance(end, int) and start >= 1 and end >= start):
                errors.append(f"mars_year_range {mars_year_range!r} must be two integers with start<=end, start>=1")

    for task_name in config.get("tasks", []):
        task_config_path = Path(config.get("configs_dir", "configs")) / f"task_{task_name}.yaml"
        if not task_config_path.exists():
            errors.append(f"Task {task_name!r} references a missing config: {task_config_path}")

    output_root = config.get("output_root")
    if output_root is not None:
        try:
            probe_dir = Path(output_root)
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_file = probe_dir / ".mars_write_probe"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
        except OSError as exc:
            errors.append(f"output_root {output_root!r} is not writable: {exc}")

    if errors:
        raise InputValidationError(
            f"Run config failed validation with {len(errors)} problem(s):\n- " + "\n- ".join(errors)
        )
