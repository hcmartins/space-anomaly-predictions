"""The reproducible entry point for real GOES + SHARP data ingestion.

    cd src/solar
    python -m ingest --config configs/solar_ingest.yaml

(Run with cwd = src/solar, verified against this environment's sys.path
behaviour -- `python -m solar.ingest` from `src/` does NOT resolve this
module's unprefixed intra-package imports, e.g. `from data_sources...`,
the same way `python -m martian.mars.run_all` from `src/` does not either;
both packages' real invocation is `python -m <entry> ...` run with cwd set
to the package directory itself.)

Orchestration only -- all real fetch/parse logic lives in
data_sources/goes_ingestion.py and data_sources/sharp_ingestion.py. This
module's job is: validate the run config, invoke each configured dataset's
ingestion function, write normalized CSVs + manifests to disk, and print a
run summary. It never invents values for a dataset that failed to fetch --
a failed dataset is reported as failed, and its output CSV is simply not
written.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

from data_sources.goes_ingestion import ingest_goes
from data_sources.schema import load_goes_config, load_sharp_config
from data_sources.sharp_ingestion import ingest_sharp_chunked
from validate_inputs import InputValidationError, validate_inputs


def load_run_config(config_path: Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_ingestion(config: dict) -> dict:
    validate_inputs(config)
    datasets = config["datasets"]
    results: dict = {}

    if "goes" in datasets:
        print("--- Ingesting GOES XRS ---")
        goes_config = load_goes_config()
        start = date.fromisoformat(config["goes"]["start_date"])
        end = date.fromisoformat(config["goes"]["end_date"])
        goes_df, goes_manifests = ingest_goes(start, end, config=goes_config)
        normalized_dir = Path(goes_config["output"]["normalized_output_dir"])
        normalized_dir.mkdir(parents=True, exist_ok=True)
        manifest_paths = [m.write(goes_config["output"]["manifest_dir"]) for m in goes_manifests]
        summary = {"n_rows": len(goes_df), "manifests": [str(p) for p in manifest_paths]}
        for m in goes_manifests:
            summary[m.dataset_name] = m.summary()
        if len(goes_df):
            out_path = normalized_dir / "goes_xrs_normalized.csv"
            goes_df.to_csv(out_path, index=False)
            summary["output_path"] = str(out_path)
        else:
            summary["output_path"] = None
            summary["status"] = "FAILED: no rows ingested -- see manifests for per-file errors"
        results["goes"] = summary
        print(f"    goes: {summary}")

    if "sharp" in datasets and config.get("sharp", {}).get("enabled", True):
        print("--- Ingesting SHARP ---")
        sharp_config = load_sharp_config()
        sharp_df, sharp_manifest = ingest_sharp_chunked(config=sharp_config)
        normalized_dir = Path(sharp_config["output"]["normalized_output_dir"])
        normalized_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = sharp_manifest.write(sharp_config["output"]["manifest_dir"])
        summary = {"n_rows": len(sharp_df), "manifest": str(manifest_path), **sharp_manifest.summary()}
        if len(sharp_df):
            out_path = normalized_dir / "sharp_normalized.csv"
            sharp_df.to_csv(out_path, index=False)
            summary["output_path"] = str(out_path)
        else:
            summary["output_path"] = None
            summary["status"] = "FAILED: no rows ingested -- see manifest for the error"
        results["sharp"] = summary
        print(f"    sharp: {summary}")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest real GOES XRS and SHARP data (Chapter 3 solar pipeline).")
    parser.add_argument("--config", required=True, help="Path to a solar ingest run config YAML (e.g. configs/solar_ingest.yaml).")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    try:
        results = run_ingestion(config)
    except InputValidationError as exc:
        print(f"INPUT VALIDATION FAILED:\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        return 1

    any_failed = any(r.get("output_path") is None for r in results.values())
    print(f"\nIngestion complete. Results: {results}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
