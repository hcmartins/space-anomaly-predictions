"""Real EMIRS observation provider.

Grounded in a real prior MSc dissertation on this exact instrument's data
(uploaded by the researcher 2026-07-29), whose Table 3.1 cites the real EMM
EMIRS Data Product Guide -- see configs/real_emirs_schema.yaml's header for
the full citation chain.

Two real, honest gaps, both documented rather than silently papered over:

1. **Schema reconciliation**: real EMIRS field names (`taudust`,
   `kinetic_temp`, `utc`, `qls`, `qlt`, ...) do not match what the
   QC/aggregation/labeling/features code downstream expects
   (`dust_optical_depth`, `surface_temperature_k`, `utc_timestamp`,
   `solar_longitude_ls`, `local_true_solar_time`, ...). `load()` derives
   `mars_year`/`sol` from the real fields first, then applies
   `configs/real_emirs_schema.yaml`'s `pipeline_column_mapping` (via
   `real_emirs_fits.apply_pipeline_column_mapping`) to rename everything
   onto the pipeline's canonical names -- see that config's own comment for
   the full mapping table and the real 19-level vertical grid's
   reconciliation (it stays a single collapsed mean, not exploded into
   per-level columns, because no real per-level uncertainty field exists to
   pair with them). `solar_longitude_ls`/`local_true_solar_time` (needed by
   `RealMCDClimatologyProvider.query()`) are NOT computed -- they are real,
   native per-footprint l3atm columns (`qls`, `qlt`), confirmed empirically
   against real downloaded data: `qls`/`qlt` values match l3emiss's
   independently-named `ltime` column row-for-row, and `qls`'s ~6.4-degree
   value for a 2021-02-20 file is physically correct (just past vernal
   equinox, Ls=0, matching EMIRS's documented MY36 science start). Note the
   EMM EMIRS Data Product Guide's Appendix A documents these as `l_sub_s`/
   `ltime` on l3atm -- the ACTUAL downloaded files (all versions checked)
   use `qls`/`qlt` instead; `pipeline_column_mapping` maps the verified real
   names, not the documented-but-absent ones.

2. **No native quality_flag**: unlike the simulated schema (which assumes
   one exists), the real product has no documented QC/quality-flag column.
   `quality_flag` is set to 0 (pass) for every real row here; `chi2temp`/
   `chi2a` (real goodness-of-fit fields) are available for a genuine QC
   rule at the `preprocessing/quality_control.py` stage later.

Real Mars Year / sol assignment uses a documented linear approximation
(`data_sources/real_emirs_fits.py::utc_to_mars_year_sol`), not a full
areocentric-Ls calculation -- RESEARCHER VERIFICATION REQUIRED before
relying on it for exact seasonal boundaries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import EMIRSObservationProvider
from .provenance import ProviderIdentity, attach_batch_provenance
from .real_emirs_fits import (
    apply_pipeline_column_mapping,
    load_real_emirs_directory,
    load_real_emirs_ingestion_config,
    load_real_emirs_schema,
    utc_to_mars_year_sol,
)
from .schema import CONFIG_DIR


class RealEMIRSObservationProvider(EMIRSObservationProvider):
    def __init__(self, ingestion_config_path: Path | None = None, schema_config_path: Path | None = None):
        self._ingestion_config = load_real_emirs_ingestion_config(ingestion_config_path)
        self._schema = load_real_emirs_schema(schema_config_path or CONFIG_DIR / "real_emirs_schema.yaml")

    def describe(self) -> dict:
        pf = self._schema["product_family"]
        return {
            "data_source": "REAL",
            "product_id": f"REAL-{pf['name'].upper().replace(' ', '-')}",
            "processing_level": pf["processing_level"],
            "version": pf["version"],
            "verification_status": "RESEARCHER VERIFICATION REQUIRED (version not stated in source document)",
            "citation": (
                "Real EMM EMIRS Data Product Guide, via a real prior MSc dissertation's "
                "Table 3.1 -- see configs/real_emirs_schema.yaml"
            ),
        }

    def _identity(self) -> ProviderIdentity:
        d = self.describe()
        return ProviderIdentity(
            data_source=d["data_source"],
            product_id=d["product_id"],
            processing_level=d["processing_level"],
            version=d["version"],
            verification_status=d["verification_status"],
        )

    def load(self, *, mars_year_range: tuple[int, int], seed: int) -> pd.DataFrame:
        """`seed` is accepted for interface compatibility with
        `EMIRSObservationProvider` but unused: real data has no generation
        randomness to seed."""
        del seed
        directory = Path(self._ingestion_config["local_fits_directory"])
        extensions = tuple(self._ingestion_config["fits_extensions"])

        l3atm = load_real_emirs_directory(directory, product="l3atm", schema=self._schema, extensions=extensions)
        l3emiss = load_real_emirs_directory(directory, product="l3emiss", schema=self._schema, extensions=extensions)

        merge_keys = self._schema["product_family"]["merge_keys"]
        df = pd.merge(l3atm, l3emiss, on=merge_keys, how="inner", suffixes=("_l3atm", "_l3emiss"))

        mars_year, sol = utc_to_mars_year_sol(df["utc"])
        df["mars_year"] = mars_year
        df["sol"] = sol
        my_start, my_end = mars_year_range
        df = df.loc[(df["mars_year"] >= my_start) & (df["mars_year"] <= my_end)].reset_index(drop=True)

        df["quality_flag"] = 0   # no native QC flag in the real product -- see module docstring

        # taudust->dust_optical_depth, kinetic_temp->surface_temperature_k, utc->utc_timestamp,
        # qls->solar_longitude_ls, qlt->local_true_solar_time, etc. -- see configs/real_emirs_schema.yaml's pipeline_column_mapping
        df = apply_pipeline_column_mapping(df, self._schema)

        identity = self._identity()
        n = len(df)
        original_filenames = df["source_fits_file_l3atm"] if "source_fits_file_l3atm" in df.columns else pd.Series([""] * n)
        retrieval_timestamps = pd.Series([pd.Timestamp.now(tz="UTC")] * n)
        physical_columns = [c for c in df.columns if c not in ("quality_flag",) and not c.startswith("source_fits_file")]
        physical_columns.append("quality_flag")

        df = attach_batch_provenance(
            df,
            identity,
            original_filenames=original_filenames,
            quality_flags=df["quality_flag"],
            retrieval_timestamps=retrieval_timestamps,
            checksum_source_columns=physical_columns,
        )
        return self._enforce_provenance_contract(df, caller="RealEMIRSObservationProvider.load")
