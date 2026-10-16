"""Abstract provider interfaces for the Martian Chapter 4 real-data pipeline.

Two interfaces, each with exactly one, real implementation:

  - EMIRSObservationProvider    physical observation records (+ provenance)
                                 -- data_sources/real_emirs_provider.py
  - MCDClimatologyProvider      climatological (mu, sigma) reference for
                                 dust_optical_depth -- data_sources/real_mcd_provider.py

There is no simulated fallback and no event/documented-event-catalogue
interface: ground-truth event injection was never wired into observation
generation, and no real dust-event catalogue has been sourced, so that
validation step (Chapter 4 Section 4.4.1) is out of scope for this pipeline
rather than stubbed or faked.

Every concrete `.load()`/`.query()` call is contractually required to return
a DataFrame that already carries full per-record provenance
(data_sources.provenance.REQUIRED_PROVENANCE_COLUMNS).
`BaseProvider._enforce_provenance_contract` is called by every concrete
implementation before it returns to a caller, and raises if any provenance column
is missing.
"""
from __future__ import annotations

import abc
from typing import Any

import pandas as pd

from .provenance import REQUIRED_PROVENANCE_COLUMNS


class ProvenanceContractError(RuntimeError):
    """Raised when a provider's output does not carry full per-record provenance."""


class BaseProvider(abc.ABC):
    """Shared provenance-enforcement mixin for all four provider interfaces."""

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return the provider-level provenance dict (data_source, product_id,
        processing_level, version, verification_status, plus any provider-
        specific extras). Called once per run and written into the immutable run
        manifest (see data_sources/manifest.py)."""

    def _enforce_provenance_contract(self, df: pd.DataFrame, *, caller: str) -> pd.DataFrame:
        missing = set(REQUIRED_PROVENANCE_COLUMNS) - set(df.columns)
        if missing:
            raise ProvenanceContractError(
                f"{caller} produced a DataFrame missing required provenance "
                f"columns: {sorted(missing)}. Every provider MUST attach these "
                f"before returning (see data_sources.provenance.attach_batch_provenance)."
            )
        return df


class EMIRSObservationProvider(BaseProvider):
    """Source of EMIRS-analogue observation records: physical variables (per
    configs/emirs_variables.yaml) plus full per-record provenance."""

    @abc.abstractmethod
    def load(self, *, mars_year_range: tuple[int, int], seed: int) -> pd.DataFrame:
        """Return one row per simulated retrieval."""


class MCDClimatologyProvider(BaseProvider):
    """Source of climatological (mu, sigma) expectations used to compute anomaly
    z-scores (Eq. 4.1 in the thesis)."""

    @abc.abstractmethod
    def query(
        self,
        *,
        latitude: Any,
        longitude: Any,
        solar_longitude_ls: Any,
        local_true_solar_time: Any,
        variable: str,
        pressure_level_pa: float | None = None,
    ) -> pd.DataFrame:
        """Return climatological mean/std for `variable` at the requested
        (lat, lon, Ls, LTST[, level]) points, plus provenance columns."""
