"""Test-collection bootstrap.

Adds the `src/` directory (the parent of the `solar` package) to sys.path
so tests can `import solar...` without requiring an editable pip install --
same pattern as martian/tests/conftest.py. Run with cwd set to `src/solar`
(`python -m pytest`) so the package's own unprefixed intra-package imports
(e.g. `from data_sources.goes_ingestion import ...` in ingest.py) also
resolve, exactly as martian/mars/run_all.py relies on.
"""
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _jsoc_email(monkeypatch):
    # configs/sharp.yaml ships with an empty jsoc_email; tests need a placeholder.
    monkeypatch.setenv("JSOC_EMAIL", "researcher@example.com")
