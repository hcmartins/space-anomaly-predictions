"""Test-collection bootstrap.

Adds the `src/` directory (the parent of the `martian` package) to sys.path so
tests can `import martian...` without requiring an editable pip install.
"""
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))  # lets tests `from split_fixtures import mechanics_splits`
