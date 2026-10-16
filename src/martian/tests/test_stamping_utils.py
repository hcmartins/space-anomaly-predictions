"""Tests for the visible data-source stamping helpers (task requirement 6)."""
import matplotlib

matplotlib.use("Agg")  # headless backend for test execution
import matplotlib.pyplot as plt
import pandas as pd

from martian.data_sources.provenance import (
    stamp_figure,
    stamp_processed_dataset,
    stamp_report,
    stamp_table_markdown,
)


def test_stamp_processed_dataset_adds_data_source_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = stamp_processed_dataset(df, "SIMULATED")
    assert (out["data_source"] == "SIMULATED").all()
    assert "x" in out.columns
    assert "data_source" not in df.columns  # original frame must be unmodified


def test_stamp_figure_adds_visible_text():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    stamp_figure(fig, "SIMULATED — seed=42")
    texts = [t.get_text() for t in fig.texts]
    assert any("SIMULATED" in t and "seed=42" in t for t in texts)
    plt.close(fig)


def test_stamp_table_markdown_contains_label_and_content():
    df = pd.DataFrame({"a": [1], "b": [2]})
    md = stamp_table_markdown(df, "SIMULATED", title="Test table")
    assert "Data source: SIMULATED" in md
    assert "Test table" in md
    assert "| a | b |" in md


def test_stamp_report_wraps_with_visible_banner():
    text = stamp_report("body content", "SIMULATED")
    assert text.count("[DATA SOURCE: SIMULATED]") == 2
    assert "body content" in text
