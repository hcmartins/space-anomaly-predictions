"""Tests for evaluation/presentation.py -- hand-built small fixtures, no
real training data (fast, deterministic), matching src/martian's own
figure-testing convention of checking the returned source_df, not pixel
content."""
from pathlib import Path

import pandas as pd
import pytest

from evaluation.presentation import (
    ablation_comparison_figure,
    failure_case_attention_figure,
    negative_transfer_figure,
    render_presentation_report,
    save_figure,
    stamp_figure,
    stamp_report,
    stamp_table_markdown,
)


def test_stamp_table_markdown_includes_data_source_and_all_rows():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    rendered = stamp_table_markdown(df, "TEST_SOURCE", title="My Table")
    assert "**My Table**" in rendered
    assert "*Data source: TEST_SOURCE*" in rendered
    assert "| a | b |" in rendered
    assert "| 1 | x |" in rendered
    assert "| 2 | y |" in rendered


def test_stamp_report_wraps_body_with_banner_top_and_bottom():
    rendered = stamp_report("body text", "TEST_SOURCE")
    assert rendered.startswith("[DATA SOURCE: TEST_SOURCE]")
    assert rendered.endswith("[DATA SOURCE: TEST_SOURCE]")
    assert "body text" in rendered


def test_negative_transfer_figure_returns_source_df_matching_input():
    df = pd.DataFrame(
        {
            "regime": ["2020", "2021", "2022"],
            "n": [10, 20, 30],
            "tss_fused": [0.1, 0.2, 0.3],
            "tss_unimodal": [0.15, 0.18, 0.25],
            "delta": [-0.05, 0.02, 0.05],
            "negative_transfer": [True, False, False],
        }
    )
    fig, source_df = negative_transfer_figure(df, domain="solar", data_source_label="TEST")
    assert list(source_df["regime"]) == ["2020", "2021", "2022"]
    assert len(fig.axes) == 1
    fig.clf()


def test_failure_case_attention_figure_handles_empty_dataframe():
    empty = pd.DataFrame(columns=["failure_type", "alpha_solar_at_issuance", "alpha_martian_at_issuance", "shared_with_unimodal"])
    fig, source_df = failure_case_attention_figure(empty, domain="martian", data_source_label="TEST")
    assert len(source_df) == 0
    fig.clf()


def test_failure_case_attention_figure_handles_real_shaped_data():
    df = pd.DataFrame(
        {
            "failure_type": ["missed_event", "false_alarm"],
            "y_true": [1, 0],
            "y_prob_fused": [0.2, 0.8],
            "y_prob_unimodal": [0.3, 0.6],
            "shared_with_unimodal": [True, False],
            "alpha_solar_at_issuance": [0.6, 0.4],
            "alpha_martian_at_issuance": [0.4, 0.6],
        }
    )
    fig, source_df = failure_case_attention_figure(df, domain="solar", data_source_label="TEST")
    assert len(source_df) == 2
    fig.clf()


def test_ablation_comparison_figure_marks_nan_as_gap_not_zero():
    results = {"A1": {"tss": 0.2}, "A2": {"tss": float("nan")}, "A7": {"tss": 0.35}}
    fig, source_df = ablation_comparison_figure(results, domain="martian", data_source_label="TEST")
    assert source_df.set_index("config_id").loc["A2", "tss"] != source_df.set_index("config_id").loc["A2", "tss"]  # NaN
    assert source_df.set_index("config_id").loc["A7", "tss"] == pytest.approx(0.35)
    fig.clf()


def test_save_figure_writes_a_file(tmp_path):
    fig, _ = negative_transfer_figure(
        pd.DataFrame({"regime": ["r1"], "n": [5], "tss_fused": [0.1], "tss_unimodal": [0.1], "delta": [0.0], "negative_transfer": [False]}),
        domain="solar", data_source_label="TEST",
    )
    out_path = save_figure(fig, tmp_path / "sub" / "fig.png", data_source_label="TEST")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_presentation_report_includes_all_sections_and_caveat():
    negative_transfer_tables = {
        "solar": pd.DataFrame({"regime": ["2020"], "n": [5], "tss_fused": [0.1], "tss_unimodal": [0.1], "delta": [0.0], "negative_transfer": [False]})
    }
    failure_case_tables = {"solar": pd.DataFrame(columns=["failure_type"])}
    report = render_presentation_report(
        run_id="run_test", negative_transfer_tables=negative_transfer_tables, failure_case_tables=failure_case_tables,
        figure_paths={}, data_source_label="REAL (solar) / REAL (martian)",
    )
    assert "Negative-transfer analysis" in report
    assert "Failure-case catalogue" in report
    assert "REAL (solar) / REAL (martian)" in report
    assert report.startswith("[DATA SOURCE:")


def test_render_presentation_report_with_ablation_results():
    report = render_presentation_report(
        run_id="run_ablation_test",
        negative_transfer_tables={}, failure_case_tables={},
        figure_paths={}, data_source_label="TEST",
        ablation_results={"A1": {"solar": {"tss": 0.1}}, "A7": {"solar": {"tss": 0.2}}},
    )
    assert "A1-A7 ablation comparison" in report
    assert "A1" in report and "A7" in report
