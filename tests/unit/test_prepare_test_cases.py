"""Unit tests for components/prepare_test_cases.py — loaders + persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.components.prepare_test_cases import (
    GroundTruthCase,
    TestCase,
    TestRunBatch,
    TestRunRecord,
    build_test_cases,
    load_batch,
    load_delivery_logs,
    load_ground_truth,
    save_batch,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


def test_load_ground_truth_consolidates_yes_wins(sample_gt_path: Path):
    gt = load_ground_truth(sample_gt_path)
    # SHP-T02 has NO then YES-row-last → YES row wins (exc_rows[-1])
    # Actually the CSV has YES row first, NO row second → YES wins (last in exc_rows list)
    assert "SHP-T02" in gt
    assert gt["SHP-T02"].is_exception == "YES"
    assert gt["SHP-T02"].expected_resolution == "REPLACE"


def test_load_ground_truth_no_case(sample_gt_path: Path):
    gt = load_ground_truth(sample_gt_path)
    assert gt["SHP-T01"].is_exception == "NO"


def test_load_ground_truth_returns_ground_truth_case_instances(sample_gt_path: Path):
    gt = load_ground_truth(sample_gt_path)
    for case in gt.values():
        assert isinstance(case, GroundTruthCase)


def test_load_ground_truth_count(sample_gt_path: Path):
    gt = load_ground_truth(sample_gt_path)
    assert len(gt) == 3  # SHP-T01, SHP-T02, SHP-T03


# ---------------------------------------------------------------------------
# load_delivery_logs
# ---------------------------------------------------------------------------


def test_load_delivery_logs_groups_by_shipment(sample_logs_path: Path):
    logs = load_delivery_logs(sample_logs_path)
    assert "SHP-T02" in logs
    # SHP-T02 has 2 rows (one duplicate)
    assert len(logs["SHP-T02"]) == 2


def test_load_delivery_logs_preserves_insertion_order(sample_logs_path: Path):
    logs = load_delivery_logs(sample_logs_path)
    assert list(logs.keys()) == ["SHP-T01", "SHP-T02", "SHP-T03"]


# ---------------------------------------------------------------------------
# build_test_cases
# ---------------------------------------------------------------------------


def test_build_test_cases_pairs_logs_with_gt(
    sample_logs_path: Path, sample_gt_path: Path
):
    cases = build_test_cases(sample_logs_path, sample_gt_path)
    assert len(cases) == 3
    ids = [c.shipment_id for c in cases]
    assert ids == ["SHP-T01", "SHP-T02", "SHP-T03"]
    assert all(isinstance(c, TestCase) for c in cases)


def test_build_test_cases_skips_orphans(tmp_path: Path, sample_logs_path: Path):
    # GT with only SHP-T01
    gt_only = tmp_path / "gt_partial.csv"
    gt_only.write_text(
        "shipment_id,is_exception,expected_resolution,expected_tone,should_escalate,ground_truth_reasoning\n"
        "SHP-T01,NO,N/A,N/A,N/A,noise\n",
        encoding="utf-8",
    )
    cases = build_test_cases(sample_logs_path, gt_only)
    assert len(cases) == 1
    assert cases[0].shipment_id == "SHP-T01"


# ---------------------------------------------------------------------------
# GroundTruthCase schema
# ---------------------------------------------------------------------------


def test_ground_truth_case_ignores_extra_fields():
    gt = GroundTruthCase(
        shipment_id="X",
        is_exception="YES",
        expected_resolution="RESCHEDULE",
        expected_tone="FORMAL",
        should_escalate="YES",
        ground_truth_reasoning="r",
        extra_column_future_proof="ok",  # type: ignore[call-arg]
    )
    assert gt.shipment_id == "X"


def test_ground_truth_case_rejects_bad_is_exception():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GroundTruthCase(shipment_id="X", is_exception="MAYBE")  # type: ignore[arg-type]


def test_ground_truth_case_applies_defaults():
    gt = GroundTruthCase(shipment_id="X", is_exception="NO")
    assert gt.expected_resolution == "N/A"
    assert gt.golden_message is None


# ---------------------------------------------------------------------------
# save_batch / load_batch round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_batch_roundtrip(tmp_path: Path, make_single_case_report):
    batch = TestRunBatch(
        runs=[
            TestRunRecord(
                shipment_id="SHP-T01",
                final_state={"shipment_id": "SHP-T01", "escalated": False},
                report=make_single_case_report(shipment_id="SHP-T01"),
                error=None,
                duration_sec=1.2,
                timestamp="2026-03-05T10:00:00+00:00",
            )
        ],
        metadata={"n": 1, "eval_llm": "FakeListLLM"},
    )
    out_dir = tmp_path / "runs"
    path = save_batch(batch, out_dir)
    assert path.exists()
    loaded = load_batch(path)
    assert len(loaded.runs) == 1
    assert loaded.runs[0].shipment_id == "SHP-T01"
    assert loaded.metadata["eval_llm"] == "FakeListLLM"
    assert loaded.runs[0].report.shipment_id == "SHP-T01"


def test_save_batch_creates_missing_directory(tmp_path: Path, make_single_case_report):
    batch = TestRunBatch(runs=[], metadata={})
    out_dir = tmp_path / "nested" / "deep" / "dir"
    assert not out_dir.exists()
    path = save_batch(batch, out_dir)
    assert path.exists()
    assert out_dir.exists()
