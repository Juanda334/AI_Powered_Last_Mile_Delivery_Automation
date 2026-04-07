"""End-to-end integration tests for the multi-agent workflow.

These tests exercise ``run_workflow`` and ``run_test_case`` using
**stub apps** (plain objects with ``.invoke``) rather than a live
compiled LangGraph app. This keeps the tests hermetic while still
verifying the error boundaries and batch isolation that production
relies on.
"""

from __future__ import annotations

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.components.multi_agent_workflow import (
    build_initial_state,
    run_workflow,
    sanitize_initial_state,
)
from AI_Powered_Last_Mile_Delivery_Automation.components.prepare_test_cases import (
    GroundTruthCase,
    TestCase,
    run_test_case,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# sanitize_initial_state + build_initial_state
# ---------------------------------------------------------------------------


def test_sanitize_initial_state_happy_path():
    sid, rows, loops = sanitize_initial_state("  SHP-1  ", [{"a": 1}], max_loops=99)
    assert sid == "SHP-1"
    assert rows == [{"a": 1}]
    assert loops == 5  # clamped from 99


def test_sanitize_initial_state_drops_non_dict_rows():
    _, rows, _ = sanitize_initial_state("S", [{"a": 1}, "bad", None, 42], max_loops=2)
    assert rows == [{"a": 1}]


def test_sanitize_initial_state_rejects_empty_shipment_id():
    with pytest.raises(ValueError):
        sanitize_initial_state("", [{"a": 1}])
    with pytest.raises(ValueError):
        sanitize_initial_state(None, [{"a": 1}])


def test_sanitize_initial_state_rejects_empty_rows():
    with pytest.raises(ValueError):
        sanitize_initial_state("S", [])
    with pytest.raises(ValueError):
        sanitize_initial_state("S", "not a list")


def test_build_initial_state_has_all_26_keys(minimal_unified_state):
    state = build_initial_state("SHP-TEST", [{"shipment_id": "SHP-TEST"}])
    assert set(state.keys()) == set(minimal_unified_state.keys())
    assert state["shipment_id"] == "SHP-TEST"
    assert state["max_loops"] == 2
    assert state["escalated"] is False


def test_build_initial_state_clamps_max_loops():
    state = build_initial_state("S", [{"a": 1}], max_loops=100)
    assert state["max_loops"] == 5
    state2 = build_initial_state("S", [{"a": 1}], max_loops=0)
    assert state2["max_loops"] == 1


# ---------------------------------------------------------------------------
# run_workflow — happy path using a stub app
# ---------------------------------------------------------------------------


class _StubApp:
    """Minimal app with an ``.invoke`` that echoes a canned state dict."""

    def __init__(self, returned_state: dict):
        self._state = returned_state
        self.calls: list[dict] = []

    def invoke(self, state, config=None):  # noqa: ARG002
        self.calls.append(dict(state))
        return {**state, **self._state}


def test_run_workflow_happy_path_stub_app():
    canned = {
        "final_actions": [{"action": "RESCHEDULE"}],
        "escalated": False,
        "trajectory_log": ["preprocessor: ok", "finalize: ok"],
    }
    app = _StubApp(canned)
    final = run_workflow(app, "SHP-001", [{"shipment_id": "SHP-001"}])
    assert final["shipment_id"] == "SHP-001"
    assert final["final_actions"] == [{"action": "RESCHEDULE"}]
    assert len(app.calls) == 1


def test_run_workflow_error_boundary_returns_fatal_state():
    class _RaisingApp:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

    final = run_workflow(_RaisingApp(), "SHP-BAD", [{"shipment_id": "SHP-BAD"}])
    # Contract: returns synthesized state with escalated=True + FATAL marker
    assert final.get("escalated") is True
    assert final.get("latency_sec") is not None
    joined_traj = " ".join(str(x) for x in final.get("trajectory_log", []))
    joined_fa = " ".join(str(x) for x in final.get("final_actions", []))
    assert "FATAL" in (joined_traj + joined_fa).upper()


# ---------------------------------------------------------------------------
# run_test_case — batch isolation
# ---------------------------------------------------------------------------


def test_run_test_case_failure_isolation(fake_eval_llm):
    """One broken shipment must not crash the batch — returns record with error."""

    class _RaisingApp:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("boom")

    # Even though `app.invoke` raises, `run_workflow` catches and returns a
    # FATAL fallback state. `run_test_case` then evaluates that state and
    # the report reflects a failure (not crash).
    tc = TestCase(
        shipment_id="SHP-BOOM",
        raw_rows=[{"shipment_id": "SHP-BOOM", "status_code": "DAMAGED"}],
        ground_truth=GroundTruthCase(
            shipment_id="SHP-BOOM",
            is_exception="YES",
            expected_resolution="REPLACE",
            expected_tone="FORMAL",
            should_escalate="YES",
        ),
    )
    record = run_test_case(_RaisingApp(), tc, eval_llm=fake_eval_llm)
    assert record.shipment_id == "SHP-BOOM"
    # The fallback path should produce a completed record (not raise)
    assert record.report is not None
    assert record.duration_sec >= 0.0
    # Task should not be marked complete — the workflow failed
    assert record.report.task_completion.task_complete is False


def test_run_test_case_happy_path(fake_eval_llm):
    """Stub app returns a plausible final state; report is built normally."""
    canned_state = {
        "shipment_id": "SHP-OK",
        "resolution_output": {"is_exception": "YES", "resolution": "REPLACE"},
        "communication_output": {"tone_label": "FORMAL", "communication_message": "hi"},
        "escalated": True,
        "noise_override": False,
        "tool_calls_log": [
            "lookup_customer_profile",
            "check_locker_availability",
            "search_playbook",
            "check_escalation_rules",
            "resolution_agent",
            "communication_agent",
        ],
        "trajectory_log": ["preprocessor: ok"],
        "final_actions": [{"action": "REPLACE"}],
        "latency_sec": 3.14,
    }
    app = _StubApp(canned_state)
    tc = TestCase(
        shipment_id="SHP-OK",
        raw_rows=[{"shipment_id": "SHP-OK", "status_code": "DAMAGED"}],
        ground_truth=GroundTruthCase(
            shipment_id="SHP-OK",
            is_exception="YES",
            expected_resolution="REPLACE",
            expected_tone="FORMAL",
            should_escalate="YES",
        ),
    )
    record = run_test_case(app, tc, eval_llm=fake_eval_llm)
    assert record.error is None
    assert record.report.task_completion.task_complete is True
    assert record.report.escalation_correct is True
    assert record.report.tool_call_correct is True
