"""Unit tests for components/evaluation_metrics.py."""

from __future__ import annotations

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.components.evaluation_metrics import (
    _strip_code_fence,
    aggregate_results,
    compute_escalation_accuracy,
    compute_failure_categories,
    compute_latency_per_agent,
    compute_task_completion,
    compute_token_efficiency,
    compute_tool_call_accuracy,
    compute_trajectory_drift,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_task_completion
# ---------------------------------------------------------------------------


def test_task_completion_exception_all_correct():
    gt = {
        "is_exception": "YES",
        "expected_resolution": "REPLACE",
        "expected_tone": "FORMAL",
    }
    pred = {
        "resolution_output": {"is_exception": "YES", "resolution": "REPLACE"},
        "communication_output": {"tone_label": "FORMAL"},
    }
    r = compute_task_completion(gt, pred)
    assert r.exception_correct is True
    assert r.resolution_correct is True
    assert r.tone_correct is True
    assert r.task_complete is True


def test_task_completion_noise_case_skips_tone():
    gt = {"is_exception": "NO", "expected_resolution": "N/A"}
    pred = {"resolution_output": {"is_exception": "NO", "resolution": "N/A"}}
    r = compute_task_completion(gt, pred)
    assert r.tone_correct is None
    assert r.task_complete is True


def test_task_completion_wrong_resolution_fails_task():
    gt = {
        "is_exception": "YES",
        "expected_resolution": "REPLACE",
        "expected_tone": "FORMAL",
    }
    pred = {
        "resolution_output": {"is_exception": "YES", "resolution": "RESCHEDULE"},
        "communication_output": {"tone_label": "FORMAL"},
    }
    r = compute_task_completion(gt, pred)
    assert r.resolution_correct is False
    assert r.task_complete is False


def test_task_completion_case_insensitive():
    gt = {
        "is_exception": "yes",
        "expected_resolution": "replace",
        "expected_tone": "formal",
    }
    pred = {
        "resolution_output": {"is_exception": "YES", "resolution": "REPLACE"},
        "communication_output": {"tone_label": "FORMAL"},
    }
    assert compute_task_completion(gt, pred).task_complete is True


# ---------------------------------------------------------------------------
# compute_escalation_accuracy
# ---------------------------------------------------------------------------


def test_escalation_accuracy_match_yes():
    assert (
        compute_escalation_accuracy({"should_escalate": "YES"}, {"escalated": True})
        is True
    )


def test_escalation_accuracy_match_no():
    assert (
        compute_escalation_accuracy({"should_escalate": "NO"}, {"escalated": False})
        is True
    )


def test_escalation_accuracy_mismatch():
    assert (
        compute_escalation_accuracy({"should_escalate": "YES"}, {"escalated": False})
        is False
    )


def test_escalation_accuracy_na_returns_none():
    assert (
        compute_escalation_accuracy({"should_escalate": "N/A"}, {"escalated": False})
        is None
    )


# ---------------------------------------------------------------------------
# compute_tool_call_accuracy
# ---------------------------------------------------------------------------


def test_tool_call_accuracy_noise_requires_no_calls():
    pred = {"noise_override": True, "tool_calls_log": []}
    gt = {"is_exception": "NO"}
    assert compute_tool_call_accuracy(gt, pred) is True


def test_tool_call_accuracy_noise_with_calls_fails():
    pred = {"noise_override": True, "tool_calls_log": ["something"]}
    gt = {"is_exception": "NO"}
    assert compute_tool_call_accuracy(gt, pred) is False


def test_tool_call_accuracy_exception_needs_all_tools_plus_comm():
    pred = {
        "noise_override": False,
        "tool_calls_log": [
            "lookup_customer_profile",
            "check_locker_availability",
            "search_playbook",
            "check_escalation_rules",
            "resolution_agent",
            "communication_agent",
        ],
    }
    gt = {"is_exception": "YES"}
    assert compute_tool_call_accuracy(gt, pred) is True


def test_tool_call_accuracy_exception_missing_comm_fails():
    pred = {
        "noise_override": False,
        "tool_calls_log": [
            "lookup_customer_profile",
            "check_locker_availability",
            "search_playbook",
            "check_escalation_rules",
            "resolution_agent",
        ],
    }
    gt = {"is_exception": "YES"}
    assert compute_tool_call_accuracy(gt, pred) is False


# ---------------------------------------------------------------------------
# compute_token_efficiency
# ---------------------------------------------------------------------------


def test_token_efficiency_empty_log_returns_zeros():
    r = compute_token_efficiency({})
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0
    assert r.estimated_cost_usd == 0.0
    assert r.tokens_per_resolution is None


def test_token_efficiency_sums_log():
    pred = {
        "token_usage_log": [
            {"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 50},
            {"model": "gpt-4o", "prompt_tokens": 200, "completion_tokens": 80},
        ],
        "_task_complete_hint": True,
    }
    r = compute_token_efficiency(pred)
    assert r.prompt_tokens == 300
    assert r.completion_tokens == 130
    assert r.total_tokens == 430
    assert r.estimated_cost_usd > 0
    assert r.tokens_per_resolution == 430.0


# ---------------------------------------------------------------------------
# compute_trajectory_drift
# ---------------------------------------------------------------------------


def test_trajectory_drift_clean_trace():
    pred = {
        "trajectory_log": ["a", "b", "c"],
        "tool_calls_log": ["x", "y"],
        "resolution_revision_count": 0,
    }
    r = compute_trajectory_drift(pred)
    assert r.drift_flag is False
    assert r.trajectory_len == 3


def test_trajectory_drift_bloated_trace_flags_drift():
    pred = {
        "trajectory_log": ["x"] * 200,
        "tool_calls_log": ["y"] * 80,
        "resolution_revision_count": 3,
    }
    r = compute_trajectory_drift(pred)
    assert r.drift_flag is True


# ---------------------------------------------------------------------------
# _strip_code_fence
# ---------------------------------------------------------------------------


def test_strip_code_fence_json_block():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_plain_block():
    assert _strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_no_fence():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------------------------------
# compute_failure_categories
# ---------------------------------------------------------------------------


def test_compute_failure_categories_all_passing(make_single_case_report):
    reports = [make_single_case_report() for _ in range(3)]
    counts = compute_failure_categories(reports)
    assert all(v == 0 for v in counts.values())


def test_compute_failure_categories_multiple_buckets(make_single_case_report):
    reports = [
        make_single_case_report(exception_correct=False, task_complete=False),
        make_single_case_report(coherence_score=1),  # low_coherence
        make_single_case_report(drift_flag=True),
        make_single_case_report(tool_call_correct=False),
        make_single_case_report(escalation_correct=False),
    ]
    counts = compute_failure_categories(reports)
    assert counts["exception_misclass"] == 1
    assert counts["low_coherence"] == 1
    assert counts["drift"] == 1
    assert counts["wrong_tools"] == 1
    assert counts["wrong_escalation"] == 1


def test_compute_failure_categories_coherence_zero_excluded(make_single_case_report):
    # score 0 means "error" — should NOT count as low_coherence (must be 1-2)
    reports = [make_single_case_report(coherence_score=0)]
    counts = compute_failure_categories(reports)
    assert counts["low_coherence"] == 0


# ---------------------------------------------------------------------------
# compute_latency_per_agent
# ---------------------------------------------------------------------------


def test_latency_per_agent_extracts_timings():
    states = [
        {
            "trajectory_log": [
                "preprocessor: finished  latency=1.5s",
                "resolution_agent: done  elapsed=2.0s",
                "finalize: latency=0.3s",
            ]
        },
        {"trajectory_log": ["preprocessor: restart  latency=2.5s"]},
    ]
    per_agent = compute_latency_per_agent(states)
    assert per_agent["preprocessor"] == pytest.approx(2.0)
    assert per_agent["resolution_agent"] == pytest.approx(2.0)
    assert per_agent["finalize"] == pytest.approx(0.3)


def test_latency_per_agent_empty_when_no_timings():
    assert compute_latency_per_agent([{"trajectory_log": ["no timing line"]}]) == {}
    assert compute_latency_per_agent([]) == {}


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------


def test_aggregate_results_empty_batch(make_single_case_report):
    br = aggregate_results([])
    assert br.n == 0
    assert br.task_completion_rate == 0.0


def test_aggregate_results_all_passing(make_single_case_report):
    reports = [make_single_case_report() for _ in range(5)]
    br = aggregate_results(reports)
    assert br.n == 5
    assert br.task_completion_rate == 1.0
    assert br.failure_count == 0
    assert br.failure_rate == 0.0
    assert br.avg_latency_sec == pytest.approx(2.5)
    assert br.median_latency_sec == pytest.approx(2.5)
    # <20 samples → p95 is max
    assert br.p95_latency_sec == pytest.approx(2.5)


def test_aggregate_results_mixed(make_single_case_report):
    reports = [
        make_single_case_report(task_complete=True, latency_sec=1.0),
        make_single_case_report(
            task_complete=False, exception_correct=False, latency_sec=3.0
        ),
        make_single_case_report(task_complete=True, latency_sec=2.0),
    ]
    br = aggregate_results(reports)
    assert br.n == 3
    assert br.task_completion_rate == pytest.approx(2 / 3)
    assert br.failure_count == 1
    assert br.failure_rate == pytest.approx(1 / 3)
    assert br.median_latency_sec == pytest.approx(2.0)
    assert br.failure_breakdown["exception_misclass"] == 1


def test_aggregate_results_total_cost(make_single_case_report):
    reports = [make_single_case_report(cost=0.001), make_single_case_report(cost=0.002)]
    br = aggregate_results(reports)
    assert br.total_cost_usd == pytest.approx(0.003)


def test_aggregate_results_p95_with_20_samples(make_single_case_report):
    # 20 samples → p95 uses quantiles[18]
    reports = [make_single_case_report(latency_sec=float(i)) for i in range(1, 21)]
    br = aggregate_results(reports)
    assert br.n == 20
    assert br.p95_latency_sec > br.median_latency_sec
