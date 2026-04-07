"""Unit tests for resolution_agent.py helpers (no LLM calls)."""

from __future__ import annotations

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.agents.resolution_agent import (
    format_playbook_context,
    sanitize_resolution_inputs,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# format_playbook_context
# ---------------------------------------------------------------------------


def test_format_playbook_context_empty():
    assert format_playbook_context([]) == "No playbook context available."


def test_format_playbook_context_formats_with_page_prefix():
    chunks = [
        {"page": 5, "content": "handle damaged perishables as follows"},
        {"page": 12, "content": "escalation criteria"},
    ]
    out = format_playbook_context(chunks)
    assert "[Page 5]" in out
    assert "[Page 12]" in out
    assert "---" in out
    assert "handle damaged perishables" in out


def test_format_playbook_context_missing_page_uses_question_mark():
    out = format_playbook_context([{"content": "x"}])
    assert "[Page ?]" in out


# ---------------------------------------------------------------------------
# sanitize_resolution_inputs
# ---------------------------------------------------------------------------


def test_sanitize_resolution_inputs_strips_pii():
    view = {
        "consolidated_event": {"status_code": "DAMAGED"},
        "customer_profile": {"tier": "VIP", "name": "Alice", "full_name": "A B"},
        "locker_availability": [{"locker_id": "L1"}],
        "playbook_context": [{"content": "abc", "page": 1}],
        "escalation_signals": {"has_triggers": True},
        "critic_feedback": "be more specific",
    }
    clean = sanitize_resolution_inputs(view)
    assert "name" not in clean["customer_profile"]
    assert "full_name" not in clean["customer_profile"]
    assert clean["customer_profile"]["tier"] == "VIP"


def test_sanitize_resolution_inputs_coerces_bad_types():
    bad = {
        "consolidated_event": "not a dict",
        "customer_profile": None,
        "locker_availability": "not a list",
        "playbook_context": {"bad": True},
        "escalation_signals": [],
        "critic_feedback": 42,
    }
    clean = sanitize_resolution_inputs(bad)
    assert clean["consolidated_event"] == {}
    assert clean["customer_profile"] == {}
    assert clean["locker_availability"] == []
    assert clean["playbook_context"] == []
    assert clean["escalation_signals"] == {}
    assert clean["critic_feedback"] == ""


def test_sanitize_resolution_inputs_filters_malformed_list_items():
    view = {
        "locker_availability": [{"id": "ok"}, "bad", None, 123],
        "playbook_context": [{"content": "ok"}, {"no_content": True}, "string"],
    }
    clean = sanitize_resolution_inputs(view)
    assert clean["locker_availability"] == [{"id": "ok"}]
    assert clean["playbook_context"] == [{"content": "ok"}]


def test_sanitize_resolution_inputs_does_not_mutate_original():
    original = {"customer_profile": {"tier": "VIP", "name": "Alice"}}
    _ = sanitize_resolution_inputs(original)
    assert "name" in original["customer_profile"]  # original untouched
