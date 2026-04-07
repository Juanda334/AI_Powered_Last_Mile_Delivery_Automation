"""Unit tests for communication_agent.py helpers (no LLM calls)."""
from __future__ import annotations

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.agents.communication_agent import (
    build_communication_context,
    sanitize_communication_inputs,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# sanitize_communication_inputs
# ---------------------------------------------------------------------------


def test_sanitize_communication_inputs_preserves_customer_name():
    view = {
        "consolidated_event": {"status_code": "DAMAGED", "tracking_number": "T123"},
        "customer_profile_full": {
            "name": "Alice",
            "tier": "VIP",
            "active_credit": 10,
            "preferred_channel": "EMAIL",
            "customer_id": "C-1",
            "random_internal": "xxx",  # dropped by _SAFE_PROFILE_KEYS filter
        },
        "resolution_output": {"resolution": "RESCHEDULE"},
        "locker_availability": [{"locker_id": "L1"}],
    }
    clean = sanitize_communication_inputs(view)
    assert clean["customer_profile_full"].get("name") == "Alice"
    assert clean["customer_profile_full"].get("tier") == "VIP"


def test_sanitize_communication_inputs_strips_noisy_event_keys():
    view = {
        "consolidated_event": {
            "status_code": "DAMAGED",
            "tracking_number": "T123",
            "raw_log": "...",
            "uuid": "abc",
            "internal_id": "42",
            "correlation_id": "xyz",
        },
    }
    clean = sanitize_communication_inputs(view)
    assert "status_code" in clean["consolidated_event"]
    assert "tracking_number" not in clean["consolidated_event"]
    assert "raw_log" not in clean["consolidated_event"]
    assert "uuid" not in clean["consolidated_event"]


def test_sanitize_communication_inputs_bad_types_coerced():
    view = {
        "consolidated_event": "bad",
        "customer_profile_full": None,
        "resolution_output": "bad",
        "locker_availability": "bad",
    }
    clean = sanitize_communication_inputs(view)
    assert clean["consolidated_event"] == {}
    assert clean["customer_profile_full"] == {}
    assert clean["resolution_output"] == {}
    assert clean["locker_availability"] == []


# ---------------------------------------------------------------------------
# build_communication_context
# ---------------------------------------------------------------------------


def test_build_communication_context_basic_shape():
    view = {
        "consolidated_event": {
            "status_code": "DAMAGED",
            "status_description": "crushed",
            "package_type": "FRAGILE",
        },
        "customer_profile_full": {
            "name": "Bob",
            "tier": "VIP",
            "preferred_channel": "SMS",
            "active_credit": 25,
        },
        "resolution_output": {"resolution": "REPLACE", "rationale": "r1"},
        "locker_availability": [],
    }
    ctx, locker_info = build_communication_context(view)
    assert ctx["customer_name"] == "Bob"
    assert ctx["customer_tier"] == "VIP"
    assert ctx["exception_type"] == "DAMAGED"
    assert ctx["resolution"] == "REPLACE"
    assert locker_info == ""  # not a reroute


def test_build_communication_context_defaults_missing_name():
    view = {
        "consolidated_event": {},
        "customer_profile_full": {},
        "resolution_output": {},
        "locker_availability": [],
    }
    ctx, locker_info = build_communication_context(view)
    assert ctx["customer_name"] == "Customer"
    assert ctx["customer_tier"] is None


def test_build_communication_context_locker_info_included_on_reroute():
    view = {
        "consolidated_event": {},
        "customer_profile_full": {"name": "Eve"},
        "resolution_output": {"resolution": "REROUTE_TO_LOCKER"},
        "locker_availability": [
            {"locker_id": "LOC-A", "eligible": False},
            {"locker_id": "LOC-B", "eligible": True, "address": "5 Main St"},
        ],
    }
    _, locker_info = build_communication_context(view)
    assert "LOC-B" in locker_info
    assert "LOCKER FOR REROUTE" in locker_info


def test_build_communication_context_reroute_no_eligible_lockers():
    view = {
        "consolidated_event": {},
        "customer_profile_full": {},
        "resolution_output": {"resolution": "REROUTE_TO_LOCKER"},
        "locker_availability": [{"eligible": False}],
    }
    _, locker_info = build_communication_context(view)
    assert locker_info == ""
