"""Unit tests for utils/agent_states_view.py — state contract + view projections."""

from __future__ import annotations

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    CommunicationAgentView,
    CriticCommunicationView,
    CriticResolutionView,
    ResolutionAgentView,
    UnifiedAgentStateModel,
    merge_back,
    project_into,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# AgentName enum
# ---------------------------------------------------------------------------


def test_agent_name_enum_values():
    assert AgentName.PREPROCESSOR.value == "preprocessor"
    assert AgentName.FINALIZE.value == "finalize"
    assert {e.value for e in AgentName} == {
        "preprocessor",
        "resolution",
        "critic_resolution",
        "communication",
        "critic_communication",
        "finalize",
    }


# ---------------------------------------------------------------------------
# UnifiedAgentStateModel — schema contract
# ---------------------------------------------------------------------------


def test_unified_state_model_defaults():
    m = UnifiedAgentStateModel()
    assert m.shipment_id == ""
    assert m.raw_rows == []
    assert m.max_loops == 2
    assert m.escalated is False
    assert m.noise_override is False
    assert m.resolution_revision_count == 0


def test_unified_state_model_roundtrip_json(minimal_unified_state):
    m = UnifiedAgentStateModel.from_typed_dict(minimal_unified_state)
    blob = m.model_dump_json()
    m2 = UnifiedAgentStateModel.model_validate_json(blob)
    assert m2.shipment_id == "SHP-T01"
    assert m2.max_loops == 2


def test_unified_state_model_rejects_out_of_range_loops():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UnifiedAgentStateModel(max_loops=99)
    with pytest.raises(ValidationError):
        UnifiedAgentStateModel(resolution_revision_count=10)


def test_unified_state_model_all_26_keys_present(minimal_unified_state):
    expected_keys = set(minimal_unified_state.keys())
    model_keys = set(UnifiedAgentStateModel.model_fields.keys())
    # every TypedDict key must exist in the Pydantic mirror
    assert expected_keys <= model_keys


# ---------------------------------------------------------------------------
# View models — frozen + PII boundaries
# ---------------------------------------------------------------------------


def test_resolution_view_has_no_pii_field():
    assert "customer_profile_full" not in ResolutionAgentView.model_fields
    assert "customer_profile" in ResolutionAgentView.model_fields


def test_communication_view_has_pii_field():
    assert "customer_profile_full" in CommunicationAgentView.model_fields


def test_critic_resolution_view_has_no_pii():
    assert "customer_profile_full" not in CriticResolutionView.model_fields


def test_critic_communication_view_has_no_pii():
    assert "customer_profile_full" not in CriticCommunicationView.model_fields


def test_views_are_frozen():
    view = ResolutionAgentView()
    with pytest.raises((TypeError, Exception)):
        view.critic_feedback = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# project_into / merge_back
# ---------------------------------------------------------------------------


def test_project_into_filters_fields(minimal_unified_state):
    state = {**minimal_unified_state, "customer_profile": {"tier": "VIP"}}
    projected = project_into(state, ResolutionAgentView)
    assert "customer_profile" in projected
    assert "customer_profile_full" not in projected  # not in resolution view
    assert "resolution_output" in projected


def test_project_into_does_not_mutate_original(minimal_unified_state):
    state_copy = dict(minimal_unified_state)
    _ = project_into(state_copy, ResolutionAgentView)
    assert state_copy == minimal_unified_state


def test_merge_back_applies_permitted_fields(minimal_unified_state):
    output = {"resolution_output": {"is_exception": "YES", "resolution": "REPLACE"}}
    new_state = merge_back(minimal_unified_state, output, ResolutionAgentView)
    assert new_state["resolution_output"]["resolution"] == "REPLACE"
    # Original state untouched
    assert minimal_unified_state["resolution_output"] == {}


def test_merge_back_drops_out_of_scope_keys(minimal_unified_state):
    output = {
        "resolution_output": {"r": "x"},
        "customer_profile_full": {"name": "DROPPED"},  # not owned by resolution view
    }
    new_state = merge_back(minimal_unified_state, output, ResolutionAgentView)
    assert new_state["resolution_output"] == {"r": "x"}
    # out-of-scope key silently dropped
    assert new_state["customer_profile_full"] == {}
