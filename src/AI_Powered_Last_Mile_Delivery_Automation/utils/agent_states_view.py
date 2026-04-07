"""Multi-agent state management for the Last-Mile Delivery pipeline.

This module formalises the state schema and agent-view system first prototyped
in ``research_trials.ipynb``.  Every piece of data that flows through the
LangGraph pipeline lives in a single **UnifiedAgentState** (TypedDict).  To
enforce data isolation each agent receives only the subset of fields it needs
via a typed *view* (frozen Pydantic model).

Public API
----------
* ``UnifiedAgentState``         — LangGraph-compatible TypedDict state schema.
* ``UnifiedAgentStateModel``    — Pydantic mirror for validation / serialisation.
* ``AgentName``                 — Enum of valid pipeline stages.
* ``RouterView``                — View for the Router / preprocessor / finalizer.
* ``ResolutionAgentView``       — View for the Resolution Agent (no PII).
* ``CommunicationAgentView``    — View for the Communication Agent (has PII).
* ``CriticResolutionView``      — View for the Critic validating resolutions.
* ``CriticCommunicationView``   — View for the Critic validating communications.
* ``project_into()``            — Project global state onto an agent view.
* ``merge_back()``              — Merge agent output back into global state.

Architecture & optimisation notes
----------------------------------
1. **Context-window management** — ``trajectory_log`` and ``tool_calls_log``
   grow with each agent hop.  Implement a ``prune_history(state, max_entries)``
   utility that truncates older entries while preserving the most recent *N*
   before the next LLM call.

2. **Persistence** — ``UnifiedAgentStateModel`` supports
   ``.model_dump_json()`` / ``.model_validate_json()`` out of the box.  For
   production, persist to Redis (fast resume) or SQLite (audit trail) between
   pipeline stages.

3. **Observability** — Every ``merge_back`` call logs the view name and changed
   keys.  The project logger already supports ``LOG_FORMAT=json`` for
   structured downstream monitoring.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("utils.agent_states_view")


__all__ = [
    "AgentName",
    "UnifiedAgentState",
    "UnifiedAgentStateModel",
    "RouterView",
    "ResolutionAgentView",
    "CommunicationAgentView",
    "CriticResolutionView",
    "CriticCommunicationView",
    "project_into",
    "merge_back",
]


# ═══════════════════════════════════════════════════════════════════════════
# Agent name enum
# ═══════════════════════════════════════════════════════════════════════════


class AgentName(str, Enum):
    """Valid pipeline stages for the ``next_agent`` routing field.

    Using a closed enum prevents the LLM from hallucinating an agent name
    that does not exist in the graph.
    """

    PREPROCESSOR = "preprocessor"
    RESOLUTION = "resolution"
    CRITIC_RESOLUTION = "critic_resolution"
    COMMUNICATION = "communication"
    CRITIC_COMMUNICATION = "critic_communication"
    FINALIZE = "finalize"


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph-compatible TypedDict state
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedAgentState(TypedDict, total=False):
    """State object passed through the LangGraph pipeline.

    Kept as a ``TypedDict`` because LangGraph's ``StateGraph`` requires this
    type for its state schema.  See ``UnifiedAgentStateModel`` for a Pydantic
    mirror with full validation.
    """

    # ── Input ────────────────────────────────────────────────────────────
    raw_rows: list[dict]                        # Raw delivery log rows for this shipment
    shipment_id: str

    # ── Preprocessor output ──────────────────────────────────────────────
    consolidated_event: dict                    # Deduplicated, consolidated event
    customer_profile: dict                      # Redacted profile for resolution (no PII)
    customer_profile_full: dict                 # Full profile with PII for communication
    locker_availability: list[dict]             # Lockers in same zip
    playbook_context: list[dict]                # Retrieved playbook chunks with page metadata
    escalation_signals: dict                    # Deterministic rule output
    noise_override: bool                        # Preprocessor guardrail flag for routine noise
    guardrail_triggered: bool                   # True if input injection detected

    # ── Resolution Agent output ──────────────────────────────────────────
    resolution_output: dict                     # {is_exception, resolution, rationale}

    # ── Critic — resolution validation ───────────────────────────────────
    critic_resolution_output: dict              # {decision, rationale}
    resolution_revision_count: int              # Track retries, max 2
    critic_feedback: str                        # Feedback for revision loop

    # ── Communication Agent output ───────────────────────────────────────
    communication_output: dict                  # {tone_label, communication_message}

    # ── Critic — communication validation ────────────────────────────────
    critic_communication_output: dict           # {decision, rationale}

    # ── Routing ──────────────────────────────────────────────────────────
    next_agent: str                             # Next node to route to (see AgentName)
    max_loops: int                              # Max revision loops

    # ── Final ────────────────────────────────────────────────────────────
    escalated: bool                             # Whether any critic node returned ESCALATE
    tool_calls_log: list[str]                   # Log of all tool invocations
    trajectory_log: list[str]                   # Audit trail of agent decisions
    start_time: Optional[float]                 # Pipeline start timestamp
    latency_sec: Optional[float]                # Total pipeline latency
    final_actions: list[dict]                   # Final packaged output


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic mirror — validation & serialisation
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedAgentStateModel(BaseModel):
    """Pydantic mirror of :class:`UnifiedAgentState`.

    Use this model when you need **validation** (e.g. at pipeline entry),
    **serialisation** (``.model_dump_json()``), or **schema export**
    (``.model_json_schema()``).  Convert to/from the LangGraph TypedDict via
    :meth:`from_typed_dict` and :meth:`to_typed_dict`.
    """

    model_config = ConfigDict(use_enum_values=True)

    # ── Input ────────────────────────────────────────────────────────────
    raw_rows: list[dict] = Field(
        default_factory=list,
        description="Raw delivery log rows for this shipment, as parsed from CSV.",
    )
    shipment_id: str = Field(
        default="",
        description="Unique shipment identifier (e.g. 'SHIP-001').",
    )

    # ── Preprocessor output ──────────────────────────────────────────────
    consolidated_event: dict = Field(
        default_factory=dict,
        description="Deduplicated, consolidated event produced by the preprocessor.",
    )
    customer_profile: dict = Field(
        default_factory=dict,
        description="Redacted customer profile (no PII). Used by Resolution and Critic agents.",
    )
    customer_profile_full: dict = Field(
        default_factory=dict,
        description="Full customer profile including PII (name). Only the Communication Agent should access this.",
    )
    locker_availability: list[dict] = Field(
        default_factory=list,
        description="Available lockers in the delivery zip code with eligibility flags.",
    )
    playbook_context: list[dict] = Field(
        default_factory=list,
        description="Retrieved playbook chunks with page metadata from vector search.",
    )
    escalation_signals: dict = Field(
        default_factory=dict,
        description="Deterministic escalation rule output: {has_triggers, trigger_count, triggers}.",
    )
    noise_override: bool = Field(
        default=False,
        description="Preprocessor guardrail flag — True when event is classified as routine noise.",
    )
    guardrail_triggered: bool = Field(
        default=False,
        description="True if prompt-injection or adversarial input was detected.",
    )

    # ── Resolution Agent output ──────────────────────────────────────────
    resolution_output: dict = Field(
        default_factory=dict,
        description="Resolution Agent output: {is_exception, resolution, rationale}.",
    )

    # ── Critic — resolution validation ───────────────────────────────────
    critic_resolution_output: dict = Field(
        default_factory=dict,
        description="Critic verdict on the resolution: {decision, rationale}.",
    )
    resolution_revision_count: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Number of resolution revision attempts (max 2 before escalation).",
    )
    critic_feedback: str = Field(
        default="",
        description="Feedback string from the Critic to guide revision.",
    )

    # ── Communication Agent output ───────────────────────────────────────
    communication_output: dict = Field(
        default_factory=dict,
        description="Communication Agent output: {tone_label, communication_message}.",
    )

    # ── Critic — communication validation ────────────────────────────────
    critic_communication_output: dict = Field(
        default_factory=dict,
        description="Critic verdict on the communication: {decision, rationale}.",
    )

    # ── Routing ──────────────────────────────────────────────────────────
    next_agent: AgentName = Field(
        default=AgentName.PREPROCESSOR,
        description="Next pipeline node to route to. Must be a valid AgentName.",
    )
    max_loops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum revision loops allowed before forced escalation.",
    )

    # ── Final ────────────────────────────────────────────────────────────
    escalated: bool = Field(
        default=False,
        description="Whether any critic node returned ESCALATE.",
    )
    tool_calls_log: list[str] = Field(
        default_factory=list,
        description="Chronological log of all tool invocations for observability.",
    )
    trajectory_log: list[str] = Field(
        default_factory=list,
        description="Audit trail of agent decisions and routing events.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Pipeline start timestamp (epoch seconds).",
    )
    latency_sec: Optional[float] = Field(
        default=None,
        description="Total pipeline latency in seconds.",
    )
    final_actions: list[dict] = Field(
        default_factory=list,
        description="Final packaged output actions for the shipment.",
    )

    # ── Validators ───────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_escalation_consistency(self) -> UnifiedAgentStateModel:
        """If the pipeline has escalated, final_actions should not be empty."""
        if self.escalated and self.next_agent == AgentName.FINALIZE and not self.final_actions:
            logger.warning(
                "State is marked escalated at FINALIZE but final_actions is empty — "
                "this may indicate the finalize node has not yet run."
            )
        return self

    # ── Conversion helpers ───────────────────────────────────────────────

    @classmethod
    def from_typed_dict(cls, state: UnifiedAgentState) -> UnifiedAgentStateModel:
        """Create a validated model instance from a LangGraph TypedDict state."""
        return cls.model_validate(dict(state))

    def to_typed_dict(self) -> UnifiedAgentState:
        """Export back to a plain dict compatible with LangGraph's TypedDict."""
        return self.model_dump()  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# Agent view models (frozen — immutable)
# ═══════════════════════════════════════════════════════════════════════════


class RouterView(BaseModel):
    """Fields accessible to the Router Agent (preprocessor, orchestrator, finalize).

    The router has **full visibility** over the pipeline state because it must
    make routing decisions based on any field.
    """

    model_config = ConfigDict(frozen=True)

    # Input
    raw_rows: list[dict] = Field(default_factory=list, description="Raw delivery log rows.")
    shipment_id: str = Field(default="", description="Unique shipment identifier.")

    # Preprocessor output
    consolidated_event: dict = Field(default_factory=dict, description="Consolidated shipment event.")
    customer_profile: dict = Field(default_factory=dict, description="Redacted customer profile (no PII).")
    customer_profile_full: dict = Field(default_factory=dict, description="Full customer profile with PII.")
    locker_availability: list[dict] = Field(default_factory=list, description="Locker options in delivery zip.")
    playbook_context: list[dict] = Field(default_factory=list, description="Playbook chunks from vector search.")
    escalation_signals: dict = Field(default_factory=dict, description="Deterministic escalation rule output.")
    noise_override: bool = Field(default=False, description="True if event is routine noise.")
    guardrail_triggered: bool = Field(default=False, description="True if injection detected.")

    # Resolution
    resolution_output: dict = Field(default_factory=dict, description="Resolution Agent output.")
    critic_resolution_output: dict = Field(default_factory=dict, description="Critic verdict on resolution.")
    resolution_revision_count: int = Field(default=0, description="Revision attempt count.")
    critic_feedback: str = Field(default="", description="Critic feedback for revision.")

    # Communication
    communication_output: dict = Field(default_factory=dict, description="Communication Agent output.")
    critic_communication_output: dict = Field(default_factory=dict, description="Critic verdict on communication.")

    # Routing & final
    next_agent: str = Field(default="preprocessor", description="Next pipeline node.")
    max_loops: int = Field(default=2, description="Max revision loops.")
    escalated: bool = Field(default=False, description="Whether escalation was triggered.")
    tool_calls_log: list[str] = Field(default_factory=list, description="Tool invocation log.")
    trajectory_log: list[str] = Field(default_factory=list, description="Agent decision audit trail.")
    start_time: Optional[float] = Field(default=None, description="Pipeline start timestamp.")
    latency_sec: Optional[float] = Field(default=None, description="Total pipeline latency.")
    final_actions: list[dict] = Field(default_factory=list, description="Final packaged output.")


class ResolutionAgentView(BaseModel):
    """Fields accessible to the Resolution Agent.  **No PII.**

    The Resolution Agent receives the redacted ``customer_profile`` (without
    name) along with playbook context, escalation signals, and any critic
    feedback from a prior revision loop.  It owns ``resolution_output``.
    """

    model_config = ConfigDict(frozen=True)

    consolidated_event: dict = Field(default_factory=dict, description="Consolidated shipment event.")
    customer_profile: dict = Field(default_factory=dict, description="Redacted customer profile — no name.")
    locker_availability: list[dict] = Field(default_factory=list, description="Locker options in delivery zip.")
    playbook_context: list[dict] = Field(default_factory=list, description="Playbook chunks from vector search.")
    escalation_signals: dict = Field(default_factory=dict, description="Deterministic escalation rule output.")
    critic_feedback: str = Field(default="", description="Critic feedback guiding this revision attempt.")
    # Output owned by this agent
    resolution_output: dict = Field(default_factory=dict, description="Resolution Agent output: {is_exception, resolution, rationale}.")


class CommunicationAgentView(BaseModel):
    """Fields accessible to the Communication Agent.  **Includes PII** for personalisation.

    This is the **only** agent view that receives ``customer_profile_full``
    (including the customer's name).  It owns ``communication_output``.
    """

    model_config = ConfigDict(frozen=True)

    consolidated_event: dict = Field(default_factory=dict, description="Consolidated shipment event.")
    customer_profile_full: dict = Field(default_factory=dict, description="Full customer profile including name — only this agent has PII access.")
    locker_availability: list[dict] = Field(default_factory=list, description="Locker options in delivery zip.")
    resolution_output: dict = Field(default_factory=dict, description="Resolution Agent output to inform messaging.")
    # Output owned by this agent
    communication_output: dict = Field(default_factory=dict, description="Communication Agent output: {tone_label, communication_message}.")


class CriticResolutionView(BaseModel):
    """Fields accessible to the Critic Agent for resolution validation.  **No PII.**

    The Critic reviews the Resolution Agent's output against the playbook
    context and escalation signals.  It owns ``critic_resolution_output``.
    """

    model_config = ConfigDict(frozen=True)

    consolidated_event: dict = Field(default_factory=dict, description="Consolidated shipment event.")
    customer_profile: dict = Field(default_factory=dict, description="Redacted customer profile — no name.")
    locker_availability: list[dict] = Field(default_factory=list, description="Locker options in delivery zip.")
    playbook_context: list[dict] = Field(default_factory=list, description="Playbook chunks from vector search.")
    escalation_signals: dict = Field(default_factory=dict, description="Deterministic escalation rule output.")
    resolution_output: dict = Field(default_factory=dict, description="Resolution Agent output to validate.")
    # Output owned by this agent
    critic_resolution_output: dict = Field(default_factory=dict, description="Critic verdict: {decision, rationale}.")


class CriticCommunicationView(BaseModel):
    """Fields accessible to the Critic Agent for communication validation.  **No PII.**

    The Critic reviews the Communication Agent's output against the resolution
    and consolidated event.  It owns ``critic_communication_output``.
    """

    model_config = ConfigDict(frozen=True)

    consolidated_event: dict = Field(default_factory=dict, description="Consolidated shipment event.")
    customer_profile: dict = Field(default_factory=dict, description="Redacted customer profile — no name.")
    resolution_output: dict = Field(default_factory=dict, description="Resolution Agent output for context.")
    communication_output: dict = Field(default_factory=dict, description="Communication Agent output to validate.")
    # Output owned by this agent
    critic_communication_output: dict = Field(default_factory=dict, description="Critic verdict: {decision, rationale}.")


# ═══════════════════════════════════════════════════════════════════════════
# Projection & merge utilities
# ═══════════════════════════════════════════════════════════════════════════


def project_into(
    state: UnifiedAgentState,
    view_class: type[BaseModel],
) -> dict[str, Any]:
    """Extract only the fields defined in the agent's view from the global state.

    Returns a **new dict** — the original state is never mutated.

    Parameters
    ----------
    state : UnifiedAgentState
        The full pipeline state (LangGraph TypedDict).
    view_class : type[BaseModel]
        One of the view models (e.g. ``ResolutionAgentView``).

    Returns
    -------
    dict[str, Any]
        Filtered state containing only the fields in ``view_class``.
    """
    view_fields = set(view_class.model_fields)
    projected = {k: state.get(k) for k in view_fields if k in state}  # type: ignore[arg-type]

    logger.info(
        "project_into  view=%s  fields_projected=%d/%d",
        view_class.__name__,
        len(projected),
        len(view_fields),
    )
    return projected


def merge_back(
    state: UnifiedAgentState,
    agent_output: dict[str, Any],
    view_class: type[BaseModel],
) -> UnifiedAgentState:
    """Write back only the fields owned by the agent's view into the global state.

    Returns a **new dict** — the original state is never mutated.  Keys in
    ``agent_output`` that fall outside the view's field set are logged as
    warnings and silently dropped.

    Parameters
    ----------
    state : UnifiedAgentState
        The current full pipeline state.
    agent_output : dict[str, Any]
        The output produced by the agent node.
    view_class : type[BaseModel]
        The agent's view model defining its writable scope.

    Returns
    -------
    UnifiedAgentState
        A new state dict with the permitted updates applied.
    """
    view_fields = set(view_class.model_fields)

    # Separate permitted and out-of-scope keys
    updates: dict[str, Any] = {}
    out_of_scope: list[str] = []
    for k, v in agent_output.items():
        if k in view_fields:
            updates[k] = v
        else:
            out_of_scope.append(k)

    if out_of_scope:
        logger.warning(
            "merge_back  view=%s  out_of_scope_keys=%s — these keys were dropped",
            view_class.__name__,
            out_of_scope,
        )

    logger.info(
        "merge_back  view=%s  keys_updated=%s",
        view_class.__name__,
        list(updates.keys()),
    )

    # Functional update — return a new dict, never mutate the original
    return {**state, **updates}  # type: ignore[return-value]
