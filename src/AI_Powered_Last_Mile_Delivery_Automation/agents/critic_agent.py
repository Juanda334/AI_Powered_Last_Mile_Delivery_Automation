"""Critic Agent — validates Resolution and Communication agent outputs.

This module implements both critic LangGraph nodes and their supporting
infrastructure: Pydantic output schemas, context-building helpers, and
production error handling.

Architecture
------------
The critic nodes sit in the hub-and-spoke pipeline managed by the Router
Agent (``router_agent.py``):

    orchestrator → resolution_agent → orchestrator → **critic_resolution**
                 → orchestrator → communication_agent → orchestrator
                 → **critic_communication** → orchestrator → finalize

Both critics use ``eval_llm`` (gpt-4o, temp=0) — a higher-quality model
than the generation agents — because validation requires stronger reasoning
about rule consistency and edge cases.

Dual-critic design
~~~~~~~~~~~~~~~~~~
* **Resolution Critic** — validates exception classification and resolution
  action against playbook rules, locker availability, escalation signals,
  and attempt history.  Can return ACCEPT, ESCALATE, or REVISE.
* **Communication Critic** — validates customer notification for tone,
  accuracy, PII leakage, and channel appropriateness.  Can return ACCEPT
  or ESCALATE only (no revision loop for messages).

Error handling
~~~~~~~~~~~~~~
Both critics default to **ESCALATE** on LLM failure.  This is the safe
direction — unlike the Resolution Agent (which must produce a usable output
and therefore retries), a critic failure should escalate for human review
rather than silently accepting a potentially flawed decision.

Optimisation suggestions
------------------------
* **No retry for critics** — A single attempt with ESCALATE fallback is
  intentional.  Adding retries would increase latency without meaningful
  benefit, since ESCALATE is already the safest failure mode.
* **Escalation authority** — The critic can recommend escalation, but the
  rule engine (``check_escalation_rules``) is authoritative for automatic
  triggers.  Critic ESCALATE without rule-engine backing is still respected
  but logged at WARNING level for observability.
* **Context pruning** — ``build_critic_communication_context`` deliberately
  extracts only the fields needed for validation (tier, channel, credit,
  resolution, exception type) rather than dumping the full state.  This
  reduces token usage and focuses the model on the validation task.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langsmith import traceable
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.prompts.prompt_library import (
    CRITIC_COMMUNICATION_SYSTEM_PROMPT,
    CRITIC_RESOLUTION_SYSTEM_PROMPT,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    CriticCommunicationView,
    CriticResolutionView,
    UnifiedAgentState,
    merge_back,
    project_into,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.resolution_agent import (
    format_playbook_context,
)

logger = get_module_logger("components.critic_agent")

__all__ = [
    "CriticResolutionOutput",
    "CriticCommunicationOutput",
    "build_critic_resolution_context",
    "build_critic_communication_context",
    "critic_resolution_node",
    "critic_communication_node",
]


# ═══════════════════════════��════════════════════════════════════���══════════
# Pydantic output schemas
# ══════��════════════════════════════════════════════════���═══════════════════


class CriticResolutionOutput(BaseModel):
    """Structured output schema for the Resolution Critic LLM call.

    Three possible decisions:

    * **ACCEPT** — resolution is correct and consistent with the playbook.
    * **ESCALATE** — genuine conflicting signals or rule-engine triggers
      require supervisor review.
    * **REVISE** — a clear error or inconsistency; the ``rationale`` field
      provides actionable feedback for the Resolution Agent's next attempt.
    """

    decision: Literal["ACCEPT", "ESCALATE", "REVISE"] = Field(
        description=(
            "ACCEPT: valid. ESCALATE: needs supervisor. "
            "REVISE: send back to Resolution Agent."
        ),
    )
    rationale: str = Field(
        description="Reasoning for the validation decision.",
    )


class CriticCommunicationOutput(BaseModel):
    """Structured output schema for the Communication Critic LLM call.

    Two possible decisions (no REVISE — communications are either acceptable
    or need supervisor review):

    * **ACCEPT** — message is appropriate, correctly toned, accurate, and
      safe to send.
    * **ESCALATE** — message has issues requiring supervisor review (wrong
      tone, inaccurate information, PII concerns).
    """

    decision: Literal["ACCEPT", "ESCALATE"] = Field(
        description=(
            "ACCEPT: message is appropriate. ESCALATE: needs supervisor review."
        ),
    )
    rationale: str = Field(
        description="Reasoning for the validation decision.",
    )


# ══════════════��═══════════════════════════��════════════════════════════════
# Context-building helpers
# ══��════════════════════���═════════════════════════════════���═════════════════


@traceable(name="build_critic_resolution_context")
def build_critic_resolution_context(view: dict[str, Any]) -> str:
    """Assemble all context fields into a formatted string for the critic.

    The resolution critic needs the full picture: delivery event, customer
    profile (redacted), locker availability, escalation signals, playbook
    chunks, and the resolution output it is validating.

    Parameters
    ----------
    view : dict[str, Any]
        Projected ``CriticResolutionView`` fields.

    Returns
    -------
    str
        Formatted multi-section string for the critic's user message.
    """
    playbook_text = format_playbook_context(view.get("playbook_context") or [])

    consolidated = view.get("consolidated_event") or {}

    return (
        f"DELIVERY EVENT:\n"
        f"{json.dumps(consolidated, indent=2)}\n\n"
        f"CUSTOMER PROFILE:\n"
        f"{json.dumps(view.get('customer_profile') or {}, indent=2)}\n\n"
        f"LOCKER AVAILABILITY:\n"
        f"{json.dumps(view.get('locker_availability') or [], indent=2)}\n\n"
        f"ESCALATION SIGNALS:\n"
        f"{json.dumps(view.get('escalation_signals') or {}, indent=2)}\n\n"
        f"PLAYBOOK CONTEXT:\n{playbook_text}\n\n"
        f"ATTEMPT NUMBER: "
        f"{consolidated.get('attempt_number', 'unknown')}\n\n"
        f"RESOLUTION AGENT OUTPUT:\n"
        f"{json.dumps(view.get('resolution_output') or {}, indent=2)}"
    )


@traceable(name="build_critic_communication_context")
def build_critic_communication_context(view: dict[str, Any]) -> str:
    """Build focused validation context for the communication critic.

    This helper deliberately extracts only the fields needed for message
    validation — customer tier, preferred channel, active credit, resolution
    action, exception type, and package type — rather than dumping the full
    state.  This reduces token usage and keeps the model focused.

    **No PII** — the communication critic does not see the customer's name.

    Parameters
    ----------
    view : dict[str, Any]
        Projected ``CriticCommunicationView`` fields.

    Returns
    -------
    str
        Formatted context string for the critic's user message.
    """
    customer_profile = view.get("customer_profile") or {}
    resolution_output = view.get("resolution_output") or {}
    consolidated = view.get("consolidated_event") or {}

    validation_context = {
        "customer_tier": customer_profile.get("tier"),
        "preferred_channel": customer_profile.get("preferred_channel"),
        "active_credit": customer_profile.get("active_credit", 0),
        "resolution": resolution_output.get("resolution"),
        "exception_type": consolidated.get("status_code"),
        "package_type": consolidated.get("package_type"),
    }

    return (
        f"VALIDATION CONTEXT:\n"
        f"{json.dumps(validation_context, indent=2)}\n\n"
        f"COMMUNICATION AGENT OUTPUT:\n"
        f"{json.dumps(view.get('communication_output') or {}, indent=2)}"
    )


# ═════���═════════════════════════���═══════════════════════════════════════════
# LangGraph nodes
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="critic_resolution_node")
def critic_resolution_node(
    state: UnifiedAgentState,
    *,
    eval_llm: Any,
) -> UnifiedAgentState:
    """Validate the Resolution Agent's output against playbook rules.

    This node is registered in the LangGraph ``StateGraph`` via
    ``functools.partial`` to bind ``eval_llm``::

        bound = partial(critic_resolution_node, eval_llm=eval_llm)
        workflow.add_node("critic_resolution", bound)

    Parameters
    ----------
    state : UnifiedAgentState
        Full pipeline state (LangGraph TypedDict).
    eval_llm : ChatOpenAI
        Evaluation LLM (gpt-4o) — injected at graph-build time.

    Returns
    -------
    UnifiedAgentState
        Updated state with ``critic_resolution_output`` populated, and
        on REVISE: ``resolution_revision_count`` incremented and
        ``critic_feedback`` stored.

    Ending Condition
    ----------------
    Always returns after a single LLM call (no retry loop).  On failure,
    defaults to ESCALATE — the safe direction for a validation step.
    """
    try:
        # 1. Project state into the critic's view (no PII, no communication)
        view = project_into(state, CriticResolutionView)

        # 2. Build the full validation context
        user_content = build_critic_resolution_context(view)

        # 3. Single LLM call with structured output
        try:
            structured_llm = eval_llm.with_structured_output(CriticResolutionOutput)
            result = structured_llm.invoke(
                [
                    SystemMessage(content=CRITIC_RESOLUTION_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )
        except Exception as exc:
            logger.error("critic_resolution LLM failed: %s", str(exc)[:200])
            result = CriticResolutionOutput(
                decision="ESCALATE",
                rationale=(
                    f"Critic validation failed: {str(exc)[:200]}. "
                    f"Escalating for human review."
                ),
            )

        # 4. Merge critic output via view-scoped merge_back
        agent_output: dict[str, Any] = {
            "critic_resolution_output": result.model_dump(),
        }
        merged = merge_back(state, agent_output, CriticResolutionView)

        # 5. Handle ESCALATE — set escalated flag
        if result.decision == "ESCALATE":
            if merged.get("escalation_signals", {}).get("has_triggers"):
                merged["escalated"] = True
            else:
                logger.warning(
                    "critic_resolution: ESCALATE recommended but no "
                    "rule-engine triggers — escalating with advisory. "
                    "Rationale: %s",
                    result.rationale[:100],
                )
                merged["escalated"] = True

        # 6. Handle REVISE — increment counter and store feedback
        if result.decision == "REVISE":
            merged["resolution_revision_count"] = (
                state.get("resolution_revision_count", 0) + 1
            )
            merged["critic_feedback"] = result.rationale

        # 7. Append audit entries (outside view scope)
        traj = list(merged.get("trajectory_log") or [])
        traj.append(f"critic_resolution: decision={result.decision}")
        merged["trajectory_log"] = traj

        tool_log = list(merged.get("tool_calls_log") or [])
        tool_log.append("AGENT: critic_resolution invoked")
        merged["tool_calls_log"] = tool_log

        merged["next_agent"] = AgentName.CRITIC_RESOLUTION

        logger.info("critic_resolution_node  decision=%s", result.decision)
        return merged  # type: ignore[return-value]

    except Exception as exc:
        logger.error("critic_resolution_node fatal error: %s", exc, exc_info=True)
        traj_fallback = list(state.get("trajectory_log") or [])
        traj_fallback.append(f"critic_resolution: FATAL ERROR — {str(exc)[:200]}")
        fallback: dict[str, Any] = {
            "critic_resolution_output": {
                "decision": "ESCALATE",
                "rationale": (
                    f"Critic encountered a fatal error: {str(exc)[:200]}. "
                    f"Escalating for human review."
                ),
            },
            "escalated": True,
            "trajectory_log": traj_fallback,
            "next_agent": AgentName.CRITIC_RESOLUTION,
        }
        return merge_back(state, fallback, CriticResolutionView)  # type: ignore[return-value]


@traceable(name="critic_communication_node")
def critic_communication_node(
    state: UnifiedAgentState,
    *,
    eval_llm: Any,
) -> UnifiedAgentState:
    """Validate the Communication Agent's customer notification.

    Checks tone (VIP/PREMIUM=FORMAL, STANDARD=CASUAL), accuracy,
    channel appropriateness, and PII safety.  No revision loop — messages
    are either accepted or escalated for supervisor review.

    This node is registered in the LangGraph ``StateGraph`` via
    ``functools.partial`` to bind ``eval_llm``::

        bound = partial(critic_communication_node, eval_llm=eval_llm)
        workflow.add_node("critic_communication", bound)

    Parameters
    ----------
    state : UnifiedAgentState
        Full pipeline state (LangGraph TypedDict).
    eval_llm : ChatOpenAI
        Evaluation LLM (gpt-4o) — injected at graph-build time.

    Returns
    -------
    UnifiedAgentState
        Updated state with ``critic_communication_output`` populated.

    Ending Condition
    ----------------
    Always returns after a single LLM call.  On failure, defaults to
    ESCALATE.  The orchestrator routes to finalize after this node.
    """
    try:
        # 1. Project state into the communication critic's view (no PII)
        view = project_into(state, CriticCommunicationView)

        # 2. Build focused validation context
        user_content = build_critic_communication_context(view)

        # 3. Single LLM call with structured output
        try:
            structured_llm = eval_llm.with_structured_output(CriticCommunicationOutput)
            result = structured_llm.invoke(
                [
                    SystemMessage(content=CRITIC_COMMUNICATION_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )
        except Exception as exc:
            logger.error("critic_communication LLM failed: %s", str(exc)[:200])
            result = CriticCommunicationOutput(
                decision="ESCALATE",
                rationale=(
                    f"Critic validation failed: {str(exc)[:200]}. "
                    f"Escalating for human review."
                ),
            )

        # 4. Merge critic output via view-scoped merge_back
        agent_output: dict[str, Any] = {
            "critic_communication_output": result.model_dump(),
        }
        merged = merge_back(state, agent_output, CriticCommunicationView)

        # 5. Handle ESCALATE — set escalated flag
        if result.decision == "ESCALATE":
            if merged.get("escalation_signals", {}).get("has_triggers"):
                merged["escalated"] = True
            else:
                logger.warning(
                    "critic_communication: ESCALATE recommended but no "
                    "rule-engine triggers — escalating with advisory. "
                    "Rationale: %s",
                    result.rationale[:100],
                )
                merged["escalated"] = True

        # 6. Append audit entries (outside view scope)
        traj = list(merged.get("trajectory_log") or [])
        traj.append(f"critic_communication: decision={result.decision}")
        merged["trajectory_log"] = traj

        tool_log = list(merged.get("tool_calls_log") or [])
        tool_log.append("AGENT: critic_communication invoked")
        merged["tool_calls_log"] = tool_log

        merged["next_agent"] = AgentName.CRITIC_COMMUNICATION

        logger.info("critic_communication_node  decision=%s", result.decision)
        return merged  # type: ignore[return-value]

    except Exception as exc:
        logger.error(
            "critic_communication_node fatal error: %s",
            exc,
            exc_info=True,
        )
        traj_fallback = list(state.get("trajectory_log") or [])
        traj_fallback.append(f"critic_communication: FATAL ERROR — {str(exc)[:200]}")
        fallback: dict[str, Any] = {
            "critic_communication_output": {
                "decision": "ESCALATE",
                "rationale": (
                    f"Critic encountered a fatal error: {str(exc)[:200]}. "
                    f"Escalating for human review."
                ),
            },
            "escalated": True,
            "trajectory_log": traj_fallback,
            "next_agent": AgentName.CRITIC_COMMUNICATION,
        }
        return merge_back(state, fallback, CriticCommunicationView)  # type: ignore[return-value]
