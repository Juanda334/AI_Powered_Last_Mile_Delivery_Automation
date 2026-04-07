"""Communication Agent — generates personalized customer notifications.

This module implements the Communication Agent LangGraph node and its
supporting infrastructure: a Pydantic structured-output schema, a context
builder, and an input-sanitisation helper.

Architecture
------------
The Communication Agent sits in the hub-and-spoke pipeline managed by the
Router Agent (``router_agent.py``):

    orchestrator → **communication_agent** → orchestrator → critic_communication → …

It receives a projected ``CommunicationAgentView`` which — uniquely among
the agent views — contains ``customer_profile_full`` (including the
customer's name) so the message can be personalised. **This is the only
node in the pipeline with PII access.** It produces a single output
field: ``communication_output``.

LLM interaction
~~~~~~~~~~~~~~~
* Uses ``gen_llm`` (gpt-4o-mini, temp=0) for cost-efficient generation.
* Structured output via ``gen_llm.with_structured_output(CommunicationOutput)``
  ensures typed, validated responses with a ``tone_label`` and a
  ``communication_message``.
* A 3-attempt retry loop handles transient LLM / validation errors.
  On exhaustion, a safe generic fallback message is returned with
  ``tone_label`` inferred from customer tier, and ``escalated=True``
  is set so a human agent follows up.

PII boundary
~~~~~~~~~~~~
Only ``customer_profile_full.{name, tier, preferred_channel,
active_credit}`` is projected into the LLM context — address, phone, and
other PII fields are stripped by :func:`sanitize_communication_inputs`
before prompt assembly, keeping LangSmith traces free of sensitive data.

Optimisation suggestions
------------------------
* **Template management** — Replace the f-string prompt assembly and the
  static fallback message with Jinja2 templates stored under
  ``prompts/templates/`` so copywriters can edit customer-facing text
  without touching Python.
* **Asynchronous execution** — The LangGraph node stays synchronous, but
  the downstream delivery handshake (email / SMS / push) should be
  decoupled: emit ``communication_output`` onto a queue
  (Redis / SQS) and let a worker call the provider with exponential
  backoff, isolating the graph from third-party latency.
* **Human-in-the-Loop (HITL)** — When ``escalated`` is ``True`` before
  dispatch, route to a ``finalize`` node that parks the message in a
  review queue (``artifacts/review_queue/<request_id>.json``).
  LangGraph's ``interrupt_before=["finalize"]`` provides the pause
  primitive for supervisor approval.
* **Feedback loop** — Persist delivery receipts and customer replies
  back into the state (new ``communication_feedback`` field) so
  analytics and future prompt tuning can learn which tones and phrasings
  produced the fewest follow-up tickets.
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
    COMMUNICATION_AGENT_SYSTEM_PROMPT,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    CommunicationAgentView,
    UnifiedAgentState,
    merge_back,
    project_into,
)

logger = get_module_logger("components.communication_agent")

__all__ = [
    "CommunicationOutput",
    "sanitize_communication_inputs",
    "build_communication_context",
    "communication_agent_node",
]


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic output schema
# ═══════════════════════════════════════════════════════════════════════════


class CommunicationOutput(BaseModel):
    """Structured output schema for the Communication Agent LLM call.

    Both fields are always required with no cross-field dependencies, so
    no ``@model_validator`` is necessary.  The tone is inferred from the
    customer's tier (VIP / PREMIUM → FORMAL, STANDARD → CASUAL) by the
    LLM following the system prompt rules.
    """

    tone_label: Literal["FORMAL", "CASUAL"] = Field(
        description="Tone of the customer message, inferred from customer tier.",
    )
    communication_message: str = Field(
        description="The customer-facing notification message.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


# Keys from the full customer profile that are safe to send to the LLM.
# Anything else (address, phone, email, etc.) is stripped by the sanitiser
# before it can leak into prompts or LangSmith traces.
_SAFE_PROFILE_KEYS: tuple[str, ...] = (
    "name",
    "tier",
    "preferred_channel",
    "active_credit",
)

# Keys from the consolidated delivery event that should never reach the
# customer-facing message — internal identifiers and raw log artefacts.
_NOISY_EVENT_KEYS: frozenset[str] = frozenset({
    "shipment_id",
    "tracking_number",
    "raw_log",
    "raw_logs",
    "uuid",
    "internal_id",
    "correlation_id",
})


@traceable(name="sanitize_communication_inputs")
def sanitize_communication_inputs(view: dict[str, Any]) -> dict[str, Any]:
    """Type-check and strip internal noise from view fields before LLM use.

    This is a defence-in-depth measure on top of the preprocessor and the
    view projection.  It ensures that even if upstream data is malformed
    or contains extra internal fields, the Communication Agent sends a
    minimal, customer-safe payload to the LLM.

    Parameters
    ----------
    view : dict[str, Any]
        Projected ``CommunicationAgentView`` fields.

    Returns
    -------
    dict[str, Any]
        A sanitised copy — the original is never mutated.
    """
    clean: dict[str, Any] = {}

    # consolidated_event — must be dict, strip internal identifier noise
    ce = view.get("consolidated_event")
    if isinstance(ce, dict):
        clean["consolidated_event"] = {
            k: v for k, v in ce.items() if k not in _NOISY_EVENT_KEYS
        }
    else:
        clean["consolidated_event"] = {}

    # customer_profile_full — must be dict, keep only message-relevant keys
    cp = view.get("customer_profile_full")
    if isinstance(cp, dict):
        clean["customer_profile_full"] = {
            k: cp[k] for k in _SAFE_PROFILE_KEYS if k in cp
        }
    else:
        clean["customer_profile_full"] = {}

    # resolution_output — must be dict
    ro = view.get("resolution_output")
    clean["resolution_output"] = ro if isinstance(ro, dict) else {}

    # locker_availability — must be list of dicts
    la = view.get("locker_availability")
    if isinstance(la, list):
        clean["locker_availability"] = [
            item for item in la if isinstance(item, dict)
        ]
    else:
        clean["locker_availability"] = []

    logger.debug(
        "sanitize_communication_inputs  keys=%s", list(clean.keys())
    )
    return clean


@traceable(name="build_communication_context")
def build_communication_context(view: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build the context dict and locker info string for the LLM prompt.

    Only uses fields available in ``CommunicationAgentView``.  If the
    resolution is ``REROUTE_TO_LOCKER``, the first eligible locker's
    details are appended for inclusion in the notification.

    Parameters
    ----------
    view : dict[str, Any]
        Sanitised ``CommunicationAgentView`` projection.

    Returns
    -------
    tuple[dict[str, Any], str]
        ``(comm_context, locker_info)`` — the context the LLM needs for
        message generation, and an optional locker details string.
    """
    event = view.get("consolidated_event", {}) or {}
    profile = view.get("customer_profile_full", {}) or {}
    resolution = view.get("resolution_output", {}) or {}
    lockers = view.get("locker_availability", []) or []

    # Include locker details only if the resolution is a reroute
    locker_info = ""
    if resolution.get("resolution") == "REROUTE_TO_LOCKER":
        eligible = [l for l in lockers if l.get("eligible")]
        if eligible:
            locker_info = (
                f"\nLOCKER FOR REROUTE:\n{json.dumps(eligible[0], indent=2)}"
            )

    # Assemble the context the LLM needs for message generation
    comm_context = {
        "customer_name": profile.get("name", "Customer"),
        "customer_tier": profile.get("tier"),
        "preferred_channel": profile.get("preferred_channel"),
        "active_credit": profile.get("active_credit", 0),
        "exception_type": event.get("status_code"),
        "status_description": event.get("status_description"),
        "package_type": event.get("package_type"),
        "resolution": resolution.get("resolution"),
        "resolution_rationale": resolution.get("rationale"),
    }

    return comm_context, locker_info


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph node
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="communication_agent_node")
def communication_agent_node(
    state: UnifiedAgentState,
    *,
    gen_llm: Any,
) -> UnifiedAgentState:
    """Generate a personalised customer notification message.

    This node is registered in the LangGraph ``StateGraph`` via
    ``functools.partial`` to bind ``gen_llm`` without polluting the state
    schema::

        bound = partial(communication_agent_node, gen_llm=gen_llm)
        workflow.add_node("communication_agent", bound)

    Communication channel
    ---------------------
    This node targets the **customer notification channel** — the tone
    and phrasing follow ``customer_profile_full.preferred_channel``
    (EMAIL = slightly longer, SMS = brief).  Actual dispatch to a
    provider (SES, Twilio, etc.) happens downstream of the graph, driven
    by the ``communication_output`` field this node writes to state.

    Expected response schema
    ------------------------
    ``CommunicationOutput`` with ``tone_label`` ∈ {FORMAL, CASUAL} and a
    3–5 sentence ``communication_message``.  The critic_communication
    node validates this against the redacted (no-PII) view.

    Parameters
    ----------
    state : UnifiedAgentState
        Full pipeline state (LangGraph TypedDict).
    gen_llm : ChatOpenAI
        Generation LLM (gpt-4o-mini) — injected at graph-build time.

    Returns
    -------
    UnifiedAgentState
        Updated state with ``communication_output`` populated.  On
        retry-exhaustion the state is also marked ``escalated=True``.

    Ending Condition
    ----------------
    The node always returns after a single invocation.  It sets
    ``next_agent`` back to the orchestrator, which routes to
    ``critic_communication`` for validation.
    """
    try:
        # 1. Project state into the communication agent's view (PII allowed)
        view = project_into(state, CommunicationAgentView)

        # 2. Sanitise inputs — strip noisy keys & non-message-relevant PII
        clean = sanitize_communication_inputs(view)

        # 3. Build the context and locker info from the sanitised view
        comm_context, locker_info = build_communication_context(clean)

        # 4. Assemble the user message for the LLM
        user_content = (
            f"CONTEXT:\n{json.dumps(comm_context, indent=2)}{locker_info}"
        )

        # 5. Invoke LLM with structured output and retry loop
        structured_llm = gen_llm.with_structured_output(CommunicationOutput)
        max_retries = 3
        result: CommunicationOutput | None = None
        last_error: Exception | None = None
        retries_used = 0

        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke([
                    SystemMessage(content=COMMUNICATION_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ])
                retries_used = attempt
                break  # Valid output
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "communication_agent LLM attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    str(exc)[:200],
                )

        # 6. Fallback on exhaustion — safe generic message + forced escalation
        fallback_used = False
        if result is None:
            fallback_used = True
            retries_used = max_retries
            tier = clean["customer_profile_full"].get("tier", "STANDARD")
            result = CommunicationOutput(
                tone_label="FORMAL" if tier in ("VIP", "PREMIUM") else "CASUAL",
                communication_message=(
                    "We're aware of an issue with your delivery and are "
                    "working to resolve it. A team member will follow up "
                    "shortly."
                ),
            )
            logger.error(
                "communication_agent: all %d LLM attempts exhausted, "
                "using generic fallback with forced escalation. "
                "Last error: %s",
                max_retries,
                str(last_error)[:200] if last_error else "n/a",
            )

        # 7. Merge communication_output via view-scoped merge_back
        agent_output: dict[str, Any] = {
            "communication_output": result.model_dump(),
        }
        merged = merge_back(state, agent_output, CommunicationAgentView)

        # 8. Force escalation if we fell back
        if fallback_used:
            merged["escalated"] = True

        # 9. Append audit entries (outside view scope)
        traj = list(merged.get("trajectory_log") or [])
        if fallback_used:
            traj.append(
                f"communication_agent: all {max_retries} retries exhausted, "
                "defaulting to generic message with forced escalation"
            )
        traj.append(
            f"communication_agent: tone={result.tone_label}"
        )
        merged["trajectory_log"] = traj

        tool_log = list(merged.get("tool_calls_log") or [])
        tool_log.append("AGENT: communication_agent invoked")
        merged["tool_calls_log"] = tool_log

        merged["next_agent"] = AgentName.COMMUNICATION  # orchestrator reads this

        logger.info(
            "communication_agent_node  tone=%s  retries_used=%d  fallback=%s",
            result.tone_label,
            retries_used,
            fallback_used,
        )
        return merged  # type: ignore[return-value]

    except Exception as exc:
        logger.error(
            "communication_agent_node fatal error: %s", exc, exc_info=True
        )
        # Fallback: generic message + escalate to human review
        traj_fallback = list(state.get("trajectory_log") or [])
        traj_fallback.append(
            f"communication_agent: FATAL ERROR — {str(exc)[:200]}"
        )
        tool_log_fallback = list(state.get("tool_calls_log") or [])
        tool_log_fallback.append("AGENT: communication_agent invoked")
        fallback: dict[str, Any] = {
            "communication_output": {
                "tone_label": "CASUAL",
                "communication_message": (
                    "We're aware of an issue with your delivery and are "
                    "working to resolve it. A team member will follow up "
                    "shortly."
                ),
            },
            "escalated": True,
            "trajectory_log": traj_fallback,
            "tool_calls_log": tool_log_fallback,
            "next_agent": AgentName.COMMUNICATION,
        }
        return merge_back(state, fallback, CommunicationAgentView)  # type: ignore[return-value]
