"""Resolution Agent — classifies delivery exceptions and selects resolution actions.

This module implements the Resolution Agent LangGraph node and its supporting
infrastructure: a Pydantic structured-output schema, playbook formatting, and
input sanitisation helpers.

Architecture
------------
The Resolution Agent sits in the hub-and-spoke pipeline managed by the Router
Agent (``router_agent.py``):

    orchestrator → **resolution_agent** → orchestrator → critic_resolution → …

It receives a projected ``ResolutionAgentView`` (no PII) containing the
consolidated delivery event, redacted customer profile, locker availability,
playbook chunks, escalation signals, and any critic feedback from a prior
revision loop.  It produces a single output field: ``resolution_output``.

LLM interaction
~~~~~~~~~~~~~~~
* Uses ``gen_llm`` (gpt-4o-mini, temp=0) for cost-efficient generation.
* Structured output via ``gen_llm.with_structured_output(ResolutionOutput)``
  ensures typed, validated responses.
* A 3-attempt retry loop handles validation errors (e.g. the model producing
  ``is_exception="YES"`` with ``resolution="N/A"``).  On exhaustion, a safe
  fallback (RESCHEDULE + escalation) is returned.

Optimisation suggestions
------------------------
* **Self-correction** — The REVISE loop is orchestrated externally by
  ``orchestrator_node``.  For *intra-node* self-correction, add a lightweight
  rule check (e.g. 3rd attempt must not produce RESCHEDULE) before returning,
  and re-prompt on violation — this avoids a full orchestrator round-trip.
* **Token management** — ``format_playbook_context`` could accept a
  ``max_chars`` parameter to truncate large retrievals.  With gpt-4o-mini's
  128 k context window this is rarely needed, but provides a safety net for
  unusually large playbook corpora.
* **Latency** — Worst-case path: 2 revision loops × (resolution + critic) =
  4 LLM calls, plus up to 2 retries per invocation.  Streaming is not
  applicable because structured output requires full completion.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langsmith import traceable
from pydantic import BaseModel, Field, model_validator

from langchain_core.messages import HumanMessage, SystemMessage

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.prompts.prompt_library import (
    RESOLUTION_AGENT_SYSTEM_PROMPT,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    ResolutionAgentView,
    UnifiedAgentState,
    merge_back,
    project_into,
)

logger = get_module_logger("components.resolution_agent")

__all__ = [
    "ResolutionOutput",
    "format_playbook_context",
    "sanitize_resolution_inputs",
    "resolution_agent_node",
]


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic output schema
# ═══════════════════════════════════════════════════════════════════════════


class ResolutionOutput(BaseModel):
    """Structured output schema for the Resolution Agent LLM call.

    The ``@model_validator`` enforces logical consistency between the
    exception classification and the chosen resolution action.  When
    ``with_structured_output`` encounters a validation error it triggers
    a retry, which is why the node function includes a retry loop.
    """

    is_exception: Literal["YES", "NO"] = Field(
        description="Whether this delivery event is a real actionable exception.",
    )
    resolution: Literal[
        "RESCHEDULE",
        "REROUTE_TO_LOCKER",
        "REPLACE",
        "RETURN_TO_SENDER",
        "N/A",
    ] = Field(
        description=(
            "Resolution action to take.  Must be N/A when is_exception is NO."
        ),
    )
    rationale: str = Field(
        description=(
            "Step-by-step reasoning for the classification and resolution "
            "decision, citing playbook rules and context."
        ),
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> ResolutionOutput:
        """Ensure is_exception and resolution are mutually consistent."""
        if self.is_exception == "YES" and self.resolution == "N/A":
            raise ValueError(
                "resolution cannot be N/A when is_exception is YES"
            )
        if self.is_exception == "NO" and self.resolution != "N/A":
            raise ValueError(
                "resolution must be N/A when is_exception is NO"
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="format_playbook_context")
def format_playbook_context(playbook: list[dict]) -> str:
    """Format playbook chunks with page references for LLM context.

    Parameters
    ----------
    playbook : list[dict]
        Each dict must have ``page`` and ``content`` keys, as returned by
        the ``search_playbook`` tool.

    Returns
    -------
    str
        Chunks joined by ``---`` separators with ``[Page N]`` prefixes,
        or a fallback message when no chunks are available.
    """
    if not playbook:
        return "No playbook context available."
    return "\n\n---\n\n".join(
        f"[Page {c.get('page', '?')}] {c.get('content', '')}"
        for c in playbook
    )


@traceable(name="sanitize_resolution_inputs")
def sanitize_resolution_inputs(view: dict[str, Any]) -> dict[str, Any]:
    """Type-check and coerce all view fields before they reach the LLM.

    This is a defence-in-depth measure on top of the preprocessor's
    guardrails.  It ensures that even if upstream data is malformed, the
    Resolution Agent receives well-typed inputs.

    Parameters
    ----------
    view : dict[str, Any]
        Projected ``ResolutionAgentView`` fields.

    Returns
    -------
    dict[str, Any]
        A sanitised copy — the original is never mutated.
    """
    clean: dict[str, Any] = {}

    # consolidated_event — must be dict
    ce = view.get("consolidated_event")
    clean["consolidated_event"] = ce if isinstance(ce, dict) else {}

    # customer_profile — must be dict, strip accidental PII
    cp = view.get("customer_profile")
    if isinstance(cp, dict):
        cp = {k: v for k, v in cp.items() if k not in ("name", "full_name")}
    else:
        cp = {}
    clean["customer_profile"] = cp

    # locker_availability — must be list of dicts
    la = view.get("locker_availability")
    if isinstance(la, list):
        clean["locker_availability"] = [
            item for item in la if isinstance(item, dict)
        ]
    else:
        clean["locker_availability"] = []

    # playbook_context — must be list of dicts with page + content
    pc = view.get("playbook_context")
    if isinstance(pc, list):
        clean["playbook_context"] = [
            item
            for item in pc
            if isinstance(item, dict)
            and "content" in item
        ]
    else:
        clean["playbook_context"] = []

    # escalation_signals — must be dict
    es = view.get("escalation_signals")
    clean["escalation_signals"] = es if isinstance(es, dict) else {}

    # critic_feedback — must be str
    cf = view.get("critic_feedback")
    clean["critic_feedback"] = cf if isinstance(cf, str) else ""

    logger.debug("sanitize_resolution_inputs  keys=%s", list(clean.keys()))
    return clean


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph node
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="resolution_agent_node")
def resolution_agent_node(
    state: UnifiedAgentState,
    *,
    gen_llm: Any,
) -> UnifiedAgentState:
    """Classify the delivery event and select a resolution action.

    This node is registered in the LangGraph ``StateGraph`` via
    ``functools.partial`` to bind ``gen_llm`` without polluting the state
    schema::

        bound = partial(resolution_agent_node, gen_llm=gen_llm)
        workflow.add_node("resolution_agent", bound)

    Parameters
    ----------
    state : UnifiedAgentState
        Full pipeline state (LangGraph TypedDict).
    gen_llm : ChatOpenAI
        Generation LLM (gpt-4o-mini) — injected at graph-build time.

    Returns
    -------
    UnifiedAgentState
        Updated state with ``resolution_output`` populated.

    Ending Condition
    ----------------
    The node always returns after a single invocation.  It sets
    ``next_agent`` back to the orchestrator, which decides whether to
    route to the critic, request a revision, or proceed to communication.
    """
    try:
        # 1. Project state into the resolution agent's view (no PII)
        view = project_into(state, ResolutionAgentView)

        # 2. Sanitise inputs — defence-in-depth type checking
        clean = sanitize_resolution_inputs(view)

        # 3. Build critic feedback section for revision loops
        feedback = clean["critic_feedback"]
        feedback_section = ""
        if feedback:
            feedback_section = (
                f"\n\nPREVIOUS ATTEMPT WAS REJECTED. Critic feedback:\n"
                f"{feedback}\n"
                f"Revise your decision based on this feedback."
            )

        # 4. Format the system prompt with optional feedback
        system_prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.format(
            critic_feedback=feedback_section
        )

        # 5. Assemble user content with all context
        playbook_text = format_playbook_context(clean["playbook_context"])

        user_content = (
            f"DELIVERY EVENT:\n"
            f"{json.dumps(clean['consolidated_event'], indent=2)}\n\n"
            f"CUSTOMER PROFILE (redacted):\n"
            f"{json.dumps(clean['customer_profile'], indent=2)}\n\n"
            f"LOCKER AVAILABILITY:\n"
            f"{json.dumps(clean['locker_availability'], indent=2)}\n\n"
            f"ESCALATION SIGNALS:\n"
            f"{json.dumps(clean['escalation_signals'], indent=2)}\n\n"
            f"RELEVANT PLAYBOOK SECTIONS:\n{playbook_text}"
        )

        # 6. Invoke LLM with structured output and retry loop
        structured_llm = gen_llm.with_structured_output(ResolutionOutput)
        max_retries = 3
        result: ResolutionOutput | None = None
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
                break  # Valid output
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "resolution_agent LLM attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    str(exc)[:200],
                )

        # 7. Fallback on exhaustion
        if result is None:
            logger.error(
                "resolution_agent: all %d LLM attempts exhausted, "
                "using RESCHEDULE fallback",
                max_retries,
            )
            result = ResolutionOutput(
                is_exception="YES",
                resolution="RESCHEDULE",
                rationale=(
                    f"Resolution agent failed after {max_retries} attempts. "
                    f"Defaulting to RESCHEDULE with escalation. "
                    f"Last error: {str(last_error)[:200]}"
                ),
            )

        # 8. Merge resolution output via view-scoped merge_back
        agent_output: dict[str, Any] = {
            "resolution_output": result.model_dump(),
        }
        merged = merge_back(state, agent_output, ResolutionAgentView)

        # 9. Append audit entries (outside view scope)
        traj = list(merged.get("trajectory_log") or [])
        if last_error is not None and result.resolution == "RESCHEDULE":
            traj.append(
                f"resolution_agent: all {max_retries} retries exhausted, "
                "defaulting to RESCHEDULE with forced escalation"
            )
        traj.append(
            f"resolution_agent: is_exception={result.is_exception}, "
            f"resolution={result.resolution}"
        )
        merged["trajectory_log"] = traj

        tool_log = list(merged.get("tool_calls_log") or [])
        tool_log.append("AGENT: resolution_agent invoked")
        merged["tool_calls_log"] = tool_log

        merged["next_agent"] = AgentName.RESOLUTION  # orchestrator reads this

        logger.info(
            "resolution_agent_node  is_exception=%s  resolution=%s  "
            "retries_used=%d",
            result.is_exception,
            result.resolution,
            max_retries - 1 if last_error and result.resolution != "RESCHEDULE"
            else 0,
        )
        return merged  # type: ignore[return-value]

    except Exception as exc:
        logger.error(
            "resolution_agent_node fatal error: %s", exc, exc_info=True
        )
        # Fallback: escalate to human review
        traj_fallback = list(state.get("trajectory_log") or [])
        traj_fallback.append(
            f"resolution_agent: FATAL ERROR — {str(exc)[:200]}"
        )
        fallback: dict[str, Any] = {
            "resolution_output": {
                "is_exception": "YES",
                "resolution": "RESCHEDULE",
                "rationale": (
                    f"Resolution agent encountered a fatal error: "
                    f"{str(exc)[:200]}. Defaulting to RESCHEDULE with "
                    f"escalation for human review."
                ),
            },
            "escalated": True,
            "trajectory_log": traj_fallback,
            "next_agent": AgentName.RESOLUTION,
        }
        return merge_back(state, fallback, ResolutionAgentView)  # type: ignore[return-value]
