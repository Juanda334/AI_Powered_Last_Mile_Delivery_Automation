"""Router Agent — primary orchestrator for the multi-agent delivery pipeline.

This module implements the three LangGraph nodes owned by the Router Agent
(preprocessor, orchestrator, finalize) together with the helper functions they
rely on: input sanitisation, event consolidation, noise detection, and
context fetching via :class:`ToolMaster`.

Architecture
------------
The pipeline follows a **hub-and-spoke** pattern:

    preprocessor → orchestrator ⇄ {resolution, critic_resolution,
                                    communication, critic_communication}
                                → finalize → END

* **Preprocessor** — deduplicates rows, consolidates the shipment event,
  runs injection / noise guardrails, and fetches tool context.
* **Orchestrator** — a purely deterministic state machine (zero LLM calls)
  that inspects the current state and routes to the next agent node.
* **Finalize** — packages the final output and records pipeline latency.

Because the orchestrator contains *no* LLM calls, routing hallucinations are
impossible by design.  All sub-agent nodes (resolution, communication,
critics) are registered as placeholders here and will be replaced by their
dedicated module implementations.

Guardrail-first design
~~~~~~~~~~~~~~~~~~~~~~
1. **Prompt-injection scan** — checked *before* any tool or LLM call.
2. **RAG chunk scan** — checked *after* vector retrieval but before the
   chunks reach a downstream agent.
3. **Noise override** — routine status codes with no anomaly indicators
   short-circuit the entire pipeline, avoiding unnecessary LLM spend.

Optimisation suggestions
------------------------
* **Decision caching** — ``check_noise_override`` and the escalation rule
  engine are deterministic.  For batch processing, cache results keyed by
  ``(status_code, description_hash)`` or the escalation-rule input tuple.
  ``functools.lru_cache`` works for in-process caching; Redis for
  cross-invocation reuse.
* **Prompt optimisation** — the orchestrator is LLM-free; for the sub-agents
  that *do* call an LLM, the prompts in ``prompt_library.py`` already
  include priority rules and typed output formats.  Adding few-shot examples
  for the most common edge cases (3rd-attempt + full locker, perishable +
  weather delay) would further reduce misclassification.
* **Parallelism** — the five tool calls inside ``fetch_context`` are
  independent and could run concurrently via ``asyncio.gather`` or a
  ``ThreadPoolExecutor``.  Parallelising would cut preprocessor latency by
  ~60 % in typical cases.  The orchestrator itself is inherently sequential
  (state-machine pattern).
"""

from __future__ import annotations

import json
import time
from functools import partial
from typing import Any

from langsmith import traceable

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import ToolMaster
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    RouterView,
    UnifiedAgentState,
    merge_back,
    project_into,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.resolution_agent import (
    resolution_agent_node,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.critic_agent import (
    critic_communication_node,
    critic_resolution_node,
)

logger = get_module_logger("components.router_agent")

__all__ = [
    "INJECTION_KEYWORDS",
    "build_router_graph",
    "check_noise_override",
    "consolidate_event",
    "deduplicate_rows",
    "fetch_context",
    "finalize_node",
    "orchestrator_node",
    "preprocessor_node",
    "scan_chunks_for_injection",
    "scan_for_injection",
    "scan_inputs_for_injection",
]


# ═══════════════════════════════════════════════════════════════════════════
# Input sanitisation
# ═══════════════════════════════════════════════════════════════════════════

INJECTION_KEYWORDS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "forget your instructions",
    "override your instructions",
    "you are now",
    "act as",
    "pretend you are",
    "new instructions",
    "system prompt",
    "reveal your prompt",
    "ignore the above",
    "ignore above",
    "do not follow",
    "bypass",
    "jailbreak",
    "prompt injection",
    "ignore safety",
    "ignore guidelines",
    "drop table",
    "delete from",
    "exec(",
    "eval(",
    "<script>",
    "javascript:",
]


@traceable(name="scan_for_injection")
def scan_for_injection(text: str) -> bool:
    """Return *True* if any prompt-injection keyword is found in *text*.

    Parameters
    ----------
    text : str
        Free-text field to scan (e.g. a driver note or RAG chunk).

    Returns
    -------
    bool
        ``True`` when at least one injection keyword matches.
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    detected = any(keyword in text_lower for keyword in INJECTION_KEYWORDS)
    if detected:
        logger.warning("Injection keyword detected in text: %r", text[:120])
    return detected


@traceable(name="scan_inputs_for_injection")
def scan_inputs_for_injection(consolidated: dict, raw_rows: list[dict]) -> bool:
    """Scan all free-text ``status_description`` fields for prompt injection.

    Checks the consolidated event's description and every raw row's
    description.  A single match is enough to trigger the guardrail.
    """
    texts = [consolidated.get("status_description", "")]
    texts.extend(row.get("status_description", "") for row in raw_rows)
    result = any(scan_for_injection(t) for t in texts)
    if result:
        logger.warning(
            "Injection detected in delivery inputs  shipment_id=%s",
            consolidated.get("shipment_id", "?"),
        )
    return result


@traceable(name="scan_chunks_for_injection")
def scan_chunks_for_injection(playbook_context: list[dict]) -> bool:
    """Scan retrieved RAG chunks for prompt injection.

    If any chunk is contaminated the preprocessor will drop *all* chunks
    and force escalation for human review.
    """
    result = any(
        scan_for_injection(chunk.get("content", "")) for chunk in playbook_context
    )
    if result:
        logger.warning(
            "Injection detected in retrieved playbook chunk  chunks=%d",
            len(playbook_context),
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Data preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="deduplicate_rows")
def deduplicate_rows(raw_rows: list[dict]) -> list[dict]:
    """Remove duplicate scan events.

    Rows whose ``is_duplicate_scan`` field equals ``"True"`` (string, as
    parsed from CSV) are filtered out.
    """
    unique = [r for r in raw_rows if r.get("is_duplicate_scan", "False") != "True"]
    logger.debug(
        "deduplicate_rows  input=%d  output=%d  removed=%d",
        len(raw_rows),
        len(unique),
        len(raw_rows) - len(unique),
    )
    return unique


@traceable(name="consolidate_event")
def consolidate_event(unique_rows: list[dict], raw_rows: list[dict]) -> dict:
    """Consolidate a multi-row shipment into a single event dict.

    Selects the row with the highest ``attempt_number`` as the primary
    record and collects driver notes from all prior attempts.
    """
    if unique_rows:
        primary = max(unique_rows, key=lambda r: int(r.get("attempt_number", 0)))
    else:
        primary = raw_rows[0]

    prior_notes = [
        f"Attempt {r['attempt_number']}: {r['status_description']}"
        for r in unique_rows
        if r is not primary
    ]

    consolidated = {
        "shipment_id": primary["shipment_id"],
        "timestamp": primary["timestamp"],
        "status_code": primary["status_code"],
        "status_description": primary["status_description"],
        "customer_id": primary["customer_id"],
        "delivery_address": primary["delivery_address"],
        "package_type": primary["package_type"],
        "package_size": primary["package_size"],
        "attempt_number": int(primary["attempt_number"]),
        "prior_attempt_notes": prior_notes,
        "total_rows": len(raw_rows),
        "duplicates_removed": len(raw_rows) - len(unique_rows),
    }

    logger.info(
        "consolidate_event  shipment_id=%s  attempt=%d  prior_notes=%d",
        consolidated["shipment_id"],
        consolidated["attempt_number"],
        len(prior_notes),
    )
    return consolidated


@traceable(name="check_noise_override")
def check_noise_override(consolidated: dict) -> bool:
    """Flag routine status codes with no anomaly indicators.

    Returns ``True`` when the event is classified as non-actionable noise,
    allowing the pipeline to skip all tool calls and LLM invocations.
    """
    routine_codes = {"DELIVERED", "IN_TRANSIT", "OUT_FOR_DELIVERY", "SCANNED"}
    if consolidated["status_code"] not in routine_codes:
        return False

    anomaly_indicators = [
        "damage",
        "wrong",
        "suspicious",
        "overdue",
        "missing",
        "unexpected",
        "misroute",
        "lost",
        "stolen",
        "abandoned",
        "leak",
        "crush",
        "broke",
        "delay",
        "late",
        "fraud",
    ]
    desc = consolidated["status_description"].lower()
    is_noise = not any(indicator in desc for indicator in anomaly_indicators)
    if is_noise:
        logger.info(
            "check_noise_override  status_code=%s  result=NOISE",
            consolidated["status_code"],
        )
    return is_noise


# ═══════════════════════════════════════════════════════════════════════════
# Context fetching via ToolMaster
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="fetch_context")
def fetch_context(
    consolidated: dict,
    tools: ToolMaster,
    tool_log: list[str],
) -> dict:
    """Fetch all enrichment context via tools.

    Makes five independent tool calls (customer profile redacted, customer
    profile full, locker availability, playbook search, escalation rules)
    and collects results into a single dict.  Each call is individually
    wrapped in ``try / except`` so one failure does not block the others.

    Parameters
    ----------
    consolidated : dict
        The consolidated shipment event produced by :func:`consolidate_event`.
    tools : ToolMaster
        Initialised tool master holding all LangChain tool callables.
    tool_log : list[str]
        Mutable log list — entries are appended in-place for observability.

    Returns
    -------
    dict
        Keys: ``customer_profile``, ``customer_profile_full``,
        ``locker_availability``, ``playbook_context``, ``escalation_signals``.
    """
    customer_id = consolidated["customer_id"]

    # 1. Customer profile (redacted — no PII)
    customer_profile: dict = {}
    try:
        customer_profile = tools.get_tool("lookup_customer_profile").invoke(
            {"customer_id": customer_id, "include_pii": False}
        )
    except Exception as exc:
        logger.error("lookup_customer_profile(pii=False) failed: %s", exc)
        tool_log.append(
            f"TOOL ERROR: lookup_customer_profile(pii=False) - {str(exc)[:100]}"
        )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=False)")

    # 2. Customer profile (full — with PII for Communication Agent)
    customer_profile_full: dict = {}
    try:
        customer_profile_full = tools.get_tool("lookup_customer_profile").invoke(
            {"customer_id": customer_id, "include_pii": True}
        )
    except Exception as exc:
        logger.error("lookup_customer_profile(pii=True) failed: %s", exc)
        tool_log.append(
            f"TOOL ERROR: lookup_customer_profile(pii=True) - {str(exc)[:100]}"
        )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=True)")

    # 3. Locker availability (zip extracted from delivery address)
    locker_availability: list[dict] = []
    try:
        address_parts = consolidated["delivery_address"].split(",")
        zip_code = address_parts[-1].strip() if address_parts else ""
        locker_availability = tools.get_tool("check_locker_availability").invoke(
            {"zip_code": zip_code, "package_size": consolidated["package_size"]}
        )
    except Exception as exc:
        logger.error("check_locker_availability failed: %s", exc)
        tool_log.append(f"TOOL ERROR: check_locker_availability - {str(exc)[:100]}")
    tool_log.append("TOOL: check_locker_availability")

    # 4. Playbook context (RAG vector search)
    playbook_context: list[dict] = []
    try:
        playbook_context = tools.get_tool("search_playbook").invoke(
            {"query": consolidated["status_description"]}
        )
    except Exception as exc:
        logger.error("search_playbook failed: %s", exc)
        tool_log.append(f"TOOL ERROR: search_playbook - {str(exc)[:100]}")
    tool_log.append("TOOL: search_playbook(query)")

    # 5. Escalation rules (deterministic rule engine)
    escalation_signals: dict = {}
    try:
        escalation_signals = tools.get_tool("check_escalation_rules").invoke(
            {
                "customer_tier": customer_profile.get("tier", "STANDARD"),
                "exceptions_last_90d": customer_profile.get("exceptions_last_90d", 0),
                "attempt_number": consolidated["attempt_number"],
                "package_type": consolidated["package_type"],
                "status_code": consolidated["status_code"],
                "status_description": consolidated["status_description"],
            }
        )
    except Exception as exc:
        logger.error("check_escalation_rules failed: %s", exc)
        tool_log.append(f"TOOL ERROR: check_escalation_rules - {str(exc)[:100]}")
    tool_log.append("TOOL: check_escalation_rules")

    logger.info(
        "fetch_context  customer_id=%s  profile=%s  lockers=%d  chunks=%d  triggers=%s",
        customer_id,
        "ok" if customer_profile else "empty",
        len(locker_availability),
        len(playbook_context),
        escalation_signals.get("has_triggers", "?"),
    )

    return {
        "customer_profile": customer_profile,
        "customer_profile_full": customer_profile_full,
        "locker_availability": locker_availability,
        "playbook_context": playbook_context,
        "escalation_signals": escalation_signals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="preprocessor_node")
def preprocessor_node(
    state: UnifiedAgentState,
    *,
    tools: ToolMaster,
) -> UnifiedAgentState:
    """Deduplicates, consolidates, assembles context, and runs input guardrails.

    Six-step pipeline:

    1. Remove duplicate scan events.
    2. Merge multi-row shipments into a single consolidated event.
    3. Scan driver notes for prompt injection — block before any LLM call.
    4. Check for routine noise — skip all tool calls if detected.
    5. Fetch context via tools (only for non-noise cases).
    6. Scan retrieved playbook chunks for injection.

    Early returns on guardrail triggers or noise detection prevent unnecessary
    LLM calls and tool invocations.
    """
    try:
        view = project_into(state, RouterView)
        tool_log: list[str] = []
        trajectory: list[str] = []
        start = time.time()

        # Step 1: Remove duplicate scan events
        unique_rows = deduplicate_rows(view["raw_rows"])
        tool_log.append("PREPROCESSOR: Deduplicated rows")
        trajectory.append(
            f"preprocessor: {len(view['raw_rows'])} raw rows -> "
            f"{len(unique_rows)} after dedup"
        )

        # Step 2: Merge multi-row shipments into a single consolidated event
        consolidated = consolidate_event(unique_rows, view["raw_rows"])

        # Step 3: Scan driver notes for prompt injection — block before any LLM call
        if scan_inputs_for_injection(consolidated, view["raw_rows"]):
            tool_log.append("GUARDRAIL: Injection detected in delivery input")
            trajectory.append(
                "preprocessor: Guardrail triggered - prompt injection detected"
            )
            output: dict[str, Any] = {
                "consolidated_event": consolidated,
                "customer_profile": {},
                "customer_profile_full": {},
                "locker_availability": [],
                "playbook_context": [],
                "escalation_signals": {},
                "tool_calls_log": tool_log,
                "trajectory_log": trajectory,
                "resolution_revision_count": 0,
                "critic_feedback": "",
                "noise_override": False,
                "guardrail_triggered": True,
                "escalated": True,
                "start_time": start,
                "next_agent": AgentName.FINALIZE,
            }
            logger.warning(
                "preprocessor_node  shipment_id=%s  result=INJECTION_BLOCKED",
                consolidated.get("shipment_id", "?"),
            )
            return merge_back(state, output, RouterView)

        # Step 4: Check for routine noise before making any tool calls
        noise_override = check_noise_override(consolidated)
        if noise_override:
            tool_log.append(
                "PREPROCESSOR: Noise guardrail - routine status with no anomaly"
            )
            trajectory.append(
                f"preprocessor: {consolidated['status_code']} flagged as noise "
                "by guardrail, skipping tool calls"
            )
            output = {
                "consolidated_event": consolidated,
                "customer_profile": {},
                "customer_profile_full": {},
                "locker_availability": [],
                "playbook_context": [],
                "escalation_signals": {},
                "tool_calls_log": tool_log,
                "trajectory_log": trajectory,
                "resolution_revision_count": 0,
                "critic_feedback": "",
                "noise_override": True,
                "guardrail_triggered": False,
                "escalated": False,
                "start_time": start,
                "next_agent": AgentName.PREPROCESSOR,  # orchestrator will handle
            }
            logger.info(
                "preprocessor_node  shipment_id=%s  result=NOISE_OVERRIDE",
                consolidated.get("shipment_id", "?"),
            )
            return merge_back(state, output, RouterView)

        # Step 5: Fetch context via tools (only for non-noise cases)
        context = fetch_context(consolidated, tools, tool_log)

        # Step 6: Scan retrieved playbook chunks for injection
        if scan_chunks_for_injection(context["playbook_context"]):
            tool_log.append("GUARDRAIL: Injection detected in retrieved playbook chunk")
            trajectory.append(
                "preprocessor: Guardrail triggered - injection in RAG chunk"
            )
            context["playbook_context"] = []  # Drop contaminated chunks
            output = {
                "consolidated_event": consolidated,
                **context,
                "tool_calls_log": tool_log,
                "trajectory_log": trajectory,
                "resolution_revision_count": 0,
                "critic_feedback": "",
                "noise_override": False,
                "guardrail_triggered": True,
                "escalated": True,
                "start_time": start,
                "next_agent": AgentName.FINALIZE,
            }
            logger.warning(
                "preprocessor_node  shipment_id=%s  result=RAG_INJECTION_BLOCKED",
                consolidated.get("shipment_id", "?"),
            )
            return merge_back(state, output, RouterView)

        # Normal path — package all context and pass to orchestrator
        output = {
            "consolidated_event": consolidated,
            **context,
            "tool_calls_log": tool_log,
            "trajectory_log": trajectory,
            "resolution_revision_count": 0,
            "critic_feedback": "",
            "noise_override": False,
            "guardrail_triggered": False,
            "escalated": False,
            "start_time": start,
            "next_agent": AgentName.PREPROCESSOR,  # orchestrator will handle
        }

        logger.info(
            "preprocessor_node  shipment_id=%s  result=OK  tools_called=%d",
            consolidated.get("shipment_id", "?"),
            sum(1 for entry in tool_log if entry.startswith("TOOL:")),
        )
        return merge_back(state, output, RouterView)

    except Exception as exc:
        logger.error("preprocessor_node failed: %s", exc, exc_info=True)
        # Fallback: escalate to human review
        fallback: dict[str, Any] = {
            "consolidated_event": {},
            "guardrail_triggered": True,
            "escalated": True,
            "next_agent": AgentName.FINALIZE,
            "trajectory_log": [f"preprocessor: FATAL ERROR — {str(exc)[:200]}"],
            "tool_calls_log": [],
            "start_time": time.time(),
        }
        return merge_back(state, fallback, RouterView)


@traceable(name="orchestrator_node")
def orchestrator_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Central router.  Determines ``next_agent`` based on current state.

    Routing order (purely deterministic — no LLM calls):

    0. Guardrail triggered                          → finalize
    1. Noise override from preprocessor             → finalize (skip LLM)
    2. Resolution not yet run                       → resolution_agent
    3. Resolution done, critic not yet run          → critic_resolution
    4. Critic returned REVISE and under loop limit  → resolution_agent (reset)
    5. Critic returned REVISE and at loop limit     → force ESCALATE
    6. Enforce automatic escalation triggers        (rule engine is authoritative)
    6b. Enforce discretionary escalation triggers
    7. Not an exception                             → finalize
    8. Communication not yet run                    → communication_agent
    9. Communication done, critic not yet run       → critic_communication
    10. All done                                    → finalize
    """
    try:
        view = project_into(state, RouterView)
        updates: dict[str, Any] = {}

        # Helper to append to trajectory without mutating the original list
        traj = list(view.get("trajectory_log") or [])

        # 0. Guardrail triggered — no LLM, force escalation, go to finalize
        if view.get("guardrail_triggered"):
            updates["resolution_output"] = {
                "is_exception": "YES",
                "resolution": "RESCHEDULE",
                "rationale": (
                    "Input flagged by guardrail - prompt injection detected. "
                    "Defaulting to RESCHEDULE with forced escalation for "
                    "human review."
                ),
            }
            updates["escalated"] = True
            updates["next_agent"] = AgentName.FINALIZE
            traj.append(
                "orchestrator: Guardrail triggered, forcing escalation to finalize"
            )
            updates["trajectory_log"] = traj
            logger.info("orchestrator_node  route=FINALIZE  reason=guardrail")
            return merge_back(state, updates, RouterView)

        # 1. Noise override — classify as non-exception, skip all agents
        if view.get("noise_override") and not view.get("resolution_output"):
            status_code = view.get("consolidated_event", {}).get(
                "status_code", "UNKNOWN"
            )
            updates["resolution_output"] = {
                "is_exception": "NO",
                "resolution": "N/A",
                "rationale": (
                    f"Status code {status_code} with routine description. "
                    "No anomaly indicators. Classified as noise by "
                    "preprocessor guardrail."
                ),
            }
            updates["next_agent"] = AgentName.FINALIZE
            traj.append(
                "orchestrator: Noise override from preprocessor, skipping to finalize"
            )
            updates["trajectory_log"] = traj
            logger.info("orchestrator_node  route=FINALIZE  reason=noise_override")
            return merge_back(state, updates, RouterView)

        # 2. Resolution Agent hasn't run yet — send it there
        if not view.get("resolution_output"):
            updates["next_agent"] = AgentName.RESOLUTION
            logger.info("orchestrator_node  route=RESOLUTION  reason=not_yet_run")
            return merge_back(state, updates, RouterView)

        # 3. Resolution done but Critic hasn't validated yet
        if not view.get("critic_resolution_output"):
            updates["next_agent"] = AgentName.CRITIC_RESOLUTION
            logger.info(
                "orchestrator_node  route=CRITIC_RESOLUTION  reason=not_yet_run"
            )
            return merge_back(state, updates, RouterView)

        critic_decision = view["critic_resolution_output"].get("decision")
        max_loops = view.get("max_loops", 2)
        rev_count = view.get("resolution_revision_count", 0)

        # 4. Critic wants a revision and we're under the retry limit
        if critic_decision == "REVISE" and rev_count < max_loops:
            updates["resolution_output"] = {}  # Clear for retry
            updates["critic_resolution_output"] = {}  # Clear for retry
            updates["next_agent"] = AgentName.RESOLUTION
            traj.append(f"orchestrator: REVISE loop {rev_count}/{max_loops}")
            updates["trajectory_log"] = traj
            logger.info(
                "orchestrator_node  route=RESOLUTION  reason=REVISE  loop=%d/%d",
                rev_count,
                max_loops,
            )
            return merge_back(state, updates, RouterView)

        # 5. Revision limit exhausted — force escalate
        if critic_decision == "REVISE" and rev_count >= max_loops:
            if view.get("escalation_signals", {}).get("has_triggers"):
                updates["escalated"] = True
            updates["critic_resolution_output"] = {
                "decision": "ESCALATE",
                "rationale": (
                    "Max revision loops reached. Accepting current "
                    "resolution with escalation."
                ),
            }
            is_exception = (
                view.get("resolution_output", {}).get("is_exception") == "YES"
            )
            updates["next_agent"] = (
                AgentName.COMMUNICATION if is_exception else AgentName.FINALIZE
            )
            traj.append("orchestrator: Max loops reached, forcing ESCALATE")
            updates["trajectory_log"] = traj
            logger.info(
                "orchestrator_node  route=%s  reason=max_loops_exhausted",
                updates["next_agent"],
            )
            return merge_back(state, updates, RouterView)

        # 6. Enforce automatic escalation triggers — rule engine is authoritative
        if view.get("escalation_signals", {}).get("has_triggers"):
            automatic = [
                t
                for t in view["escalation_signals"].get("triggers", [])
                if t.startswith("AUTOMATIC")
            ]
            if (
                automatic
                and view.get("resolution_output", {}).get("is_exception") == "YES"
            ):
                updates["escalated"] = True
                traj.append(
                    f"orchestrator: Forced escalation from rule engine - {automatic}"
                )

        # 6b. Enforce discretionary escalation triggers
        if view.get("escalation_signals", {}).get("has_triggers"):
            discretionary = [
                t
                for t in view["escalation_signals"].get("triggers", [])
                if t.startswith("DISCRETIONARY")
            ]
            if (
                discretionary
                and view.get("resolution_output", {}).get("is_exception") == "YES"
            ):
                updates["escalated"] = True
                traj.append(
                    f"orchestrator: Escalation from discretionary triggers "
                    f"- {discretionary}"
                )

        # 7. Not an exception — no customer message needed, go to finalize
        if view.get("resolution_output", {}).get("is_exception") == "NO":
            updates["next_agent"] = AgentName.FINALIZE
            traj.append("orchestrator: Not an exception, skipping to finalize")
            updates["trajectory_log"] = traj
            logger.info("orchestrator_node  route=FINALIZE  reason=not_exception")
            return merge_back(state, updates, RouterView)

        # 8. Communication Agent hasn't run yet
        if not view.get("communication_output"):
            updates["next_agent"] = AgentName.COMMUNICATION
            if traj != list(view.get("trajectory_log") or []):
                updates["trajectory_log"] = traj
            logger.info("orchestrator_node  route=COMMUNICATION  reason=not_yet_run")
            return merge_back(state, updates, RouterView)

        # 9. Communication done but Critic hasn't validated yet
        if not view.get("critic_communication_output"):
            updates["next_agent"] = AgentName.CRITIC_COMMUNICATION
            if traj != list(view.get("trajectory_log") or []):
                updates["trajectory_log"] = traj
            logger.info(
                "orchestrator_node  route=CRITIC_COMMUNICATION  reason=not_yet_run"
            )
            return merge_back(state, updates, RouterView)

        # 10. Everything complete — finalize
        updates["next_agent"] = AgentName.FINALIZE
        if traj != list(view.get("trajectory_log") or []):
            updates["trajectory_log"] = traj
        logger.info("orchestrator_node  route=FINALIZE  reason=all_complete")
        return merge_back(state, updates, RouterView)

    except Exception as exc:
        logger.error("orchestrator_node failed: %s", exc, exc_info=True)
        fallback: dict[str, Any] = {
            "escalated": True,
            "next_agent": AgentName.FINALIZE,
        }
        traj_fallback = list(state.get("trajectory_log") or [])
        traj_fallback.append(f"orchestrator: FATAL ERROR — {str(exc)[:200]}")
        fallback["trajectory_log"] = traj_fallback
        return merge_back(state, fallback, RouterView)


@traceable(name="finalize_node")
def finalize_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Package final results and record end time.

    Two output paths:

    * **Guardrail-blocked** — produces ``is_exception="BLOCKED"``,
      ``resolution="ESCALATED"``, ``guardrail_blocked=True``.
    * **Normal** — packages resolution, communication, escalation status,
      and revision count from the pipeline state.
    """
    try:
        view = project_into(state, RouterView)
        traj = list(view.get("trajectory_log") or [])

        if view.get("guardrail_triggered"):
            final = {
                "shipment_id": view.get("shipment_id", ""),
                "is_exception": "BLOCKED",
                "resolution": "ESCALATED",
                "escalated": True,
                "tone": "N/A",
                "message": (
                    "This shipment was flagged by the input guardrail "
                    "and requires human review."
                ),
                "revision_count": 0,
                "guardrail_blocked": True,
            }
        else:
            final = {
                "shipment_id": view.get("shipment_id", ""),
                "is_exception": view.get("resolution_output", {}).get(
                    "is_exception", "ERROR"
                ),
                "resolution": view.get("resolution_output", {}).get(
                    "resolution", "ERROR"
                ),
                "escalated": view.get("escalated", False),
                "tone": view.get("communication_output", {}).get("tone_label", "N/A"),
                "message": view.get("communication_output", {}).get(
                    "communication_message", ""
                ),
                "revision_count": view.get("resolution_revision_count", 0),
                "guardrail_blocked": False,
            }

        latency = time.time() - view["start_time"] if view.get("start_time") else 0.0

        traj.append(f"finalize: actions={json.dumps(final)}; latency={latency:.3f}s")

        output: dict[str, Any] = {
            "final_actions": [final],
            "latency_sec": latency,
            "next_agent": "END",
            "trajectory_log": traj,
        }

        logger.info(
            "finalize_node  shipment_id=%s  is_exception=%s  escalated=%s  "
            "latency=%.3fs",
            final["shipment_id"],
            final["is_exception"],
            final["escalated"],
            latency,
        )
        return merge_back(state, output, RouterView)

    except Exception as exc:
        logger.error("finalize_node failed: %s", exc, exc_info=True)
        error_final = {
            "shipment_id": state.get("shipment_id", ""),
            "is_exception": "ERROR",
            "resolution": "ERROR",
            "escalated": True,
            "tone": "N/A",
            "message": f"Pipeline error in finalize: {str(exc)[:200]}",
            "revision_count": 0,
            "guardrail_blocked": False,
        }
        return merge_back(
            state,
            {
                "final_actions": [error_final],
                "latency_sec": 0.0,
                "next_agent": "END",
            },
            RouterView,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════


def _placeholder_node(name: str):
    """Factory for placeholder pass-through nodes.

    Sub-agent nodes (resolution, communication, critics) will be
    implemented in dedicated modules and registered into the graph by
    replacing these placeholders.
    """

    def _node(state: UnifiedAgentState) -> UnifiedAgentState:
        logger.warning(
            "Placeholder node '%s' invoked — implement in dedicated module",
            name,
        )
        return state

    _node.__name__ = name
    return _node


def _route_from_orchestrator(state: UnifiedAgentState) -> str:
    """Deterministic routing based on the orchestrator's ``next_agent`` decision."""
    next_agent = state.get("next_agent", AgentName.FINALIZE)
    # Normalise enum to string for LangGraph edge matching
    return next_agent.value if isinstance(next_agent, AgentName) else str(next_agent)


@traceable(name="build_router_graph")
def build_router_graph(
    tools: ToolMaster,
    *,
    gen_llm: Any = None,
    eval_llm: Any = None,
) -> Any:
    """Build and compile the LangGraph ``StateGraph`` workflow.

    Parameters
    ----------
    tools : ToolMaster
        Initialised tool master — bound into the preprocessor node via
        ``functools.partial`` so that tools flow through without polluting
        the LangGraph state schema.
    gen_llm : ChatOpenAI, optional
        Generation LLM (e.g. gpt-4o-mini).  When provided, the real
        ``resolution_agent_node`` replaces the placeholder.
    eval_llm : ChatOpenAI, optional
        Evaluation LLM (e.g. gpt-4o).  When provided, the real
        ``critic_resolution_node`` and ``critic_communication_node``
        replace their placeholders.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph application ready for ``.invoke()``.
    """
    workflow = StateGraph(UnifiedAgentState)

    # Bind tools into the preprocessor node
    bound_preprocessor = partial(preprocessor_node, tools=tools)
    bound_preprocessor.__name__ = "preprocessor"  # type: ignore[attr-defined]

    # Register nodes
    workflow.add_node("preprocessor", bound_preprocessor)
    workflow.add_node("orchestrator", orchestrator_node)

    # Resolution agent — uses gen_llm
    if gen_llm is not None:
        bound_resolution = partial(resolution_agent_node, gen_llm=gen_llm)
        bound_resolution.__name__ = "resolution_agent"  # type: ignore[attr-defined]
        workflow.add_node("resolution_agent", bound_resolution)
    else:
        workflow.add_node("resolution_agent", _placeholder_node("resolution_agent"))

    # Communication agent — placeholder until dedicated module is implemented
    workflow.add_node("communication_agent", _placeholder_node("communication_agent"))

    # Critic resolution — uses eval_llm
    if eval_llm is not None:
        bound_critic_res = partial(critic_resolution_node, eval_llm=eval_llm)
        bound_critic_res.__name__ = "critic_resolution"  # type: ignore[attr-defined]
        workflow.add_node("critic_resolution", bound_critic_res)
    else:
        workflow.add_node("critic_resolution", _placeholder_node("critic_resolution"))

    # Critic communication — uses eval_llm
    if eval_llm is not None:
        bound_critic_comm = partial(critic_communication_node, eval_llm=eval_llm)
        bound_critic_comm.__name__ = "critic_communication"  # type: ignore[attr-defined]
        workflow.add_node("critic_communication", bound_critic_comm)
    else:
        workflow.add_node(
            "critic_communication", _placeholder_node("critic_communication")
        )

    workflow.add_node("finalize", finalize_node)

    # Entry point
    workflow.set_entry_point("preprocessor")

    # Preprocessor always routes to orchestrator
    workflow.add_edge("preprocessor", "orchestrator")

    # Orchestrator routes conditionally based on state["next_agent"]
    workflow.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            "resolution": "resolution_agent",
            "communication": "communication_agent",
            "critic_resolution": "critic_resolution",
            "critic_communication": "critic_communication",
            "finalize": "finalize",
        },
    )

    # All sub-agent nodes return to the orchestrator for re-routing
    workflow.add_edge("resolution_agent", "orchestrator")
    workflow.add_edge("communication_agent", "orchestrator")
    workflow.add_edge("critic_resolution", "orchestrator")
    workflow.add_edge("critic_communication", "orchestrator")

    # Finalize is the terminal node
    workflow.add_edge("finalize", END)

    # Compile with in-memory checkpointer for state persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info(
        "Router graph compiled  nodes=%d  entry=preprocessor  terminal=finalize",
        len(workflow.nodes),
    )
    return app
