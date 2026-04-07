"""Multi-Agent Workflow — centralized LangGraph orchestration.

This module is the **single entry point** that assembles the compiled
``StateGraph`` for the Last-Mile Delivery Automation pipeline and
exposes a :func:`run_workflow` convenience for executing that graph on
one shipment.  It supersedes the older ``build_router_graph`` helper in
[router_agent.py](../agents/router_agent.py) (which predates
``communication_agent.py`` and therefore uses a placeholder for the
communication node) and wires every real agent into place.

Pipeline topology
-----------------
Hub-and-spoke routing managed by a purely deterministic
``orchestrator_node`` (no LLM calls, no hallucinated routing):

    preprocessor → orchestrator ⇄ { resolution_agent, critic_resolution,
                                     communication_agent, critic_communication }
                                → finalize → END

Per-agent logic lives in the sibling modules — this file only owns
**assembly** and **external invocation**:

* [router_agent.py](../agents/router_agent.py) — ``preprocessor_node``,
  ``orchestrator_node``, ``finalize_node`` and guardrails.
* [resolution_agent.py](../agents/resolution_agent.py) — classification
  + resolution selection.
* [communication_agent.py](../agents/communication_agent.py) — PII-aware
  customer notification generation.
* [critic_agent.py](../agents/critic_agent.py) — the two validator
  nodes that gate the resolution / communication outputs.

Dependency injection
--------------------
External runtime dependencies — the tool master and the two LLM
clients — are bound into their respective nodes via
``functools.partial`` at graph-build time, identical to how each agent
module registers itself.  The ``UnifiedAgentState`` schema is *not*
polluted with these runtime handles.

Observability
-------------
Every public function is decorated with :func:`langsmith.traceable` so
assembly and per-shipment runs surface as distinct spans in LangSmith,
and a module-level logger (``components.multi_agent_workflow``) emits
structured records for compile / success / failure paths.

Optimisation suggestions
------------------------
* **Parallelism in ``preprocessor_node``** — the five tool calls inside
  ``fetch_context`` are independent; wrapping them in
  ``asyncio.gather`` and promoting the node to ``async def`` would cut
  preprocessor latency by ~60%.  The orchestrator itself is inherently
  sequential (state-machine pattern) — no parallelism opportunity there.
* **Memory / context management** — ``trajectory_log`` and
  ``tool_calls_log`` grow unbounded across revision loops.  Trim them
  inside ``finalize_node`` (e.g. keep the last 50 entries) or expose a
  ``max_log_entries`` knob so long-running threads do not bloat the
  checkpointer store.
* **Fault tolerance / HITL** — when ``escalated=True`` *before*
  ``finalize``, route to a pre-finalize review node compiled with
  ``interrupt_before=["finalize"]``.  Combined with a persistent
  checkpointer (SQLite/Postgres instead of ``MemorySaver``) this gives
  pausable, resumable runs for human approval of customer-facing copy.
* **Retry on transient graph failures** — wrap ``app.invoke`` in
  :func:`run_workflow` with ``tenacity`` (exponential backoff,
  max 2 retries) for ``TimeoutError`` / connection errors only.
  LLM-level retries already happen inside each agent node.
* **Checkpointer swap** — ``build_workflow`` already accepts a
  ``checkpointer`` argument so production deployments can drop in
  ``SqliteSaver`` / ``PostgresSaver`` without touching this module.
* **Decision caching** — ``check_noise_override`` and the escalation
  rule engine in ``preprocessor_node`` are deterministic.  An
  ``functools.lru_cache`` keyed on
  ``(status_code, description_hash)`` eliminates redundant evaluation
  during batch runs.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from langsmith import traceable

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import ToolMaster
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    AgentName,
    UnifiedAgentState,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.router_agent import (
    finalize_node,
    orchestrator_node,
    preprocessor_node,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.resolution_agent import (
    resolution_agent_node,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.communication_agent import (
    communication_agent_node,
)
from AI_Powered_Last_Mile_Delivery_Automation.agents.critic_agent import (
    critic_communication_node,
    critic_resolution_node,
)

logger = get_module_logger("components.multi_agent_workflow")

__all__ = [
    "sanitize_initial_state",
    "build_initial_state",
    "build_workflow",
    "run_workflow",
]


# ═══════════════════════════════════════════════════════════════════════════
# Internal — edge routing
# ═══════════════════════════════════════════════════════════════════════════


def _route_from_orchestrator(state: UnifiedAgentState) -> str:
    """Deterministic conditional-edge selector.

    Reads ``state["next_agent"]`` (set by ``orchestrator_node``) and
    returns the string edge key LangGraph matches against the mapping
    passed to ``add_conditional_edges``.
    """
    next_agent = state.get("next_agent", AgentName.FINALIZE)
    return next_agent.value if isinstance(next_agent, AgentName) else str(next_agent)


# ═══════════════════════════════════════════════════════════════════════════
# Input sanitisation (structural — domain guardrails live in preprocessor)
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="sanitize_initial_state")
def sanitize_initial_state(
    shipment_id: Any,
    raw_rows: Any,
    *,
    max_loops: Any = 2,
) -> tuple[str, list[dict], int]:
    """Coerce and validate workflow-entry inputs.

    This runs **before** the LangGraph app is invoked and is the
    workflow's structural boundary check — it complements (does not
    duplicate) the injection / noise scans that already run inside
    ``preprocessor_node``.

    Responsibilities
    ----------------
    * Coerce ``shipment_id`` to a stripped ``str``; reject empty.
    * Coerce ``raw_rows`` to ``list[dict]``; drop non-dict entries;
      reject an empty list.
    * Clamp ``max_loops`` into ``[1, 5]``.

    Raises
    ------
    ValueError
        When ``shipment_id`` is empty / non-string-coercible or
        ``raw_rows`` contains no dict entries.

    Returns
    -------
    tuple[str, list[dict], int]
        ``(clean_shipment_id, clean_rows, clean_max_loops)``.
    """
    # shipment_id
    try:
        clean_id = str(shipment_id).strip() if shipment_id is not None else ""
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"shipment_id not coercible to str: {exc!r}") from exc
    if not clean_id:
        raise ValueError("shipment_id must be a non-empty string")

    # raw_rows
    if not isinstance(raw_rows, list):
        raise ValueError(f"raw_rows must be a list, got {type(raw_rows).__name__}")
    clean_rows: list[dict] = []
    dropped = 0
    for row in raw_rows:
        if isinstance(row, dict):
            clean_rows.append(row)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "sanitize_initial_state  dropped %d non-dict row(s) for shipment_id=%s",
            dropped,
            clean_id,
        )
    if not clean_rows:
        raise ValueError(
            f"raw_rows contained no dict entries for shipment_id={clean_id}"
        )

    # max_loops
    try:
        ml_int = int(max_loops)
    except (TypeError, ValueError):
        logger.warning(
            "sanitize_initial_state  max_loops=%r invalid, defaulting to 2",
            max_loops,
        )
        ml_int = 2
    clamped = max(1, min(5, ml_int))
    if clamped != ml_int:
        logger.warning(
            "sanitize_initial_state  max_loops=%d clamped to %d", ml_int, clamped
        )

    return clean_id, clean_rows, clamped


# ═══════════════════════════════════════════════════════════════════════════
# Initial-state construction
# ═══════════════════════════════════════════════════════════════════════════


def build_initial_state(
    shipment_id: Any,
    raw_rows: Any,
    *,
    max_loops: int = 2,
) -> UnifiedAgentState:
    """Return a fully-populated initial ``UnifiedAgentState``.

    Every field of the ``TypedDict(total=False)`` schema is populated
    with its empty-default so downstream nodes that read with
    ``state.get(...)`` never see a ``KeyError`` even on the first pass.
    Inputs are run through :func:`sanitize_initial_state` so this helper
    is safe to call directly from HTTP handlers.
    """
    clean_id, clean_rows, clean_loops = sanitize_initial_state(
        shipment_id, raw_rows, max_loops=max_loops
    )

    initial: UnifiedAgentState = {  # type: ignore[typeddict-item]
        # Input
        "shipment_id": clean_id,
        "raw_rows": clean_rows,
        "max_loops": clean_loops,
        # Preprocessor outputs
        "consolidated_event": {},
        "customer_profile": {},
        "customer_profile_full": {},
        "locker_availability": [],
        "playbook_context": [],
        "escalation_signals": {},
        "noise_override": False,
        "guardrail_triggered": False,
        # Resolution
        "resolution_output": {},
        # Critic resolution
        "critic_resolution_output": {},
        "resolution_revision_count": 0,
        "critic_feedback": "",
        # Communication
        "communication_output": {},
        # Critic communication
        "critic_communication_output": {},
        # Routing
        "next_agent": "",
        # Final
        "escalated": False,
        "tool_calls_log": [],
        "trajectory_log": [],
        "start_time": None,
        "latency_sec": None,
        "final_actions": [],
    }
    return initial


# ═══════════════════════════════════════════════════════════════════════════
# Graph builder — canonical workflow assembly
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="build_workflow")
def build_workflow(
    tools: ToolMaster,
    *,
    gen_llm: Any,
    eval_llm: Any,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the Last-Mile Delivery multi-agent ``StateGraph``.

    Parameters
    ----------
    tools : ToolMaster
        Initialised tool master — bound into the preprocessor node via
        ``functools.partial`` so it flows through without polluting the
        LangGraph state schema.
    gen_llm : ChatOpenAI
        Generation LLM (e.g. gpt-4o-mini) used by the resolution and
        communication agents.
    eval_llm : ChatOpenAI
        Evaluation LLM (e.g. gpt-4o) used by both critic nodes.
    checkpointer : BaseCheckpointSaver, optional
        LangGraph checkpointer. Defaults to an in-process
        :class:`MemorySaver`.  Swap in ``SqliteSaver`` / ``PostgresSaver``
        for persistent / multi-process deployments.

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph application ready for ``.invoke()`` /
        ``.stream()``.
    """
    workflow: StateGraph = StateGraph(UnifiedAgentState)

    # Bind runtime dependencies into each node via functools.partial so
    # LangGraph's state schema stays clean and node functions remain
    # directly unit-testable.
    bound_preprocessor = partial(preprocessor_node, tools=tools)
    bound_preprocessor.__name__ = "preprocessor"  # type: ignore[attr-defined]

    bound_resolution = partial(resolution_agent_node, gen_llm=gen_llm)
    bound_resolution.__name__ = "resolution_agent"  # type: ignore[attr-defined]

    bound_communication = partial(communication_agent_node, gen_llm=gen_llm)
    bound_communication.__name__ = "communication_agent"  # type: ignore[attr-defined]

    bound_critic_res = partial(critic_resolution_node, eval_llm=eval_llm)
    bound_critic_res.__name__ = "critic_resolution"  # type: ignore[attr-defined]

    bound_critic_comm = partial(critic_communication_node, eval_llm=eval_llm)
    bound_critic_comm.__name__ = "critic_communication"  # type: ignore[attr-defined]

    # Register nodes
    workflow.add_node("preprocessor", bound_preprocessor)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("resolution_agent", bound_resolution)
    workflow.add_node("communication_agent", bound_communication)
    workflow.add_node("critic_resolution", bound_critic_res)
    workflow.add_node("critic_communication", bound_critic_comm)
    workflow.add_node("finalize", finalize_node)

    # Entry point
    workflow.set_entry_point("preprocessor")

    # preprocessor → orchestrator (always)
    workflow.add_edge("preprocessor", "orchestrator")

    # orchestrator → (conditional routing)
    workflow.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            AgentName.RESOLUTION.value: "resolution_agent",
            AgentName.COMMUNICATION.value: "communication_agent",
            AgentName.CRITIC_RESOLUTION.value: "critic_resolution",
            AgentName.CRITIC_COMMUNICATION.value: "critic_communication",
            AgentName.FINALIZE.value: "finalize",
        },
    )

    # All sub-agents return to the orchestrator for re-routing
    workflow.add_edge("resolution_agent", "orchestrator")
    workflow.add_edge("communication_agent", "orchestrator")
    workflow.add_edge("critic_resolution", "orchestrator")
    workflow.add_edge("critic_communication", "orchestrator")

    # finalize is terminal
    workflow.add_edge("finalize", END)

    effective_checkpointer = checkpointer or MemorySaver()
    app = workflow.compile(checkpointer=effective_checkpointer)

    logger.info(
        "Workflow compiled  nodes=%d  gen_llm=%s  eval_llm=%s  checkpointer=%s",
        len(workflow.nodes),
        type(gen_llm).__name__,
        type(eval_llm).__name__,
        type(effective_checkpointer).__name__,
    )
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Single-shipment invocation wrapper (global error boundary)
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="run_workflow")
def run_workflow(
    app: Any,
    shipment_id: Any,
    raw_rows: Any,
    *,
    max_loops: int = 2,
    thread_id: str | None = None,
) -> UnifiedAgentState:
    """Execute the compiled workflow for a single shipment.

    Wraps initial-state construction, the ``app.invoke`` call and
    top-level error handling so callers never have to deal with
    exceptions thrown from inside the graph.

    Parameters
    ----------
    app : CompiledStateGraph
        Result of :func:`build_workflow`.
    shipment_id : str
        Logical shipment identifier; also used as the default
        checkpointer ``thread_id``.
    raw_rows : list[dict]
        Raw delivery-log rows for this shipment.
    max_loops : int, optional
        Maximum number of resolution REVISE loops before forced
        escalation. Defaults to 2.
    thread_id : str, optional
        Checkpointer thread namespace. Defaults to ``shipment_id``.

    Returns
    -------
    UnifiedAgentState
        The final state dict.  On fatal error the returned dict is
        synthesised: ``escalated=True``, a ``FATAL`` entry appended to
        ``final_actions`` / ``trajectory_log``, and the elapsed
        wall-clock latency recorded — so downstream consumers have a
        uniform contract regardless of success or failure.
    """
    start = time.time()
    try:
        initial_state = build_initial_state(shipment_id, raw_rows, max_loops=max_loops)
    except ValueError as exc:
        logger.error(
            "run_workflow  rejected input for shipment_id=%r: %s",
            shipment_id,
            exc,
        )
        return {  # type: ignore[return-value]
            "shipment_id": str(shipment_id) if shipment_id is not None else "",
            "raw_rows": raw_rows if isinstance(raw_rows, list) else [],
            "escalated": True,
            "final_actions": [
                {
                    "action": "FATAL",
                    "message": f"workflow input rejected: {exc}",
                }
            ],
            "trajectory_log": [f"run_workflow: input validation failed — {exc}"],
            "tool_calls_log": [],
            "latency_sec": round(time.time() - start, 4),
            "next_agent": "END",
        }

    clean_shipment_id = initial_state.get("shipment_id", "")
    effective_thread = thread_id or clean_shipment_id
    config = {"configurable": {"thread_id": effective_thread}}

    try:
        result: UnifiedAgentState = app.invoke(initial_state, config=config)
    except Exception as exc:
        elapsed = round(time.time() - start, 4)
        logger.error(
            "run_workflow  fatal error  shipment_id=%s  thread_id=%s  elapsed=%.3fs",
            clean_shipment_id,
            effective_thread,
            elapsed,
            exc_info=True,
        )
        traj = list(initial_state.get("trajectory_log") or [])
        traj.append(f"run_workflow: FATAL ERROR — {str(exc)[:200]}")
        return {  # type: ignore[return-value]
            **initial_state,
            "escalated": True,
            "final_actions": [
                {
                    "action": "FATAL",
                    "message": f"workflow error — {str(exc)[:200]}",
                }
            ],
            "trajectory_log": traj,
            "latency_sec": elapsed,
            "next_agent": "END",
        }

    elapsed = round(time.time() - start, 4)
    logger.info(
        "run_workflow  complete  shipment_id=%s  next_agent=%s  "
        "escalated=%s  trajectory_entries=%d  latency=%.3fs",
        clean_shipment_id,
        result.get("next_agent"),
        result.get("escalated"),
        len(result.get("trajectory_log") or []),
        elapsed,
    )
    return result
