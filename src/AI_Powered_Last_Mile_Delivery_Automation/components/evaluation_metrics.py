"""Evaluation Metrics — multi-agent workflow scoring.

This module quantifies the performance and reliability of the
multi-agent delivery-exception pipeline defined in
``components/multi_agent_workflow.py``.  It is a faithful port of
``notebooks/research_trials.ipynb`` §"Evaluation Metrics" (line 4005)
and §"Aggregated Evaluation Metrics" (line 5175), with four upgrades:

1. Pydantic output models for every metric (validation, JSON export,
   LangSmith-friendly shapes).
2. ``@traceable`` decoration on every public function.
3. A hardened error boundary — one broken metric never takes down a
   whole batch evaluation.
4. Three additional LLMOps-grade metrics the notebook does not have:
   token/cost efficiency, trajectory drift, semantic similarity.

Consumed by
-----------
* ``components/multi_agent_workflow.py`` — post-run scoring.
* Offline batch evaluators / CI regression runs.

Metric catalogue (public API)
-----------------------------
1. ``compute_task_completion``      — 3-way exception / resolution / tone check.
2. ``compute_escalation_accuracy``  — noise-aware escalation match.
3. ``compute_tool_call_accuracy``   — required-tool invocation check.
4. ``compute_coherence_score``      — LLM-as-a-judge trajectory coherence (1–5).
5. ``compute_token_efficiency``     — tokens + cost per successful resolution.
6. ``compute_trajectory_drift``     — log bloat / loop-out detection.
7. ``compute_semantic_similarity``  — final message vs. golden answer cosine.
8. ``evaluate_single_case``         — run all seven on one pipeline result.
9. ``aggregate_results``            — batch-level rollup.
10. ``log_to_langsmith``            — best-effort LangSmith feedback push.

Design notes
------------
* ``eval_llm`` / ``embedder`` are **injected** into metric functions — the
  module never constructs LLMs itself (same DI pattern as the agents).
* The pipeline ``UnifiedAgentState`` stays as a ``TypedDict``; only
  *metric outputs* use Pydantic.
* Noise-aware aggregation: escalation accuracy excludes GT rows whose
  ``should_escalate == "N/A"``.

Optimisation & operations (see plan file for rationale)
-------------------------------------------------------
* **Async batch** — wrap ``evaluate_single_case`` calls in
  ``asyncio.gather`` with a semaphore (5–10) for N-shipment regression
  runs; requires an async ``ainvoke`` on ``eval_llm``.
* **Background evaluation** — decouple scoring from the request path
  via a worker queue so ``run_workflow`` returns immediately and
  LangSmith feedback posts land out-of-band.
* **Golden Set management** — ``ground_truth.csv`` should be PR-reviewed
  and carry a ``version`` column; stratify aggregation by version as
  the set grows.
* **Coherence caching** — cache gpt-4o coherence scores by
  ``(shipment_id, state_hash)``.
* **Cost/latency dashboard** — plot ``avg_tokens_per_case`` +
  ``avg_latency_sec`` over time to catch regressions that accuracy
  metrics miss.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("components.evaluation_metrics")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic result models
# ═══════════════════════════════════════════════════════════════════════════


class TaskCompletionResult(BaseModel):
    """3-criterion pass/fail breakdown for a single shipment."""

    model_config = ConfigDict(extra="forbid")

    exception_correct: bool
    resolution_correct: bool
    tone_correct: bool | None  # None for non-exception / noise cases
    task_complete: bool


class CoherenceResult(BaseModel):
    """LLM-as-a-judge reasoning trajectory coherence score (1–5; 0 = error)."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=5)
    justification: str


class TokenEfficiencyResult(BaseModel):
    """Token / cost accounting for a single pipeline run."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    tokens_per_resolution: float | None  # None when task not complete


class TrajectoryDriftResult(BaseModel):
    """Context-bloat / looping detector for long or unstable runs."""

    model_config = ConfigDict(extra="forbid")

    trajectory_len: int
    tool_calls_len: int
    revision_count: int
    drift_flag: bool


class SemanticSimilarityResult(BaseModel):
    """Cosine similarity of the generated customer message vs. golden answer."""

    model_config = ConfigDict(extra="forbid")

    similarity: float = Field(ge=0.0, le=1.0)
    golden_present: bool


class SingleCaseReport(BaseModel):
    """Aggregate metric report for one pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: str
    task_completion: TaskCompletionResult
    escalation_correct: bool | None
    tool_call_correct: bool
    coherence: CoherenceResult
    token_efficiency: TokenEfficiencyResult
    trajectory_drift: TrajectoryDriftResult
    semantic_similarity: SemanticSimilarityResult
    latency_sec: float | None
    citations: list[str]


class BatchReport(BaseModel):
    """Rollup across a list of ``SingleCaseReport`` entries."""

    model_config = ConfigDict(extra="forbid")

    n: int
    task_completion_rate: float
    exception_detection_rate: float
    resolution_accuracy: float
    tone_accuracy: float
    tool_call_accuracy: float
    escalation_accuracy: float | None
    avg_coherence: float
    avg_latency_sec: float
    avg_tokens_per_case: float | None
    avg_similarity: float | None
    drift_rate: float
    # --- Aggregated-metrics extensions (notebook §5175 port) ---
    median_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    latency_per_agent: dict[str, float] = Field(default_factory=dict)
    failure_count: int = 0
    failure_rate: float = 0.0
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    total_cost_usd: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Default pricing (USD per 1K tokens) — used by compute_token_efficiency.
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.00500, "completion": 0.01500},
}


# ═══════════════════════════════════════════════════════════════════════════
# Notebook ports — core accuracy metrics
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="evaluation.compute_task_completion")
def compute_task_completion(gt: dict, pred: dict) -> TaskCompletionResult:
    """3-way pass/fail across exception / resolution / tone.

    Mirrors notebook lines 4050-4073.  For noise cases
    (``gt["is_exception"] == "NO"``) the tone check is skipped — the
    communication agent is not invoked.
    """
    try:
        gt_is_exc = str(gt.get("is_exception", "")).strip().upper()
        pred_is_exc = (
            str((pred.get("resolution_output") or {}).get("is_exception", ""))
            .strip()
            .upper()
        )

        exception_correct = gt_is_exc == pred_is_exc

        gt_res = str(gt.get("expected_resolution", "")).strip().upper()
        pred_res = (
            str((pred.get("resolution_output") or {}).get("resolution", ""))
            .strip()
            .upper()
        )
        resolution_correct = gt_res == pred_res

        tone_correct: bool | None
        if gt_is_exc == "YES":
            gt_tone = str(gt.get("expected_tone", "")).strip().upper()
            pred_tone = (
                str((pred.get("communication_output") or {}).get("tone_label", ""))
                .strip()
                .upper()
            )
            tone_correct = gt_tone == pred_tone
            task_complete = exception_correct and resolution_correct and tone_correct
        else:
            tone_correct = None
            task_complete = exception_correct and resolution_correct

        return TaskCompletionResult(
            exception_correct=exception_correct,
            resolution_correct=resolution_correct,
            tone_correct=tone_correct,
            task_complete=task_complete,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("compute_task_completion failed: %s", exc)
        return TaskCompletionResult(
            exception_correct=False,
            resolution_correct=False,
            tone_correct=None,
            task_complete=False,
        )


@traceable(name="evaluation.compute_escalation_accuracy")
def compute_escalation_accuracy(gt: dict, pred: dict) -> bool | None:
    """Check if escalation decision matches ground truth.

    Returns ``None`` when ``gt["should_escalate"]`` is neither "YES" nor
    "NO" (noise cases are excluded from escalation aggregation).
    Mirrors notebook lines 4104-4114.
    """
    try:
        gt_esc = str(gt.get("should_escalate", "")).strip().upper()
        if gt_esc not in {"YES", "NO"}:
            return None
        pred_esc = "YES" if bool(pred.get("escalated", False)) else "NO"
        return gt_esc == pred_esc
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_escalation_accuracy failed: %s", exc)
        return None


# Required tool keywords per the notebook 4146-4175 logic.
_CORE_TOOLS: tuple[str, ...] = (
    "lookup_customer_profile",
    "check_locker_availability",
    "search_playbook",
    "check_escalation_rules",
    "resolution_agent",
)
_COMMUNICATION_TOOL: str = "communication_agent"


@traceable(name="evaluation.compute_tool_call_accuracy")
def compute_tool_call_accuracy(gt: dict, pred: dict) -> bool:
    """Check that the right tools were invoked.

    Three paths (mirrors notebook 4146-4175):

    1. **Noise-overridden** — no tool calls allowed (preprocessor
       short-circuited).
    2. **Exception** — all 5 core tools + ``communication_agent``.
    3. **Non-exception, non-noise** — all 5 core tools, and
       ``communication_agent`` must **not** have fired.
    """
    try:
        noise_override = bool(pred.get("noise_override", False))
        tool_calls_log = pred.get("tool_calls_log") or []
        joined = " ".join(str(x) for x in tool_calls_log).lower()

        if noise_override:
            return len(tool_calls_log) == 0

        has_all_core = all(tool.lower() in joined for tool in _CORE_TOOLS)
        has_comm = _COMMUNICATION_TOOL.lower() in joined

        gt_is_exc = str(gt.get("is_exception", "")).strip().upper()
        if gt_is_exc == "YES":
            return has_all_core and has_comm
        return has_all_core and not has_comm
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_tool_call_accuracy failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Coherence (LLM-as-a-judge)
# ---------------------------------------------------------------------------

_COHERENCE_SYSTEM_PROMPT = """You are a rigorous evaluator of multi-agent \
reasoning traces for a last-mile delivery exception-resolution system. \
You will receive a JSON trace summarising one shipment's full pipeline \
execution. Score the *reasoning trajectory coherence* on a 1-5 scale:

5 (Excellent) — all decisions logically consistent, clear rationale chain.
4 (Good)      — mostly consistent with minor gaps.
3 (Adequate)  — some inconsistencies but outcome reasonable.
2 (Poor)      — significant inconsistencies (resolution contradicts event, \
                tone mismatches tier, unjustified critic decisions).
1 (Incoherent) — contradictory or nonsensical.

Evaluate: exception classification match, resolution appropriateness, \
critic-decision justification, revision improvements, tone consistency, \
overall trajectory logic.

Respond with **only** a JSON object of the form:
{"score": <int 1-5>, "justification": "<one-sentence rationale>"}
"""


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` wrappers from an LLM response."""
    t = text.strip()
    if t.startswith("```"):
        # drop opening fence (``` or ```json) and trailing ```
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


@traceable(name="evaluation.compute_coherence_score")
def compute_coherence_score(pred: dict, *, eval_llm: Any) -> CoherenceResult:
    """LLM-as-a-judge 1–5 coherence scoring (ports notebook 4253-4289).

    Sends a sanitised, PII-free subset of the final state to ``eval_llm``
    (typically gpt-4o, temperature=0) and parses its JSON verdict.
    Returns ``CoherenceResult(score=0, justification="error: ...")`` on
    any failure so callers can safely ignore the exception path.
    """
    if eval_llm is None:
        logger.warning("compute_coherence_score: eval_llm is None")
        return CoherenceResult(score=0, justification="error: eval_llm is None")

    try:
        trace = {
            "shipment_id": pred.get("shipment_id", ""),
            "consolidated_event": pred.get("consolidated_event", {}),
            "customer_tier": (pred.get("customer_profile") or {}).get("tier", ""),
            "escalation_signals": pred.get("escalation_signals", {}),
            "resolution_output": pred.get("resolution_output", {}),
            "critic_resolution_output": pred.get("critic_resolution_output", {}),
            "resolution_revision_count": pred.get("resolution_revision_count", 0),
            "communication_output": pred.get("communication_output", {}),
            "critic_communication_output": pred.get("critic_communication_output", {}),
            "trajectory_log": pred.get("trajectory_log", []),
        }

        messages = [
            SystemMessage(content=_COHERENCE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(trace, default=str, indent=2)),
        ]
        response = eval_llm.invoke(messages)
        raw = getattr(response, "content", str(response))
        parsed = json.loads(_strip_code_fence(raw))
        score = int(parsed.get("score", 0))
        if not 1 <= score <= 5:
            score = 0
        return CoherenceResult(
            score=score,
            justification=str(parsed.get("justification", "")),
        )
    except Exception as exc:
        logger.warning("compute_coherence_score failed: %s", exc)
        return CoherenceResult(score=0, justification=f"error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# LLMOps metrics (new — not in notebook)
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="evaluation.compute_token_efficiency")
def compute_token_efficiency(
    pred: dict,
    *,
    model_pricing: dict[str, dict[str, float]] | None = None,
) -> TokenEfficiencyResult:
    """Sum prompt/completion tokens and estimate USD cost.

    Reads ``pred["token_usage_log"]`` — a list of
    ``{"model": str, "prompt_tokens": int, "completion_tokens": int}``
    dicts emitted by each agent (if wired). Returns zeros when the log
    is missing so the metric is safe to call on older traces.
    """
    try:
        pricing = model_pricing or _DEFAULT_PRICING
        log = pred.get("token_usage_log") or []

        prompt_total = 0
        completion_total = 0
        cost = 0.0
        for entry in log:
            if not isinstance(entry, dict):
                continue
            model = str(entry.get("model", "")).strip()
            pt = int(entry.get("prompt_tokens", 0) or 0)
            ct = int(entry.get("completion_tokens", 0) or 0)
            prompt_total += pt
            completion_total += ct
            rates = pricing.get(model, {"prompt": 0.0, "completion": 0.0})
            cost += (pt / 1000.0) * rates["prompt"]
            cost += (ct / 1000.0) * rates["completion"]

        total = prompt_total + completion_total
        task_complete = bool(
            (pred.get("_task_complete_hint") is True)
            or (pred.get("final_actions") and not pred.get("escalated", False))
        )
        tokens_per_resolution = float(total) if task_complete and total > 0 else None

        return TokenEfficiencyResult(
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
            total_tokens=total,
            estimated_cost_usd=round(cost, 6),
            tokens_per_resolution=tokens_per_resolution,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_token_efficiency failed: %s", exc)
        return TokenEfficiencyResult(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            tokens_per_resolution=None,
        )


@traceable(name="evaluation.compute_trajectory_drift")
def compute_trajectory_drift(
    pred: dict,
    *,
    max_log_entries: int = 50,
) -> TrajectoryDriftResult:
    """Detect context bloat / looping.

    Flags as drifted when ``trajectory_log`` exceeds ``max_log_entries``
    or when revision count has hit ``max_loops``.
    """
    try:
        traj_len = len(pred.get("trajectory_log") or [])
        tool_len = len(pred.get("tool_calls_log") or [])
        revisions = int(pred.get("resolution_revision_count", 0) or 0)
        max_loops = int(pred.get("max_loops", 2) or 2)

        drift_flag = traj_len > max_log_entries or (
            revisions > 0 and revisions >= max_loops
        )
        return TrajectoryDriftResult(
            trajectory_len=traj_len,
            tool_calls_len=tool_len,
            revision_count=revisions,
            drift_flag=drift_flag,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_trajectory_drift failed: %s", exc)
        return TrajectoryDriftResult(
            trajectory_len=0,
            tool_calls_len=0,
            revision_count=0,
            drift_flag=False,
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = dot / (na * nb)
    # clamp into [0, 1] (negative cosines are rounded to 0 for the
    # purposes of reporting — embeddings for short messages are nearly
    # always non-negative in practice).
    return max(0.0, min(1.0, sim))


@traceable(name="evaluation.compute_semantic_similarity")
def compute_semantic_similarity(
    pred: dict,
    gt: dict,
    *,
    embedder: Any,
) -> SemanticSimilarityResult:
    """Cosine similarity of generated message vs. golden answer.

    Returns ``(similarity=0.0, golden_present=False)`` when
    ``gt["golden_message"]`` is absent so callers can filter before
    aggregation. The ``embedder`` is expected to expose
    ``embed_query(text) -> list[float]`` (HuggingFace / LangChain API).
    """
    try:
        golden = str(gt.get("golden_message", "") or "").strip()
        if not golden or embedder is None:
            return SemanticSimilarityResult(similarity=0.0, golden_present=False)

        generated = str(
            (pred.get("communication_output") or {}).get("communication_message", "")
            or ""
        ).strip()
        if not generated:
            return SemanticSimilarityResult(similarity=0.0, golden_present=True)

        v_pred = embedder.embed_query(generated)
        v_gold = embedder.embed_query(golden)
        return SemanticSimilarityResult(
            similarity=_cosine(list(v_pred), list(v_gold)),
            golden_present=True,
        )
    except Exception as exc:
        logger.warning("compute_semantic_similarity failed: %s", exc)
        return SemanticSimilarityResult(similarity=0.0, golden_present=False)


# ═══════════════════════════════════════════════════════════════════════════
# Single-case helper & aggregation
# ═══════════════════════════════════════════════════════════════════════════


def _extract_citations(pred: dict) -> list[str]:
    """Return formatted ``Page N`` citations from ``playbook_context``."""
    try:
        chunks = pred.get("playbook_context") or []
        pages: list[str] = []
        for chunk in chunks:
            meta = (chunk or {}).get("metadata") or {}
            page = meta.get("page")
            if page is None:
                page = (chunk or {}).get("page")
            if page is not None:
                label = (
                    f"Page {int(page) + 1}" if isinstance(page, int) else f"Page {page}"
                )
                if label not in pages:
                    pages.append(label)
        return pages
    except Exception:
        return []


@traceable(name="evaluation.evaluate_single_case")
def evaluate_single_case(
    pred: dict,
    gt: dict,
    *,
    eval_llm: Any,
    embedder: Any | None = None,
) -> SingleCaseReport:
    """Run every metric on one pipeline result and return a unified report.

    Each inner call is wrapped in ``try/except`` via the metric functions
    themselves, so this helper is safe to call from batch loops: one
    broken case will not crash the batch.
    """
    shipment_id = str(pred.get("shipment_id", "") or gt.get("shipment_id", ""))
    task_completion = compute_task_completion(gt, pred)
    escalation_correct = compute_escalation_accuracy(gt, pred)
    tool_call_correct = compute_tool_call_accuracy(gt, pred)
    # Hint downstream cost metric about successful completion
    pred_with_hint = {**pred, "_task_complete_hint": task_completion.task_complete}
    coherence = compute_coherence_score(pred, eval_llm=eval_llm)
    token_eff = compute_token_efficiency(pred_with_hint)
    drift = compute_trajectory_drift(pred)
    similarity = compute_semantic_similarity(pred, gt, embedder=embedder)
    citations = _extract_citations(pred)
    latency_sec = pred.get("latency_sec")
    latency_sec = float(latency_sec) if isinstance(latency_sec, (int, float)) else None

    return SingleCaseReport(
        shipment_id=shipment_id,
        task_completion=task_completion,
        escalation_correct=escalation_correct,
        tool_call_correct=tool_call_correct,
        coherence=coherence,
        token_efficiency=token_eff,
        trajectory_drift=drift,
        semantic_similarity=similarity,
        latency_sec=latency_sec,
        citations=citations,
    )


def _rate(num: int, den: int) -> float:
    return (num / den) if den > 0 else 0.0


_FAILURE_BUCKETS: tuple[str, ...] = (
    "exception_misclass",
    "wrong_resolution",
    "wrong_tone",
    "wrong_escalation",
    "wrong_tools",
    "drift",
    "low_coherence",
)


@traceable(name="evaluation.compute_failure_categories")
def compute_failure_categories(
    reports: list[SingleCaseReport],
) -> dict[str, int]:
    """Bucket failures by category; one report can hit multiple buckets.

    Buckets
    -------
    * ``exception_misclass`` — YES/NO classification mismatch.
    * ``wrong_resolution``  — resolution label mismatch.
    * ``wrong_tone``        — tone label mismatch (eligible reports only).
    * ``wrong_escalation``  — escalation decision mismatch (eligible only).
    * ``wrong_tools``       — required-tools check failed.
    * ``drift``             — trajectory-drift flag set.
    * ``low_coherence``     — LLM-judge coherence score < 3.
    """
    counts: dict[str, int] = {b: 0 for b in _FAILURE_BUCKETS}
    try:
        for r in reports:
            if not r.task_completion.exception_correct:
                counts["exception_misclass"] += 1
            if not r.task_completion.resolution_correct:
                counts["wrong_resolution"] += 1
            if (
                r.task_completion.tone_correct is not None
                and not r.task_completion.tone_correct
            ):
                counts["wrong_tone"] += 1
            if r.escalation_correct is False:
                counts["wrong_escalation"] += 1
            if not r.tool_call_correct:
                counts["wrong_tools"] += 1
            if r.trajectory_drift.drift_flag:
                counts["drift"] += 1
            if 0 < r.coherence.score < 3:
                counts["low_coherence"] += 1
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_failure_categories failed: %s", exc)
    return counts


_LATENCY_RE = re.compile(r"latency\s*=\s*([0-9]+\.?[0-9]*)\s*s", re.IGNORECASE)
_ELAPSED_RE = re.compile(r"elapsed\s*=\s*([0-9]+\.?[0-9]*)\s*s", re.IGNORECASE)


@traceable(name="evaluation.compute_latency_per_agent")
def compute_latency_per_agent(
    final_states: list[dict],
) -> dict[str, float]:
    """Best-effort mean wall-clock seconds per agent node.

    Parses ``trajectory_log`` lines of the form
    ``"<agent>: ... latency=X.Xs"`` or ``"<agent>: elapsed=X.Xs"``.
    Returns ``{}`` when traces lack per-node timings so callers can
    filter the result.
    """
    buckets: dict[str, list[float]] = {}
    try:
        for state in final_states or []:
            log = (state or {}).get("trajectory_log") or []
            for entry in log:
                line = str(entry)
                colon = line.find(":")
                if colon <= 0:
                    continue
                agent = line[:colon].strip()
                m = _LATENCY_RE.search(line) or _ELAPSED_RE.search(line)
                if not m:
                    continue
                try:
                    buckets.setdefault(agent, []).append(float(m.group(1)))
                except ValueError:
                    continue
    except Exception as exc:  # pragma: no cover
        logger.warning("compute_latency_per_agent failed: %s", exc)
        return {}
    return {agent: statistics.mean(vals) for agent, vals in buckets.items() if vals}


@traceable(name="evaluation.aggregate_results")
def aggregate_results(
    reports: list[SingleCaseReport],
    *,
    final_states: list[dict] | None = None,
) -> BatchReport:
    """Batch-level rollup (ports notebook inline block 5217-5248).

    Extended with median/p95 latency, per-agent latency breakdown,
    failure-rate analysis, and total cost. The ``final_states`` kwarg
    is optional — when omitted, ``latency_per_agent`` stays ``{}``.
    """
    n = len(reports)
    if n == 0:
        empty = BatchReport(
            n=0,
            task_completion_rate=0.0,
            exception_detection_rate=0.0,
            resolution_accuracy=0.0,
            tone_accuracy=0.0,
            tool_call_accuracy=0.0,
            escalation_accuracy=None,
            avg_coherence=0.0,
            avg_latency_sec=0.0,
            avg_tokens_per_case=None,
            avg_similarity=None,
            drift_rate=0.0,
        )
        logger.info("aggregate_results: empty batch")
        return empty

    task_complete = sum(1 for r in reports if r.task_completion.task_complete)
    exc_correct = sum(1 for r in reports if r.task_completion.exception_correct)
    res_correct = sum(1 for r in reports if r.task_completion.resolution_correct)
    tone_eligible = [r for r in reports if r.task_completion.tone_correct is not None]
    tone_correct = sum(1 for r in tone_eligible if r.task_completion.tone_correct)
    tool_correct = sum(1 for r in reports if r.tool_call_correct)

    esc_eligible = [r for r in reports if r.escalation_correct is not None]
    esc_correct = sum(1 for r in esc_eligible if r.escalation_correct)
    escalation_accuracy = (
        _rate(esc_correct, len(esc_eligible)) if esc_eligible else None
    )

    avg_coherence = statistics.mean(r.coherence.score for r in reports)
    latencies = [r.latency_sec for r in reports if r.latency_sec is not None]
    avg_latency_sec = statistics.mean(latencies) if latencies else 0.0

    tokens = [
        r.token_efficiency.total_tokens
        for r in reports
        if r.token_efficiency.total_tokens > 0
    ]
    avg_tokens_per_case = statistics.mean(tokens) if tokens else None

    sims = [
        r.semantic_similarity.similarity
        for r in reports
        if r.semantic_similarity.golden_present
    ]
    avg_similarity = statistics.mean(sims) if sims else None

    drift_rate = _rate(sum(1 for r in reports if r.trajectory_drift.drift_flag), n)

    # --- Aggregated-metrics extensions ---
    median_latency_sec = statistics.median(latencies) if latencies else 0.0
    if len(latencies) >= 20:
        p95_latency_sec = statistics.quantiles(latencies, n=20)[18]
    elif latencies:
        p95_latency_sec = max(latencies)
    else:
        p95_latency_sec = 0.0

    total_cost_usd = round(
        sum(r.token_efficiency.estimated_cost_usd for r in reports), 6
    )
    failure_count = sum(1 for r in reports if not r.task_completion.task_complete)
    failure_rate = _rate(failure_count, n)
    failure_breakdown = compute_failure_categories(reports)
    latency_per_agent = compute_latency_per_agent(final_states) if final_states else {}

    batch = BatchReport(
        n=n,
        task_completion_rate=_rate(task_complete, n),
        exception_detection_rate=_rate(exc_correct, n),
        resolution_accuracy=_rate(res_correct, n),
        tone_accuracy=_rate(tone_correct, len(tone_eligible)) if tone_eligible else 0.0,
        tool_call_accuracy=_rate(tool_correct, n),
        escalation_accuracy=escalation_accuracy,
        avg_coherence=avg_coherence,
        avg_latency_sec=avg_latency_sec,
        avg_tokens_per_case=avg_tokens_per_case,
        avg_similarity=avg_similarity,
        drift_rate=drift_rate,
        median_latency_sec=median_latency_sec,
        p95_latency_sec=p95_latency_sec,
        latency_per_agent=latency_per_agent,
        failure_count=failure_count,
        failure_rate=failure_rate,
        failure_breakdown=failure_breakdown,
        total_cost_usd=total_cost_usd,
    )
    logger.info("Batch evaluation complete: %s", batch.model_dump())
    return batch


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printers (notebook-parity console output)
# ═══════════════════════════════════════════════════════════════════════════


def print_batch_report(batch_report: BatchReport) -> None:
    """Print the aggregate metrics banner (ports notebook 5217-5248).

    Mirrors the notebook's ``AGGREGATE EVALUATION METRICS`` box so
    interactive and CI output look identical. Also emits the full
    ``model_dump`` at INFO for log aggregators.
    """
    n = batch_report.n
    bar = "=" * 60
    print(bar)
    print("AGGREGATE EVALUATION METRICS")
    print(bar)
    if n == 0:
        print("(empty batch)")
        print(bar)
        logger.info("print_batch_report: %s", batch_report.model_dump())
        return

    tcc = int(round(batch_report.task_completion_rate * n))
    exc = int(round(batch_report.exception_detection_rate * n))
    res = int(round(batch_report.resolution_accuracy * n))
    tool = int(round(batch_report.tool_call_accuracy * n))
    print(f"Task Completion:      {tcc}/{n} ({batch_report.task_completion_rate:.0%})")
    print(
        f"  Exception Detection:  {exc}/{n} ({batch_report.exception_detection_rate:.0%})"
    )
    print(f"  Resolution Accuracy:  {res}/{n} ({batch_report.resolution_accuracy:.0%})")
    print(f"  Tone Accuracy:        {batch_report.tone_accuracy:.0%}")
    print(f"Tool Call Accuracy:   {tool}/{n} ({batch_report.tool_call_accuracy:.0%})")
    if batch_report.escalation_accuracy is None:
        print("Escalation Accuracy:  N/A (no applicable cases)")
    else:
        print(f"Escalation Accuracy:  {batch_report.escalation_accuracy:.0%}")
    print(f"Avg Coherence Score:  {batch_report.avg_coherence:.2f}/5")
    print(f"Avg Latency:          {batch_report.avg_latency_sec:.2f}s")
    print(f"Median Latency:       {batch_report.median_latency_sec:.2f}s")
    print(f"P95 Latency:          {batch_report.p95_latency_sec:.2f}s")
    print(
        f"Failure Rate:         {batch_report.failure_count}/{n} "
        f"({batch_report.failure_rate:.0%})"
    )
    if batch_report.failure_breakdown:
        nonzero = {k: v for k, v in batch_report.failure_breakdown.items() if v}
        if nonzero:
            print(f"  Failure Breakdown:  {nonzero}")
    if batch_report.latency_per_agent:
        rendered = {k: round(v, 3) for k, v in batch_report.latency_per_agent.items()}
        print(f"Latency per Agent:    {rendered}")
    if batch_report.avg_tokens_per_case is not None:
        print(f"Avg Tokens/Case:      {batch_report.avg_tokens_per_case:.0f}")
    print(f"Total Cost (USD):     ${batch_report.total_cost_usd:.4f}")
    print(f"Drift Rate:           {batch_report.drift_rate:.0%}")
    print(bar)
    logger.info("print_batch_report: %s", batch_report.model_dump())


# ═══════════════════════════════════════════════════════════════════════════
# LangSmith integration (best-effort)
# ═══════════════════════════════════════════════════════════════════════════


def log_to_langsmith(report: SingleCaseReport, run_id: str | None = None) -> None:
    """Push per-metric scores to LangSmith as feedback (silent no-op on failure).

    Wraps ``langsmith.Client().create_feedback`` — if LangSmith is not
    configured (missing ``LANGCHAIN_API_KEY``) or the client raises, we
    log at DEBUG and return. This keeps LangSmith a soft dependency.
    """
    try:
        from langsmith import Client  # local import — optional dep

        client = Client()
        feedback_map: dict[str, float] = {
            "task_complete": float(report.task_completion.task_complete),
            "tool_call_correct": float(report.tool_call_correct),
            "coherence": float(report.coherence.score),
            "estimated_cost_usd": float(report.token_efficiency.estimated_cost_usd),
            "total_tokens": float(report.token_efficiency.total_tokens),
            "drift_flag": float(report.trajectory_drift.drift_flag),
        }
        if report.escalation_correct is not None:
            feedback_map["escalation_correct"] = float(report.escalation_correct)
        if report.semantic_similarity.golden_present:
            feedback_map["semantic_similarity"] = report.semantic_similarity.similarity
        if report.latency_sec is not None:
            feedback_map["latency_sec"] = report.latency_sec

        if run_id is None:
            logger.debug("log_to_langsmith: no run_id provided; skipping feedback push")
            return

        for key, value in feedback_map.items():
            client.create_feedback(run_id=run_id, key=key, score=value)
    except Exception as exc:
        logger.debug("log_to_langsmith no-op: %s", exc)


__all__ = [
    "TaskCompletionResult",
    "CoherenceResult",
    "TokenEfficiencyResult",
    "TrajectoryDriftResult",
    "SemanticSimilarityResult",
    "SingleCaseReport",
    "BatchReport",
    "compute_task_completion",
    "compute_escalation_accuracy",
    "compute_tool_call_accuracy",
    "compute_coherence_score",
    "compute_token_efficiency",
    "compute_trajectory_drift",
    "compute_semantic_similarity",
    "evaluate_single_case",
    "aggregate_results",
    "compute_failure_categories",
    "compute_latency_per_agent",
    "print_batch_report",
    "log_to_langsmith",
]
