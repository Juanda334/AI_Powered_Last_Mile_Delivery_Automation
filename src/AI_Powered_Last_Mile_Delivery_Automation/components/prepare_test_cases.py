"""Test-case preparation & batch evaluation harness.

Ports notebook §"Prepare Ground Truth" (line 4400) and
§"Helper Function for Test Case Runs" (line 4320) from
``notebooks/research_trials.ipynb`` into a production-grade batch runner.

Responsibilities
----------------
1. Load ``ground_truth.csv`` into typed ``GroundTruthCase`` rows and
   consolidate multi-row shipments.
2. Load ``delivery_logs.csv`` and group rows by shipment ID.
3. Pair the two into ``TestCase`` objects.
4. Execute every case through the compiled multi-agent workflow and
   score it via ``components.evaluation_metrics.evaluate_single_case``.
5. Persist/reload the resulting ``TestRunBatch`` as JSON for versioning
   and regression testing.

Design notes
------------
* ``app``, ``eval_llm``, ``embedder`` are **injected** (DI) — the runner
  never constructs LLMs itself.
* ``run_test_case`` is wrapped in ``try/except`` so one broken shipment
  never aborts a 100-case batch; the failing record carries
  ``error=str(exc)`` and neutral metric values.
* Every public runner function is ``@traceable`` so LangSmith's
  "Batch Runs" view lights up automatically.

Optimisation & operations (captured here, not implemented now)
--------------------------------------------------------------
* **Parallelisation** — ``run_batch`` is LLM-bound; wrap
  ``run_test_case`` in ``asyncio.gather`` with a semaphore (5-8) for
  10× speedup on 100-case regressions. Requires ``app.ainvoke`` +
  async ``eval_llm.ainvoke``.
* **Parameterised testing** — all models are DI args, so a pytest
  parametrize sweep over ``[("gpt-4o-mini","gpt-4o"),
  ("claude-haiku","claude-opus"), …]`` can score the same golden set
  across providers.
* **CI integration** — ``.github/workflows/eval.yml`` should checkout,
  install, run the CLI here, parse the persisted
  ``batch_<ts>.json``, post ``BatchReport.model_dump()`` as a PR
  comment, and fail on ``task_completion_rate`` regressions > 5%.
* **Run caching** — key batch results by
  ``(git_sha, gt_csv_hash, model_versions)`` so unchanged CI runs
  return cached numbers in < 1 s.
* **Golden-set versioning** — add a ``version`` column to
  ``ground_truth.csv`` and stratify ``aggregate_results`` by version
  once the set grows heterogeneous. PRs touching the CSV should need
  two reviewers so baselines don't silently drift.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

from AI_Powered_Last_Mile_Delivery_Automation.components.evaluation_metrics import (
    CoherenceResult,
    SemanticSimilarityResult,
    SingleCaseReport,
    TaskCompletionResult,
    TokenEfficiencyResult,
    TrajectoryDriftResult,
    evaluate_single_case,
)
from AI_Powered_Last_Mile_Delivery_Automation.components.multi_agent_workflow import (
    run_workflow,
)
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("components.prepare_test_cases")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════


class GroundTruthCase(BaseModel):
    """One consolidated ground-truth row for a single shipment.

    Mirrors the columns of ``data/processed/ground_truth.csv``. Extra
    columns are ignored so adding new ones (e.g. ``version``,
    ``golden_message``) will not break existing loaders.
    """

    model_config = ConfigDict(extra="ignore")

    shipment_id: str
    is_exception: Literal["YES", "NO"]
    expected_resolution: str = "N/A"
    expected_tone: str = "N/A"
    should_escalate: str = "N/A"
    ground_truth_reasoning: str = ""
    golden_message: str | None = None


class TestCase(BaseModel):
    """Executable batch-test unit: input rows + expected output."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: str
    raw_rows: list[dict]
    ground_truth: GroundTruthCase


class TestRunRecord(BaseModel):
    """One completed workflow execution + scoring outcome."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: str
    final_state: dict
    report: SingleCaseReport
    error: str | None = None
    duration_sec: float = 0.0
    timestamp: str


class TestRunBatch(BaseModel):
    """Container for an ordered batch of ``TestRunRecord``s."""

    model_config = ConfigDict(extra="forbid")

    runs: list[TestRunRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════


@traceable(name="batch.load_ground_truth")
def load_ground_truth(csv_path: Path | str) -> dict[str, GroundTruthCase]:
    """Load ``ground_truth.csv`` and consolidate multi-row shipments.

    Mirrors notebook lines 4418-4428: for shipments with multiple GT
    rows, the last row with ``is_exception == "YES"`` wins; otherwise
    the first row is used.
    """
    path = Path(csv_path)
    rows_by_id: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = str(row.get("shipment_id", "")).strip()
            if not sid:
                continue
            rows_by_id[sid].append(row)

    consolidated: dict[str, GroundTruthCase] = {}
    for sid, rows in rows_by_id.items():
        exc_rows = [
            r for r in rows if str(r.get("is_exception", "")).strip().upper() == "YES"
        ]
        chosen = exc_rows[-1] if exc_rows else rows[0]
        try:
            consolidated[sid] = GroundTruthCase(**chosen)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("ground_truth row %s failed validation: %s", sid, exc)
    logger.info(
        "load_ground_truth: %d shipments (from %d rows) in %s",
        len(consolidated),
        sum(len(v) for v in rows_by_id.values()),
        path,
    )
    return consolidated


@traceable(name="batch.load_delivery_logs")
def load_delivery_logs(csv_path: Path | str) -> dict[str, list[dict]]:
    """Group ``delivery_logs.csv`` rows by ``shipment_id`` preserving order."""
    path = Path(csv_path)
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = str(row.get("shipment_id", "")).strip()
            if not sid:
                continue
            groups[sid].append(row)
    logger.info(
        "load_delivery_logs: %d shipments (%d rows) in %s",
        len(groups),
        sum(len(v) for v in groups.values()),
        path,
    )
    return dict(groups)


@traceable(name="batch.build_test_cases")
def build_test_cases(
    logs_csv: Path | str,
    gt_csv: Path | str,
) -> list[TestCase]:
    """Pair delivery-log rows with consolidated ground truth."""
    logs = load_delivery_logs(logs_csv)
    gt = load_ground_truth(gt_csv)

    cases: list[TestCase] = []
    for sid, rows in logs.items():
        if sid not in gt:
            logger.warning(
                "build_test_cases: shipment %s has logs but no ground truth — skipping",
                sid,
            )
            continue
        cases.append(TestCase(shipment_id=sid, raw_rows=rows, ground_truth=gt[sid]))
    logger.info("build_test_cases: %d executable test cases", len(cases))
    return cases


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════


def _empty_report(shipment_id: str) -> SingleCaseReport:
    """Neutral SingleCaseReport for records that failed to execute."""
    return SingleCaseReport(
        shipment_id=shipment_id,
        task_completion=TaskCompletionResult(
            exception_correct=False,
            resolution_correct=False,
            tone_correct=None,
            task_complete=False,
        ),
        escalation_correct=None,
        tool_call_correct=False,
        coherence=CoherenceResult(score=0, justification="error: workflow failed"),
        token_efficiency=TokenEfficiencyResult(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            tokens_per_resolution=None,
        ),
        trajectory_drift=TrajectoryDriftResult(
            trajectory_len=0,
            tool_calls_len=0,
            revision_count=0,
            drift_flag=False,
        ),
        semantic_similarity=SemanticSimilarityResult(
            similarity=0.0, golden_present=False
        ),
        latency_sec=None,
        citations=[],
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@traceable(name="batch.run_test_case")
def run_test_case(
    app: Any,
    test_case: TestCase,
    *,
    eval_llm: Any,
    embedder: Any | None = None,
    max_loops: int = 2,
) -> TestRunRecord:
    """Execute one shipment through the workflow + scoring, isolating failures.

    The ``run_workflow`` call is wrapped in ``try/except`` so one
    broken shipment never aborts a batch. On failure, returns a
    record with neutral metrics and ``error`` populated.
    """
    start = time.perf_counter()
    ts = _iso_now()
    try:
        final_state = run_workflow(
            app,
            test_case.shipment_id,
            test_case.raw_rows,
            max_loops=max_loops,
        )
        # ``run_workflow`` returns a TypedDict; cast to plain dict for
        # Pydantic / JSON serialisation.
        state_dict = dict(final_state) if final_state is not None else {}
        report = evaluate_single_case(
            state_dict,
            test_case.ground_truth.model_dump(),
            eval_llm=eval_llm,
            embedder=embedder,
        )
        return TestRunRecord(
            shipment_id=test_case.shipment_id,
            final_state=state_dict,
            report=report,
            error=None,
            duration_sec=round(time.perf_counter() - start, 4),
            timestamp=ts,
        )
    except Exception as exc:
        logger.error(
            "run_test_case: shipment %s failed: %s", test_case.shipment_id, exc
        )
        return TestRunRecord(
            shipment_id=test_case.shipment_id,
            final_state={},
            report=_empty_report(test_case.shipment_id),
            error=str(exc),
            duration_sec=round(time.perf_counter() - start, 4),
            timestamp=ts,
        )


@traceable(name="batch.run_batch")
def run_batch(
    app: Any,
    test_cases: list[TestCase],
    *,
    eval_llm: Any,
    embedder: Any | None = None,
    max_loops: int = 2,
) -> TestRunBatch:
    """Execute every test case sequentially and return a ``TestRunBatch``."""
    runs: list[TestRunRecord] = []
    total = len(test_cases)
    for i, tc in enumerate(test_cases, start=1):
        rec = run_test_case(
            app, tc, eval_llm=eval_llm, embedder=embedder, max_loops=max_loops
        )
        status = "FAIL" if rec.error else (
            "PASS" if rec.report.task_completion.task_complete else "MISS"
        )
        logger.info(
            "[%d/%d] %s %s task_complete=%s coherence=%d (%.2fs)",
            i,
            total,
            rec.shipment_id,
            status,
            rec.report.task_completion.task_complete,
            rec.report.coherence.score,
            rec.duration_sec,
        )
        runs.append(rec)

    metadata = {
        "timestamp": _iso_now(),
        "n": total,
        "max_loops": max_loops,
        "eval_llm": type(eval_llm).__name__,
        "embedder": type(embedder).__name__ if embedder is not None else None,
    }
    logger.info(
        "run_batch complete: n=%d failures=%d",
        total,
        sum(1 for r in runs if r.error),
    )
    return TestRunBatch(runs=runs, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


def save_batch(batch: TestRunBatch, out_dir: Path | str) -> Path:
    """Serialise a ``TestRunBatch`` to ``<out_dir>/batch_<timestamp>.json``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"batch_{stamp}.json"
    path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    logger.info("save_batch: wrote %d runs to %s", len(batch.runs), path)
    return path


def load_batch(path: Path | str) -> TestRunBatch:
    """Deserialise a previously-saved batch JSON file."""
    p = Path(path)
    return TestRunBatch.model_validate_json(p.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printer (ports notebook 4476-4518)
# ═══════════════════════════════════════════════════════════════════════════


def print_test_case_output(
    record: TestRunRecord, gt: GroundTruthCase
) -> None:
    """Console-print a single test case (notebook-parity layout)."""
    state = record.final_state or {}
    res = state.get("resolution_output", {}) or {}
    comm = state.get("communication_output", {}) or {}
    tc = record.report.task_completion

    # --- System predictions vs ground truth ---
    print("--- Predictions ---")
    print(f"  Exception:  {res.get('is_exception', 'N/A')} (GT: {gt.is_exception})")
    print(
        f"  Resolution: {res.get('resolution', 'N/A')} "
        f"(GT: {gt.expected_resolution})"
    )
    pred_esc = "YES" if state.get("escalated") else "NO"
    print(f"  Escalated:  {pred_esc} (GT: {gt.should_escalate})")
    print(
        f"  Tone:       {comm.get('tone_label', 'N/A')} (GT: {gt.expected_tone})"
    )
    print(f"  Revisions:  {state.get('resolution_revision_count', 0)}")
    print(
        f"  Guardrail:  "
        f"{'TRIGGERED' if state.get('guardrail_triggered') else 'CLEAR'}"
    )

    # --- Per-metric pass/fail breakdown ---
    print("\n--- Metrics ---")
    print(f"  Task Complete:       {'PASS' if tc.task_complete else 'FAIL'}")
    print(f"    Exception ID:      {'PASS' if tc.exception_correct else 'FAIL'}")
    print(f"    Resolution:        {'PASS' if tc.resolution_correct else 'FAIL'}")
    tone_str = (
        "N/A"
        if tc.tone_correct is None
        else ("PASS" if tc.tone_correct else "FAIL")
    )
    print(f"    Tone:              {tone_str}")
    esc = record.report.escalation_correct
    esc_str = "N/A" if esc is None else ("PASS" if esc else "FAIL")
    print(f"  Escalation Accuracy: {esc_str}")
    print(
        f"  Tool Call Accuracy:  "
        f"{'PASS' if record.report.tool_call_correct else 'FAIL'}"
    )
    print(f"  Coherence Score:     {record.report.coherence.score}/5")
    latency = record.report.latency_sec
    print(f"  Latency:             {latency:.2f}s" if latency is not None else "  Latency:             N/A")

    # --- Full agent decision trail ---
    print("\n--- Trajectory ---")
    for entry in state.get("trajectory_log", []) or []:
        print(f"  {entry}")

    # --- Which playbook pages were retrieved ---
    print("\n--- Document Citations ---")
    cites = record.report.citations
    print(
        f"  Playbook pages referenced: {', '.join(cites) if cites else 'None'}"
    )

    # --- Preview of the customer notification if generated ---
    msg = comm.get("communication_message") or ""
    if msg:
        print("\n--- Customer Message ---")
        preview = msg[:200] + ("..." if len(msg) > 200 else "")
        print(f"  {preview}")

    if record.error:
        print(f"\n--- Error ---\n  {record.error}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════


def _main() -> int:  # pragma: no cover — CLI glue
    import argparse
    import sys

    from AI_Powered_Last_Mile_Delivery_Automation.components.evaluation_metrics import (
        aggregate_results,
        print_batch_report,
    )
    from AI_Powered_Last_Mile_Delivery_Automation.components.multi_agent_workflow import (
        build_workflow,
    )
    from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import (
        ToolMaster,
    )
    from AI_Powered_Last_Mile_Delivery_Automation.utils.model_loader import (
        ModelLoader,
    )

    parser = argparse.ArgumentParser(
        description="Run the multi-agent workflow across ground_truth.csv and score it."
    )
    parser.add_argument("--logs", default="data/processed/delivery_logs.csv")
    parser.add_argument("--gt", default="data/processed/ground_truth.csv")
    parser.add_argument("--out-dir", default="data/processed/test_runs")
    parser.add_argument("--max-loops", type=int, default=2)
    parser.add_argument("--print-each", action="store_true")
    parser.add_argument("--pass-threshold", type=float, default=0.8)
    args = parser.parse_args()

    cases = build_test_cases(args.logs, args.gt)

    loader = ModelLoader()
    gen_llm, eval_llm = loader.load_llm()
    embedder = loader.load_embeddings()

    tools = ToolMaster()
    app = build_workflow(tools, gen_llm=gen_llm, eval_llm=eval_llm)

    batch = run_batch(
        app,
        cases,
        eval_llm=eval_llm,
        embedder=embedder,
        max_loops=args.max_loops,
    )
    save_batch(batch, args.out_dir)

    if args.print_each:
        for rec in batch.runs:
            print(f"\n=== {rec.shipment_id} ===")
            # Look up the matching GT case
            gt = next(
                (
                    c.ground_truth
                    for c in cases
                    if c.shipment_id == rec.shipment_id
                ),
                None,
            )
            if gt is not None:
                print_test_case_output(rec, gt)

    report = aggregate_results(
        [r.report for r in batch.runs],
        final_states=[r.final_state for r in batch.runs],
    )
    print_batch_report(report)

    if report.task_completion_rate < args.pass_threshold:
        sys.stderr.write(
            f"FAIL: task_completion_rate={report.task_completion_rate:.2%} "
            f"< threshold={args.pass_threshold:.0%}\n"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "GroundTruthCase",
    "TestCase",
    "TestRunRecord",
    "TestRunBatch",
    "load_ground_truth",
    "load_delivery_logs",
    "build_test_cases",
    "run_test_case",
    "run_batch",
    "save_batch",
    "load_batch",
    "print_test_case_output",
]
