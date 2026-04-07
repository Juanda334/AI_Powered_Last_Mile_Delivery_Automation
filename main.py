"""AI_Powered_Last_Mile_Delivery_Automation — centralized pipeline entry point.

Orchestrates the full multi-agent workflow for last-mile delivery exception
resolution.  Supports two modes via CLI subcommands:

    python main.py query --shipment-id SHP-002
    python main.py batch --print-each

Operational suggestions
-----------------------
* **Async execution** — ``run_batch`` is LLM-bound.  Wrapping
  ``run_test_case`` in ``asyncio.gather`` with a semaphore (5-8
  concurrent) and switching to ``app.ainvoke`` / ``eval_llm.ainvoke``
  would yield ~10x speedup on large batches.
* **Artifact exporting** — ``save_batch()`` already writes timestamped
  JSON.  Add a ``--export-csv`` flag to dump ``BatchReport.model_dump()``
  to ``outputs/eval_<timestamp>.csv`` for spreadsheet consumption.
* **Health checks** — ``_health_check()`` below verifies critical
  dependencies (SQLite, CSV, PDF, API key) before any LLM calls fire.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env early so health checks can see API keys in local mode.
if os.getenv("ENV", "local").lower() != "production":
    load_dotenv()

from AI_Powered_Last_Mile_Delivery_Automation.components.evaluation_metrics import (
    aggregate_results,
    print_batch_report,
)
from AI_Powered_Last_Mile_Delivery_Automation.components.prepare_test_cases import (
    build_test_cases,
    load_delivery_logs,
    load_ground_truth,
    print_test_case_output,
    run_batch,
    save_batch,
)
from AI_Powered_Last_Mile_Delivery_Automation.core.pipeline import PipelineManager
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("main")


# ═══════════════════════════════════════════════════════════════════════════
# Health check (delegates to PipelineManager)
# ═══════════════════════════════════════════════════════════════════════════


def _health_check(logs_csv: str) -> bool:
    """Verify critical dependencies before pipeline init.

    Returns ``True`` if all checks pass, ``False`` otherwise.
    """
    pm = PipelineManager.get()
    checks = pm.health_check(logs_csv, include_pipeline_status=False)

    all_ok = all(checks.values())
    if all_ok:
        logger.info(">>>>>> Stage: Health Check passed <<<<<<")
    else:
        logger.error(">>>>>> Stage: Health Check FAILED — aborting <<<<<<")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Result printer (single-query mode)
# ═══════════════════════════════════════════════════════════════════════════


def _print_single_result(result: dict) -> None:
    """Pretty-print the final state from a single workflow run."""
    res = result.get("resolution_output", {}) or {}
    comm = result.get("communication_output", {}) or {}

    print(f"\n{'=' * 60}")
    print(f"  Shipment: {result.get('shipment_id', 'N/A')}")
    print(f"{'=' * 60}")

    # Resolution
    print("\n--- Resolution ---")
    print(f"  Exception:  {res.get('is_exception', 'N/A')}")
    print(f"  Resolution: {res.get('resolution', 'N/A')}")
    print(f"  Reasoning:  {res.get('reasoning', 'N/A')}")
    pred_esc = "YES" if result.get("escalated") else "NO"
    print(f"  Escalated:  {pred_esc}")
    print(f"  Revisions:  {result.get('resolution_revision_count', 0)}")
    guardrail = "TRIGGERED" if result.get("guardrail_triggered") else "CLEAR"
    print(f"  Guardrail:  {guardrail}")

    # Communication
    msg = comm.get("communication_message") or ""
    if msg:
        print("\n--- Customer Message ---")
        print(f"  Tone: {comm.get('tone_label', 'N/A')}")
        preview = msg[:300] + ("..." if len(msg) > 300 else "")
        print(f"  {preview}")

    # Trajectory
    print("\n--- Trajectory ---")
    for entry in result.get("trajectory_log", []) or []:
        print(f"  {entry}")

    # Final actions
    actions = result.get("final_actions", []) or []
    if actions:
        print("\n--- Final Actions ---")
        for act in actions:
            print(f"  {act}")

    # Latency
    latency = result.get("latency_sec")
    if latency is not None:
        print(f"\n  Latency: {latency:.2f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: query
# ═══════════════════════════════════════════════════════════════════════════


def cmd_query(args: argparse.Namespace) -> int:
    """Run the multi-agent workflow for a single shipment."""
    if not _health_check(args.logs):
        return 1

    pm = PipelineManager.get()
    pm.initialize()
    try:
        logs = load_delivery_logs(args.logs)
        raw_rows = logs.get(args.shipment_id)
        if not raw_rows:
            logger.error("Shipment %s not found in %s", args.shipment_id, args.logs)
            return 1

        logger.info(
            ">>>>>> Stage: Execution Started (mode=query, shipment=%s) <<<<<<",
            args.shipment_id,
        )
        result = pm.run_single_sync(
            args.shipment_id, raw_rows, max_loops=args.max_loops
        )
        _print_single_result(dict(result) if result else {})
        logger.info(">>>>>> Stage: Execution Complete <<<<<<")
        return 0
    finally:
        pm.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: batch
# ═══════════════════════════════════════════════════════════════════════════


def cmd_batch(args: argparse.Namespace) -> int:
    """Run the full test suite and optionally evaluate metrics."""
    if not _health_check(args.logs):
        return 1

    pm = PipelineManager.get()
    pm.initialize()
    try:
        cases = build_test_cases(args.logs, args.gt)
        if not cases:
            logger.error("No test cases built — check logs and ground-truth CSVs")
            return 1

        logger.info(
            ">>>>>> Stage: Execution Started (mode=batch, n=%d) <<<<<<",
            len(cases),
        )
        batch = run_batch(
            pm.app,
            cases,
            eval_llm=pm.eval_llm,
            embedder=pm.embedder,
            max_loops=args.max_loops,
        )
        save_batch(batch, args.out_dir)
        logger.info(">>>>>> Stage: Batch Execution Complete <<<<<<")

        # Per-case output
        if args.print_each:
            gt_map = load_ground_truth(args.gt)
            for rec in batch.runs:
                print(f"\n{'=' * 60}")
                print(f"  {rec.shipment_id}")
                print(f"{'=' * 60}")
                gt = gt_map.get(rec.shipment_id)
                if gt is not None:
                    print_test_case_output(rec, gt)

        # Evaluation metrics
        if not args.no_eval:
            report = aggregate_results(
                [r.report for r in batch.runs],
                final_states=[r.final_state for r in batch.runs],
            )
            print_batch_report(report)
            logger.info(">>>>>> Stage: Evaluation Complete <<<<<<")

            if report.task_completion_rate < args.pass_threshold:
                logger.warning(
                    "task_completion_rate=%.2f%% < threshold=%.0f%%",
                    report.task_completion_rate * 100,
                    args.pass_threshold * 100,
                )
                return 1
        return 0
    finally:
        pm.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parser
# ═══════════════════════════════════════════════════════════════════════════


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="AI-Powered Last Mile Delivery — multi-agent pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── query ──────────────────────────────────────────────────
    q = sub.add_parser("query", help="Run workflow for a single shipment")
    q.add_argument(
        "--shipment-id", required=True, help="Shipment ID to process (e.g. SHP-002)"
    )
    q.add_argument(
        "--logs",
        default="data/processed/delivery_logs.csv",
        help="Path to delivery logs CSV",
    )
    q.add_argument(
        "--max-loops", type=int, default=2, help="Max revision loops (default: 2)"
    )

    # ── batch ──────────────────────────────────────────────────
    b = sub.add_parser("batch", help="Run full test suite with evaluation")
    b.add_argument(
        "--logs",
        default="data/processed/delivery_logs.csv",
        help="Path to delivery logs CSV",
    )
    b.add_argument(
        "--gt",
        default="data/processed/ground_truth.csv",
        help="Path to ground truth CSV",
    )
    b.add_argument(
        "--out-dir",
        default="data/processed/test_runs",
        help="Directory for batch result JSON",
    )
    b.add_argument(
        "--max-loops", type=int, default=2, help="Max revision loops (default: 2)"
    )
    b.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-case predictions and metrics",
    )
    b.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip aggregate evaluation metrics",
    )
    b.add_argument(
        "--pass-threshold",
        type=float,
        default=0.8,
        help="Fail (exit 1) if task_completion_rate < this (default: 0.8)",
    )

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    dispatch = {
        "query": cmd_query,
        "batch": cmd_batch,
    }
    sys.exit(dispatch[args.command](args))
