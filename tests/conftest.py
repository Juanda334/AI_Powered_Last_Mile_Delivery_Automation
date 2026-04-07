"""Shared test fixtures for the AI-Powered Last-Mile Delivery suite.

All fixtures are deterministic and free of network I/O. Smoke tests
that need real infrastructure define their own fixtures inline and
skip when creds are absent.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure the package is importable even without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Session-scoped path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def sample_logs_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_logs.csv"


@pytest.fixture(scope="session")
def sample_gt_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_ground_truth.csv"


# ---------------------------------------------------------------------------
# Sample raw rows (function-scoped so mutation never leaks)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_row() -> dict:
    return {
        "shipment_id": "SHP-T01",
        "timestamp": "2026-03-05T10:14:00",
        "status_code": "DELIVERED",
        "status_description": "Left at front door, signed by resident",
        "customer_id": "CUST-100",
        "delivery_address": "1 Test Street, Testville, 99001",
        "package_type": "STANDARD",
        "package_size": "SMALL",
        "attempt_number": "1",
        "is_duplicate_scan": "False",
    }


@pytest.fixture
def sample_exception_rows() -> list[dict]:
    return [
        {
            "shipment_id": "SHP-T02",
            "timestamp": "2026-03-05T11:00:00",
            "status_code": "DAMAGED",
            "status_description": "Box crushed contents visibly damaged",
            "customer_id": "CUST-200",
            "delivery_address": "2 Test Ave, Testville, 99002",
            "package_type": "PERISHABLE",
            "package_size": "MEDIUM",
            "attempt_number": "1",
            "is_duplicate_scan": "False",
        },
        {
            "shipment_id": "SHP-T02",
            "timestamp": "2026-03-05T11:02:00",
            "status_code": "DAMAGED",
            "status_description": "Box crushed contents visibly damaged",
            "customer_id": "CUST-200",
            "delivery_address": "2 Test Ave, Testville, 99002",
            "package_type": "PERISHABLE",
            "package_size": "MEDIUM",
            "attempt_number": "1",
            "is_duplicate_scan": "True",
        },
    ]


# ---------------------------------------------------------------------------
# FakeListLLM fixtures — canned JSON responses for deterministic testing
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gen_llm():
    from langchain_core.language_models.fake import FakeListLLM

    return FakeListLLM(
        responses=[
            json.dumps(
                {
                    "is_exception": "YES",
                    "resolution": "RESCHEDULE",
                    "rationale": "test",
                }
            ),
            json.dumps(
                {
                    "tone_label": "FORMAL",
                    "communication_message": "We will reschedule your delivery.",
                }
            ),
        ]
    )


@pytest.fixture
def fake_eval_llm():
    from langchain_core.language_models.fake import FakeListLLM

    return FakeListLLM(
        responses=[
            json.dumps({"score": 5, "justification": "coherent"}),
            json.dumps({"decision": "APPROVE", "rationale": "ok"}),
            json.dumps({"decision": "APPROVE", "rationale": "ok"}),
        ]
    )


@pytest.fixture
def fake_embedder():
    class _FakeEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 384

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

    return _FakeEmbedder()


# ---------------------------------------------------------------------------
# In-memory sqlite for tool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_sqlite_customer_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "customers.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT,
            tier TEXT,
            exceptions_last_90d INTEGER,
            preferred_channel TEXT,
            active_credit INTEGER
        );
        CREATE TABLE lockers (
            locker_id TEXT PRIMARY KEY,
            zip_code TEXT,
            max_package_size TEXT,
            capacity_status TEXT
        );
        INSERT INTO customers VALUES
            ('CUST-100', 'Alice', 'STANDARD', 0, 'EMAIL', 0),
            ('CUST-200', 'Bob', 'VIP', 4, 'SMS', 25);
        INSERT INTO lockers VALUES
            ('LOC-001', '99001', 'LARGE', 'AVAILABLE'),
            ('LOC-002', '99001', 'LARGE', 'LIMITED'),
            ('LOC-003', '99002', 'MEDIUM', 'FULL');
        """
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Minimal UnifiedAgentState — all 26 keys, sane defaults
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_unified_state() -> dict:
    return {
        "raw_rows": [],
        "shipment_id": "SHP-T01",
        "consolidated_event": {},
        "customer_profile": {},
        "customer_profile_full": {},
        "locker_availability": [],
        "playbook_context": [],
        "escalation_signals": {},
        "noise_override": False,
        "guardrail_triggered": False,
        "resolution_output": {},
        "critic_resolution_output": {},
        "resolution_revision_count": 0,
        "critic_feedback": "",
        "communication_output": {},
        "critic_communication_output": {},
        "next_agent": "preprocessor",
        "max_loops": 2,
        "escalated": False,
        "tool_calls_log": [],
        "trajectory_log": [],
        "start_time": None,
        "latency_sec": None,
        "final_actions": [],
    }


# ---------------------------------------------------------------------------
# SingleCaseReport factory — for aggregation tests
# ---------------------------------------------------------------------------


@pytest.fixture
def make_single_case_report():
    """Factory that builds SingleCaseReport with overridable fields."""
    from AI_Powered_Last_Mile_Delivery_Automation.components.evaluation_metrics import (
        CoherenceResult,
        SemanticSimilarityResult,
        SingleCaseReport,
        TaskCompletionResult,
        TokenEfficiencyResult,
        TrajectoryDriftResult,
    )

    def _factory(
        *,
        shipment_id: str = "SHP-T01",
        task_complete: bool = True,
        exception_correct: bool = True,
        resolution_correct: bool = True,
        tone_correct: Any = True,
        escalation_correct: Any = True,
        tool_call_correct: bool = True,
        coherence_score: int = 5,
        total_tokens: int = 1000,
        cost: float = 0.002,
        drift_flag: bool = False,
        latency_sec: float = 2.5,
    ) -> SingleCaseReport:
        return SingleCaseReport(
            shipment_id=shipment_id,
            task_completion=TaskCompletionResult(
                exception_correct=exception_correct,
                resolution_correct=resolution_correct,
                tone_correct=tone_correct,
                task_complete=task_complete,
            ),
            escalation_correct=escalation_correct,
            tool_call_correct=tool_call_correct,
            coherence=CoherenceResult(score=coherence_score, justification=""),
            token_efficiency=TokenEfficiencyResult(
                prompt_tokens=total_tokens // 2,
                completion_tokens=total_tokens - (total_tokens // 2),
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                tokens_per_resolution=None,
            ),
            trajectory_drift=TrajectoryDriftResult(
                trajectory_len=10,
                tool_calls_len=5,
                revision_count=0,
                drift_flag=drift_flag,
            ),
            semantic_similarity=SemanticSimilarityResult(
                similarity=0.0, golden_present=False
            ),
            latency_sec=latency_sec,
            citations=[],
        )

    return _factory
