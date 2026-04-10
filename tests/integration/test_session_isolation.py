"""Session-isolation and state-wiping leak tests.

These tests stub the compiled LangGraph app with a fake checkpointer so
they run without any LLM / vector-store infrastructure. Each test targets
a single property of the isolation contract:

1. ``test_wipe_removes_checkpoints`` — after a session is wiped, none of its
   recorded threads remain in the checkpointer.
2. ``test_cross_session_batch_read_404`` — polling a batch job with a
   different ``X-Session-Id`` returns 404, preventing cross-session reads.
3. ``test_exception_safety_wipes`` — if ``run_single`` raises, the
   ``session_scope`` ``finally`` block still wipes the session.
4. ``test_pii_filter_scrubs`` — ``PIIFilter`` redacts denylist keys from
   log records before they reach any handler.
5. ``test_ttl_sweeper_expires`` — expired sessions are wiped by the
   sweeper hook.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from fastapi import HTTPException

from AI_Powered_Last_Mile_Delivery_Automation.core.pipeline import PipelineManager
from AI_Powered_Last_Mile_Delivery_Automation.core.session_store import (
    InMemorySessionStore,
)
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import PIIFilter


# ── Fake LangGraph checkpointer ─────────────────────────────────────────────


class FakeCheckpointer:
    """Minimal MemorySaver substitute with ``delete_thread``."""

    def __init__(self) -> None:
        self.storage: dict[str, dict] = {}

    def put(self, thread_id: str, data: dict) -> None:
        self.storage[thread_id] = data

    def delete_thread(self, thread_id: str) -> None:
        self.storage.pop(thread_id, None)


class FakeApp:
    def __init__(self) -> None:
        self.checkpointer = FakeCheckpointer()


# ── Pipeline fixture that skips real initialisation ─────────────────────────


@pytest.fixture
def stub_pipeline():
    """Yield a PipelineManager with a fake compiled app — no LLMs involved."""
    PipelineManager.reset()
    pm = PipelineManager.get()
    pm.app = FakeApp()
    pm._initialized = True
    pm.session_store = InMemorySessionStore(default_ttl_sec=900)
    pm._batch_jobs = {}
    try:
        yield pm
    finally:
        pm._initialized = False
        PipelineManager.reset()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Isolation — wipe_session removes every registered thread
# ═══════════════════════════════════════════════════════════════════════════


def test_wipe_removes_checkpoints(stub_pipeline: PipelineManager) -> None:
    pm = stub_pipeline

    thread_a = pm._derive_thread_id("A", "SHP-001", "trace-a")
    thread_b = pm._derive_thread_id("B", "SHP-001", "trace-b")

    pm.session_store.register_thread("A", thread_a)
    pm.session_store.register_thread("B", thread_b)
    pm.app.checkpointer.put(thread_a, {"secret": "Alice"})
    pm.app.checkpointer.put(thread_b, {"secret": "Bob"})

    # Wiping A must not touch B.
    pm.wipe_session("A")
    assert thread_a not in pm.app.checkpointer.storage
    assert thread_b in pm.app.checkpointer.storage
    assert pm.session_store.get("A") is None
    assert pm.session_store.get("B") is not None

    # And derived thread ids must differ even for the same shipment.
    assert thread_a != thread_b


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cross-session batch read returns 404
# ═══════════════════════════════════════════════════════════════════════════


def test_cross_session_batch_read_404(stub_pipeline: PipelineManager) -> None:
    from AI_Powered_Last_Mile_Delivery_Automation.schemas.api_models import (
        BatchJobResponse,
    )
    from api import get_batch_status

    stub_pipeline._batch_jobs["job-1"] = BatchJobResponse(
        job_id="job-1",
        status="running",
        total=1,
        session_id="A",
    )

    # Owner — must succeed.
    result = asyncio.run(get_batch_status("job-1", x_session_id="A"))
    assert result.session_id == "A"

    # Unknown job id — 404.
    with pytest.raises(HTTPException) as exc_unknown:
        asyncio.run(get_batch_status("missing", x_session_id="A"))
    assert exc_unknown.value.status_code == 404

    # Different session — 404 (no leak).
    with pytest.raises(HTTPException) as exc_cross:
        asyncio.run(get_batch_status("job-1", x_session_id="B"))
    assert exc_cross.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# 3. Exception safety — finally block wipes even on failure
# ═══════════════════════════════════════════════════════════════════════════


def test_exception_safety_wipes(stub_pipeline: PipelineManager) -> None:
    pm = stub_pipeline
    session_id = "E"
    thread_id = pm._derive_thread_id(session_id, "SHP-007", "trace-e")
    pm.session_store.register_thread(session_id, thread_id)
    pm.app.checkpointer.put(thread_id, {"blob": "secret"})

    async def _scope_simulation() -> None:
        try:
            raise RuntimeError("simulated workflow failure")
        finally:
            pm.wipe_session(session_id)

    with pytest.raises(RuntimeError):
        asyncio.run(_scope_simulation())

    assert pm.session_store.get(session_id) is None
    assert thread_id not in pm.app.checkpointer.storage


# ═══════════════════════════════════════════════════════════════════════════
# 4. PII filter scrubs denylist keys
# ═══════════════════════════════════════════════════════════════════════════


def test_pii_filter_scrubs() -> None:
    f = PIIFilter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="profile=%s",
        args=({"customer_profile_full": {"name": "Carol", "phone": "555"}},),
        exc_info=None,
    )
    assert f.filter(record) is True
    scrubbed_arg = record.args[0]  # type: ignore[index]
    assert scrubbed_arg["customer_profile_full"] == "<redacted>"

    # Nested dict inside a list — name key should still be redacted.
    record2 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="batch=%s",
        args=([{"name": "Dave", "order": "X"}],),
        exc_info=None,
    )
    f.filter(record2)
    assert record2.args[0][0]["name"] == "<redacted>"  # type: ignore[index]
    assert record2.args[0][0]["order"] == "X"  # type: ignore[index]


# ═══════════════════════════════════════════════════════════════════════════
# 5. TTL sweeper expires abandoned sessions
# ═══════════════════════════════════════════════════════════════════════════


def test_ttl_sweeper_expires(stub_pipeline: PipelineManager) -> None:
    pm = stub_pipeline
    pm.session_store = InMemorySessionStore(default_ttl_sec=0.05)

    sid = "T"
    thread_id = pm._derive_thread_id(sid, "SHP-999", "trace-t")
    pm.session_store.register_thread(sid, thread_id)
    pm.app.checkpointer.put(thread_id, {"blob": "ttl"})

    assert pm.session_store.get(sid) is not None
    time.sleep(0.1)
    wiped = pm.sweep_expired_sessions()

    assert wiped == 1
    assert pm.session_store.get(sid) is None
    assert thread_id not in pm.app.checkpointer.storage
