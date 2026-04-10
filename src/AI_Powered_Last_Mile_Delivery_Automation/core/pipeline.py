"""Singleton pipeline manager for the multi-agent workflow.

Extracts the initialisation and health-check logic from ``main.py`` into a
thread-safe singleton reusable by the CLI, the FastAPI backend, and tests.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

from AI_Powered_Last_Mile_Delivery_Automation.components.data_ingestion import (
    DataIngestor,
    _EXTERNAL_DB,
    _EXTERNAL_DOC,
)
from AI_Powered_Last_Mile_Delivery_Automation.components.multi_agent_workflow import (
    build_workflow,
    run_workflow,
    wipe_thread,
)
from AI_Powered_Last_Mile_Delivery_Automation.core.session_store import (
    SessionStore,
    build_session_store,
)
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import ToolMaster
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import (
    UnifiedAgentState,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.model_loader import ModelLoader

logger = get_module_logger("core.pipeline")

_DEFAULT_LOGS_CSV = "data/processed/delivery_logs.csv"


class PipelineManager:
    """Thread-safe singleton holding the compiled workflow and shared resources.

    Usage::

        pm = PipelineManager.get()
        pm.initialize()           # expensive — call once at startup
        result = await pm.run_single("SHP-002", raw_rows)
        pm.shutdown()             # release DB connections
    """

    _instance: PipelineManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.app: Any = None
        self.eval_llm: Any = None
        self.embedder: Any = None
        self.tools: ToolMaster | None = None
        self.session_store: SessionStore = build_session_store()
        self._batch_jobs: dict[str, Any] = {}
        self._initialized = False

    # ── Singleton accessor ────────────────────────────────────────────

    @classmethod
    def get(cls) -> PipelineManager:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (for tests)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load models, build the ChromaDB retriever, compile the workflow.

        Safe to call multiple times — only the first call does work.
        """
        if self._initialized:
            logger.info("PipelineManager already initialized — skipping")
            return

        logger.info(">>>>>> PipelineManager: Init started <<<<<<")

        loader = ModelLoader()
        gen_llm, eval_llm = loader.load_llm()
        embedder = loader.load_embeddings()
        logger.info(
            ">>>>>> PipelineManager: Models loaded (gen=%s, eval=%s) <<<<<<",
            type(gen_llm).__name__,
            type(eval_llm).__name__,
        )

        ingestor = DataIngestor()
        retriever = ingestor.build_retriever()
        logger.info(">>>>>> PipelineManager: Retriever built (ChromaDB) <<<<<<")

        tools = ToolMaster(retriever=retriever)
        logger.info(">>>>>> PipelineManager: ToolMaster initialized <<<<<<")

        app = build_workflow(tools, gen_llm=gen_llm, eval_llm=eval_llm)
        logger.info(">>>>>> PipelineManager: Workflow compiled <<<<<<")

        self.app = app
        self.eval_llm = eval_llm
        self.embedder = embedder
        self.tools = tools
        self._initialized = True

    def shutdown(self) -> None:
        """Release resources (DB connections, etc.) and wipe any live sessions."""
        # Wipe all sessions still in the store so no checkpoints leak across
        # reloads / test runs that reuse the same process.
        try:
            active = list(getattr(self.session_store, "_records", {}).keys())
            for sid in active:
                self.wipe_session(sid)
        except Exception as exc:
            logger.warning("shutdown: session flush failed  err=%s", exc)

        if self.tools is not None:
            self.tools.close()
            logger.info("PipelineManager: ToolMaster closed")
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Health check ──────────────────────────────────────────────────

    def health_check(
        self,
        logs_csv: str = _DEFAULT_LOGS_CSV,
        *,
        include_pipeline_status: bool = True,
    ) -> dict[str, bool]:
        """Verify critical dependencies. Returns a dict of check_name -> pass/fail.

        Set *include_pipeline_status* to ``False`` for pre-init checks (CLI)
        where the pipeline hasn't been initialised yet.
        """
        checks: dict[str, bool] = {
            "sqlite_db": _EXTERNAL_DB.exists(),
            "playbook_pdf": _EXTERNAL_DOC.exists(),
            "delivery_logs_csv": Path(logs_csv).exists(),
            "openai_api_key": bool(os.environ.get("OPENAI_API_KEY")),
        }
        if include_pipeline_status:
            checks["pipeline_initialized"] = self._initialized
        for name, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            logger.info("Health check  %-30s  %s", name, status)
        return checks

    # ── Execution wrappers ────────────────────────────────────────────

    async def run_single(
        self,
        shipment_id: str,
        raw_rows: list[dict],
        *,
        max_loops: int = 2,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> UnifiedAgentState:
        """Run the workflow for a single shipment (async-safe).

        Offloads the synchronous ``run_workflow()`` call to a thread so the
        FastAPI event loop stays responsive. Registers the derived LangGraph
        thread with the session store so ``wipe_session`` can reach it.
        """
        if not self._initialized:
            raise RuntimeError(
                "PipelineManager not initialized — call initialize() first"
            )

        thread_id = self._derive_thread_id(session_id, shipment_id, trace_id)
        if session_id:
            self.session_store.register_thread(session_id, thread_id)

        return await asyncio.to_thread(
            run_workflow,
            self.app,
            shipment_id,
            raw_rows,
            max_loops=max_loops,
            thread_id=thread_id,
        )

    def run_single_sync(
        self,
        shipment_id: str,
        raw_rows: list[dict],
        *,
        max_loops: int = 2,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> UnifiedAgentState:
        """Synchronous variant for CLI usage."""
        if not self._initialized:
            raise RuntimeError(
                "PipelineManager not initialized — call initialize() first"
            )
        thread_id = self._derive_thread_id(session_id, shipment_id, trace_id)
        if session_id:
            self.session_store.register_thread(session_id, thread_id)
        return run_workflow(
            self.app, shipment_id, raw_rows, max_loops=max_loops, thread_id=thread_id
        )

    # ── Session isolation ─────────────────────────────────────────────

    @staticmethod
    def _derive_thread_id(
        session_id: str | None, shipment_id: str, trace_id: str | None
    ) -> str:
        """Namespaced LangGraph thread id.

        Concurrent runs for the same shipment under different sessions must
        never collide — including the session_id and trace_id guarantees a
        fresh checkpoint scope per request.
        """
        parts = [
            session_id or "no-session",
            str(shipment_id),
            trace_id or "no-trace",
        ]
        return ":".join(parts)

    def wipe_session(self, session_id: str) -> dict[str, int]:
        """Purge every local artifact tied to ``session_id``.

        Removes LangGraph checkpoints for every registered thread and drops
        any batch-job entries owned by the session. LangSmith remote traces
        are unaffected — ``@traceable`` ships them to the LangSmith service
        before this runs, so audit history survives the local wipe.
        """
        record = self.session_store.pop(session_id)
        if record is None:
            return {"threads": 0, "batch_jobs": 0}

        thread_hits = 0
        for thread_id in record.thread_ids:
            try:
                if wipe_thread(self.app, thread_id):
                    thread_hits += 1
            except Exception as exc:
                logger.warning(
                    "wipe_session  wipe_thread failed  session_id=%s  tid=%s  err=%s",
                    session_id,
                    thread_id,
                    exc,
                )

        job_hits = 0
        for job_id in record.batch_job_ids:
            if self._batch_jobs.pop(job_id, None) is not None:
                job_hits += 1

        logger.info(
            "session wiped  session_id=%s  threads=%d/%d  batch_jobs=%d/%d",
            session_id,
            thread_hits,
            len(record.thread_ids),
            job_hits,
            len(record.batch_job_ids),
        )
        return {"threads": thread_hits, "batch_jobs": job_hits}

    def sweep_expired_sessions(self) -> int:
        """TTL sweeper hook — called from the lifespan background task."""
        expired_sids = self.session_store.expired(time.time())
        for sid in expired_sids:
            self.wipe_session(sid)
        return len(expired_sids)
