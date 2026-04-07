"""Singleton pipeline manager for the multi-agent workflow.

Extracts the initialisation and health-check logic from ``main.py`` into a
thread-safe singleton reusable by the CLI, the FastAPI backend, and tests.
"""

from __future__ import annotations

import asyncio
import os
import threading
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
        """Release resources (DB connections, etc.)."""
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
    ) -> UnifiedAgentState:
        """Run the workflow for a single shipment (async-safe).

        Offloads the synchronous ``run_workflow()`` call to a thread so the
        FastAPI event loop stays responsive.
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized — call initialize() first")

        config_metadata: dict[str, Any] = {}
        if trace_id:
            config_metadata["run_id"] = trace_id

        return await asyncio.to_thread(
            run_workflow,
            self.app,
            shipment_id,
            raw_rows,
            max_loops=max_loops,
            thread_id=trace_id or shipment_id,
        )

    def run_single_sync(
        self,
        shipment_id: str,
        raw_rows: list[dict],
        *,
        max_loops: int = 2,
    ) -> UnifiedAgentState:
        """Synchronous variant for CLI usage."""
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized — call initialize() first")
        return run_workflow(self.app, shipment_id, raw_rows, max_loops=max_loops)
