"""AI_Powered_Last_Mile_Delivery_Automation — FastAPI backend (api.py).

Orchestration gateway that wraps the multi-agent workflow as a REST API.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env early so health checks can see API keys in local mode.
if os.getenv("ENV", "local").lower() != "production":
    load_dotenv()

from AI_Powered_Last_Mile_Delivery_Automation.components.prepare_test_cases import (
    load_delivery_logs,
)
from AI_Powered_Last_Mile_Delivery_Automation.core.pipeline import PipelineManager
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
    trace_id_var,
)
from AI_Powered_Last_Mile_Delivery_Automation.schemas.api_models import (
    BatchJobResponse,
    BatchQueryRequest,
    HealthCheckDetail,
    HealthResponse,
    HomeResponse,
    PredictRequest,
    PredictResponse,
    SingleQueryRequest,
    SingleQueryResponse,
    state_to_response,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.agent_states_view import AgentName

logger = get_module_logger("api")

# Batch jobs live on ``PipelineManager._batch_jobs`` so ``wipe_session`` can
# reach them — there is no module-level store.

_DEFAULT_LOGS_CSV = "data/processed/delivery_logs.csv"
_SWEEPER_INTERVAL_SEC = float(os.environ.get("SESSION_SWEEP_INTERVAL_SEC", "60"))


# ═══════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════


async def _ttl_sweeper(pipeline) -> None:
    """Background loop that wipes sessions past their TTL."""
    logger.info("session TTL sweeper started  interval=%.1fs", _SWEEPER_INTERVAL_SEC)
    while True:
        try:
            await asyncio.sleep(_SWEEPER_INTERVAL_SEC)
            wiped = pipeline.sweep_expired_sessions()
            if wiped:
                logger.info("TTL sweeper  wiped=%d sessions", wiped)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("TTL sweeper iteration failed  err=%s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline, start the sweeper, release on shutdown."""
    pipeline = PipelineManager.get()
    try:
        pipeline.initialize()
        logger.info("Pipeline initialized — server ready")
    except Exception:
        logger.exception(
            "Pipeline initialization failed — server starting in degraded mode"
        )

    sweeper_task = asyncio.create_task(_ttl_sweeper(pipeline))
    try:
        yield
    finally:
        sweeper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper_task
        pipeline.shutdown()
        logger.info("Pipeline shut down")


# ═══════════════════════════════════════════════════════════════════════════
# App instance
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI-Powered Last-Mile Delivery Automation",
    description=(
        "Multi-agent orchestration gateway for delivery exception resolution. "
        "Supports single-query and batch prediction modes."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Middleware — Trace ID
# ═══════════════════════════════════════════════════════════════════════════


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """Extract or generate a trace ID and propagate it through the request."""
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
    trace_id_var.set(trace_id)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


# ═══════════════════════════════════════════════════════════════════════════
# Global exception handler
# ═══════════════════════════════════════════════════════════════════════════


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = trace_id_var.get("")
    logger.error(
        "Unhandled exception  trace_id=%s  path=%s  error=%s",
        trace_id,
        request.url.path,
        str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "trace_id": trace_id,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health():
    """Lightweight health check verifying critical dependencies."""
    pipeline = PipelineManager.get()
    checks = pipeline.health_check()

    all_ok = all(checks.values())
    any_ok = any(checks.values())

    if all_ok:
        status = "healthy"
    elif any_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        checks=HealthCheckDetail(**checks),
        version="0.2.0",
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /home
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/home", response_model=HomeResponse)
async def home():
    """Return system metadata — available agents, endpoints, version."""
    return HomeResponse(
        agents=[agent.value for agent in AgentName],
        endpoints={
            "/health": "GET",
            "/home": "GET",
            "/predict": "POST",
            "/predict/batch/{job_id}": "GET",
            "/sessions/{session_id}/wipe": "POST",
            "/docs": "GET",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Session scope dependency
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_session_id(
    payload: PredictRequest, header_session_id: str | None
) -> str:
    """Pick session_id from payload → header → fresh UUID."""
    candidates: list[str | None] = [header_session_id]
    if payload.query is not None:
        candidates.append(payload.query.session_id)
    if payload.batch is not None:
        candidates.append(payload.batch.session_id)
    for sid in candidates:
        if sid:
            return sid
    return str(uuid.uuid4())


async def session_scope(
    payload: PredictRequest,
    x_session_id: str | None = Header(default=None),
) -> AsyncIterator[str]:
    """Dependency that binds a request to a session and wipes on exit.

    Single-query mode: auto-wipe once the response is serialized.
    Batch mode: wipe is deferred to ``_run_batch_job``'s ``finally`` so the
    background task keeps access to the checkpoints it needs.
    """
    session_id = _resolve_session_id(payload, x_session_id)
    pipeline = PipelineManager.get()
    pipeline.session_store.touch(session_id)
    try:
        yield session_id
    finally:
        if payload.query is not None:
            try:
                pipeline.wipe_session(session_id)
            except Exception as exc:
                logger.warning(
                    "session_scope  wipe failed  session_id=%s  err=%s",
                    session_id,
                    exc,
                )


# ═══════════════════════════════════════════════════════════════════════════
# POST /predict
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    x_trace_id: str | None = Header(default=None),
    session_id: str = Depends(session_scope),
):
    """Primary execution endpoint — single query or batch mode."""
    pipeline = PipelineManager.get()
    if not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    trace_id = x_trace_id or trace_id_var.get("") or str(uuid.uuid4())
    response.headers["X-Session-Id"] = session_id

    # ── Single-query mode ─────────────────────────────────────────
    if payload.query:
        raw_rows = payload.query.raw_rows
        if raw_rows is None:
            logs = load_delivery_logs(_DEFAULT_LOGS_CSV)
            raw_rows = logs.get(payload.query.shipment_id)
            if not raw_rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"Shipment {payload.query.shipment_id} not found in delivery logs",
                )

        result = await pipeline.run_single(
            payload.query.shipment_id,
            raw_rows,
            max_loops=payload.query.max_loops,
            trace_id=trace_id,
            session_id=session_id,
        )
        return PredictResponse(
            result=state_to_response(dict(result), trace_id, session_id)
        )

    # ── Batch mode ────────────────────────────────────────────────
    if payload.batch:
        job_id = str(uuid.uuid4())
        batch_req = payload.batch

        if batch_req.queries:
            total = len(batch_req.queries)
        else:
            total = 0  # updated once the dataset is loaded

        job = BatchJobResponse(
            job_id=job_id,
            status="accepted",
            total=total,
            session_id=session_id,
        )
        pipeline._batch_jobs[job_id] = job
        pipeline.session_store.register_batch(session_id, job_id)
        background_tasks.add_task(
            _run_batch_job, job_id, batch_req, trace_id, session_id
        )
        return PredictResponse(job=job)


# ═══════════════════════════════════════════════════════════════════════════
# GET /predict/batch/{job_id}
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/predict/batch/{job_id}", response_model=BatchJobResponse)
async def get_batch_status(
    job_id: str,
    x_session_id: str | None = Header(default=None),
):
    """Poll for batch job progress.

    If ``X-Session-Id`` is supplied it must match the job's owning session —
    prevents cross-session reads of in-flight batch results.
    """
    pipeline = PipelineManager.get()
    job = pipeline._batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")
    if x_session_id and job.session_id and x_session_id != job.session_id:
        raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")
    return job


# ═══════════════════════════════════════════════════════════════════════════
# POST /sessions/{session_id}/wipe
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/sessions/{session_id}/wipe")
async def wipe_session_endpoint(session_id: str):
    """Client-initiated teardown — drops checkpoints and batch entries."""
    pipeline = PipelineManager.get()
    result = pipeline.wipe_session(session_id)
    return {"session_id": session_id, "wiped": result}


# ═══════════════════════════════════════════════════════════════════════════
# Background batch runner
# ═══════════════════════════════════════════════════════════════════════════


async def _run_batch_job(
    job_id: str,
    batch_req: BatchQueryRequest,
    trace_id: str,
    session_id: str,
) -> None:
    """Execute a batch of queries in the background, updating job status.

    On exit — success, failure, or cancellation — the session is wiped,
    which drops all LangGraph checkpoints for threads opened during the
    batch plus the job entry itself. ``BATCH_RETAIN_RESULTS_SEC`` can delay
    the wipe so clients still have time to poll results.
    """
    pipeline = PipelineManager.get()
    job = pipeline._batch_jobs[job_id]
    job.status = "running"
    results: list[SingleQueryResponse] = []

    try:
        queries = batch_req.queries
        if queries is None and batch_req.dataset_path:
            logs = load_delivery_logs(batch_req.dataset_path)
            queries = []
            for sid, rows in logs.items():
                queries.append(
                    SingleQueryRequest(
                        shipment_id=sid,
                        raw_rows=rows,
                        max_loops=batch_req.max_loops,
                    )
                )
            job.total = len(queries)

        if not queries:
            job.status = "failed"
            job.error = "No queries resolved from the batch request"
            return

        for i, q in enumerate(queries):
            per_query_trace = f"{trace_id}:batch:{i}"
            try:
                raw_rows = q.raw_rows
                if raw_rows is None:
                    logs = load_delivery_logs(_DEFAULT_LOGS_CSV)
                    raw_rows = logs.get(q.shipment_id, [])
                    if not raw_rows:
                        results.append(
                            SingleQueryResponse(
                                shipment_id=q.shipment_id,
                                escalated=True,
                                final_actions=[
                                    {"action": "ERROR", "message": "Shipment not found"}
                                ],
                                trace_id=per_query_trace,
                                session_id=session_id,
                            )
                        )
                        job.failed += 1
                        job.completed = i + 1
                        continue

                state = await pipeline.run_single(
                    q.shipment_id,
                    raw_rows,
                    max_loops=q.max_loops,
                    trace_id=per_query_trace,
                    session_id=session_id,
                )
                results.append(
                    state_to_response(dict(state), per_query_trace, session_id)
                )
            except Exception as exc:
                logger.error(
                    "Batch job %s  query %d/%d failed: %s",
                    job_id, i + 1, job.total, exc,
                )
                results.append(
                    SingleQueryResponse(
                        shipment_id=q.shipment_id,
                        escalated=True,
                        final_actions=[
                            {"action": "ERROR", "message": str(exc)[:200]}
                        ],
                        trace_id=per_query_trace,
                        session_id=session_id,
                    )
                )
                job.failed += 1

            job.completed = i + 1

        job.status = "completed"
        job.results = results

    except Exception as exc:
        logger.error("Batch job %s  fatal error: %s", job_id, exc, exc_info=True)
        job.status = "failed"
        job.error = str(exc)[:500]
        job.results = results if results else None
    finally:
        retain_sec = float(os.environ.get("BATCH_RETAIN_RESULTS_SEC", "0"))
        if retain_sec > 0:
            await asyncio.sleep(retain_sec)
        try:
            pipeline.wipe_session(session_id)
        except Exception as exc:
            logger.warning(
                "batch cleanup  wipe_session failed  session_id=%s  err=%s",
                session_id,
                exc,
            )
