"""AI_Powered_Last_Mile_Delivery_Automation — FastAPI backend (api.py).

Orchestration gateway that wraps the multi-agent workflow as a REST API.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
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

# ── In-memory batch job store ─────────────────────────────────────────────
_batch_jobs: dict[str, BatchJobResponse] = {}

_DEFAULT_LOGS_CSV = "data/processed/delivery_logs.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline on startup, release on shutdown."""
    pipeline = PipelineManager.get()
    try:
        pipeline.initialize()
        logger.info("Pipeline initialized — server ready")
    except Exception:
        logger.exception("Pipeline initialization failed — server starting in degraded mode")
    yield
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
            "/docs": "GET",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# POST /predict
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    background_tasks: BackgroundTasks,
    x_trace_id: str | None = Header(default=None),
):
    """Primary execution endpoint — single query or batch mode."""
    pipeline = PipelineManager.get()
    if not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    trace_id = x_trace_id or trace_id_var.get("") or str(uuid.uuid4())

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
        )
        return PredictResponse(result=state_to_response(dict(result), trace_id))

    # ── Batch mode ────────────────────────────────────────────────
    if payload.batch:
        job_id = str(uuid.uuid4())
        batch_req = payload.batch

        # Resolve total count
        if batch_req.queries:
            total = len(batch_req.queries)
        else:
            total = 0  # will be updated once dataset is loaded

        job = BatchJobResponse(
            job_id=job_id,
            status="accepted",
            total=total,
        )
        _batch_jobs[job_id] = job
        background_tasks.add_task(
            _run_batch_job, job_id, batch_req, trace_id
        )
        return PredictResponse(job=job)


# ═══════════════════════════════════════════════════════════════════════════
# GET /predict/batch/{job_id}
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/predict/batch/{job_id}", response_model=BatchJobResponse)
async def get_batch_status(job_id: str):
    """Poll for batch job progress."""
    job = _batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")
    return job


# ═══════════════════════════════════════════════════════════════════════════
# Background batch runner
# ═══════════════════════════════════════════════════════════════════════════


async def _run_batch_job(
    job_id: str,
    batch_req: BatchQueryRequest,
    trace_id: str,
) -> None:
    """Execute a batch of queries in the background, updating job status."""
    pipeline = PipelineManager.get()
    job = _batch_jobs[job_id]
    job.status = "running"
    results: list[SingleQueryResponse] = []

    try:
        # Resolve queries
        queries = batch_req.queries
        if queries is None and batch_req.dataset_path:
            # Load shipment IDs from dataset file
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

        # Execute each query
        for i, q in enumerate(queries):
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
                                trace_id=trace_id,
                            )
                        )
                        job.failed += 1
                        job.completed = i + 1
                        continue

                state = await pipeline.run_single(
                    q.shipment_id,
                    raw_rows,
                    max_loops=q.max_loops,
                    trace_id=f"{trace_id}:batch:{i}",
                )
                results.append(state_to_response(dict(state), f"{trace_id}:batch:{i}"))
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
                        trace_id=f"{trace_id}:batch:{i}",
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
