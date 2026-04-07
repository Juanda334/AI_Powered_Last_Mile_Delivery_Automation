"""Pydantic request / response models for the FastAPI backend.

Defines a clean API contract that flattens the internal
``UnifiedAgentState`` into user-friendly response shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Request models
# ═══════════════════════════════════════════════════════════════════════════


class SingleQueryRequest(BaseModel):
    """A single shipment query."""

    shipment_id: str = Field(..., description="Shipment identifier, e.g. 'SHP-002'")
    raw_rows: list[dict] | None = Field(
        default=None,
        description="Optional raw delivery-log rows. If omitted, loaded from the default CSV.",
    )
    max_loops: int = Field(default=2, ge=1, le=5, description="Max revision loops")


class BatchQueryRequest(BaseModel):
    """Batch prediction — either inline queries or a dataset path."""

    queries: list[SingleQueryRequest] | None = Field(
        default=None, description="Inline list of queries"
    )
    dataset_path: str | None = Field(
        default=None, description="Path to a CSV/JSON file with shipment queries"
    )
    max_loops: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> BatchQueryRequest:
        if not self.queries and not self.dataset_path:
            raise ValueError("Provide either 'queries' or 'dataset_path'")
        if self.queries and self.dataset_path:
            raise ValueError(
                "Provide only one of 'queries' or 'dataset_path', not both"
            )
        return self


class PredictRequest(BaseModel):
    """Unified /predict request — discriminated by presence of query vs. batch."""

    query: SingleQueryRequest | None = None
    batch: BatchQueryRequest | None = None

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> PredictRequest:
        if not self.query and not self.batch:
            raise ValueError("Provide either 'query' (single) or 'batch'")
        if self.query and self.batch:
            raise ValueError("Provide only one of 'query' or 'batch', not both")
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Response models
# ═══════════════════════════════════════════════════════════════════════════


class ResolutionResult(BaseModel):
    is_exception: str | None = None
    resolution: str | None = None
    rationale: str | None = None


class CommunicationResult(BaseModel):
    tone_label: str | None = None
    communication_message: str | None = None


class SingleQueryResponse(BaseModel):
    """Flattened response for a single shipment run."""

    shipment_id: str
    resolution: ResolutionResult = Field(default_factory=ResolutionResult)
    communication: CommunicationResult = Field(default_factory=CommunicationResult)
    escalated: bool = False
    guardrail_triggered: bool = False
    resolution_revision_count: int = 0
    trajectory_log: list[str] = Field(default_factory=list)
    tool_calls_log: list[str] = Field(default_factory=list)
    final_actions: list[dict] = Field(default_factory=list)
    latency_sec: float | None = None
    trace_id: str | None = None


class BatchJobResponse(BaseModel):
    """Returned for batch requests — immediately with status='accepted', later with results."""

    job_id: str
    status: Literal["accepted", "running", "completed", "failed"] = "accepted"
    total: int = 0
    completed: int = 0
    failed: int = 0
    results: list[SingleQueryResponse] | None = None
    error: str | None = None


class PredictResponse(BaseModel):
    """Unified /predict response."""

    result: SingleQueryResponse | None = None
    job: BatchJobResponse | None = None


class HealthCheckDetail(BaseModel):
    sqlite_db: bool = False
    playbook_pdf: bool = False
    delivery_logs_csv: bool = False
    openai_api_key: bool = False
    pipeline_initialized: bool = False


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    checks: HealthCheckDetail = Field(default_factory=HealthCheckDetail)
    version: str = "0.2.0"


class HomeResponse(BaseModel):
    project: str = "AI-Powered Last-Mile Delivery Automation"
    version: str = "0.2.0"
    agents: list[str] = Field(default_factory=list)
    endpoints: dict[str, str] = Field(default_factory=dict)
    docs_url: str = "/docs"


# ═══════════════════════════════════════════════════════════════════════════
# Conversion helper
# ═══════════════════════════════════════════════════════════════════════════


def state_to_response(
    state: dict[str, Any],
    trace_id: str | None = None,
) -> SingleQueryResponse:
    """Convert a ``UnifiedAgentState`` dict into a ``SingleQueryResponse``."""
    res = state.get("resolution_output") or {}
    comm = state.get("communication_output") or {}

    return SingleQueryResponse(
        shipment_id=state.get("shipment_id", ""),
        resolution=ResolutionResult(
            is_exception=res.get("is_exception"),
            resolution=res.get("resolution"),
            rationale=res.get("rationale"),
        ),
        communication=CommunicationResult(
            tone_label=comm.get("tone_label"),
            communication_message=comm.get("communication_message"),
        ),
        escalated=state.get("escalated", False),
        guardrail_triggered=state.get("guardrail_triggered", False),
        resolution_revision_count=state.get("resolution_revision_count", 0),
        trajectory_log=state.get("trajectory_log") or [],
        tool_calls_log=state.get("tool_calls_log") or [],
        final_actions=state.get("final_actions") or [],
        latency_sec=state.get("latency_sec"),
        trace_id=trace_id,
    )
