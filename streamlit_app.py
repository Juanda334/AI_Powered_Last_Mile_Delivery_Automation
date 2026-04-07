"""AI-Powered Last-Mile Delivery Automation — Streamlit dashboard.

Provides a user-friendly interface to the FastAPI backend (api.py).

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import time
import uuid
from io import StringIO

import httpx
import pandas as pd
import streamlit as st
import yaml

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

_CONFIG_PATH = "src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml"

def _load_api_url() -> str:
    """Read the API base URL from config.yaml, falling back to localhost."""
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("server", {}).get("api_base_url", "http://localhost:8080")
    except Exception:
        return "http://localhost:8080"


@st.cache_resource
def _get_http_client() -> httpx.Client:
    """Reusable HTTP client (connection pooling)."""
    return httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))


def _api_url() -> str:
    """Return the (possibly user-overridden) API URL from session state."""
    return st.session_state.get("api_url", _load_api_url())


# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Last-Mile Delivery Automation",
    page_icon="🚚",
    layout="wide",
)

st.title("AI-Powered Last-Mile Delivery Automation")

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — config, health, state viewer
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input(
        "API Base URL",
        value=_load_api_url(),
        key="api_url",
        help="The FastAPI backend URL",
    )

    # Health check
    st.subheader("System Health")
    if st.button("Check Health", key="health_btn"):
        try:
            resp = _get_http_client().get(f"{_api_url()}/health")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "unknown")
            if status == "healthy":
                st.success(f"Status: {status}")
            elif status == "degraded":
                st.warning(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

            checks = data.get("checks", {})
            for check_name, passed in checks.items():
                icon = "✅" if passed else "❌"
                st.text(f"  {icon} {check_name}")
        except httpx.ConnectError:
            st.error("Backend unavailable. Is the API server running?")
        except httpx.TimeoutException:
            st.warning("Health check timed out.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    # Agent state viewer
    st.subheader("Agent State")
    if "last_response" in st.session_state and st.session_state["last_response"]:
        with st.expander("View Full Response", expanded=False):
            st.json(st.session_state["last_response"])

    # Execution logs
    st.subheader("Execution Logs")
    if "last_trajectory" in st.session_state and st.session_state["last_trajectory"]:
        with st.expander("Trajectory Log", expanded=False):
            for entry in st.session_state["last_trajectory"]:
                st.text(entry)

    if "last_tool_calls" in st.session_state and st.session_state["last_tool_calls"]:
        with st.expander("Tool Calls Log", expanded=False):
            for entry in st.session_state["last_tool_calls"]:
                st.text(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Main — tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_single, tab_batch = st.tabs(["Single Query", "Batch Mode"])


# ── Tab 1: Single Query ──────────────────────────────────────────────────

with tab_single:
    st.subheader("Run a Single Shipment Query")

    col1, col2 = st.columns([3, 1])
    with col1:
        shipment_id = st.text_input(
            "Shipment ID",
            value="SHP-002",
            placeholder="e.g. SHP-002",
        )
    with col2:
        max_loops = st.slider("Max Revision Loops", 1, 5, 2)

    if st.button("Run Query", type="primary", key="run_single"):
        if not shipment_id.strip():
            st.error("Please enter a Shipment ID")
        else:
            trace_id = str(uuid.uuid4())
            with st.spinner("Running multi-agent workflow..."):
                try:
                    resp = _get_http_client().post(
                        f"{_api_url()}/predict",
                        json={
                            "query": {
                                "shipment_id": shipment_id.strip(),
                                "max_loops": max_loops,
                            }
                        },
                        headers={"X-Trace-Id": trace_id},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    result = data.get("result", {})

                    # Store in session state for sidebar
                    st.session_state["last_response"] = result
                    st.session_state["last_trajectory"] = result.get("trajectory_log", [])
                    st.session_state["last_tool_calls"] = result.get("tool_calls_log", [])

                    # Display results
                    st.success(f"Completed in {result.get('latency_sec', 'N/A')}s  |  Trace: {trace_id}")

                    col_res, col_comm = st.columns(2)

                    with col_res:
                        st.markdown("#### Resolution")
                        resolution = result.get("resolution", {})
                        st.metric("Exception", resolution.get("is_exception", "N/A"))
                        st.metric("Resolution", resolution.get("resolution", "N/A"))
                        escalated = "YES" if result.get("escalated") else "NO"
                        st.metric("Escalated", escalated)
                        guardrail = "TRIGGERED" if result.get("guardrail_triggered") else "CLEAR"
                        st.metric("Guardrail", guardrail)
                        st.metric("Revisions", result.get("resolution_revision_count", 0))
                        if resolution.get("rationale"):
                            with st.expander("Rationale"):
                                st.write(resolution["rationale"])

                    with col_comm:
                        st.markdown("#### Customer Communication")
                        comm = result.get("communication", {})
                        st.metric("Tone", comm.get("tone_label", "N/A"))
                        msg = comm.get("communication_message", "")
                        if msg:
                            st.info(msg)
                        else:
                            st.text("No message generated")

                    # Final actions
                    actions = result.get("final_actions", [])
                    if actions:
                        st.markdown("#### Final Actions")
                        st.dataframe(pd.DataFrame(actions), use_container_width=True)

                    # Trajectory timeline
                    trajectory = result.get("trajectory_log", [])
                    if trajectory:
                        st.markdown("#### Trajectory")
                        for i, entry in enumerate(trajectory, 1):
                            st.text(f"  {i}. {entry}")

                except httpx.ConnectError:
                    st.error("Backend unavailable. Is the API server running at " + _api_url() + "?")
                except httpx.TimeoutException:
                    st.warning(
                        "Request timed out. The LLM pipeline may be under heavy load. "
                        f"Trace ID: {trace_id}"
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 422:
                        st.error(f"Invalid input: {exc.response.json().get('detail', exc)}")
                    elif exc.response.status_code == 404:
                        st.error(f"Shipment not found: {shipment_id}")
                    elif exc.response.status_code == 503:
                        st.error("Pipeline not initialized. The backend may still be starting up.")
                    else:
                        st.error(f"Server error ({exc.response.status_code}). Trace ID: {trace_id}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")


# ── Tab 2: Batch Mode ────────────────────────────────────────────────────

with tab_batch:
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV or JSON file with shipment queries, or process all shipments from the default dataset.")

    uploaded_file = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"],
        key="batch_upload",
    )

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        batch_max_loops = st.slider("Max Revision Loops (Batch)", 1, 5, 2, key="batch_loops")
    with col_b2:
        use_default = st.checkbox("Use default delivery_logs.csv", value=True, key="use_default")

    if st.button("Run Batch", type="primary", key="run_batch"):
        trace_id = str(uuid.uuid4())
        batch_payload: dict = {"max_loops": batch_max_loops}

        # Build batch request
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    if "shipment_id" not in df.columns:
                        st.error("CSV must have a 'shipment_id' column")
                        st.stop()
                    queries = []
                    for sid in df["shipment_id"].unique():
                        rows = df[df["shipment_id"] == sid].to_dict("records")
                        queries.append({
                            "shipment_id": str(sid),
                            "raw_rows": rows,
                            "max_loops": batch_max_loops,
                        })
                    batch_payload["queries"] = queries
                elif uploaded_file.name.endswith(".json"):
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        batch_payload["queries"] = [
                            {"shipment_id": q.get("shipment_id", ""), "max_loops": batch_max_loops}
                            for q in data
                        ]
                    else:
                        st.error("JSON must be a list of objects with 'shipment_id'")
                        st.stop()
            except Exception as exc:
                st.error(f"Error parsing file: {exc}")
                st.stop()
        elif use_default:
            batch_payload["dataset_path"] = "data/processed/delivery_logs.csv"
        else:
            st.error("Please upload a file or select 'Use default delivery_logs.csv'")
            st.stop()

        # Submit batch
        with st.spinner("Submitting batch job..."):
            try:
                resp = _get_http_client().post(
                    f"{_api_url()}/predict",
                    json={"batch": batch_payload},
                    headers={"X-Trace-Id": trace_id},
                )
                resp.raise_for_status()
                data = resp.json()
                job = data.get("job", {})
                job_id = job.get("job_id", "")

                if not job_id:
                    st.error("No job ID returned from the API")
                    st.stop()

                st.info(f"Batch job submitted. Job ID: {job_id}  |  Trace: {trace_id}")

            except httpx.ConnectError:
                st.error("Backend unavailable. Is the API server running?")
                st.stop()
            except httpx.TimeoutException:
                st.warning("Submission timed out.")
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(f"Server error ({exc.response.status_code}): {exc.response.text[:200]}")
                st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        # Poll for results
        progress_bar = st.progress(0, text="Waiting for results...")
        status_text = st.empty()
        results_container = st.empty()

        poll_count = 0
        max_polls = 600  # 10 min max at 1s intervals

        while poll_count < max_polls:
            time.sleep(2)
            poll_count += 1
            try:
                resp = _get_http_client().get(
                    f"{_api_url()}/predict/batch/{job_id}",
                    headers={"X-Trace-Id": trace_id},
                )
                resp.raise_for_status()
                job_status = resp.json()

                total = job_status.get("total", 1) or 1
                completed = job_status.get("completed", 0)
                status = job_status.get("status", "unknown")
                progress = min(completed / total, 1.0)

                progress_bar.progress(progress, text=f"{completed}/{total} completed")
                status_text.text(f"Status: {status}")

                if status in ("completed", "failed"):
                    progress_bar.progress(1.0, text="Done")

                    results = job_status.get("results", [])
                    if results:
                        # Build summary table
                        rows = []
                        for r in results:
                            res = r.get("resolution", {})
                            comm = r.get("communication", {})
                            rows.append({
                                "Shipment": r.get("shipment_id", ""),
                                "Exception": res.get("is_exception", ""),
                                "Resolution": res.get("resolution", ""),
                                "Tone": comm.get("tone_label", ""),
                                "Escalated": "YES" if r.get("escalated") else "NO",
                                "Latency (s)": r.get("latency_sec", ""),
                            })
                        df_results = pd.DataFrame(rows)
                        results_container.dataframe(df_results, use_container_width=True)

                        # Store for sidebar
                        st.session_state["last_response"] = job_status

                    if job_status.get("error"):
                        st.error(f"Batch error: {job_status['error']}")

                    if status == "failed" and job_status.get("failed", 0) > 0:
                        st.warning(
                            f"{job_status['failed']} of {total} queries failed. "
                            "Check individual results for details."
                        )
                    elif status == "completed":
                        st.success(f"Batch complete: {completed}/{total} processed")
                    break

            except httpx.ConnectError:
                status_text.text("Lost connection to backend — retrying...")
            except Exception as exc:
                status_text.text(f"Poll error: {exc}")

        else:
            st.warning("Batch polling timed out after 10 minutes. Check the API directly.")
