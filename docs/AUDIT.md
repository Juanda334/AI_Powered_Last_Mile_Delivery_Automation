# Systemic Audit — Multi-Agent Delivery Pipeline

**Date**: 2026-04-05
**Scope**: `src/AI_Powered_Last_Mile_Delivery_Automation/**`,
`notebooks/`, `config/`, `.github/`, `requirements.txt`.
**Auditor**: Lead MLOps Architect review

---

## Executive Summary

| # | Finding | Severity | Where |
|---|---|---|---|
| 1 | API keys committed in plain YAML | **CRITICAL** | [src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml](../src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml) lines 30, 34, 39, 40 |
| 2 | No timeouts on OpenAI / SQLite / ChromaDB clients | **HIGH** | [model_loader.py:108-109](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/model_loader.py#L108-L109), [tools_library.py:105](../src/AI_Powered_Last_Mile_Delivery_Automation/tools/tools_library.py#L105), [data_ingestion.py:58-62](../src/AI_Powered_Last_Mile_Delivery_Automation/components/data_ingestion.py#L58-L62) |
| 3 | Zero test / CI infrastructure (addressed by this PR) | **HIGH** | `tests/` (was empty), `.github/workflows/` (empty) |
| 4 | All dependencies unpinned (`>=` implicit) | **MEDIUM** | [requirements.txt](../requirements.txt) |
| 5 | No caching on deterministic tools / fan-out of preprocessor fetches is serial | **MEDIUM** | [tools_library.py:387-498](../src/AI_Powered_Last_Mile_Delivery_Automation/tools/tools_library.py#L387-L498), [router_agent.py:303-401](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/router_agent.py#L303-L401) |
| 6 | Inconsistent retry policy between agents | **MEDIUM** | 3-retry on resolution/communication, 0 on critics |
| 7 | Unbounded growth of `trajectory_log` / `tool_calls_log` | **LOW** | [agent_states_view.py:24-26](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/agent_states_view.py#L24-L26) |

---

## 1. Architecture Review

### 1.1 Data flow — END-TO-END

```
START
  │
  ▼
preprocessor_node (router_agent.py:427)
  • dedupe rows → consolidate_event
  • scan_inputs_for_injection (guardrail)
  • fetch_context → 5 tool calls (SERIAL)
  • check_noise_override (skip routing when routine)
  │
  ▼
orchestrator_node (router_agent.py:596)
  • deterministic routing (NO LLM)
  • enforces max_loops and revision_count
  │
  ├─▶ resolution_agent ──▶ (back to orchestrator)
  ├─▶ critic_resolution ──▶ (back to orchestrator)
  ├─▶ communication_agent ──▶ (back to orchestrator)
  ├─▶ critic_communication ──▶ (back to orchestrator)
  │
  ▼
finalize_node (router_agent.py:807)
  • packages final_actions
  │
  ▼
END
```

State contract: 26 keys in **UnifiedAgentState** (see
[agent_states_view.py:92-139](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/agent_states_view.py#L92-L139)),
split into 5 groups:

* **Input** (2): `raw_rows`, `shipment_id`
* **Preprocessor output** (8): `consolidated_event`,
  `customer_profile`, `customer_profile_full`, `locker_availability`,
  `playbook_context`, `escalation_signals`, `noise_override`,
  `guardrail_triggered`
* **Resolution + critic** (4): `resolution_output`,
  `critic_resolution_output`, `resolution_revision_count`,
  `critic_feedback`
* **Communication + critic** (2): `communication_output`,
  `critic_communication_output`
* **Routing + final** (10): `next_agent`, `max_loops`, `escalated`,
  `tool_calls_log`, `trajectory_log`, `start_time`, `latency_sec`,
  `final_actions`

**Data-isolation**: PII (`customer_profile_full.name`) is projected
**only** into `CommunicationAgentView`; all other views receive the
redacted `customer_profile`. This is enforced by
[`project_into`](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/agent_states_view.py#L424)
+ [`merge_back`](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/agent_states_view.py#L456),
which drop out-of-scope keys silently (logged at WARNING).

### 1.2 Tool inventory (ToolMaster)

| Tool | Side effect | Timeout? | Cached? |
|---|---|---|---|
| `read_delivery_logs` | CSV read | None | No |
| `lookup_customer_profile` | SQLite SELECT | None | No |
| `check_locker_availability` | SQLite SELECT + compute | None | No |
| `search_playbook` | ChromaDB MMR search | None | No |
| `check_escalation_rules` | **Pure** — no I/O | N/A | No |

### 1.3 Loop safety

* `max_loops` defaults to 2, clamped to `[1, 5]` at entry
  ([multi_agent_workflow.py:217](../src/AI_Powered_Last_Mile_Delivery_Automation/components/multi_agent_workflow.py#L217)).
* Orchestrator enforces revision_count < max_loops before routing
  back to resolution.
* **No infinite-loop risk** under review — all cycles terminate at
  orchestrator's revision gate.

---

## 2. Latency & Cost Analysis

### 2.1 LLM call budget per shipment

| Path | Calls | Models |
|---|---|---|
| **Noise override** | 0 | — |
| **Non-exception** (happy) | 4 | 2× gen, 2× eval |
| **Exception** (1 revision) | 6 | 4× gen, 2× eval |
| **Exception** (max retries) | 8 | 6× gen, 2× eval |

### 2.2 Optimization candidates

| Lever | Target | Expected win |
|---|---|---|
| **Async preprocessor fan-out** | Parallelise the 5 tool calls in [`fetch_context`](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/router_agent.py#L304) | 3-4× preprocessor wall-clock |
| **Cache deterministic tools** | `@lru_cache` on `check_escalation_rules`, memoize `search_playbook(query)` per batch | ~10% per-case | 
| **Batch critic calls** | Combine critic_resolution + critic_communication into one LLM call when both fire in the same loop | 1× eval LLM saved |
| **Lower eval_llm for critics** | Swap gpt-4o → gpt-4o-mini on critics | ~33× cost cut (see pricing table in [evaluation_metrics.py:186-189](../src/AI_Powered_Last_Mile_Delivery_Automation/components/evaluation_metrics.py#L186-L189)) |
| **Skip critic on high-confidence resolutions** | Add rationale-length / confidence gate | ~25% of runs can skip critic |

---

## 3. Robustness & Error Handling

### 3.1 Try/except coverage — complete table

| File | Function | Lines | Coverage |
|---|---|---|---|
| [data_ingestion.py](../src/AI_Powered_Last_Mile_Delivery_Automation/components/data_ingestion.py) | `DataIngestor.__init__` | 48-50 | ✓ |
| [data_ingestion.py](../src/AI_Powered_Last_Mile_Delivery_Automation/components/data_ingestion.py) | `_connect_chromadb` | 66-68 | ✓ |
| [data_ingestion.py](../src/AI_Powered_Last_Mile_Delivery_Automation/components/data_ingestion.py) | `build_retriever` | 146-148 | ✓ |
| [tools_library.py](../src/AI_Powered_Last_Mile_Delivery_Automation/tools/tools_library.py) | all 5 tools | per-tool try/except returning error dict | ✓ |
| [router_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/router_agent.py) | `fetch_context` (5 tool calls) | 335-400 | ✓ per-call |
| [router_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/router_agent.py) | `preprocessor_node` | 578-592 | ✓ outer |
| [resolution_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/resolution_agent.py) | `resolution_agent_node` | 258/369-392 | ✓ inner + outer |
| [communication_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/communication_agent.py) | `communication_agent_node` | 304/401-426 | ✓ inner + outer |
| [critic_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/critic_agent.py) | `critic_resolution_node` | 269/339-359 | ✓ |
| [critic_agent.py](../src/AI_Powered_Last_Mile_Delivery_Automation/agents/critic_agent.py) | `critic_communication_node` | 397/462-484 | ✓ |
| [multi_agent_workflow.py](../src/AI_Powered_Last_Mile_Delivery_Automation/components/multi_agent_workflow.py) | `run_workflow` | 438/468-493 | ✓ — synthesises FATAL state on exception |

**Verdict**: no missing try/except on LLM/tool boundaries. The
`run_workflow` fallback contract (escalated=True, FATAL action,
latency populated) is the linchpin that lets batch evaluation
isolate broken shipments without crashing.

### 3.2 Timeout gaps (HIGH severity)

```python
# model_loader.py:108-109 — BEFORE
gen_llm = ChatOpenAI(model=gen_model_name, temperature=gen_temperature)
eval_llm = ChatOpenAI(model=eval_model_name, temperature=eval_temperature)

# RECOMMENDED
gen_llm = ChatOpenAI(
    model=gen_model_name,
    temperature=gen_temperature,
    timeout=30.0,         # seconds
    max_retries=2,
)
```

Same gap for `sqlite3.connect(...)` (no `timeout=` kwarg) and
`chromadb.CloudClient(...)` (no request timeout). A stuck vendor
API today blocks the pipeline indefinitely.

### 3.3 Retry policy — inconsistency

| Agent | Retry loop | Fallback |
|---|---|---|
| Resolution | 3 attempts | RESCHEDULE |
| Communication | 3 attempts | Generic message |
| Critic resolution | **0** | ESCALATE |
| Critic communication | **0** | ESCALATE |

Recommendation: add a 1-retry loop on critics too — an eval_llm
timeout currently forces an unnecessary escalation.

---

## 4. Security Findings

### 4.1 Committed secrets — **CRITICAL**

[src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml](../src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml):

| Line | Secret | Status |
|---|---|---|
| 30 | `OPENAI_API_KEY: "gl-..."` | Committed in plaintext |
| 34 | `LANGCHAIN_API_KEY: "lsv2_pt_..."` | Committed in plaintext |
| 39 | `CHROMA_API_KEY: "ck-..."` | Committed in plaintext |
| 40 | `CHROMA_TENANT: <uuid>` | Committed in plaintext |

`.env` is in [.gitignore](../.gitignore) but `config.yaml` is **not**.
These keys must be rotated immediately (see remediation below).

### 4.2 Notebook outputs — HIGH

[notebooks/research_trials.ipynb](../notebooks/research_trials.ipynb)
was used for ad-hoc experimentation. Output cells may contain API
responses with customer data or key fragments. Recommend running
`nbstripout --install` and re-committing scrubbed notebooks.

### 4.3 Recommended rotation runbook

1. In OpenAI console → revoke the committed key → issue new key.
2. In LangSmith → revoke + reissue.
3. In ChromaDB Cloud → rotate API key and tenant if possible.
4. Remove the 4 secret lines from `config.yaml` and commit an
   empty-string / `${ENV_VAR}` placeholder.
5. Add `config.yaml` to the secrets runbook and ensure
   `.env.example` documents every required variable.
6. `git filter-repo` the historical blob with the committed keys
   (irreversible — coordinate with team).

---

## 5. Observability Baseline

### 5.1 LangSmith coverage

Every public agent + orchestration function is `@traceable`-decorated.
Verified across: `router_agent.py` (9 functions), `resolution_agent.py`
(3), `communication_agent.py` (3), `critic_agent.py` (4),
`multi_agent_workflow.py` (3), `evaluation_metrics.py` (12),
`prepare_test_cases.py` (6). **Coverage grade: A.**

### 5.2 Logging

[logger/logging_config.py](../src/AI_Powered_Last_Mile_Delivery_Automation/logger/logging_config.py):

* JSON-capable formatter via `LOG_FORMAT=json` env var
* RotatingFileHandler (10 MB, 5 backups) — per-module log files
* Timestamped run dir: `logs/ai_powered_last_mile_delivery_automation_YYYYMMDD_HHMMSS/`
* Thread-safe setup under a module-level `_lock`
* Writes to both stdout and disk (container-friendly)

### 5.3 Missing metrics

The pipeline emits structured logs and LangSmith traces but does
**not** emit any aggregate production metric (prom / OTel /
CloudWatch). Recommend wiring `print_batch_report`-style aggregates
to a time-series store on each batch-evaluation run in CI (see
HARDENING.md §3).

---

## 6. Prioritized Remediation Backlog

### P0 — before next release
1. **Rotate leaked keys** and move secrets to `.env`.
2. **Add timeouts** on `ChatOpenAI`, `sqlite3.connect`, `chromadb.CloudClient`.

### P1 — next sprint
3. **Pin dependencies** via `pip-compile` → `requirements.lock`.
4. **Wire the new test suite into CI** (see HARDENING.md §2).
5. **Parallelize** `fetch_context` tool calls with `asyncio.gather`.

### P2 — backlog
6. Cache deterministic tools + playbook queries per batch.
7. Add 1-retry loop to critic agents.
8. Implement `prune_history(state, max_entries=50)` for long-running
   threads (per existing docstring TODO in agent_states_view.py:24-26).

---

*End of audit.*
