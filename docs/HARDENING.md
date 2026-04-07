# Production Hardening Recommendations

Companion to [AUDIT.md](AUDIT.md). This document is **prescriptive**
— each section gives the exact change to make, the command to run,
or the config snippet to drop in. None of the changes below are
applied to the repo yet; they require explicit user go-ahead.

---

## 1. Dependency Management

### Current state
- [requirements.txt](../requirements.txt) uses **no version pins**.
  Any pin-free install is non-reproducible and vulnerable to
  dependency-confusion attacks.

### Recommendation: lock with `pip-compile` (pip-tools)

```bash
conda activate general
pip install pip-tools
# Create requirements.in with your top-level deps (copy from requirements.txt)
pip-compile --resolver=backtracking --output-file=requirements.lock requirements.in
pip-sync requirements.lock  # installs exact versions
```

Commit `requirements.in` (human-edited) AND `requirements.lock`
(generated). Re-run `pip-compile --upgrade` weekly to absorb CVE
patches. Alternative tools: `uv`, `poetry`, or `pdm`.

### Add a supply-chain scan

```bash
pip install pip-audit
pip-audit --requirement requirements.lock --strict
```

Wire into CI (below). Fails the build on known CVEs.

---

## 2. CI/CD Integration

### Drop-in GitHub Actions workflow

Save as `.github/workflows/tests.yml` **after** keys are rotated:

```yaml
name: tests

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  unit:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -r requirements.lock -e .
      - run: pytest -m "not smoke and not integration" --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml

  integration:
    needs: unit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13", cache: pip }
      - run: pip install -r requirements.lock -e .
      - run: pytest -m integration

  smoke-nightly:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      OPENAI_API_BASE: ${{ secrets.OPENAI_API_BASE }}
      CHROMA_API_KEY: ${{ secrets.CHROMA_API_KEY }}
      CHROMA_TENANT: ${{ secrets.CHROMA_TENANT }}
      CHROMA_DATABASE: ${{ secrets.CHROMA_DATABASE }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13", cache: pip }
      - run: pip install -r requirements.lock -e .
      - run: pytest -m smoke
```

Add a `schedule:` trigger (e.g. `cron: "0 6 * * *"`) for the
nightly smoke job.

### Batch evaluation gate (regression guard)

Add a second job that runs `prepare_test_cases.py` on every PR and
fails if `task_completion_rate` drops >5% vs `main`. The CLI
already exits non-zero on threshold breach; wire it into CI:

```yaml
  eval-regression:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13", cache: pip }
      - run: pip install -r requirements.lock -e .
      - run: |
          python -m AI_Powered_Last_Mile_Delivery_Automation.components.prepare_test_cases \
            --logs data/processed/delivery_logs.csv \
            --gt   data/processed/ground_truth.csv \
            --out-dir data/processed/test_runs \
            --pass-threshold 0.80
      - uses: actions/upload-artifact@v4
        with: { name: batch-report, path: data/processed/test_runs/ }
```

---

## 3. Observability

### 3.1 LangSmith project separation

Set env var per environment so traces don't mix:

| Env | `LANGCHAIN_PROJECT` |
|---|---|
| Local dev | `delivery-dev-<username>` |
| CI unit | (disabled — set `LANGCHAIN_TRACING_V2=false`) |
| CI batch-eval | `delivery-ci-eval` |
| Production | `delivery-prod` |

### 3.2 Dashboard — metrics to track

From `BatchReport` (see
[evaluation_metrics.py:155-179](../src/AI_Powered_Last_Mile_Delivery_Automation/components/evaluation_metrics.py#L155)):

| Panel | Metric | Alert threshold |
|---|---|---|
| Accuracy | `task_completion_rate` | < 0.80 |
| Latency | `p95_latency_sec` | > 30 s |
| Cost | `total_cost_usd` / batch | > previous × 1.5 |
| Drift | `drift_rate` | > 0.10 |
| Failure modes | `failure_breakdown` stacked bar | any bucket > 20 % |

### 3.3 Production metrics emission

Add a hook in `run_workflow` to emit a CloudWatch / StatsD metric
on every shipment (currently only logs). Example:

```python
# After "run_workflow complete" log line in multi_agent_workflow.py:504
metrics_client.timing("delivery.workflow.latency_ms", elapsed * 1000)
metrics_client.incr(
    "delivery.workflow.outcome",
    tags=[f"escalated:{result.get('escalated')}"]
)
```

---

## 4. Security

### 4.1 Secrets rotation runbook

1. **Revoke + reissue** every key listed in
   [AUDIT.md §4.1](AUDIT.md#41-committed-secrets--critical).
2. Update team's `.env` files with new values.
3. Remove the 4 plaintext lines from
   [config/config.yaml](../src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml)
   lines 30, 34, 39, 40.
4. Commit a scrubbed `config.yaml` (with empty strings or
   `${ENV_VAR}` placeholders) + force-update downstream deploys.
5. (Optional but recommended) `git filter-repo` the historical
   blobs — coordinate with the team since this rewrites history.

### 4.2 Move to `.env` + pydantic-settings

```python
# src/AI_Powered_Last_Mile_Delivery_Automation/config/settings.py  (NEW)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    LANGCHAIN_API_KEY: str | None = None
    CHROMA_API_KEY: str | None = None
    CHROMA_TENANT: str | None = None
    CHROMA_DATABASE: str | None = None


settings = Settings()
```

Replace all `os.getenv(...)` calls in
[model_loader.py](../src/AI_Powered_Last_Mile_Delivery_Automation/utils/model_loader.py)
with `from ...config.settings import settings; settings.OPENAI_API_KEY`.

### 4.3 Pre-commit hook: detect-secrets

```yaml
# .pre-commit-config.yaml  (NEW)
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

### 4.4 Strip notebook outputs

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
nbstripout notebooks/*.ipynb  # scrub existing
git add notebooks/ && git commit -m "strip notebook outputs"
```

---

## 5. Summary checklist

- [ ] Rotate the 4 keys in `config.yaml`
- [ ] Move secrets to `.env` + add `settings.py`
- [ ] Add timeouts to `ChatOpenAI` / `sqlite3.connect` / `chromadb.CloudClient`
- [ ] Lock deps with `pip-compile` → `requirements.lock`
- [ ] Drop in `.github/workflows/tests.yml` (above)
- [ ] Wire `pip-audit` into the CI unit job
- [ ] Install `detect-secrets` + `nbstripout` pre-commit hooks
- [ ] Strip existing notebook outputs
- [ ] Set `LANGCHAIN_PROJECT` per environment
- [ ] Create LangSmith dashboard for `BatchReport` metrics

*When all items are checked, the pipeline is considered production-hardened.*
