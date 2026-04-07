# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered last-mile delivery automation system with a RAG (Retrieval-Augmented Generation) pipeline for exception resolution in logistics. Uses a dual LLM architecture: GPT-4o-mini for generation and GPT-4o for evaluation. Python 3.13+ required.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .                    # editable install from src/

# Run the ML pipeline
python main.py

# Start the FastAPI inference server
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# DVC pipeline
dvc repro          # run full pipeline
dvc dag            # visualize stage dependencies
dvc metrics show   # view evaluation metrics
```

No test framework is configured yet (`test/` directory is empty).

## Architecture

### Pipeline (`main.py` + `dvc.yaml`)
Five-stage DVC pipeline: data_ingestion → data_validation → data_transformation → model_training → model_evaluation. Artifacts output to `artifacts/<stage_name>/`. Stages 2–5 are placeholder TODOs.

### RAG Ingestion (`src/.../components/data_ingestion.py`)
`DataIngestor` chunks documents and builds FAISS vector indices. `FaissManager` handles idempotent index creation/loading/updating. Source documents live in `data/external/` (e.g., `exception_resolution_playbook.pdf`).

### API (`app.py`)
FastAPI server with endpoints: `GET /` (web UI from `templates/index.html`), `GET /health`, `POST /predict` (placeholder using `sum()`—not wired to a model yet).

### Configuration
- `src/.../config/config.yaml` — embedding model (HuggingFace `all-MiniLM-L6-v2`), LLM providers, retriever params (MMR, top_k=5, fetch_k=20)
- `params.yaml` — training hyperparameters (test_size, n_estimators, learning_rate)
- `schema.yaml` — data schema (3 float64 features + float64 target)

### Utilities (`src/.../utils/`)
- `config_loader.py` — YAML config resolution (env-based or default)
- `model_loader.py` — loads HuggingFace embeddings and OpenAI LLMs, manages API keys
- `document_ops.py` — PDF/DOCX/TXT loading via PyMuPDF and LangChain

### Logging (`src/.../logger/logging_config.py`)
Centralized logging with JSON/text output, timestamped run directories, rotating file handlers. Use `get_module_logger()` for module-level loggers.

### Custom Exceptions (`src/.../exceptions/exception.py`)
`DocumentPortalException` with automatic traceback capture.

## Key Dependencies

LangChain (RAG framework), FAISS (vector search), OpenAI API, HuggingFace embeddings, FastAPI/Uvicorn, DVC, MLflow, Pandas, Scikit-learn, PyMuPDF.

### Token & Tool Efficiency
- Do not re-read files already accessed during this session unless explicitly requested.
- Minimize tool calls by prioritizing information already present in the current context.
- Work with existing context to solve tasks before seeking external data or performing redundant reads.