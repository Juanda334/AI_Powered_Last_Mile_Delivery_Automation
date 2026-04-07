"""Smoke tests — verify real infrastructure is reachable.

These tests are **opt-in** via ``pytest -m smoke`` and skip cleanly
when credentials or data files are absent, so they never block a CI
job that lacks secrets.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# OpenAI connectivity
# ---------------------------------------------------------------------------


def test_openai_connectivity_tiny_completion():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live OpenAI smoke test")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        pytest.skip("langchain_openai not installed")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5)
    resp = llm.invoke("reply with the word pong")
    assert resp.content, "expected non-empty response"


# ---------------------------------------------------------------------------
# ChromaDB connectivity
# ---------------------------------------------------------------------------


def test_chromadb_connectivity():
    needed = ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
    if not all(os.getenv(k) for k in needed):
        pytest.skip("ChromaDB creds not set — skipping live smoke test")
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    # Trivial handshake — listing collections should not raise
    collections = client.list_collections()
    assert isinstance(collections, list)


# ---------------------------------------------------------------------------
# SQLite connectivity
# ---------------------------------------------------------------------------


def test_sqlite_customers_db_reachable(project_root: Path):
    db = project_root / "data" / "external" / "customers.db"
    if not db.exists():
        pytest.skip(f"customers.db not present at {db}")
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 10"
        )
        tables = {r[0] for r in cur.fetchall()}
        # We expect at least one of these tables in a valid db
        assert tables, "customers.db has no tables"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ModelLoader can construct
# ---------------------------------------------------------------------------


def test_model_loader_constructs():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_BASE"):
        pytest.skip("OPENAI_API_KEY / OPENAI_API_BASE not set")
    from AI_Powered_Last_Mile_Delivery_Automation.utils.model_loader import ModelLoader

    loader = ModelLoader()
    gen_llm, eval_llm = loader.load_llm()
    assert gen_llm is not None
    assert eval_llm is not None
