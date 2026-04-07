"""Unit tests for utils/document_ops.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.utils.document_ops import (
    SUPPORTED_EXTENSIONS,
    load_documents,
)


pytestmark = pytest.mark.unit


def test_supported_extensions_set():
    assert SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".txt"}


def test_load_documents_txt(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("hello world\nline two", encoding="utf-8")
    docs = load_documents([f])
    assert len(docs) == 1
    assert "hello world" in docs[0].page_content


def test_load_documents_skips_unsupported(tmp_path: Path):
    bad = tmp_path / "thing.xyz"
    bad.write_text("ignored", encoding="utf-8")
    ok = tmp_path / "good.txt"
    ok.write_text("kept", encoding="utf-8")
    docs = load_documents([bad, ok])
    assert len(docs) == 1
    assert "kept" in docs[0].page_content


def test_load_documents_handles_missing_pdf_gracefully(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pdf"
    # Does not raise — just logs a warning and returns []
    docs = load_documents([missing])
    assert docs == []
