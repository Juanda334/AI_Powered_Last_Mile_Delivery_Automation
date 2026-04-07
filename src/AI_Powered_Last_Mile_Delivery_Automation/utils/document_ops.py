from __future__ import annotations
import fitz  # PyMuPDF — PDF text extraction
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.exceptions.exception import (
    DocumentPortalException,
)
from fastapi import UploadFile

logger = get_module_logger("utils.document_ops")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_pdf_with_pymupdf(pdf_path) -> list:
    """Extract text from a PDF and return LangChain Document objects."""
    documents = []

    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata

        for page_num, page in enumerate(doc):
            text = page.get_text()

            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "total_pages": len(doc),
                            "title": metadata.get("title", ""),
                            "author": metadata.get("author", ""),
                        },
                    )
                )

    return documents


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                try:
                    pdf_docs = load_pdf_with_pymupdf(str(p))
                    docs.extend(pdf_docs)
                except Exception as pdf_err:
                    logger.warning(
                        f"Failed to load PDF (corrupted or unreadable): {p} — {pdf_err}"
                    )
                    continue
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
                docs.extend(loader.load())
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
                docs.extend(loader.load())
            else:
                logger.warning(f"Unsupported extension skipped: {str(p)}")
                continue
        logger.info(f"Documents loaded: {len(docs)}")
        return docs
    except Exception as e:
        logger.error(f"Failed loading documents: {str(e)}")
        raise DocumentPortalException("Error loading documents", e) from e


class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
