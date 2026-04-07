from __future__ import annotations
import chromadb
from chromadb.api import ClientAPI
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from AI_Powered_Last_Mile_Delivery_Automation.utils.model_loader import ModelLoader
from AI_Powered_Last_Mile_Delivery_Automation.utils.config_loader import load_config
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)
from AI_Powered_Last_Mile_Delivery_Automation.exceptions.exception import (
    DocumentPortalException,
)
from AI_Powered_Last_Mile_Delivery_Automation.utils.document_ops import load_documents
import json
import os
import sqlite3
import hashlib
import sys

logger = get_module_logger("components.data_ingestion")

_PROJECT_ROOT = (
    Path(os.environ["PROJECT_ROOT"])
    if "PROJECT_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)
_EXTERNAL_DOC = (
    _PROJECT_ROOT / "data" / "external" / "exception_resolution_playbook.pdf"
)
_EXTERNAL_DB = _PROJECT_ROOT / "data" / "external" / "customers.db"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

COLLECTION_NAME = "Exception_Resolution_Playbook"


class DataIngestor:
    """Orchestrates document ingestion, chunking, vector storage (ChromaDB), and retrieval."""

    def __init__(self):
        try:
            self.model_loader = ModelLoader()
            self.config = load_config()

            self.embeddings = self.model_loader.load_embeddings()

            self.text_splitter_cfg = self.config.get("text_splitter", {})
            self.retriever_cfg = self.config.get("retriever", {})

            logger.info(
                f"DataIngestor initialized: chunk_size={self.text_splitter_cfg.get('chunk_size')}, "
                f"search_type={self.retriever_cfg.get('search_type')}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DataIngestor: {e}")
            raise DocumentPortalException(
                "Initialization error in DataIngestor", e
            ) from e

    # ── ChromaDB connection ──────────────────────────────────────────────

    def _connect_chromadb(self) -> ClientAPI:
        """Connect to ChromaDB Cloud using credentials from env/config."""
        keys = self.model_loader.load_chromadb_keys()
        try:
            client = chromadb.CloudClient(
                api_key=keys["CHROMA_API_KEY"],
                tenant=keys["CHROMA_TENANT"],
                database=keys["CHROMA_DATABASE"],
            )
            client.heartbeat()
            logger.info("Connected to ChromaDB Cloud")
            return client
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {e}")
            raise ConnectionError(f"ChromaDB connection failed: {e}") from e

    # ── Text splitting ───────────────────────────────────────────────────

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks using config-driven parameters."""
        chunk_size = self.text_splitter_cfg.get("chunk_size", 1500)
        chunk_overlap = self.text_splitter_cfg.get("chunk_overlap", 300)
        separators = self.text_splitter_cfg.get(
            "separators", ["\n\n", "\n", ". ", " ", ""]
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = splitter.split_documents(docs)
        logger.info(
            f"Documents split: {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
        return chunks

    # ── Idempotent ChromaDB collection ───────────────────────────────────

    def _get_or_create_collection(
        self, client: ClientAPI, chunks: List[Document]
    ) -> Chroma:
        """Get existing ChromaDB collection or create from chunks (idempotent)."""
        existing_collections = [c.name for c in client.list_collections()]

        if COLLECTION_NAME in existing_collections:
            vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            count = vector_store._collection.count()
            logger.info(
                f"Loaded existing collection '{COLLECTION_NAME}' ({count} chunks)"
            )
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=client,
                collection_name=COLLECTION_NAME,
            )
            count = vector_store._collection.count()
            logger.info(f"Created new collection '{COLLECTION_NAME}' ({count} chunks)")

        return vector_store

    # ── Build retriever (main entry point) ───────────────────────────────

    def build_retriever(self):
        """Load documents, chunk, store in ChromaDB, and return a configured retriever."""
        try:
            # Validate source file
            if not _EXTERNAL_DOC.exists():
                raise FileNotFoundError(f"Source document not found: {_EXTERNAL_DOC}")

            # Load and split documents
            docs = load_documents([_EXTERNAL_DOC])
            if not docs:
                raise ValueError("No valid documents loaded from source")
            logger.info(f"Loaded {len(docs)} pages from {_EXTERNAL_DOC.name}")

            chunks = self._split_documents(docs)

            # Connect to ChromaDB and get/create collection
            client = self._connect_chromadb()
            vector_store = self._get_or_create_collection(client, chunks)

            # Configure retriever from config
            search_type = self.retriever_cfg.get("search_type", "mmr")
            k = self.retriever_cfg.get("top_k", 5)
            search_kwargs = {"k": k}

            if search_type == "mmr":
                search_kwargs["fetch_k"] = self.retriever_cfg.get("fetch_k", 20)
                search_kwargs["lambda_mult"] = self.retriever_cfg.get(
                    "lambda_mult", 0.5
                )

            logger.info(
                f"Retriever configured: search_type={search_type}, search_kwargs={search_kwargs}"
            )
            return vector_store.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

        except Exception as e:
            logger.error(f"Failed to build retriever: {e}")
            raise DocumentPortalException("Failed to build retriever", e) from e

    # ── Tabular data loading ─────────────────────────────────────────────

    def load_tabular_data(self) -> Dict[str, pd.DataFrame]:
        """Load CSV files from the processed data directory."""
        try:
            result = {}
            csv_files = {
                "delivery_logs": _PROCESSED_DIR / "delivery_logs.csv",
                "ground_truth": _PROCESSED_DIR / "ground_truth.csv",
            }

            for name, path in csv_files.items():
                if not path.exists():
                    logger.warning(f"CSV file not found, skipping: {path}")
                    continue
                df = pd.read_csv(path)
                result[name] = df
                logger.info(
                    f"Loaded '{name}': {df.shape[0]} rows, {df.shape[1]} columns"
                )

            return result
        except Exception as e:
            logger.error(f"Failed to load tabular data: {e}")
            raise DocumentPortalException("Failed to load tabular data", e) from e

    # ── SQLite data loading ─────────────────────────────────────────────

    def load_sqlite_data(self) -> Dict[str, pd.DataFrame]:
        """Load tables from the external SQLite database."""
        try:
            if not _EXTERNAL_DB.exists():
                raise FileNotFoundError(f"SQLite database not found: {_EXTERNAL_DB}")

            logger.info(f"Loading SQLite database: {_EXTERNAL_DB.name}")

            result = {}
            tables = ["customers", "lockers"]

            with sqlite3.connect(str(_EXTERNAL_DB)) as conn:
                for table in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    result[table] = df
                    logger.info(
                        f"Loaded '{table}': {df.shape[0]} rows, {df.shape[1]} columns"
                    )

            logger.info(f"SQLite load complete: {len(result)} tables")
            return result

        except Exception as e:
            logger.error(f"Failed to load SQLite data: {e}")
            raise DocumentPortalException("Failed to load SQLite data", e) from e


# ══════════════════════════════════════════════════════════════════════════
# FAISS Manager (kept as alternative vector store backend)
# ══════════════════════════════════════════════════════════════════════════


class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {
                    "rows": {}
                }
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "index.pkl"
        ).exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(
        self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None
    ):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException(
                "No existing FAISS index and no data to create one", sys
            )
        self.vs = FAISS.from_texts(
            texts=texts, embedding=self.emb, metadatas=metadatas or []
        )
        self.vs.save_local(str(self.index_dir))
        return self.vs
