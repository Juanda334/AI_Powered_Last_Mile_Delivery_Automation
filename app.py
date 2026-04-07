"""Backwards-compatible alias for the FastAPI backend.

The canonical backend is now ``api.py``.  This module re-exports the
``app`` instance so that ``uvicorn app:app`` continues to work.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload
    # or preferably:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

from api import app  # noqa: F401

__all__ = ["app"]
