"""Session registry for per-request isolation and state wiping.

The ``SessionStore`` tracks which LangGraph checkpointer threads and which
background batch jobs belong to a given logical session. The pipeline uses
it to wipe every trace of a session from local process state once the
caller is done (success, failure, or TTL expiry).

Two backends are shipped:

* :class:`InMemorySessionStore` — default; a thread-safe dict. Correct for
  the v1 single-process deployment because LangGraph's ``MemorySaver`` is
  also in-process.
* :class:`RedisSessionStore` — stub for the future, activated once the
  LangGraph checkpointer is externalised to Redis / Postgres so both
  layers share the same backing store across uvicorn workers.

Select the backend via the ``SESSION_STORE_BACKEND`` env var
(``memory`` | ``redis``). The TTL defaults to 900s and can be tuned via
``SESSION_TTL_SEC``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("core.session_store")


_DEFAULT_TTL_SEC = float(os.environ.get("SESSION_TTL_SEC", "900"))


@dataclass
class SessionRecord:
    session_id: str
    thread_ids: set[str] = field(default_factory=set)
    batch_job_ids: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    ttl_sec: float = _DEFAULT_TTL_SEC

    def is_expired(self, now: float) -> bool:
        return (now - self.last_seen) > self.ttl_sec


class SessionStore(Protocol):
    """Backend-agnostic registry of active session artifacts."""

    def touch(self, session_id: str) -> SessionRecord: ...

    def register_thread(self, session_id: str, thread_id: str) -> None: ...

    def register_batch(self, session_id: str, job_id: str) -> None: ...

    def get(self, session_id: str) -> SessionRecord | None: ...

    def pop(self, session_id: str) -> SessionRecord | None: ...

    def expired(self, now: float) -> list[str]: ...

    def active_count(self) -> int: ...

    def backend_name(self) -> str: ...


class InMemorySessionStore:
    """Thread-safe in-process session registry."""

    def __init__(self, default_ttl_sec: float = _DEFAULT_TTL_SEC) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_sec

    def touch(self, session_id: str) -> SessionRecord:
        now = time.time()
        with self._lock:
            rec = self._records.get(session_id)
            if rec is None:
                rec = SessionRecord(session_id=session_id, ttl_sec=self._default_ttl)
                self._records[session_id] = rec
                logger.info("session opened  session_id=%s", session_id)
            else:
                rec.last_seen = now
            return rec

    def register_thread(self, session_id: str, thread_id: str) -> None:
        with self._lock:
            rec = self._records.get(session_id)
            if rec is None:
                rec = SessionRecord(session_id=session_id, ttl_sec=self._default_ttl)
                self._records[session_id] = rec
            rec.thread_ids.add(thread_id)
            rec.last_seen = time.time()

    def register_batch(self, session_id: str, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(session_id)
            if rec is None:
                rec = SessionRecord(session_id=session_id, ttl_sec=self._default_ttl)
                self._records[session_id] = rec
            rec.batch_job_ids.add(job_id)
            rec.last_seen = time.time()

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._records.get(session_id)

    def pop(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._records.pop(session_id, None)

    def expired(self, now: float) -> list[str]:
        with self._lock:
            return [sid for sid, rec in self._records.items() if rec.is_expired(now)]

    def active_count(self) -> int:
        with self._lock:
            return len(self._records)

    def backend_name(self) -> str:
        return "memory"


class RedisSessionStore:
    """Redis-backed registry — stub. Activated once LangGraph uses an
    external checkpointer (RedisSaver / PostgresSaver). Wiping local
    checkpoints from a Redis session registry cannot reach a remote
    uvicorn worker's in-process ``MemorySaver``, so this backend is
    intentionally rejected until that precondition holds.
    """

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "RedisSessionStore is not yet enabled. It requires the LangGraph "
            "checkpointer to be externalised (RedisSaver / PostgresSaver) so "
            "the session store and checkpointer share the same backing store."
        )


def build_session_store() -> SessionStore:
    """Factory — picks the backend from ``SESSION_STORE_BACKEND``."""
    backend = os.environ.get("SESSION_STORE_BACKEND", "memory").lower()
    if backend == "redis":
        return RedisSessionStore()  # type: ignore[return-value]
    return InMemorySessionStore()
