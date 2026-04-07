"""Centralized tool library for the Last-Mile Delivery agent framework.

All LangChain ``@tool``-decorated callables live inside **ToolMaster**, which
owns the shared resources (DB connection, retriever, delivery-logs path) and
exposes each tool via the ``.tools`` property.  Instantiate once at startup and
pass ``.tools`` to your agent executor.

Architecture notes
------------------
* ``DataIngestor`` (components.data_ingestion) handles heavy-lifting for vector
  store creation, SQLite ingestion, and CSV loading.  ToolMaster reuses the same
  path constants rather than reimplementing connection logic.
* Each tool is wrapped in ``try / except`` so the LLM receives an actionable
  error string instead of an unhandled traceback.
* ``check_escalation_rules`` is pure business logic with zero I/O — it can be
  unit-tested in isolation.

Optimisation suggestions
------------------------
* **Input validation** — For tools with structured multi-field inputs
  (``check_escalation_rules`` takes 6 args), consider defining Pydantic
  ``args_schema`` models to let the framework validate *before* execution and
  give the LLM a typed error when it hallucinates an invalid value.
* **Tool granularity** — ``check_locker_availability`` combines querying *and*
  eligibility evaluation.  Splitting into ``list_lockers(zip_code)`` and
  ``evaluate_locker_eligibility(locker_id, package_size)`` would let the agent
  inspect raw data before filtering, improving transparency.
* **Connection pooling** — The single ``sqlite3.Connection`` is fine for
  single-threaded agent loops.  If you move to async or multi-agent
  concurrency, swap to an ``aiosqlite`` pool or SQLAlchemy async engine.
"""

from __future__ import annotations

import csv
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

from AI_Powered_Last_Mile_Delivery_Automation.components.data_ingestion import (
    _EXTERNAL_DB,
    _PROCESSED_DIR,
)
from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import (
    get_module_logger,
)

logger = get_module_logger("tools.tools_library")


class ToolMaster:
    """Manages shared resources and exposes LangChain-compatible tool callables.

    Parameters
    ----------
    retriever : VectorStoreRetriever, optional
        Pre-built retriever returned by ``DataIngestor.build_retriever()``.
        If *None*, ``search_playbook`` will return an error to the agent.
    db_path : Path | str | None
        Path to the SQLite database.  Defaults to the project-level
        ``_EXTERNAL_DB`` constant (``data/external/customers.db``).
    delivery_logs_path : Path | str | None
        Path to the delivery-logs CSV.  Defaults to
        ``data/processed/delivery_logs.csv``.
    """

    def __init__(
        self,
        retriever: Optional[VectorStoreRetriever] = None,
        db_path: Path | str | None = None,
        delivery_logs_path: Path | str | None = None,
    ) -> None:
        self._retriever = retriever
        self._db_path = Path(db_path) if db_path else _EXTERNAL_DB
        self._delivery_logs_path = (
            Path(delivery_logs_path)
            if delivery_logs_path
            else _PROCESSED_DIR / "delivery_logs.csv"
        )

        # Open a shared read-only SQLite connection
        self._db_conn: Optional[sqlite3.Connection] = None
        self._connect_db()

        # Build tool callables (closures that capture *self*)
        self._tools = self._build_tools()

        logger.info(
            "ToolMaster initialized  db=%s  retriever=%s  logs=%s",
            self._db_path.name,
            "ready" if self._retriever else "not set",
            self._delivery_logs_path.name,
        )

    # -- resource helpers ---------------------------------------------------

    def _connect_db(self) -> None:
        """Open (or re-open) the SQLite connection with Row factory."""
        if not self._db_path.exists():
            logger.warning("SQLite database not found: %s", self._db_path)
            return
        self._db_conn = sqlite3.connect(str(self._db_path))
        self._db_conn.row_factory = sqlite3.Row
        logger.info("SQLite connection opened: %s", self._db_path.name)

    def close(self) -> None:
        """Release the database connection.  Safe to call multiple times."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
            logger.info("SQLite connection closed")

    # -- public interface ---------------------------------------------------

    @property
    def tools(self) -> list:
        """Return all tool callables for agent registration.

        Returns
        -------
        list
            LangChain ``StructuredTool`` objects ready for an agent executor.
        """
        return list(self._tools.values())

    def get_tool(self, name: str):
        """Look up a single tool by its registered name.

        Parameters
        ----------
        name : str
            Tool name, e.g. ``"read_delivery_logs"``.

        Returns
        -------
        StructuredTool

        Raises
        ------
        KeyError
            If no tool with that name exists.
        """
        return self._tools[name]

    # ======================================================================
    # Tool factory
    # ======================================================================

    def _build_tools(self) -> dict:
        """Create ``@tool``-decorated closures that capture *self*.

        LangChain's ``@tool`` decorator inspects the function signature to
        build an input schema.  Using closures (rather than decorating instance
        methods) keeps ``self`` out of the schema.
        """
        owner = self  # captured by every closure below

        # -- 1. read_delivery_logs -----------------------------------------

        @tool("read_delivery_logs")
        def read_delivery_logs() -> list[dict]:
            """Read all delivery log rows from CSV.  Used by the preprocessor
            agent to parse raw shipment events before any other tool is invoked.

            Description:
                Opens the delivery-logs CSV and returns every row as a
                dictionary keyed by column header.  The output feeds downstream
                consolidation logic that groups events by shipment ID.

            Returns:
                list[dict]: One dict per CSV row with string values.
            """
            start = time.perf_counter()
            try:
                if not owner._delivery_logs_path.exists():
                    raise FileNotFoundError(
                        f"Delivery logs CSV not found: {owner._delivery_logs_path}"
                    )
                with open(owner._delivery_logs_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                elapsed = time.perf_counter() - start
                logger.info(
                    "read_delivery_logs  rows=%d  elapsed=%.3fs", len(rows), elapsed
                )
                return rows
            except Exception as e:
                logger.error("read_delivery_logs failed: %s", e)
                return [{"error": f"read_delivery_logs failed: {e}"}]

        # -- 2. lookup_customer_profile ------------------------------------

        @tool("lookup_customer_profile")
        def lookup_customer_profile(
            customer_id: str,
            include_pii: bool = False,
        ) -> dict:
            """Fetch a customer profile from the SQLite database.

            Description:
                Queries the ``customers`` table by primary key.  By default the
                customer's **name** is redacted (PII).  Only the Communication
                Agent should set ``include_pii=True`` when personalising
                messages.

            Args:
                customer_id: Unique identifier (e.g. ``"CUST-001"``).
                include_pii: When *True*, include the customer's name in the
                    response.  Defaults to *False*.

            Returns:
                dict: Customer profile fields, or ``{}`` if not found, or an
                ``{"error": "..."}`` dict on failure.
            """
            start = time.perf_counter()
            try:
                if owner._db_conn is None:
                    raise ConnectionError("SQLite connection is not available")

                cursor = owner._db_conn.cursor()
                cursor.execute(
                    "SELECT * FROM customers WHERE customer_id = ?",
                    (customer_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    logger.warning(
                        "lookup_customer_profile  customer_id=%s  result=NOT_FOUND",
                        customer_id,
                    )
                    return {}

                profile: dict = dict(row)
                if not include_pii:
                    profile.pop("name", None)

                elapsed = time.perf_counter() - start
                logger.info(
                    "lookup_customer_profile  customer_id=%s  include_pii=%s  elapsed=%.3fs",
                    customer_id,
                    include_pii,
                    elapsed,
                )
                return profile

            except Exception as e:
                logger.error("lookup_customer_profile failed: %s", e)
                return {"error": f"lookup_customer_profile failed: {e}"}

        # -- 3. check_locker_availability ----------------------------------

        @tool("check_locker_availability")
        def check_locker_availability(
            zip_code: str,
            package_size: str,
        ) -> list[dict]:
            """Find compatible lockers in a given zip code.

            Description:
                Queries the ``lockers`` table for entries matching *zip_code*,
                then evaluates three eligibility constraints per locker:

                1. **Size** — the locker's ``max_package_size`` must be >=
                   the package size (SMALL < MEDIUM < LARGE).
                2. **Capacity** — the locker must not be ``FULL``.
                3. **Limited capacity** — a ``LIMITED`` locker only accepts
                   ``SMALL`` packages.

                Each result dict includes ``eligible`` (bool) and ``reason``
                (str).

            Args:
                zip_code: Five-digit ZIP code for the delivery area.
                package_size: One of ``"SMALL"``, ``"MEDIUM"``, ``"LARGE"``.

            Returns:
                list[dict]: One dict per locker with all DB columns plus
                ``eligible`` and ``reason``.
            """
            start = time.perf_counter()
            try:
                if owner._db_conn is None:
                    raise ConnectionError("SQLite connection is not available")

                size_hierarchy: dict[str, int] = {"SMALL": 1, "MEDIUM": 2, "LARGE": 3}
                pkg_level = size_hierarchy.get(package_size, 0)

                cursor = owner._db_conn.cursor()
                cursor.execute("SELECT * FROM lockers WHERE zip_code = ?", (zip_code,))
                rows = cursor.fetchall()

                results: list[dict] = []
                for row in rows:
                    locker: dict = dict(row)
                    locker_max = size_hierarchy.get(locker["max_package_size"], 0)

                    if locker_max < pkg_level:
                        locker["eligible"] = False
                        locker["reason"] = (
                            f"Locker max {locker['max_package_size']} < package {package_size}"
                        )
                    elif locker["capacity_status"] == "FULL":
                        locker["eligible"] = False
                        locker["reason"] = "Locker is FULL"
                    elif (
                        locker["capacity_status"] == "LIMITED"
                        and package_size != "SMALL"
                    ):
                        locker["eligible"] = False
                        locker["reason"] = (
                            "Locker is LIMITED - only SMALL packages accepted"
                        )
                    else:
                        locker["eligible"] = True
                        locker["reason"] = "Compatible"

                    results.append(locker)

                elapsed = time.perf_counter() - start
                logger.info(
                    "check_locker_availability  zip=%s  size=%s  lockers=%d  eligible=%d  elapsed=%.3fs",
                    zip_code,
                    package_size,
                    len(results),
                    sum(1 for r in results if r["eligible"]),
                    elapsed,
                )
                return results

            except Exception as e:
                logger.error("check_locker_availability failed: %s", e)
                return [{"error": f"check_locker_availability failed: {e}"}]

        # -- 4. search_playbook --------------------------------------------

        @tool("search_playbook")
        def search_playbook(query: str) -> list[dict]:
            """Retrieve relevant playbook sections via vector search.

            Description:
                Runs an MMR similarity search against the *Exception Resolution
                Playbook* stored in ChromaDB.  Returns the top-k chunks with
                page metadata so the agent can cite its source.

                **Prerequisite:** A ``retriever`` must have been passed to
                ``ToolMaster.__init__`` (typically via
                ``DataIngestor.build_retriever()``).

            Args:
                query: Natural-language question or keyword phrase describing
                    the exception scenario (e.g. ``"damaged perishable
                    rerouting"``).

            Returns:
                list[dict]: Each dict has ``"content"`` (chunk text) and
                ``"page"`` (source page number or ``"?"``).
            """
            start = time.perf_counter()
            try:
                if owner._retriever is None:
                    raise RuntimeError(
                        "Retriever not configured. Pass a retriever to ToolMaster "
                        "or call DataIngestor.build_retriever() first."
                    )

                docs = owner._retriever.invoke(query)
                results = [
                    {"content": d.page_content, "page": d.metadata.get("page", "?")}
                    for d in docs
                ]

                elapsed = time.perf_counter() - start
                logger.info(
                    "search_playbook  query=%r  chunks=%d  elapsed=%.3fs",
                    query[:80],
                    len(results),
                    elapsed,
                )
                return results

            except Exception as e:
                logger.error("search_playbook failed: %s", e)
                return [{"error": f"search_playbook failed: {e}"}]

        # -- 5. check_escalation_rules -------------------------------------

        @tool("check_escalation_rules")
        def check_escalation_rules(
            customer_tier: str,
            exceptions_last_90d: int,
            attempt_number: int,
            package_type: str,
            status_code: str,
            status_description: str,
        ) -> dict:
            """Deterministic escalation rule engine.

            Description:
                Evaluates hard-coded business rules against shipment and
                customer attributes.  Returns a dict indicating whether
                escalation triggers fired, how many, and a human-readable list
                of trigger descriptions.

                Rules fall into two categories:

                * **AUTOMATIC** — conditions that *must* escalate (e.g. 3rd
                  failed attempt, damaged perishable, VIP with >=3 recent
                  exceptions).
                * **DISCRETIONARY** — conditions flagged for agent judgement
                  (e.g. standard customer with unusually high exception count).

                This tool performs **no I/O**; it is pure business logic.

            Args:
                customer_tier: ``"VIP"``, ``"PREMIUM"``, or ``"STANDARD"``.
                exceptions_last_90d: Number of exceptions in the last 90 days.
                attempt_number: Current delivery attempt number.
                package_type: ``"PERISHABLE"``, ``"STANDARD"``, etc.
                status_code: Exception status code (e.g. ``"DAMAGED"``,
                    ``"WEATHER_DELAY"``, ``"ADDRESS_ISSUE"``).
                status_description: Free-text description of the exception
                    event.

            Returns:
                dict: ``{"has_triggers": bool, "trigger_count": int,
                "triggers": list[str]}``.
            """
            start = time.perf_counter()
            try:
                triggers: list[str] = []

                # --- Automatic triggers ---

                if attempt_number >= 3:
                    triggers.append("AUTOMATIC: 3rd failed delivery attempt")

                if customer_tier == "VIP" and exceptions_last_90d >= 3:
                    triggers.append(
                        f"AUTOMATIC: VIP customer with {exceptions_last_90d} exceptions in 90d (>=3)"
                    )

                if status_code == "DAMAGED" and package_type == "PERISHABLE":
                    triggers.append("AUTOMATIC: Damaged perishable package")

                if status_code == "WEATHER_DELAY" and package_type == "PERISHABLE":
                    hour_matches = re.findall(
                        r"(\d+(?:\.\d+)?)\s*(?:hr|hour|hours)",
                        status_description.lower(),
                    )
                    if hour_matches:
                        hours = float(hour_matches[0])
                        if hours > 4:
                            triggers.append(
                                f"AUTOMATIC: Perishable with {hours}hr delay (>4hr threshold)"
                            )

                fraud_keywords = [
                    "vacant",
                    "demolished",
                    "construction site",
                    "empty lot",
                ]
                if status_code == "ADDRESS_ISSUE" and any(
                    kw in status_description.lower() for kw in fraud_keywords
                ):
                    triggers.append(
                        "AUTOMATIC: Potential fraud - address is vacant/demolished"
                    )

                # --- Discretionary triggers ---

                if customer_tier == "STANDARD" and exceptions_last_90d > 5:
                    triggers.append(
                        f"DISCRETIONARY: Standard customer with {exceptions_last_90d} exceptions in 90d (>5)"
                    )

                if (
                    customer_tier == "PREMIUM"
                    and package_type == "PERISHABLE"
                    and status_code == "WEATHER_DELAY"
                ):
                    triggers.append(
                        "DISCRETIONARY: Premium customer with perishable in weather delay"
                    )

                elapsed = time.perf_counter() - start
                logger.info(
                    "check_escalation_rules  tier=%s  attempt=%d  triggers=%d  elapsed=%.3fs",
                    customer_tier,
                    attempt_number,
                    len(triggers),
                    elapsed,
                )

                return {
                    "has_triggers": len(triggers) > 0,
                    "trigger_count": len(triggers),
                    "triggers": triggers,
                }

            except Exception as e:
                logger.error("check_escalation_rules failed: %s", e)
                return {"error": f"check_escalation_rules failed: {e}"}

        # -- Return all tools keyed by name --------------------------------

        return {
            "read_delivery_logs": read_delivery_logs,
            "lookup_customer_profile": lookup_customer_profile,
            "check_locker_availability": check_locker_availability,
            "search_playbook": search_playbook,
            "check_escalation_rules": check_escalation_rules,
        }
