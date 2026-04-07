"""Unit tests for tools/tools_library.py.

The tools are LangChain @tool-decorated — they must be invoked via
``.invoke({...})`` rather than called with positional args.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import ToolMaster


pytestmark = pytest.mark.unit


@pytest.fixture
def tool_master(tmp_sqlite_customer_db: Path, sample_logs_path: Path) -> ToolMaster:
    tm = ToolMaster(
        retriever=None,
        db_path=tmp_sqlite_customer_db,
        delivery_logs_path=sample_logs_path,
    )
    yield tm
    tm.close()


def test_toolmaster_exposes_five_tools(tool_master: ToolMaster):
    names = {t.name for t in tool_master.tools}
    assert names == {
        "read_delivery_logs",
        "lookup_customer_profile",
        "check_locker_availability",
        "search_playbook",
        "check_escalation_rules",
    }


def test_get_tool_by_name(tool_master: ToolMaster):
    tool = tool_master.get_tool("read_delivery_logs")
    assert tool.name == "read_delivery_logs"
    with pytest.raises(KeyError):
        tool_master.get_tool("nonexistent_tool")


def test_read_delivery_logs(tool_master: ToolMaster):
    tool = tool_master.get_tool("read_delivery_logs")
    rows = tool.invoke({})
    assert isinstance(rows, list)
    assert any(r.get("shipment_id") == "SHP-T01" for r in rows)


def test_lookup_customer_profile_redacts_pii_by_default(tool_master: ToolMaster):
    tool = tool_master.get_tool("lookup_customer_profile")
    profile = tool.invoke({"customer_id": "CUST-100"})
    assert profile["customer_id"] == "CUST-100"
    assert "name" not in profile  # PII redacted
    assert profile["tier"] == "STANDARD"


def test_lookup_customer_profile_includes_pii_when_opted_in(tool_master: ToolMaster):
    tool = tool_master.get_tool("lookup_customer_profile")
    profile = tool.invoke({"customer_id": "CUST-200", "include_pii": True})
    assert profile.get("name") == "Bob"
    assert profile["tier"] == "VIP"


def test_lookup_customer_profile_missing_returns_empty(tool_master: ToolMaster):
    tool = tool_master.get_tool("lookup_customer_profile")
    assert tool.invoke({"customer_id": "CUST-DOES-NOT-EXIST"}) == {}


def test_check_locker_availability_eligibility(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_locker_availability")
    results = tool.invoke({"zip_code": "99001", "package_size": "SMALL"})
    assert len(results) == 2
    loc1 = next(r for r in results if r["locker_id"] == "LOC-001")
    assert loc1["eligible"] is True
    loc2 = next(r for r in results if r["locker_id"] == "LOC-002")
    # LIMITED locker accepts only SMALL — and we asked for SMALL
    assert loc2["eligible"] is True


def test_check_locker_availability_limited_rejects_medium(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_locker_availability")
    results = tool.invoke({"zip_code": "99001", "package_size": "MEDIUM"})
    loc2 = next(r for r in results if r["locker_id"] == "LOC-002")
    assert loc2["eligible"] is False
    assert "LIMITED" in loc2["reason"]


def test_check_locker_availability_full_rejects_all(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_locker_availability")
    results = tool.invoke({"zip_code": "99002", "package_size": "MEDIUM"})
    loc3 = next(r for r in results if r["locker_id"] == "LOC-003")
    assert loc3["eligible"] is False
    assert "FULL" in loc3["reason"]


def test_check_locker_availability_size_mismatch(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_locker_availability")
    # MEDIUM package in 99002 — LOC-003 is MEDIUM but FULL
    results = tool.invoke({"zip_code": "99002", "package_size": "LARGE"})
    loc3 = next(r for r in results if r["locker_id"] == "LOC-003")
    assert loc3["eligible"] is False
    assert "Locker max MEDIUM" in loc3["reason"]


def test_search_playbook_no_retriever_returns_error(tool_master: ToolMaster):
    tool = tool_master.get_tool("search_playbook")
    result = tool.invoke({"query": "damaged package"})
    assert isinstance(result, list) and len(result) == 1
    assert "error" in result[0]


def test_check_escalation_rules_third_attempt_triggers(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_escalation_rules")
    out = tool.invoke({
        "customer_tier": "STANDARD",
        "exceptions_last_90d": 0,
        "attempt_number": 3,
        "package_type": "STANDARD",
        "status_code": "ATTEMPTED",
        "status_description": "nobody home",
    })
    assert out["has_triggers"] is True
    assert any("3rd failed" in t for t in out["triggers"])


def test_check_escalation_rules_vip_with_many_exceptions(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_escalation_rules")
    out = tool.invoke({
        "customer_tier": "VIP",
        "exceptions_last_90d": 5,
        "attempt_number": 1,
        "package_type": "STANDARD",
        "status_code": "ATTEMPTED",
        "status_description": "x",
    })
    assert out["has_triggers"] is True
    assert any("VIP" in t for t in out["triggers"])


def test_check_escalation_rules_damaged_perishable(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_escalation_rules")
    out = tool.invoke({
        "customer_tier": "STANDARD",
        "exceptions_last_90d": 0,
        "attempt_number": 1,
        "package_type": "PERISHABLE",
        "status_code": "DAMAGED",
        "status_description": "crushed",
    })
    assert out["has_triggers"] is True
    assert any("Damaged perishable" in t for t in out["triggers"])


def test_check_escalation_rules_fraud_address(tool_master: ToolMaster):
    tool = tool_master.get_tool("check_escalation_rules")
    out = tool.invoke({
        "customer_tier": "STANDARD",
        "exceptions_last_90d": 0,
        "attempt_number": 1,
        "package_type": "STANDARD",
        "status_code": "ADDRESS_ISSUE",
        "status_description": "address is a vacant lot",
    })
    assert out["has_triggers"] is True
    assert any("fraud" in t.lower() for t in out["triggers"])


def test_check_escalation_rules_no_triggers_routine():
    from AI_Powered_Last_Mile_Delivery_Automation.tools.tools_library import ToolMaster
    tm = ToolMaster(retriever=None, db_path=Path("/nonexistent"), delivery_logs_path=Path("/nonexistent"))
    tool = tm.get_tool("check_escalation_rules")
    out = tool.invoke({
        "customer_tier": "STANDARD",
        "exceptions_last_90d": 1,
        "attempt_number": 1,
        "package_type": "STANDARD",
        "status_code": "ATTEMPTED",
        "status_description": "routine miss",
    })
    assert out["has_triggers"] is False
    assert out["trigger_count"] == 0
    tm.close()


def test_toolmaster_close_is_idempotent(tool_master: ToolMaster):
    tool_master.close()
    tool_master.close()  # no exception
