"""Tests for budget and quota enforcement in license-authz.

Token budgets are decided by billing-service over HTTP (its Postgres
``token_budgets`` row is the single counter); license-authz reads no Redis
budget key. Request quotas are decided by billing over HTTP too
(``/api/quota/check``). A deny returns 402 instead of 200.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import redis.asyncio
import redis.exceptions
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan

TENANT = "0192a3b4-c5d6-7e8f-9012-345678901234"
NIL = "00000000-0000-0000-0000-000000000000"
BUDGET_URL = f"{main._settings.metering_service_url.rstrip('/')}/api/budget/check"
QUOTA_URL = f"{main._settings.metering_service_url.rstrip('/')}/api/quota/check"


@pytest.fixture(autouse=True)
def _enable_budget_enforcement():
    """Enable budget enforcement for all tests in this module."""
    original = main.BUDGET_ENFORCEMENT_ENABLED
    main.BUDGET_ENFORCEMENT_ENABLED = True
    yield
    main.BUDGET_ENFORCEMENT_ENABLED = original


def _active_response(
    *,
    tenant_id: str = "0192a3b4-c5d6-7e8f-9012-345678901234",
    license_id: str = "lic_abc",
) -> dict:
    return {
        "active": True,
        "tenant_id": tenant_id,
        "subject_id": "sub_123",
        "license_id": license_id,
        "app": {"id": "app_workers"},
        "plan": {"id": "premium"},
    }


def _row(license_id: str | None, budget_tokens: int, used_tokens: int) -> dict:
    """One billing ``TokenBudgetStatus`` row."""
    return {
        "budget_id": f"b-{license_id}-{budget_tokens}",
        "tenant_id": TENANT,
        "license_id": license_id,
        "period": "monthly",
        "budget_tokens": budget_tokens,
        "used_tokens": used_tokens,
        "remaining_tokens": max(0, budget_tokens - used_tokens),
        "usage_pct": 0.0,
        "over_budget": used_tokens >= budget_tokens,
    }


def _quota(*, over_quota: bool) -> respx.Route:
    """Serve billing's ``/api/quota/check``."""
    return respx.get(QUOTA_URL).respond(
        200,
        json={
            "data": {
                "license_id": "lic",
                "plan_id": "premium",
                "quota_monthly": 500,
                "used_this_month": 500 if over_quota else 1,
                "remaining": 0 if over_quota else 499,
                "over_quota": over_quota,
            },
            "meta": {"request_id": "r"},
        },
    )


def _billing(*rows: dict) -> respx.Route:
    """Serve billing's ``/api/budget/check`` with these active rows (quota: under)."""
    _quota(over_quota=False)
    return respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {
                "tenant_id": TENANT,
                "total_remaining": sum(r["remaining_tokens"] for r in rows),
                "budget_count": len(rows),
                "next_expiry": None,
                "budgets": list(rows),
            },
            "meta": {"request_id": "r"},
        },
    )


def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _mock_redis(budget_data: dict | None = None) -> AsyncMock:
    """Create a mock Redis client with configurable GET responses."""
    mock = AsyncMock(spec=redis.asyncio.Redis)
    store: dict[str, str] = {}
    if budget_data:
        store.update(budget_data)

    async def mock_get(key: str) -> str | None:
        return store.get(key)

    mock.get = AsyncMock(side_effect=mock_get)
    mock.ping = AsyncMock(return_value=True)
    mock.xadd = AsyncMock(return_value="stream-id")
    mock.aclose = AsyncMock()
    return mock


# -- Budget enforcement tests --


async def _get(redis_client=None) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.redis = redis_client if redis_client is not None else _mock_redis()
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                return await ac.get(
                    "/api/resource", headers={"Authorization": "Bearer valid_token"}
                )
        finally:
            app.state.redis = None


@respx.mock
async def test_budget_allows_when_billing_has_no_budgets():
    respx.post(INTROSPECT_URL).respond(200, json=_active_response())
    route = _billing()
    resp = await _get()
    assert resp.status_code == 200
    assert route.called, "billing was never asked — this test proves nothing"
    assert route.calls.last.request.url.params["tenant_id"] == TENANT


@respx.mock
async def test_budget_allows_when_under_budget():
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-under"))
    _billing(_row("lic-under", 100_000, 50_000))
    resp = await _get()
    assert resp.status_code == 200


@respx.mock
async def test_budget_blocks_when_licence_budget_exhausted():
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-over"))
    _billing(_row("lic-over", 100_000, 100_000))
    resp = await _get()
    assert resp.status_code == 402
    body = resp.json()
    assert body["error"] == "budget_exceeded"
    assert body["kind"] == "budget"
    assert body["reason"] == "token_budget_exhausted"
    assert resp.headers.get("x-error-reason") == "token_budget_exhausted"


@respx.mock
async def test_licence_over_its_cap_inside_a_big_tenant_pool_still_denies():
    """The live case: licence at 10,649,903 / 10,000,000 while the tenant pool
    has 1.45B left. A pooled verdict would ALLOW; the per-licence cap denies."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-live"))
    _billing(_row("lic-live", 10_000_000, 10_649_903), _row(NIL, 1_450_000_000, 0))
    resp = await _get()
    assert resp.status_code == 402
    assert resp.json()["reason"] == "token_budget_exhausted"


@respx.mock
async def test_another_licence_over_its_cap_does_not_deny_this_one():
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-mine"))
    _billing(_row("lic-other", 10, 999), _row("lic-mine", 1_000, 1))
    resp = await _get()
    assert resp.status_code == 200


@respx.mock
async def test_budget_tenant_wide_blocks():
    """A tenant-scoped row (license_id null) binds every licence of the tenant."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-t"))
    _billing(_row(None, 50_000, 50_000), _row("lic-t", 1_000_000, 0))
    resp = await _get()
    assert resp.status_code == 402
    assert resp.json()["reason"] == "token_budget_exhausted"


@respx.mock
async def test_redis_budget_and_quota_keys_are_no_longer_read():
    """billing is the ONLY budget and quota counter: an exhausted Redis key must
    neither deny nor even be requested."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-redis"))
    _billing(_row("lic-redis", 100_000, 10))
    mk = _month_key()
    redis_client = _mock_redis(
        {
            "budget:limit:lic-redis:monthly": json.dumps({"budget_tokens": 1, "active": True}),
            f"tokens:lic-redis:{mk}": "999",
            f"budget:limit:tenant:{TENANT}:monthly": json.dumps({"budget_tokens": 1}),
            f"tokens:tenant:{TENANT}:{mk}": "999",
            "quota:limit:lic-redis": json.dumps({"quota_monthly": 1}),
            f"quota:lic-redis:{mk}": "999",
        }
    )
    resp = await _get(redis_client)
    assert resp.status_code == 200
    redis_client.get.assert_not_called()


@respx.mock
async def test_budget_is_still_enforced_when_redis_is_down():
    """The token budget no longer depends on Redis at all."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-noredis"))
    _billing(_row("lic-noredis", 10, 10))
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.redis = None
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/resource", headers={"Authorization": "Bearer valid_token"})
    assert resp.status_code == 402


@respx.mock
async def test_licence_without_a_tenant_skips_the_budget_and_is_counted():
    """billing's budget check is tenant-keyed; no tenant ⇒ nothing to ask."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(tenant_id=""))
    route = _billing(_row(None, 1, 1))
    labels = {"reason": "no_tenant_id"}
    before = REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) or 0.0
    resp = await _get()
    assert resp.status_code == 200
    assert not route.called
    assert REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) == before + 1


@respx.mock
async def test_budget_skipped_when_enforcement_disabled():
    """Budget check should be skipped when BUDGET_ENFORCEMENT_ENABLED is false."""
    main.BUDGET_ENFORCEMENT_ENABLED = False
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-disabled"))
    route = _billing(_row("lic-disabled", 100, 99_999))
    resp = await _get()
    assert resp.status_code == 200
    assert not route.called


@respx.mock
async def test_budget_metering_event_emitted_on_deny():
    """When budget is exceeded, a metering event with status=denied_budget should be emitted."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-emit"))
    _billing(_row("lic-emit", 100, 200))

    mock_redis = _mock_redis()

    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.redis = mock_redis
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/resource", headers={"Authorization": "Bearer valid_token"})
    assert resp.status_code == 402

    found = False
    for call in mock_redis.xadd.call_args_list:
        event_data = call.args[1] if len(call.args) > 1 else {}
        if isinstance(event_data, dict) and event_data.get("status") == "denied_budget":
            assert event_data["license_id"] == "lic-emit"
            found = True
            break
    assert found, "Expected metering event with status=denied_budget"


@respx.mock
async def test_quota_metering_event_emitted_as_denied_quota():
    """When request quota is exhausted, metering event should use status=denied_quota."""
    _billing()
    _quota(over_quota=True)
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-quota-evt"))

    mock_redis = _mock_redis()

    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.redis = mock_redis
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/resource", headers={"Authorization": "Bearer valid_token"})
    assert resp.status_code == 402
    assert resp.json()["reason"] == "request_quota_exhausted"

    # Verify metering event uses denied_quota (not denied_budget)
    xadd_calls = mock_redis.xadd.call_args_list
    found = False
    for call in xadd_calls:
        event_data = call.args[1] if len(call.args) > 1 else {}
        if isinstance(event_data, dict) and event_data.get("status") == "denied_quota":
            assert event_data["license_id"] == "lic-quota-evt"
            found = True
            break
    assert found, "Expected metering event with status=denied_quota"


# === Round 8: a skipped budget check leaves a record ==========================
#
# The call-site guard is `if BUDGET_ENFORCEMENT_ENABLED and result.license_id:`.
# BOTH ways of failing it were invisible: a deployment with the entire quota gate
# switched off (the DEFAULT) produced telemetry byte-identical to one enforcing
# every request. Driven through the REAL handler — a test that re-implemented the
# `if` would be grading a copy of the predicate.

SKIPPED_METRIC = "license_authz_budget_enforcement_skipped_total"


def _skipped(reason: str) -> float:
    return REGISTRY.get_sample_value(SKIPPED_METRIC, {"reason": reason}) or 0.0


@respx.mock
async def test_a_disabled_gate_is_counted(monkeypatch):
    """BUDGET_ENFORCEMENT_ENABLED=false — the default — now leaves a record."""
    monkeypatch.setattr(main, "BUDGET_ENFORCEMENT_ENABLED", False)
    _billing()
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-skip1"))
    before = _skipped("enforcement_disabled")
    before_other = _skipped("no_license_id")

    resp = await _get()

    assert resp.status_code == 200, "counting must not change the decision"
    assert _skipped("enforcement_disabled") == before + 1
    assert _skipped("no_license_id") == before_other, "the two skip reasons were folded together"


@respx.mock
async def test_an_active_licence_with_no_id_is_counted(monkeypatch):
    """Enforcement on, but the introspect result carries no licence id.

    This slipped the gate with no record at all — and it is the same shape the
    service-principal path has, so it is the arm that will move if service
    principals ever reach this call site.
    """
    monkeypatch.setattr(main, "BUDGET_ENFORCEMENT_ENABLED", True)
    body = _active_response()
    body.pop("license_id")
    _billing()
    respx.post(INTROSPECT_URL).respond(200, json=body)
    before = _skipped("no_license_id")
    before_other = _skipped("enforcement_disabled")

    resp = await _get()

    assert resp.status_code == 200
    assert _skipped("no_license_id") == before + 1
    assert _skipped("enforcement_disabled") == before_other


@respx.mock
async def test_an_enforced_request_is_counted_as_neither(monkeypatch):
    """THE CONTROL. Enforcement on, with a licence id → neither reason moves.

    Without this, the instrument could be counting every request and every other
    test here would still pass — it would then be unable to distinguish a
    disengaged gate from an engaged one, which is the whole property it buys.
    """
    monkeypatch.setattr(main, "BUDGET_ENFORCEMENT_ENABLED", True)
    _billing()
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(license_id="lic-enforced8"))
    before = {r: _skipped(r) for r in ("enforcement_disabled", "no_license_id")}

    resp = await _get()

    assert resp.status_code == 200
    assert {r: _skipped(r) for r in before} == before
