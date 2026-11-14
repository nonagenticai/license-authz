"""ext_authz call-site tests for the billing HTTP checks (token budget + request quota).

These drive the REAL :class:`MeteringClient` (not a MagicMock standing in for
it) through the real ``ext_authz_check`` handler against respx-served billing
payloads in billing's REAL shape, so the whole chain is under test: HTTP →
envelope unwrap → verdict → 402/200.

The landmine this file exists to defuse
---------------------------------------
The call site used to catch ``httpx.HTTPError`` ONLY, and applied
``METERING_FAIL_OPEN`` inside that handler. Anything else raised by
the metering client — and the client once raised ``AttributeError`` on EVERY
real billing response, because it iterated an object as if it were a list —
escaped the handler, became a 500 from the authz service, and Envoy turns a 500
from ext_authz into a DENY. Turning on the billing HTTP path would therefore not have degraded the
platform, it would have taken it down.

So: a shape error must be governed by ``METERING_FAIL_OPEN`` exactly like a
transport error, and both directions are asserted here.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import redis.asyncio
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan
from metering_client import BudgetDecision, MeteringClient

TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"
BILLING = "http://billing.test"
BUDGET_URL = f"{BILLING}/api/budget/check"
QUOTA_URL = f"{BILLING}/api/quota/check"


def _active(**overrides) -> dict:
    base = {
        "active": True,
        "tenant_id": TENANT_UUID,
        "subject_id": "sub_123",
        "license_id": "lic_abc",
        "app": {"id": "app_workers", "key": "workers"},
        "plan": {"id": "premium", "code": "PRM"},
    }
    base.update(overrides)
    return base


def _budget_body(*, budget_tokens: int, used_tokens: int) -> dict:
    """billing's real ``Envelope[BudgetAggregateStatus]``."""
    remaining = budget_tokens - used_tokens
    return {
        "data": {
            "tenant_id": TENANT_UUID,
            "total_remaining": remaining,
            "budget_count": 1,
            "next_expiry": None,
            "budgets": [
                {
                    "budget_id": "019d3076-2a1f-739b-93e0-e18604846447",
                    "tenant_id": TENANT_UUID,
                    "license_id": "lic_abc",
                    "period": "monthly",
                    "budget_tokens": budget_tokens,
                    "used_tokens": used_tokens,
                    "remaining_tokens": remaining,
                    "usage_pct": 0.0,
                    "over_budget": remaining <= 0,
                    "priority": 100,
                    "starts_at": None,
                    "expires_at": None,
                    "source": "manual",
                }
            ],
        },
        "meta": {"request_id": "r1", "next_cursor": None, "total": None},
    }


def _quota_body(*, quota_monthly, used_this_month: int, over_quota: bool) -> dict:
    """billing's real ``Envelope[QuotaStatus]``."""
    return {
        "data": {
            "license_id": "lic_abc",
            "plan_id": "premium",
            "quota_monthly": quota_monthly,
            "used_this_month": used_this_month,
            "remaining": None if quota_monthly is None else quota_monthly - used_this_month,
            "over_quota": over_quota,
        },
        "meta": {"request_id": "r2", "next_cursor": None, "total": None},
    }


def _quota_ok() -> None:
    """Serve a healthy quota check, so a test isolates the budget check."""
    respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=None, used_this_month=0, over_quota=False)
    )


def _lookup_counter(result: str) -> float:
    return (
        REGISTRY.get_sample_value("license_authz_metering_lookup_total", {"result": result}) or 0.0
    )


async def _drive(metering_client, *, fail_open: bool, monkeypatch, **active) -> httpx.Response:
    """Run one request through ext_authz with billing HTTP armed.

    The token budget, then the request quota, are asked of billing.
    """
    monkeypatch.setattr(main, "BUDGET_ENFORCEMENT_ENABLED", True)
    monkeypatch.setattr(main, "METERING_FAIL_OPEN", fail_open)

    respx.post(INTROSPECT_URL).respond(200, json=_active(**active))

    async with lifespan(app):
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        app.state.redis = mock_redis
        app.state.metering_client = metering_client
        # Service-to-service auth is not what these tests exercise; a real
        # provider would try to reach Keycloak through respx.
        sa = MagicMock()
        sa.auth_header = AsyncMock(return_value={"Authorization": "Bearer svc"})
        app.state.service_auth = sa
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                return await ac.get("/workers/api/tasks", headers={"Authorization": "Bearer tok"})
        finally:
            app.state.redis = None
            app.state.metering_client = None


# -- Real client, real billing shapes, both verdicts --------------------------


@respx.mock
async def test_real_client_allows_when_billing_reports_headroom(monkeypatch):
    budget = respx.get(BUDGET_URL).respond(
        200, json=_budget_body(budget_tokens=100_000, used_tokens=10)
    )
    quota = respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=1000, used_this_month=1, over_quota=False)
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    # Non-vacuous: prove both billing endpoints were actually queried.
    assert budget.called, "the budget check never ran — this test proves nothing"
    assert quota.called, "the quota check never ran — this test proves nothing"


@respx.mock
async def test_real_client_denies_402_when_billing_reports_exhausted_budget(monkeypatch):
    """DENY direction. A suite where nothing can deny proves nothing."""
    respx.get(BUDGET_URL).respond(
        200, json=_budget_body(budget_tokens=100_000, used_tokens=100_000)
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    body = resp.json()
    assert body["error"] == "budget_exceeded"
    assert body["kind"] == "budget"
    assert body["reason"] == "token_budget_exhausted"


@respx.mock
async def test_real_client_denies_402_when_billing_reports_over_quota(monkeypatch):
    respx.get(BUDGET_URL).respond(200, json=_budget_body(budget_tokens=100_000, used_tokens=0))
    respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=500, used_this_month=500, over_quota=True)
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    body = resp.json()
    assert body["kind"] == "quota"
    assert body["reason"] == "request_quota_exhausted"


@pytest.mark.parametrize(("plan", "sent_plan_id"), [({"id": "premium"}, "premium"), (None, None)])
@respx.mock
async def test_quota_is_always_asked_of_billing_with_plan_id_only_when_known(
    plan, sent_plan_id, monkeypatch
):
    """No Redis quota tier: billing decides every request, with or without a plan."""
    respx.get(BUDGET_URL).respond(200, json=_budget_body(budget_tokens=100_000, used_tokens=0))
    quota = respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=500, used_this_month=500, over_quota=True)
    )
    resp = await _drive(
        MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch, plan=plan
    )
    assert resp.status_code == 402
    assert resp.json()["reason"] == "request_quota_exhausted"
    assert quota.call_count == 1
    params = quota.calls.last.request.url.params
    assert params["license_id"] == "lic_abc"
    assert params.get("plan_id") == sent_plan_id


@respx.mock
async def test_the_exact_live_deployed_payload_does_not_500(monkeypatch):
    """The body deployed billing served for tenant n8n-dev on 2026-08-10.

    Under the old client this produced ``AttributeError: 'str' object has no
    attribute 'get'`` → uncaught → 500 → Envoy DENY. It must be a clean 200.
    """
    respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {
                "tenant_id": "n8n-dev",
                "total_remaining": 200000,
                "budget_count": 2,
                "next_expiry": None,
                "budgets": [
                    {
                        "budget_id": "019d3076-2a1f-739b-93e0-e18604846447",
                        "tenant_id": "n8n-dev",
                        "license_id": "069c6c62-f856-7383-9c71-5c9d54200a01",
                        "period": "monthly",
                        "budget_tokens": 100000,
                        "used_tokens": 0,
                        "remaining_tokens": 100000,
                        "usage_pct": 0.0,
                        "over_budget": False,
                        "priority": 100,
                        "starts_at": "2026-04-26T10:00:06.454876Z",
                        "expires_at": None,
                        "source": "manual",
                    },
                    {
                        "budget_id": "019d30c3-731a-7476-a702-2dfe997c05b4",
                        "tenant_id": "n8n-dev",
                        "license_id": "069c6d9f-88c0-78dd-9c95-b9855df6bd51",
                        "period": "monthly",
                        "budget_tokens": 100000,
                        "used_tokens": 0,
                        "remaining_tokens": 100000,
                        "usage_pct": 0.0,
                        "over_budget": False,
                        "priority": 100,
                        "starts_at": "2026-04-26T10:00:06.454876Z",
                        "expires_at": None,
                        "source": "manual",
                    },
                ],
            },
            "meta": {"request_id": "019fea27-b5bd-7181-bc8b-d644f0000000"},
        },
    )
    quota = respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=None, used_this_month=0, over_quota=False)
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    assert quota.called, "the live payload never reached the client — this test proves nothing"


# -- A shape error must never escape ext_authz --------------------------------


@respx.mock
async def test_billing_shape_error_fail_open_allows_and_never_500s(monkeypatch):
    """billing 200s with an uninterpretable body; FAIL_OPEN=true → allow."""
    _quota_ok()
    respx.get(BUDGET_URL).respond(200, json={"data": ["not", "an", "object"], "meta": {}})
    before = _lookup_counter("contract_error")
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200, "a parse failure must not become an ext_authz 500"
    assert _lookup_counter("contract_error") == before + 1.0


@respx.mock
async def test_billing_shape_error_fail_closed_denies_402(monkeypatch):
    """Same body, FAIL_OPEN=false → deny 402. The kill switch governs shape errors."""
    respx.get(BUDGET_URL).respond(200, json={"data": ["not", "an", "object"], "meta": {}})
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    assert resp.json()["reason"] == "metering_unavailable"


@respx.mock
async def test_an_arbitrary_programming_error_is_contained_fail_open(monkeypatch):
    """Not just MeteringContractError — ANY non-BaseException must be contained.

    This is the literal 2026-08 landmine: ``AttributeError: 'str' object has no
    attribute 'get'`` raised out of the client. Narrowing the except clause to a
    known error type would leave the platform one rename away from a 500-storm.
    """
    broken = MagicMock()
    broken.check_budget = AsyncMock(
        side_effect=AttributeError("'str' object has no attribute 'get'")
    )
    broken.check_quota = AsyncMock(
        return_value=BudgetDecision(allowed=True, kind=None, reason=None)
    )
    before = _lookup_counter("contract_error")
    resp = await _drive(broken, fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    assert _lookup_counter("contract_error") == before + 1.0


@respx.mock
async def test_an_arbitrary_programming_error_is_contained_fail_closed(monkeypatch):
    broken = MagicMock()
    broken.check_budget = AsyncMock(
        side_effect=AttributeError("'str' object has no attribute 'get'")
    )
    broken.check_quota = AsyncMock(
        return_value=BudgetDecision(allowed=True, kind=None, reason=None)
    )
    resp = await _drive(broken, fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    assert resp.json()["reason"] == "metering_unavailable"


@respx.mock
async def test_a_renamed_required_field_does_not_silently_allow_a_drained_tenant(monkeypatch):
    """The realistic drift case, end to end, at FAIL_OPEN=FALSE.

    billing reports active budgets but omits `budgets` (which it may legally
    omit only when empty — the field carries `default: []`). Reading that as
    "no rows" answers "not exceeded", so a tenant with nothing left is allowed
    through *regardless of the kill switch* — an ungovernable silent allow.
    Refusing it makes it a contract error, which the operator then governs:
    here, a 402.
    """
    respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {"tenant_id": TENANT_UUID, "remaining_total": 0, "budget_count": 3},
            "meta": {"request_id": "r1"},
        },
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402, "a drained tenant must not slip through a renamed field"
    assert resp.json()["reason"] == "metering_unavailable"


@respx.mock
async def test_a_renamed_required_field_is_at_least_observable_when_failing_open(monkeypatch):
    """At the platform's live FAIL_OPEN=true the request is still allowed — but
    LOUDLY, on the contract_error counter, instead of silently."""
    _quota_ok()
    respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {"tenant_id": TENANT_UUID, "remaining_total": 0, "budget_count": 3},
            "meta": {"request_id": "r1"},
        },
    )
    before = _lookup_counter("contract_error")
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    assert _lookup_counter("contract_error") == before + 1.0


@respx.mock
async def test_a_non_integer_used_tokens_is_governed_by_the_kill_switch(monkeypatch):
    """`used_tokens` decides the verdict, so a non-integer one is a contract
    error — never guessed at — and FAIL_OPEN=false turns it into a 402."""
    body = _budget_body(budget_tokens=100_000, used_tokens=100_000)
    body["data"]["budgets"][0]["used_tokens"] = "100000.5"
    respx.get(BUDGET_URL).respond(200, json=body)
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    assert resp.json()["reason"] == "metering_unavailable"


@respx.mock
async def test_false_budget_count_is_governed_by_the_kill_switch(monkeypatch):
    """`budget_count: false` → int 0 → "no active budgets" → ALLOW for a tenant at
    zero remaining, and nothing raised, so METERING_FAIL_OPEN could not override it
    in EITHER position. It must now be a governed contract error: 402 when closed."""
    respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {"tenant_id": TENANT_UUID, "total_remaining": 0, "budget_count": False},
            "meta": {"request_id": "r1"},
        },
    )
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    assert resp.json()["reason"] == "metering_unavailable"


@respx.mock
async def test_false_budget_count_is_observable_when_failing_open(monkeypatch):
    _quota_ok()
    respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {"tenant_id": TENANT_UUID, "total_remaining": 0, "budget_count": False},
            "meta": {"request_id": "r1"},
        },
    )
    before = _lookup_counter("contract_error")
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    assert _lookup_counter("contract_error") == before + 1.0


# -- Transport errors keep their existing, separately-labelled semantics ------


@respx.mock
async def test_transport_error_fail_open_allows(monkeypatch):
    _quota_ok()
    respx.get(BUDGET_URL).mock(side_effect=httpx.ConnectError("billing unreachable"))
    before = _lookup_counter("error")
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)
    assert resp.status_code == 200
    assert _lookup_counter("error") == before + 1.0, "transport errors keep result=error"


@respx.mock
async def test_transport_error_fail_closed_denies(monkeypatch):
    respx.get(BUDGET_URL).mock(side_effect=httpx.ConnectError("billing unreachable"))
    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)
    assert resp.status_code == 402
    assert resp.json()["reason"] == "metering_unavailable"


# -- Breadth must not swallow cancellation ------------------------------------


@respx.mock
async def test_cancellation_is_not_swallowed_by_the_widened_except(monkeypatch):
    """``except Exception`` must still let asyncio cancellation through.

    ``asyncio.CancelledError`` derives from ``BaseException``, so a client
    disconnect or server shutdown still unwinds instead of being logged as a
    metering contract error and silently converted into an allow.
    """
    cancelling = MagicMock()
    cancelling.check_budget = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _drive(cancelling, fail_open=True, monkeypatch=monkeypatch)


# === Round 8: the fallback's fail-open ALLOW is now counted as a fail-open =====
#
# THE GAP. A billing HTTP check is the ONLY enforcement for its decision — and
# when it fails with
# METERING_FAIL_OPEN=true the request is allowed with no gate having run at all.
# It incremented `metering_lookup{result="error"|"contract_error"}`, which names
# the TRANSPORT/SHAPE defect, not the ALLOW it produced, so a query asking "how
# often did the gate stop gating?" missed every one of these.

FAIL_OPEN_METRIC = "license_authz_budget_fail_open_total"


def _http_fail_open() -> float:
    return REGISTRY.get_sample_value(FAIL_OPEN_METRIC, {"reason": "metering_http_error"}) or 0.0


@respx.mock
async def test_transport_error_fail_open_counts_a_budget_fail_open(monkeypatch):
    """A transport failure that allows the request is a fail-open, and says so.

    The pre-existing `metering_lookup{result="error"}` assertion is kept alongside
    so the defect counter is not traded for the fail-open counter.
    """
    _quota_ok()
    respx.get(BUDGET_URL).mock(side_effect=httpx.ConnectError("billing unreachable"))
    before = _http_fail_open()
    before_lookup = _lookup_counter("error")

    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)

    assert resp.status_code == 200
    assert _http_fail_open() == before + 1.0
    assert _lookup_counter("error") == before_lookup + 1.0


@respx.mock
async def test_shape_error_fail_open_counts_a_budget_fail_open(monkeypatch):
    """A response the client cannot parse disengages the last gate too.

    Same body the pre-existing `contract_error` tests use, so this exercises the
    broad-Exception arm rather than a shape of my own invention.
    """
    _quota_ok()
    respx.get(BUDGET_URL).respond(200, json={"data": ["not", "an", "object"], "meta": {}})
    before = _http_fail_open()
    before_lookup = _lookup_counter("contract_error")

    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)

    assert resp.status_code == 200
    assert _lookup_counter("contract_error") == before_lookup + 1.0, "the shape arm was not reached"
    assert _http_fail_open() == before + 1.0


@respx.mock
async def test_a_fail_closed_denial_is_not_counted_as_a_fail_open(monkeypatch):
    """THE CONTROL. With METERING_FAIL_OPEN=false the same fault DENIES.

    A deny is not a fail-open, and counting it as one would make the instrument
    read high exactly where enforcement is working — inverting its meaning.
    """
    respx.get(BUDGET_URL).mock(side_effect=httpx.ConnectError("billing unreachable"))
    before = _http_fail_open()

    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=False, monkeypatch=monkeypatch)

    assert resp.status_code == 402
    assert _http_fail_open() == before, "a DENY was counted as a fail-open"


@respx.mock
async def test_a_successful_fallback_allow_is_not_counted_as_a_fail_open(monkeypatch):
    """THE SECOND CONTROL. A billing check that WORKS and allows is enforcement, not a
    fail-open — without this, counting every allow would be indistinguishable."""
    budget = respx.get(BUDGET_URL).respond(
        200, json=_budget_body(budget_tokens=100_000, used_tokens=10)
    )
    respx.get(QUOTA_URL).respond(
        200, json=_quota_body(quota_monthly=1000, used_this_month=1, over_quota=False)
    )
    before = _http_fail_open()

    resp = await _drive(MeteringClient(base_url=BILLING), fail_open=True, monkeypatch=monkeypatch)

    assert resp.status_code == 200
    assert budget.called, "the fallback never ran — this control proves nothing"
    assert _http_fail_open() == before
