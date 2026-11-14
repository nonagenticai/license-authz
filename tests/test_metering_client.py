"""Unit tests for the billing-service HTTP client (token budget + quota fallback).

⚠️ Every payload here is billing's REAL shape. This file previously asserted
against a server that does not exist — ``[{"used": ..., "budget": ...}]`` for
``/api/budget/check`` and ``{"used": ..., "quota": ...}`` for
``/api/quota/check`` — and was green for the whole life of the metering HTTP
fallback while the client could not parse a single genuine response. Billing
declares ``Envelope[BudgetAggregateStatus]`` and ``Envelope[QuotaStatus]``.

The shapes below are pinned against billing's published ``openapi.json`` by
``tests/contract/test_billing_metering_contract.py``, which synthesises its
payloads FROM that spec and refuses any field name billing does not declare.
Keep the two in agreement: if the contract guard goes red, this file is wrong,
not the guard.
"""

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from metering_client import (
    MeteringClient,
    MeteringContractError,
    _as_int,
    _quota_exceeded,
    _scoped_budget_decision,
    _unwrap_envelope,
)


def _resp(status_code: int, body):
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request("GET", "http://test"),
    )


def _budget(
    *,
    budget_tokens: int = 100_000,
    used_tokens: int = 0,
    over_budget: bool = False,
    **extra,
) -> dict:
    """One ``TokenBudgetStatus`` entry, billing's real field names."""
    item = {
        "budget_id": "019d3076-2a1f-739b-93e0-e18604846447",
        "tenant_id": "t",
        "license_id": "L",
        "period": "monthly",
        "budget_tokens": budget_tokens,
        "used_tokens": used_tokens,
        "remaining_tokens": budget_tokens - used_tokens,
        "usage_pct": 0.0,
        "over_budget": over_budget,
        "priority": 100,
        "starts_at": None,
        "expires_at": None,
        "source": "manual",
    }
    item.update(extra)
    return item


def _budget_body(*budgets: dict, next_expiry: str | None = None) -> dict:
    """``Envelope[BudgetAggregateStatus]`` — an OBJECT whose list is `budgets`."""
    return {
        "data": {
            "tenant_id": "t",
            "total_remaining": sum(b["remaining_tokens"] for b in budgets),
            "budget_count": len(budgets),
            "next_expiry": next_expiry,
            "budgets": list(budgets),
        },
        "meta": {"request_id": "r1", "next_cursor": None, "total": None},
    }


def _quota_body(
    *,
    quota_monthly: int | None = None,
    used_this_month: int = 0,
    over_quota: bool = False,
) -> dict:
    """``Envelope[QuotaStatus]``."""
    remaining = None if quota_monthly is None else quota_monthly - used_this_month
    return {
        "data": {
            "license_id": "L",
            "plan_id": "P",
            "quota_monthly": quota_monthly,
            "used_this_month": used_this_month,
            "remaining": remaining,
            "over_quota": over_quota,
        },
        "meta": {"request_id": "r2", "next_cursor": None, "total": None},
    }


_UNLIMITED_QUOTA = _quota_body()
_NO_BUDGETS = _budget_body()


NIL = "00000000-0000-0000-0000-000000000000"


def _get(*responses: httpx.Response) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = list(responses)
    return client


@pytest.mark.asyncio
class TestCheckBudget:
    async def test_no_budgets_allows(self):
        client = _get(_resp(200, _NO_BUDGETS))
        d = await MeteringClient(base_url="http://meter").check_budget(
            client, tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is True
        assert d.kind is None

    async def test_sends_only_tenant_id_to_billing(self):
        """billing declares only ``tenant_id``; scoping is done client-side."""
        client = _get(_resp(200, _NO_BUDGETS))
        await MeteringClient(base_url="http://meter").check_budget(
            client, tenant_id="t", license_id="L", auth_headers={"Authorization": "Bearer x"}
        )
        args, kwargs = client.get.call_args
        assert args[0] == "http://meter/api/budget/check"
        assert kwargs["params"] == {"tenant_id": "t"}
        assert kwargs["headers"] == {"X-Tenant-Id": "t", "Authorization": "Bearer x"}

    async def test_licence_over_its_cap_inside_a_big_tenant_pool_denies(self):
        """The live case: licence at 10,649,903 / 10,000,000, tenant pool 1.45B.

        Pooled semantics (``total_remaining > 0``) would ALLOW this. The
        per-licence cap must keep denying.
        """
        client = _get(
            _resp(
                200,
                _budget_body(
                    _budget(license_id="L", budget_tokens=10_000_000, used_tokens=10_649_903),
                    _budget(license_id=NIL, budget_tokens=1_450_000_000, used_tokens=1_000),
                    next_expiry="2026-11-01T00:00:00Z",
                ),
            )
        )
        d = await MeteringClient().check_budget(
            client, tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False
        assert d.kind == "budget"
        assert d.reason == "token_budget_exhausted"
        assert (d.used, d.limit) == (10_649_903, 10_000_000)
        assert d.reset_at == "2026-11-01T00:00:00Z"

    async def test_another_licence_over_its_cap_does_not_deny_this_one(self):
        client = _get(
            _resp(
                200,
                _budget_body(
                    _budget(license_id="OTHER", budget_tokens=10, used_tokens=99),
                    _budget(license_id="L", budget_tokens=1_000, used_tokens=10),
                ),
            )
        )
        d = await MeteringClient().check_budget(
            client, tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is True

    async def test_tenant_scoped_row_over_its_cap_denies_every_licence(self):
        body = _budget_body(
            _budget(license_id=None, budget_tokens=100, used_tokens=100),
            _budget(license_id="L1", budget_tokens=1_000_000, used_tokens=0),
        )
        for lic in ("L1", "L2", None):
            d = await MeteringClient().check_budget(
                _get(_resp(200, body)), tenant_id="t", license_id=lic, auth_headers={}
            )
            assert d.allowed is False, lic
            assert (d.used, d.limit) == (100, 100)

    async def test_nil_uuid_row_is_tenant_scoped(self):
        body = _budget_body(_budget(license_id=NIL, budget_tokens=50, used_tokens=60))
        d = await MeteringClient().check_budget(
            _get(_resp(200, body)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False

    async def test_service_principal_ignores_licence_rows(self):
        """license_id=None checks the tenant scope only."""
        body = _budget_body(_budget(license_id="L", budget_tokens=10, used_tokens=10))
        d = await MeteringClient().check_budget(
            _get(_resp(200, body)), tenant_id="t", license_id=None, auth_headers={}
        )
        assert d.allowed is True

    async def test_stacked_rows_in_one_scope_are_summed(self):
        """A drained top-up next to a funded one in the same scope still allows."""
        funded = _budget_body(
            _budget(license_id="L", budget_tokens=100, used_tokens=100),
            _budget(license_id="L", budget_tokens=50_000, used_tokens=0),
        )
        d = await MeteringClient().check_budget(
            _get(_resp(200, funded)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is True
        drained = _budget_body(
            _budget(license_id="L", budget_tokens=100, used_tokens=100),
            _budget(license_id="L", budget_tokens=900, used_tokens=900),
        )
        d = await MeteringClient().check_budget(
            _get(_resp(200, drained)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False
        assert (d.used, d.limit) == (1000, 1000)

    async def test_one_token_left_allows_and_at_cap_denies(self):
        under = _budget_body(_budget(license_id="L", budget_tokens=1000, used_tokens=999))
        at = _budget_body(_budget(license_id="L", budget_tokens=1000, used_tokens=1000))
        m = MeteringClient()
        assert (
            await m.check_budget(
                _get(_resp(200, under)), tenant_id="t", license_id="L", auth_headers={}
            )
        ).allowed is True
        assert (
            await m.check_budget(
                _get(_resp(200, at)), tenant_id="t2", license_id="L", auth_headers={}
            )
        ).allowed is False

    async def test_zero_cap_denies_at_zero_usage(self):
        body = _budget_body(_budget(license_id="L", budget_tokens=0, used_tokens=0))
        d = await MeteringClient().check_budget(
            _get(_resp(200, body)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False

    async def test_weekly_rows_count_too(self):
        body = _budget_body(
            _budget(license_id="L", period="weekly", budget_tokens=10, used_tokens=10)
        )
        d = await MeteringClient().check_budget(
            _get(_resp(200, body)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False

    async def test_bool_counter_is_a_contract_error(self):
        for field in ("used_tokens", "budget_tokens"):
            row = _budget(license_id="L")
            row[field] = False
            with pytest.raises(MeteringContractError):
                await MeteringClient().check_budget(
                    _get(_resp(200, _budget_body(row))),
                    tenant_id="t",
                    license_id="L",
                    auth_headers={},
                )

    async def test_cache_is_keyed_on_tenant_and_licence(self):
        """A tenant-only (SP) verdict must never be served to a licence, or back."""
        body = _budget_body(_budget(license_id="L", budget_tokens=10, used_tokens=10))
        client = _get(_resp(200, body), _resp(200, body))
        m = MeteringClient()
        sp = await m.check_budget(client, tenant_id="t", license_id=None, auth_headers={})
        lic = await m.check_budget(client, tenant_id="t", license_id="L", auth_headers={})
        assert sp.allowed is True and lic.allowed is False
        assert client.get.call_count == 2
        again = await m.check_budget(client, tenant_id="t", license_id="L", auth_headers={})
        assert again.allowed is False
        assert client.get.call_count == 2

    async def test_cache_expires(self):
        client = _get(_resp(200, _NO_BUDGETS), _resp(200, _NO_BUDGETS))
        m = MeteringClient(ttl=0)
        await m.check_budget(client, tenant_id="t", license_id="L", auth_headers={})
        time.sleep(0.01)
        await m.check_budget(client, tenant_id="t", license_id="L", auth_headers={})
        assert client.get.call_count == 2

    async def test_cache_max_entries_evicts_oldest(self):
        m = MeteringClient(max_entries=1000)
        from metering_client import BudgetDecision

        for i in range(1001):
            m._put(("budget", "t", f"L{i}"), BudgetDecision(allowed=True, kind=None, reason=None))
        assert len(m._cache) == 1000
        assert ("budget", "t", "L0") not in m._cache
        assert ("budget", "t", "L1000") in m._cache

    async def test_5xx_retried_once_then_raises(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _resp(503, {"detail": "down"})
        with pytest.raises(httpx.HTTPStatusError):
            await MeteringClient().check_budget(
                client, tenant_id="t", license_id="L", auth_headers={}
            )
        assert client.get.call_count == 2

    async def test_5xx_then_200_succeeds(self):
        client = _get(_resp(503, {"detail": "transient"}), _resp(200, _NO_BUDGETS))
        d = await MeteringClient().check_budget(
            client, tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is True
        assert client.get.call_count == 2

    async def test_transport_error_then_200_succeeds(self):
        """Keep-alive race: a reused connection closed by billing is retried once."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            _resp(200, _NO_BUDGETS),
        ]
        d = await MeteringClient().check_budget(
            client, tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is True
        assert client.get.call_count == 2

    async def test_transport_error_twice_raises(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("refused")
        with pytest.raises(httpx.ConnectError):
            await MeteringClient().check_budget(
                client, tenant_id="t", license_id="L", auth_headers={}
            )
        assert client.get.call_count == 2

    async def test_4xx_not_retried(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _resp(404, {"detail": "missing"})
        with pytest.raises(httpx.HTTPStatusError):
            await MeteringClient().check_budget(
                client, tenant_id="t", license_id="L", auth_headers={}
            )
        assert client.get.call_count == 1

    async def test_unparsable_body_raises_contract_error_not_attribute_error(self):
        """A 200 with the WRONG shape must be a typed, catchable contract error,
        so the call site can apply METERING_FAIL_OPEN instead of a 500 (= Envoy DENY)."""
        client = _get(_resp(200, {"data": ["not", "an", "object"], "meta": {}}))
        with pytest.raises(MeteringContractError):
            await MeteringClient().check_budget(
                client, tenant_id="t", license_id="L", auth_headers={}
            )

    async def test_bare_unenveloped_body_still_parses(self):
        bare = _budget_body(_budget(license_id="L", budget_tokens=10, used_tokens=10))["data"]
        d = await MeteringClient().check_budget(
            _get(_resp(200, bare)), tenant_id="t", license_id="L", auth_headers={}
        )
        assert d.allowed is False


@pytest.mark.asyncio
class TestCheckQuota:
    async def test_quota_exhausted_denies(self):
        client = _get(
            _resp(200, _quota_body(quota_monthly=500, used_this_month=500, over_quota=True))
        )
        d = await MeteringClient(base_url="http://meter").check_quota(
            client, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
        )
        assert d.allowed is False
        assert d.kind == "quota"
        assert d.reason == "request_quota_exhausted"
        assert d.used == 500 and d.limit == 500
        args, kwargs = client.get.call_args
        assert args[0] == "http://meter/api/quota/check"
        assert kwargs["params"] == {"license_id": "L", "plan_id": "P"}

    async def test_quota_under_limit_allows(self):
        client = _get(_resp(200, _quota_body(quota_monthly=500, used_this_month=499)))
        d = await MeteringClient().check_quota(
            client, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
        )
        assert d.allowed is True

    async def test_billing_over_quota_wins_over_local_arithmetic(self):
        """billing derives over_quota from a counter that can be ahead of the rollup."""
        client = _get(
            _resp(200, _quota_body(quota_monthly=1000, used_this_month=3, over_quota=True))
        )
        d = await MeteringClient().check_quota(
            client, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
        )
        assert d.allowed is False

    async def test_deny_is_cached(self):
        client = _get(
            _resp(200, _quota_body(quota_monthly=10, used_this_month=10, over_quota=True))
        )
        m = MeteringClient()
        for _ in range(2):
            d = await m.check_quota(
                client, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
            )
            assert d.allowed is False
        assert client.get.call_count == 1

    async def test_unparsable_quota_body_raises_contract_error(self):
        client = _get(_resp(200, {"data": ["nope"], "meta": {}}))
        with pytest.raises(MeteringContractError):
            await MeteringClient().check_quota(
                client, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
            )


# -- _unwrap_envelope --


def test_unwrap_envelope_with_data():
    raw = {"data": {"items": [1, 2, 3], "total": 3}, "meta": {"request_id": "r1"}}
    assert _unwrap_envelope(raw) == {"items": [1, 2, 3], "total": 3}


def test_unwrap_envelope_pre_t32_fallback_dict():
    raw = {"items": [1, 2, 3], "total": 3}
    assert _unwrap_envelope(raw) == {"items": [1, 2, 3], "total": 3}


def test_unwrap_envelope_none():
    assert _unwrap_envelope(None) is None


def test_unwrap_envelope_data_without_meta_returns_as_is():
    """No `meta` key → not a T3.2 envelope; return bare-T as-is."""
    raw = {"data": "x"}
    assert _unwrap_envelope(raw) == {"data": "x"}


def test_unwrap_envelope_data_null_returns_as_is():
    """`data` is None inside an otherwise valid envelope → don't unwrap to None."""
    raw = {"data": None, "meta": {"request_id": "r1"}}
    assert _unwrap_envelope(raw) == {"data": None, "meta": {"request_id": "r1"}}


def test_unwrap_envelope_bare_response_with_data_field_returns_as_is():
    """Legitimate bare response that happens to carry a `data` field (no `meta`) →
    must NOT be wrongly unwrapped to the inner string."""
    raw = {"used_this_month": 5, "quota_monthly": 10, "data": "some-string"}
    assert _unwrap_envelope(raw) == {
        "used_this_month": 5,
        "quota_monthly": 10,
        "data": "some-string",
    }


# -- _scoped_budget_decision: required fields are refused, never guessed --


def test_missing_budget_count_is_refused():
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision({"tenant_id": "t", "budgets": []}, "L")


def test_an_empty_object_is_refused():
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision({}, "L")


def test_a_null_data_envelope_is_refused():
    """``{data: null, meta}`` is not unwrapped, and the raw object is not a budget answer."""
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision(_unwrap_envelope({"data": None, "meta": {}}), "L")


def test_budget_rejects_a_list_payload():
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision([], "L")


def test_a_positive_count_with_no_rows_is_refused_not_read_as_no_budgets():
    """Reading this as "no rows" would turn every exhausted budget into an ALLOW."""
    for budgets in (None, []):
        body = {"tenant_id": "t", "total_remaining": 0, "budget_count": 2, "budgets": budgets}
        with pytest.raises(MeteringContractError):
            _scoped_budget_decision(body, "L")


def test_a_row_missing_a_verdict_field_is_refused():
    for field in ("license_id", "budget_tokens", "used_tokens"):
        row = _budget(license_id="L")
        del row[field]
        body = _budget_body(row)["data"]
        with pytest.raises(MeteringContractError):
            _scoped_budget_decision(body, "L")


def test_a_non_string_licence_id_is_refused():
    body = _budget_body(_budget(license_id=123))["data"]
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision(body, "L")


def test_a_non_integral_string_counter_is_refused():
    """``"100.5"`` is not billing's declared integer; refuse rather than guess."""
    row = _budget(license_id="L")
    row["used_tokens"] = "100.5"
    body = _budget_body(row)["data"]
    with pytest.raises(MeteringContractError):
        _scoped_budget_decision(body, "L")


# -- bool is not an integer, however much Python disagrees --


def test_as_int_rejects_bools():
    for value in (True, False):
        with pytest.raises(MeteringContractError):
            _as_int(value, "budget_count")


def test_bool_budget_count_is_refused():
    for value in (True, False):
        with pytest.raises(MeteringContractError):
            _scoped_budget_decision({"tenant_id": "t", "budget_count": value, "budgets": []}, "L")


def test_bool_quota_fields_are_refused():
    q = {"quota_monthly": True, "used_this_month": 5}
    with pytest.raises(MeteringContractError):
        _quota_exceeded(q)


# -- _quota_exceeded: billing's authoritative over_quota --


def test_quota_over_quota_true_blocks():
    assert (
        _quota_exceeded(_quota_body(quota_monthly=10, used_this_month=10, over_quota=True)["data"])
        is True
    )


def test_quota_over_quota_false_allows():
    assert _quota_exceeded(_quota_body(quota_monthly=10, used_this_month=3)["data"]) is False


def test_billing_over_quota_false_is_authoritative_at_the_limit():
    """The GREEN half of "billing's boolean wins": `false` must beat a local `true`.

    Without this, `if over_quota is not None:` can be narrowed to
    `if over_quota:` and nothing notices — a payload billing marks NOT over
    would fall through to the local derivation, which blocks at `used >= cap`,
    and a paying customer gets a 402. A missing green half reads as a missing
    capability, and this one fails in the false-DENY direction.
    """
    q = _quota_body(quota_monthly=500, used_this_month=500, over_quota=False)["data"]
    assert q["over_quota"] is False
    assert _quota_exceeded(q) is False


def test_billing_over_quota_false_is_authoritative_past_the_limit():
    """Same, one step further: billing's counter can lag the rollup either way."""
    q = _quota_body(quota_monthly=500, used_this_month=900, over_quota=False)["data"]
    assert _quota_exceeded(q) is False


def test_quota_unlimited_allows():
    """quota_monthly: null is billing's UNLIMITED branch."""
    assert _quota_exceeded(_quota_body()["data"]) is False


def test_quota_fallback_zero_cap_blocks_when_over_quota_absent():
    """quota_monthly == 0 → BLOCK ALL, matching billing's own derivation."""
    q = _quota_body(quota_monthly=0)["data"]
    del q["over_quota"]
    assert _quota_exceeded(q) is True


def test_quota_fallback_negative_cap_blocks():
    q = _quota_body(quota_monthly=-5)["data"]
    del q["over_quota"]
    assert _quota_exceeded(q) is True


def test_quota_fallback_at_limit_blocks():
    """billing's boundary is `remaining <= 0`, so used == cap is OVER quota."""
    q = _quota_body(quota_monthly=10, used_this_month=10)["data"]
    del q["over_quota"]
    assert _quota_exceeded(q) is True


def test_quota_fallback_under_limit_allows():
    q = _quota_body(quota_monthly=10, used_this_month=9)["data"]
    del q["over_quota"]
    assert _quota_exceeded(q) is False


def test_quota_fallback_unlimited_allows():
    q = _quota_body()["data"]
    del q["over_quota"]
    assert _quota_exceeded(q) is False


def test_quota_rejects_a_list_payload():
    with pytest.raises(MeteringContractError):
        _quota_exceeded([{"used": 1, "quota": 0}])


def test_quota_rejects_a_non_integer_cap():
    q = {"quota_monthly": "many", "used_this_month": 1}
    with pytest.raises(MeteringContractError):
        _quota_exceeded(q)


def test_quota_ignores_the_fictional_fields_the_old_client_read():
    """A body carrying ONLY the old fictional names must not be readable.

    Positive control on the rewrite: if `used`/`quota` still drove the verdict,
    this 501-of-500 payload would deny and the assertion would fail.
    """
    fictional = {"used": 501, "quota": 500, "period_end": "2026-09-01"}
    assert _quota_exceeded(fictional) is False
