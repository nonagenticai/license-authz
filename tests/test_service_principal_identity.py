"""Tests for service-principal metering IDENTITY resolution (azp -> tenant_id).

Wave B1: the service-principal fast path (`test_service_principal.py`) emitted
metering with `tenant_id=""` unconditionally — attributable to a caller, never
billable to a tenant. `_resolve_service_principal_tenant` looks the verified
azp's SERVICE_PRINCIPAL_TENANT_NAME_MAP candidate name up, by exact local
match, in the `name -> id` map built from licences `GET /internal/tenants`
(the sanctioned internal route — licences refuses SAs on `GET /tenants*`).

⛔ The one invariant every test here protects: an unresolved azp still emits
`tenant_id=""`. Nothing is ever invented — see
`test_name_absent_from_list_does_not_fabricate_a_tenant_id`.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from jwt_mode import VerifiedClaims
from main import app, lifespan
from metering_client import MeteringClient

TENANTS_URL = "http://licenses.test/internal/tenants"
MARKETING_ID = "0192a3b4-c5d6-7e8f-9012-345678901234"


def _tenant_rows(**names_to_ids: str) -> list[dict]:
    """A `GET /internal/tenants` body — the real InternalTenantResponse shape."""
    return [
        {
            "id": tid,
            "name": name,
            "keycloak_realm": "nonagentic",
            "archived": False,
            "kind": "service",
            "cost_centre": None,
        }
        for name, tid in names_to_ids.items()
    ]


# A realistic registry that does NOT contain any of the five candidates.
UNRELATED_ROWS = _tenant_rows(**{"svc-kag": "t-kag", "marketing": "t-marketing-business"})


def _sa_claims(azp: str) -> VerifiedClaims:
    return VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={"azp": azp, "realm_access": {"roles": ["platform_service"]}},
    )


@pytest.fixture
async def client_with_caller_jwks(monkeypatch):
    """Same shape as test_service_principal.py's fixture — duplicated rather
    than imported, so this file stays runnable/movable on its own."""
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.caller_jwks = MagicMock()
        monkeypatch.setattr(
            main._settings,
            "keycloak_secure_apps_issuer",
            "https://auth.nonagentic.ai/realms/secure-apps",
        )
        monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
        monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
        monkeypatch.setattr(main, "LICENSES_INTERNAL_TENANTS_URL", TENANTS_URL)
        # Fresh cache per test — module-level state otherwise leaks a
        # resolution (or a backoff) from one test into the next.
        main._sp_tenant_map_cache.clear()
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        main._sp_tenant_map_cache.clear()


async def _sp_request(ac, azp: str):
    with patch("main.verify_jwt", return_value=_sa_claims(azp)):
        return await ac.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )


async def _emitted_tenant_id(ac, azp: str) -> str:
    emit_calls = []
    orig = main._emit_metering_event

    async def _spy(*args, **kwargs):
        emit_calls.append(kwargs)
        return await orig(*args, **kwargs)

    with patch("main._emit_metering_event", _spy):
        resp = await _sp_request(ac, azp)
    assert resp.status_code == 200
    assert len(emit_calls) == 1
    return emit_calls[0]["tenant_id"]


def _identity_sample(client_id: str, result: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value(
            "license_authz_service_principal_identity_total",
            {"client_id": client_id, "result": result},
        )
        or 0.0
    )


# -- Resolved: the candidate name is in licences' list -----------------------


@respx.mock
async def test_name_present_in_list_resolves_to_that_rows_id(client_with_caller_jwks):
    respx.get(TENANTS_URL).respond(
        200, json=[*UNRELATED_ROWS, *_tenant_rows(**{"svc-marketing": MARKETING_ID})]
    )
    before = _identity_sample("marketing-service-account", "resolved")

    tenant_id = await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")

    assert tenant_id == MARKETING_ID
    assert _identity_sample("marketing-service-account", "resolved") == before + 1


@respx.mock
async def test_whole_map_is_cached_across_requests_and_callers(client_with_caller_jwks):
    """ONE licences GET per TTL window for the whole map — not per request,
    and not per azp."""
    route = respx.get(TENANTS_URL).respond(
        200,
        json=_tenant_rows(**{"svc-marketing": MARKETING_ID, "svc-sales": "t-sales"}),
    )
    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == (
        MARKETING_ID
    )
    assert await _emitted_tenant_id(client_with_caller_jwks, "sales-service-account") == "t-sales"
    # An azp whose name is absent is answered from the same cached map.
    assert await _emitted_tenant_id(client_with_caller_jwks, "control-relay-account") == ""
    assert route.call_count == 1


# -- Single-flight: the lock actually serializes concurrent cold-cache fetches


@respx.mock
async def test_concurrent_cold_cache_resolves_share_exactly_one_fetch(client_with_caller_jwks):
    """N concurrent requests racing a cold cache must single-flight: exactly
    ONE GET to licences, and every caller sees the same resolved tenant id.
    `test_whole_map_is_cached_across_requests_and_callers` above proves the
    CACHE (sequential awaits never contend the lock); this proves the LOCK."""
    call_count = 0

    async def _delayed(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)  # widen the race window past cache-check
        return httpx.Response(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))

    respx.get(TENANTS_URL).mock(side_effect=_delayed)

    results = await asyncio.gather(
        *(
            _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")
            for _ in range(5)
        )
    )

    assert call_count == 1, f"single-flight lock did not hold: licences fetched {call_count} times"
    assert results == [MARKETING_ID] * 5


# The pair below reproduces the pre-fix bug outside the suite: a module-level
# `asyncio.Lock()` binds to whichever event loop first contends it, then
# raises `RuntimeError: ... is bound to a different event loop` the next time
# a DIFFERENT loop contends it. pytest-asyncio gives each test function its
# own fresh loop (`asyncio_default_fixture_loop_scope = "function"`), so two
# back-to-back tests that each contend the lock are two back-to-back loops —
# equivalent to the verifier's two `asyncio.run()` calls. Order matters: both
# must run, in this order, for the pair to be a proof.
@respx.mock
async def test_lock_contention_survives_a_fresh_event_loop_1_of_2(client_with_caller_jwks):
    respx.get(TENANTS_URL).respond(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))
    results = await asyncio.gather(
        *(
            _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")
            for _ in range(5)
        )
    )
    assert results == [MARKETING_ID] * 5


@respx.mock
async def test_lock_contention_survives_a_fresh_event_loop_2_of_2(client_with_caller_jwks):
    """Paired with `..._1_of_2` above, run under a DIFFERENT event loop (see
    that test's docstring). Must not raise "bound to a different event
    loop" — a module-level lock raises here even though this test, in
    isolation, looks identical to its pair."""
    respx.get(TENANTS_URL).respond(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))
    results = await asyncio.gather(
        *(
            _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")
            for _ in range(5)
        )
    )
    assert results == [MARKETING_ID] * 5


# -- Unresolved: a usable list that does not contain the name ----------------


@respx.mock
async def test_name_absent_from_list_does_not_fabricate_a_tenant_id(client_with_caller_jwks):
    """⛔ THE INVARIANT. No SA's `svc-*` candidate tenant exists in licences
    today (measured live 2026-09-16). A look-alike row (`marketing`, the
    business tenant) must NOT be matched — exact name only."""
    respx.get(TENANTS_URL).respond(200, json=UNRELATED_ROWS)

    tenant_id = await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")

    assert tenant_id == ""
    assert _identity_sample("marketing-service-account", "unresolved") >= 1


# -- Unresolved: licences gave no usable list (non-definitive) ---------------


@respx.mock
async def test_empty_list_is_unresolved_and_not_cached(client_with_caller_jwks, monkeypatch):
    """An EMPTY list is a non-answer, not "there are no tenants": it must not be
    cached as a map, so once licences answers properly (after the backoff) the
    SA resolves."""
    monkeypatch.setattr(main, "_SP_TENANT_NEGATIVE_BACKOFF", 0.0)
    route = respx.get(TENANTS_URL).respond(200, json=[])

    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == ""
    assert _identity_sample("marketing-service-account", "unresolved") >= 1

    route.respond(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))
    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == (
        MARKETING_ID
    ), "an empty list was cached as the tenant map"
    assert route.call_count == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json={"detail": "forbidden"}),
        httpx.Response(500),
        httpx.Response(200, json={"id": MARKETING_ID, "name": "svc-marketing"}),  # not a list
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=[{"name": "svc-marketing"}, "junk"]),  # no usable row
    ],
    ids=["403", "500", "object-not-list", "non-json", "no-usable-row"],
)
async def test_unusable_licences_answer_is_unresolved(client_with_caller_jwks, response):
    with respx.mock:
        respx.get(TENANTS_URL).mock(return_value=response)
        tenant_id = await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")
    assert tenant_id == ""
    assert main._sp_tenant_map_cache.names_to_ids is None


@respx.mock
async def test_transient_failure_is_unresolved_backs_off_then_recovers(
    client_with_caller_jwks, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    route = respx.get(TENANTS_URL)
    route.side_effect = httpx.ConnectTimeout("boom")

    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == ""
    assert route.call_count == 1

    # Inside the negative backoff: licences is NOT asked again on every request.
    clock[0] += main._SP_TENANT_NEGATIVE_BACKOFF - 1
    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == ""
    assert route.call_count == 1

    # Licences recovers; after the backoff the failure was not pinned for the TTL.
    route.side_effect = None
    route.respond(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))
    clock[0] += 2
    assert clock[0] < 1000.0 + main._SP_TENANT_CACHE_TTL
    assert await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account") == (
        MARKETING_ID
    ), "a transient failure was cached for the TTL"
    assert route.call_count == 2


def test_duplicate_name_is_dropped_not_guessed():
    rows = [
        {"id": "t-1", "name": "svc-marketing"},
        {"id": "t-2", "name": "svc-marketing"},
        {"id": "t-3", "name": "svc-sales"},
    ]
    assert main._parse_internal_tenants(rows) == {"svc-sales": "t-3"}


# -- Unresolved: no candidate name for this azp -------------------------------


async def test_unknown_client_id_attempts_no_lookup(client_with_caller_jwks):
    """An azp with no entry in SERVICE_PRINCIPAL_TENANT_NAME_MAP is unresolved
    WITHOUT ever calling licences — nothing to ask it."""
    with respx.mock:
        # No route registered: any HTTP call at all raises inside respx.
        resp = await _sp_request(client_with_caller_jwks, "some-unmapped-service")
    assert resp.status_code == 200
    assert _identity_sample("other", "unresolved") >= 1


# -- Unconfigured: byte-identical to before this change -----------------------


async def test_unconfigured_url_makes_no_call_and_is_byte_identical(
    client_with_caller_jwks, monkeypatch, caplog
):
    """With the setting unset: no HTTP call, tenant_id="", the log_access reason
    is exactly `caller=<azp>`, and the structured event carries no `identity`."""
    monkeypatch.setattr(main, "LICENSES_INTERNAL_TENANTS_URL", "")
    caplog.set_level(logging.INFO, logger="main")
    with respx.mock:  # no route: any HTTP call raises
        tenant_id = await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")
    assert tenant_id == ""
    assert _identity_sample("marketing-service-account", "unresolved") >= 1

    records = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    access = [r for r in records if r.get("status") == "allowed_service_principal"]
    assert access and access[-1]["reason"] == "caller=marketing-service-account"
    assert access[-1]["tenant_id"] is None
    sp_events = [r for r in records if r.get("event") == "authz_service_principal"]
    assert sp_events and "identity" not in sp_events[-1]


@respx.mock
async def test_configured_url_keeps_reason_and_adds_identity_field(client_with_caller_jwks, caplog):
    respx.get(TENANTS_URL).respond(200, json=UNRELATED_ROWS)
    caplog.set_level(logging.INFO, logger="main")
    await _emitted_tenant_id(client_with_caller_jwks, "marketing-service-account")

    records = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    access = [r for r in records if r.get("status") == "allowed_service_principal"]
    assert access[-1]["reason"] == "caller=marketing-service-account"
    sp_events = [r for r in records if r.get("event") == "authz_service_principal"]
    assert sp_events[-1]["identity"] == "unresolved"


# -- The counter is bounded and born at zero ----------------------------------


def test_identity_counter_born_at_zero_for_the_known_census():
    from prometheus_client import REGISTRY

    for client_id in (*main.SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS, "other"):
        for result in ("resolved", "unresolved"):
            assert (
                REGISTRY.get_sample_value(
                    "license_authz_service_principal_identity_total",
                    {"client_id": client_id, "result": result},
                )
                is not None
            ), f"client_id={client_id},result={result} has no series until it fires"


def test_the_five_known_callers_are_exactly_the_measured_census():
    """Pins the census this map was built from — see the constant's docstring."""
    assert main.SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS == frozenset(
        {
            "marketing-service-account",
            "sales-service-account",
            "growth-service-account",
            "control-relay-account",
            "platform-console-backend",
        }
    )


def test_every_default_candidate_is_a_service_tenant_name():
    """Every default candidate follows the registry's `svc-<service>` convention
    for a service's OWN tenant — never a business/internal tenant such as
    `marketing` or `sales`, which would silently bill SA traffic to a cost
    centre the registry never bound it to (see the map's docstring). A unit
    test cannot measure whether the rows exist; it can pin that the code asks
    only for rows of the right KIND."""
    assert main.SERVICE_PRINCIPAL_TENANT_NAME_MAP == {
        "marketing-service-account": "svc-marketing",
        "sales-service-account": "svc-sales",
        "growth-service-account": "svc-growth",
        "control-relay-account": "svc-control-relay",
        "platform-console-backend": "svc-platform-console",
    }
    for name in main.SERVICE_PRINCIPAL_TENANT_NAME_MAP.values():
        assert name.startswith("svc-"), name


def test_identity_caller_label_clamps_unknown_azp():
    assert main._sp_identity_caller_label("marketing-service-account") == (
        "marketing-service-account"
    )
    assert main._sp_identity_caller_label("some-new-service") == "other"


# -- Round 16: the resolved tenant now actually reaches the budget check -----
#
# Everything above this section pins RESOLUTION in isolation
# (SERVICE_PRINCIPAL_BUDGET_ENFORCED=False in the fixture). These two pin the
# END-TO-END wiring: a resolved tenant id must be the tenant billing's budget
# check is keyed on, and an unresolved one must still fail open, distinguishably.

BUDGET_URL = f"{main._settings.metering_service_url.rstrip('/')}/api/budget/check"


def _budget_row(license_id: str | None, budget_tokens: int, used_tokens: int) -> dict:
    return {
        "budget_id": f"b-{license_id}",
        "tenant_id": MARKETING_ID,
        "license_id": license_id,
        "period": "monthly",
        "budget_tokens": budget_tokens,
        "used_tokens": used_tokens,
        "remaining_tokens": max(0, budget_tokens - used_tokens),
        "usage_pct": 0.0,
        "over_budget": used_tokens >= budget_tokens,
    }


@respx.mock
async def test_flag_on_with_resolved_tenant_denies_when_tenant_budget_exhausted(
    client_with_caller_jwks, monkeypatch
):
    """THE FIX, end to end: the azp resolves to MARKETING_ID and that tenant's
    TENANT-scoped billing budget is exhausted — the request is denied. A
    licence-scoped row of the same tenant has headroom and must not rescue it."""
    respx.get(TENANTS_URL).respond(200, json=_tenant_rows(**{"svc-marketing": MARKETING_ID}))
    rows = [_budget_row(None, 10, 10), _budget_row("lic-x", 1_000_000, 0)]
    budget = respx.get(BUDGET_URL).respond(
        200,
        json={
            "data": {
                "tenant_id": MARKETING_ID,
                "total_remaining": 1_000_000,
                "budget_count": 2,
                "budgets": rows,
            },
            "meta": {},
        },
    )
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", True)
    app.state.metering_client = MeteringClient()

    resp = await _sp_request(client_with_caller_jwks, "marketing-service-account")

    assert resp.status_code == 402
    body = resp.json()
    assert body["error"] == "budget_exceeded"
    assert body["kind"] == "budget"
    assert body["reason"] == "token_budget_exhausted"
    assert budget.calls.last.request.url.params["tenant_id"] == MARKETING_ID


@respx.mock
async def test_flag_on_with_unresolved_tenant_still_fails_open_and_is_counted(
    client_with_caller_jwks, monkeypatch
):
    """An unresolved SP (candidate name absent from licences' list) armed with
    SERVICE_PRINCIPAL_BUDGET_ENFORCED must still ALLOW — never denied by a
    check with no identity to key on — and must be counted distinguishably on
    the SAME series this combination already produced pre-fix: exact label
    values `license_authz_budget_fail_open_total{reason="no_license_id"}`."""
    respx.get(TENANTS_URL).respond(200, json=UNRELATED_ROWS)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", True)
    labels = {"reason": "no_license_id"}
    before = REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) or 0.0
    healthy_redis = AsyncMock()
    healthy_redis.get = AsyncMock(return_value=None)

    with patch("main._get_redis", AsyncMock(return_value=healthy_redis)):
        resp = await _sp_request(client_with_caller_jwks, "marketing-service-account")

    assert resp.status_code == 200
    assert REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) == before + 1
    healthy_redis.get.assert_not_awaited()
