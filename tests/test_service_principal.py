"""Tests for the service-principal auth path in ext_authz_check.

When SERVICE_PRINCIPAL_AUTH_ENABLED is ON, a request carrying a *verified*
secure-apps SA token in X-Caller-Authorization but NO primary license token in
Authorization is PERMITted as a trusted first-party service principal (identity
only: caller overlay injected, tenant/license headers empty). Default OFF keeps
the shared license-token path byte-for-byte unchanged for every other waypoint.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.exceptions
import respx
from httpx import ASGITransport, AsyncClient

import main
from jwt_mode import InvalidJWT, VerifiedClaims
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan


def _active() -> dict:
    return {
        "active": True,
        # Canonical tenant UUID — license-authz injects it directly as x-tenant-id.
        "tenant_id": "0192a3b4-c5d6-7e8f-9012-345678901234",
        "subject_id": "sub_x",
        "license_id": "lic_x",
        "app": {"id": "app_workers", "key": "workers"},
        "plan": {"id": "premium", "code": "PRM"},
    }


@pytest.fixture
async def client_with_caller_jwks(monkeypatch):
    """Client with caller_jwks stubbed + the secure-apps issuer set, so a
    patched verify_jwt is exercised through _extract_caller_identity."""
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.caller_jwks = MagicMock()
        monkeypatch.setattr(
            main._settings,
            "keycloak_secure_apps_issuer",
            "https://auth.nonagentic.ai/realms/secure-apps",
        )
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _sa_claims() -> VerifiedClaims:
    return VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={
            "azp": "marketing-service-account",
            "realm_access": {"roles": ["platform_service"]},
        },
    )


# -- Flag ON: SA-only request is PERMITted as a service principal -----------


async def test_service_principal_allows_sa_only_when_enabled(client_with_caller_jwks, monkeypatch):
    """Flag on + verified SA token + NO Authorization → 200, caller overlay
    injected, tenant/license headers empty. No introspect call happens."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.headers["x-caller-client-id"] == "marketing-service-account"
    assert resp.headers["x-caller-roles"] == "platform_service"
    # Identity-only: no tenant/license leaks into the trusted headers.
    assert resp.headers.get("x-tenant-id", "") == ""
    assert resp.headers.get("x-license-id", "") == ""
    # Service-principal mode has no verified end-user identity — x-actor-id
    # must never be emitted here.
    assert "x-actor-id" not in resp.headers


# -- Flag OFF (default): SA-only request still 401s -------------------------


async def test_service_principal_disabled_sa_only_401(client_with_caller_jwks, monkeypatch):
    """Flag off → a verified SA token without a license token is NOT sufficient;
    falls through to the existing missing_bearer_token 401 (unchanged behavior)."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", False)
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "missing_bearer_token"
    assert resp.headers["x-error-reason"] == "missing_bearer_token"


# -- Flag ON but invalid SA token → 401 (verification still fails loud) ------


async def test_service_principal_invalid_sa_token_401(client_with_caller_jwks, monkeypatch):
    """A present-but-invalid SA token is rejected by _extract_caller_identity
    before the service-principal branch is ever reached, flag or not."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with patch("main.verify_jwt", side_effect=InvalidJWT("bad sig")):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer bad-jwt"},
        )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "caller_token_invalid"
    assert resp.headers["x-error-reason"] == "caller_token_invalid"


# -- Flag ON but NO caller token → 401 (nothing to authorize the hop) -------


async def test_service_principal_no_caller_no_primary_401(client, monkeypatch):
    """Flag on but neither Authorization nor X-Caller-Authorization → 401.
    caller_client_id is empty so the branch is skipped."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    resp = await client.post("/api/action-intents/decide")
    assert resp.status_code == 401
    assert resp.json()["reason"] == "missing_bearer_token"


# -- Regression: license-token path is unchanged when the flag is ON --------


# -- Allowlist: caller on the list is allowed; off the list falls through to 401 --


async def test_service_principal_allowlisted_caller_allowed(client_with_caller_jwks, monkeypatch):
    """Flag on + a non-empty allowlist that INCLUDES the caller → 200, overlay
    injected, and the `allowed` counter increments for that caller."""
    from prometheus_client import REGISTRY

    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(
        main, "SERVICE_PRINCIPAL_ALLOWED_CLIENTS", frozenset({"marketing-service-account"})
    )
    labels = {"result": "allowed", "caller": "marketing-service-account"}
    before = REGISTRY.get_sample_value("license_authz_service_principal_total", labels) or 0.0
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 200
    assert resp.headers["x-caller-client-id"] == "marketing-service-account"
    after = REGISTRY.get_sample_value("license_authz_service_principal_total", labels)
    assert after == before + 1


async def test_service_principal_non_allowlisted_caller_denied(
    client_with_caller_jwks, monkeypatch
):
    """Flag on + a non-empty allowlist that EXCLUDES the caller → falls through to
    the missing_bearer_token 401 (no overlay leaks); the denial counter increments."""
    from prometheus_client import REGISTRY

    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(
        main, "SERVICE_PRINCIPAL_ALLOWED_CLIENTS", frozenset({"some-other-service"})
    )
    labels = {"result": "denied_not_allowlisted", "caller": "marketing-service-account"}
    before = REGISTRY.get_sample_value("license_authz_service_principal_total", labels) or 0.0
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "missing_bearer_token"
    assert resp.headers.get("x-caller-client-id", "") == ""
    after = REGISTRY.get_sample_value("license_authz_service_principal_total", labels)
    assert after == before + 1


async def test_service_principal_empty_allowlist_allows_any(client_with_caller_jwks, monkeypatch):
    """An empty allowlist preserves the original #22 behavior: any verified SA is
    PERMITted (the lockdown is strictly opt-in)."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_ALLOWED_CLIENTS", frozenset())
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 200
    assert resp.headers["x-caller-client-id"] == "marketing-service-account"


@respx.mock
async def test_license_token_path_unchanged_with_flag_on(client_with_caller_jwks, monkeypatch):
    """A normal primary-license-token request behaves identically with the flag
    on: introspect runs, tenant headers are injected, 200. The service-principal
    branch only governs the no-primary-token case."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    resp = await client_with_caller_jwks.get(
        "/workers/x",
        headers={"Authorization": "Bearer license-tok"},
    )
    assert resp.status_code == 200
    # Real tenant resolved + injected (autouse _stub_tenant_resolver supplies it).
    assert resp.headers.get("x-tenant-id", "") != ""
    assert resp.headers["x-license-id"] == "lic_x"


@respx.mock
async def test_service_principal_allowed_on_registered_route_without_app_key(
    client_with_caller_jwks, monkeypatch
):
    """The SP branch returns BEFORE the app-route check — pin it as executable.

    App-route enforcement now DENIES a missing/empty app_key on a registered
    prefix, and the service-principal path injects x-app-key="" by design. The
    only reason that is not an outage is the early return at the SP branch. If
    someone ever moves the app-route check above it, every service-principal
    call to /workers/, /vllm/, /kanban/, ... 403s at once.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/workers/tasks",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )
    assert resp.status_code == 200, "the app-route check now runs before the SP return"
    assert resp.json()["status"] == "ok"
    assert resp.headers["x-caller-client-id"] == "marketing-service-account"
    assert resp.headers.get("x-app-key", "") == ""


# === Round 8: the service-principal fast path, measured and optionally enforced =
#
# THE AUDIT FINDING: over 24h all 1,487 ext_authz checks returned 200 from this
# branch, which sits BEFORE the budget check and emits no metering.
#
# ⚠️ AND THE CENSUS THAT REFUTES THE OBVIOUS FIX: the only two callers on this
# path (`marketing-service-account` 1,474, `platform-console-backend` 17) carry NO
# licence and NO tenant — the live log line is `license_id: None, tenant_id: None`.
# Unless the caller's `azp` resolves to a tenant, the budget check has nothing
# to key on and cannot bite. What ships is
# therefore ATTRIBUTION (a caller-dimensioned counter, always on) plus the
# enforcement SCAFFOLDING behind a default-off flag. The flag-off tests are the
# important ones.

SP_BUDGET_METRIC = "license_authz_service_principal_budget_total"


def _sp_sample(result: str, caller: str = "marketing-service-account") -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(SP_BUDGET_METRIC, {"result": result, "caller": caller}) or 0.0


async def test_service_principal_budget_enforcement_defaults_off():
    """Arming this by default would change live authz on 100% of traffic."""
    assert main.SERVICE_PRINCIPAL_BUDGET_ENFORCED is False


async def test_flag_off_allows_exactly_as_before(client_with_caller_jwks, monkeypatch):
    """FLAG OFF: 200 + overlay, byte-for-byte the pre-change response."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)

    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.headers["x-caller-client-id"] == "marketing-service-account"


async def test_flag_off_never_reads_the_budget(client_with_caller_jwks, monkeypatch):
    """FLAG OFF must not touch the budget check at all — not "call it and ignore it".

    A call with a discarded result would still pay a billing round trip on the
    ext_authz hot path and would move the fail-open counters.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    spy = AsyncMock(return_value=(True, None))

    with (
        patch("main._resolve_service_principal_tenant", AsyncMock(return_value="t-sp")),
        patch("main._metering_http_verdict", spy),
    ):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 200
    spy.assert_not_awaited()


async def test_flag_off_counts_the_unenforced_allow(client_with_caller_jwks, monkeypatch):
    """THE INSTRUMENT THE AUDIT ASKED FOR. Behaviour unchanged, allow now counted.

    `license_authz_service_principal_total{result="allowed"}` already counted the
    VOLUME; it cannot distinguish "allowed after passing its quota" from "allowed
    because this path has no quota gate", and 100% of production was the latter.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    before = _sp_sample("not_enforced")

    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/api/action-intents/decide",
            headers={"X-Caller-Authorization": "Bearer sa-jwt"},
        )

    assert resp.status_code == 200
    assert _sp_sample("not_enforced") == before + 1


async def test_service_principal_allow_emits_a_metering_event(client_with_caller_jwks, monkeypatch):
    """THE DEFECT: 1,491 measured allows over 24h produced NO metering row at all.

    That traffic was not "unattributed" — it was ABSENT from the measurement.
    This test replaces `test_flag_off_emits_no_metering_event`, which pinned the
    previous round's decision NOT to emit. That decision rested on the claim that
    an honest tenantless row would be dropped downstream and fire
    MeteringEventsDroppedNoTenant; `test_tenantless_metering_rows_are_not_a_new_class`
    below refutes it against this same module.

    ⛔ Reverting the emit makes this fail: `emit.assert_awaited_once()` sees zero.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    emit = AsyncMock()

    with patch("main._emit_metering_event", emit):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 200
    emit.assert_awaited_once()
    kwargs = emit.await_args.kwargs
    assert kwargs["status"] == "allowed_service_principal"
    assert kwargs["caller_client_id"] == "marketing-service-account"
    assert kwargs["path"] == "/api/action-intents/decide"
    assert kwargs["method"] == "POST"


async def test_service_principal_metering_event_fabricates_no_identity(
    client_with_caller_jwks, monkeypatch
):
    """⛔ THE HALF THAT MUST NOT DRIFT. This path has no tenant, no licence, no
    subject, no app and no plan, and every one of those must stay ABSENT.

    A false identity is unrecoverable once billing aggregates it — the same
    defect the `tenant_id or subject_id` fallbacks were removed for. A future
    "improvement" that defaults tenant_id to the caller id, a nil UUID or a
    literal is exactly what this test exists to red.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    emit = AsyncMock()

    with patch("main._emit_metering_event", emit):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 200
    kwargs = emit.await_args.kwargs
    assert kwargs["tenant_id"] == "", "a tenant was fabricated for a path that has none"
    for absent in ("license_id", "subject_id", "app_id", "plan_id", "plan_code"):
        assert kwargs[absent] is None, f"{absent} was fabricated"


async def test_service_principal_metering_row_reaches_the_stream_with_the_caller(
    client_with_caller_jwks, monkeypatch
):
    """End to end through the REAL `_emit_metering_event` onto a stubbed Redis.

    The two tests above patch the emit helper, so they prove the CALL and not the
    ROW. This one proves the row: `caller_client_id` has to survive into the
    actual xadd field map, because that field is the only true identity the event
    carries and a helper signature alone cannot show it is written.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    app.state.redis = mock_redis
    try:
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )
        assert resp.status_code == 200
        mock_redis.xadd.assert_awaited_once()
        fields = mock_redis.xadd.await_args[0][1]
        assert fields["caller_client_id"] == "marketing-service-account"
        assert fields["status"] == "allowed_service_principal"
        # Still no fabricated identity once the row is serialised.
        assert fields["tenant_id"] == ""
        assert fields["license_id"] == ""
        assert fields["subject_id"] == ""
    finally:
        app.state.redis = None


async def test_tenantless_metering_rows_are_not_a_new_class(client):
    """The refutation the service-principal emit rests on, measured not asserted.

    The previous round declined to emit because "an honest tenantless row is
    DROPPED downstream and fires MeteringEventsDroppedNoTenant, so it would have
    to be silenced". That is an argument against a class of event this service
    ALREADY writes: a request with no Authorization header at all emits a
    metering row with `tenant_id=""` and has done for as long as that path has
    existed. So whatever the consumer does with a tenantless row, it already does
    it, and the service-principal emit introduces nothing new.

    ⚠️ DRIVEN THROUGH THE REAL HANDLER ON PURPOSE. The first version of this test
    grepped `inspect.getsource(ext_authz_check)` for `tenant_id=""` — which the
    NEW service-principal emit satisfies all by itself, so it would have passed
    even if the pre-existing tenantless emit had been deleted. It was a test that
    could not fail for its own subject.

    ⛔ If this ever reddens, the refutation has expired and the service-principal
    emit needs re-arguing, not re-asserting.
    """
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    main.app.state.redis = mock_redis
    try:
        resp = await client.get("/anything")  # no Authorization header at all
        assert resp.status_code == 401
        mock_redis.xadd.assert_awaited_once()
        fields = mock_redis.xadd.await_args[0][1]
        assert fields["status"] == "denied_missing_token"
        assert fields["tenant_id"] == "", (
            "the pre-existing tenantless emit is gone — the premise that a "
            "tenantless row is not a new class no longer holds"
        )
    finally:
        main.app.state.redis = None


async def test_metering_emit_does_not_change_the_authz_outcome(
    client_with_caller_jwks, monkeypatch
):
    """NEGATIVE CONTROL: a metering emit that fails must not turn an allow into a
    deny. Metering is best-effort; the auth decision is already made.

    Redis raising inside the emit is the realistic failure, and the response must
    still be the byte-identical 200 + caller overlay.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", False)
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(side_effect=redis.exceptions.RedisError("stream down"))
    app.state.redis = mock_redis
    try:
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )
        assert resp.status_code == 200
        assert resp.headers["x-caller-client-id"] == "marketing-service-account"
        assert resp.headers.get("x-tenant-id", "") == ""
    finally:
        app.state.redis = None


async def test_flag_on_runs_the_budget_check_with_the_identity_the_path_has(
    client_with_caller_jwks, monkeypatch
):
    """FLAG ON: billing's budget check runs with the RESOLVED tenant and NO licence.

    Passing a made-up licence id to make the check "bite" would charge one
    caller's usage to an identity nobody assigned.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", True)
    spy = AsyncMock(return_value=(True, None))

    with (
        patch("main._resolve_service_principal_tenant", AsyncMock(return_value="t-sp")),
        patch("main._metering_http_verdict", spy),
    ):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 200
    spy.assert_awaited_once()
    assert spy.await_args.kwargs == {"license_id": None, "tenant_id": "t-sp"}, (
        "a licence id was fabricated, or the resolved tenant was not used"
    )


async def test_flag_on_denies_with_402_when_the_budget_check_denies(
    client_with_caller_jwks, monkeypatch
):
    """FLAG ON: a deny becomes a 402 with the reason surfaced to Envoy.

    The billing verdict is stubbed here to pin the 402 WIRING; the real
    billing-backed deny is driven end to end in test_service_principal_identity.
    """
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", True)
    before = _sp_sample("denied")

    with (
        patch("main._resolve_service_principal_tenant", AsyncMock(return_value="t-sp")),
        patch(
            "main._metering_http_verdict",
            AsyncMock(return_value=(False, "token_budget_exhausted")),
        ),
    ):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 402
    body = resp.json()
    assert body["error"] == "budget_exceeded"
    assert body["kind"] == "budget"
    assert body["reason"] == "token_budget_exhausted"
    assert resp.headers["x-error-reason"] == "token_budget_exhausted"
    # No overlay leaks on a denial.
    assert resp.headers.get("x-caller-client-id", "") == ""
    assert _sp_sample("denied") == before + 1


async def test_flag_on_alone_is_a_measured_no_op_against_the_real_check(
    client_with_caller_jwks, monkeypatch
):
    """⚠️ An UNRESOLVED caller (no tenant) is allowed, never denied by a check
    with nothing to key on, and counted `budget_fail_open{reason="no_license_id"}`.

    Nothing is read from Redis (this path has no quota). A healthy Redis client
    is supplied so "nothing was read" is measured, not assumed.
    """
    from prometheus_client import REGISTRY

    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_BUDGET_ENFORCED", True)
    healthy_redis = AsyncMock()
    healthy_redis.get = AsyncMock(return_value=None)
    labels = {"reason": "no_license_id"}
    before = REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) or 0.0
    before_allowed = _sp_sample("allowed")

    with patch("main._get_redis", AsyncMock(return_value=healthy_redis)):
        with patch("main.verify_jwt", return_value=_sa_claims()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 200, "arming the flag alone changed a live decision"
    assert REGISTRY.get_sample_value("license_authz_budget_fail_open_total", labels) == before + 1
    assert _sp_sample("allowed") == before_allowed + 1
    # Plain statement, not `assert x(), "msg"` — that builds a truthy tuple and
    # never fails, which is how a mutation-proof check quietly stops checking.
    healthy_redis.get.assert_not_awaited()


# -- The caller label is clamped ----------------------------------------------


def test_caller_label_keeps_the_measured_census():
    """Attribution is the deliverable — the two known callers must stay readable."""
    assert main._sp_caller_label("marketing-service-account") == "marketing-service-account"
    assert main._sp_caller_label("platform-console-backend") == "platform-console-backend"


def test_an_unknown_caller_is_bucketed_rather_than_minting_a_series():
    """`azp` is remote-controlled on a per-request label: unclamped it is an
    unbounded-cardinality hazard, verified token or not."""
    assert main._sp_caller_label("some-new-service") == "other"
    assert main._sp_caller_label("") == "other"


def test_the_census_setting_is_what_drives_the_clamp(monkeypatch):
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_METRIC_CALLERS", frozenset({"only-this"}))
    assert main._sp_caller_label("only-this") == "only-this"
    assert main._sp_caller_label("marketing-service-account") == "other"


def test_every_sp_budget_series_is_born_at_zero():
    """A labelled Counter exports NO series until its first .inc().

    Without birth-at-import, "this caller has never been denied" and "this build
    cannot report a denial" are the same empty vector.
    """
    from prometheus_client import REGISTRY

    for result in ("not_enforced", "allowed", "denied"):
        for caller in (*main.SERVICE_PRINCIPAL_METRIC_CALLERS, "other"):
            assert (
                REGISTRY.get_sample_value(SP_BUDGET_METRIC, {"result": result, "caller": caller})
                is not None
            ), f"{SP_BUDGET_METRIC}{{result={result},caller={caller}}} has no series until it fires"


# -- The unsafe-header deny must leave an audit row -------------------------
#
# Both `unsafe_header_value` branches in ext_authz_check used to return without
# calling `log_access`, which is this service's ONLY structured audit emitter.
# The deny is real (it fails CLOSED, so nothing was bypassed) but it was
# invisible to every audit query keyed on the `status` field — and a
# header-injection attempt against the authz boundary is the event you most
# want in that trail. These tests pin the row, and pin that the attacker-
# controlled header VALUE never reaches the log.

# Deliberately placed in `realm_access.roles`, NOT in `azp`: `azp` is echoed
# into the audit `reason` by design (it is the one VERIFIED identity this path
# holds), so a canary there could not distinguish "value leaked" from "verified
# caller recorded". A role string is attacker-influenced and is named by NOTHING
# in the row, so its absence is a real measurement.
_ROLE_CANARY = "bad role;X-SPCANARY: true"


def _sa_claims_with_unsafe_role() -> VerifiedClaims:
    return VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={
            "azp": "marketing-service-account",
            "realm_access": {"roles": ["platform_service", _ROLE_CANARY]},
        },
    )


def _audit_rows(caplog) -> list[dict]:
    """Every structured log_access row emitted during the block."""
    rows = []
    for record in caplog.records:
        try:
            entry = json.loads(record.getMessage())
        except ValueError, TypeError:
            continue
        if isinstance(entry, dict) and "status" in entry and "source_ip" in entry:
            rows.append(entry)
    return rows


async def test_sp_unsafe_header_deny_is_audited(client_with_caller_jwks, monkeypatch, caplog):
    """Service-principal fast path: a role that fails the safety regex denies
    401 *and* writes an audit row naming the offending header."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with caplog.at_level(logging.INFO, logger="main"):
        with patch("main.verify_jwt", return_value=_sa_claims_with_unsafe_role()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    # Behaviour is unchanged: still a bare 401, still fails closed.
    assert resp.status_code == 401
    assert resp.json()["reason"] == "unsafe_header_value"

    rows = [
        r for r in _audit_rows(caplog) if r["status"] == "denied_service_principal_unsafe_header"
    ]
    assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
    row = rows[0]
    assert row["reason"] == "caller=marketing-service-account,unsafe_header=x-caller-roles"
    assert row["path"] == "/api/action-intents/decide"
    assert row["method"] == "POST"
    # No licence identity exists on this path and none is invented. `None`, not
    # "" — an empty string would read as "looked it up and it was blank".
    assert row["license_id"] is None
    assert row["tenant_id"] is None
    assert row["app_id"] is None
    assert row["plan_id"] is None
    # There is no licence token here to hash; the file's sentinel for that.
    assert row["token_hash"] == "service_principal"


async def test_sp_unsafe_header_value_never_reaches_the_log(
    client_with_caller_jwks, monkeypatch, caplog
):
    """⛔ The rejected value is attacker-influenced. Logging it would turn the
    log-injection guard into the log-injection vector."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with caplog.at_level(logging.INFO, logger="main"):
        with patch("main.verify_jwt", return_value=_sa_claims_with_unsafe_role()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )
    assert resp.status_code == 401
    # Positive control: the row we are searching exists, so "canary not found"
    # cannot be satisfied by an empty log.
    assert any(r["status"] == "denied_service_principal_unsafe_header" for r in _audit_rows(caplog))
    for record in caplog.records:
        assert "SPCANARY" not in record.getMessage(), record.getMessage()


# -- The rejected value must not be recorded AS the caller's identity -------
#
# ⚠️ The `_ROLE_CANARY` test above cannot fail for this class, by construction.
# It puts its payload in `realm_access.roles`, which the row names nowhere — the
# very property that makes its absence a real measurement also makes it blind to
# a leak through a field the row DOES carry. `caller_client_id` is such a field:
# `sp_headers["x-caller-client-id"]` IS `caller_client_id`, and the row puts
# `caller_client_id` in `reason`. So when the header that trips the regex is the
# azp itself, the row used to assert, as the verified caller, the exact string
# this service had just refused to propagate.

_AZP_CANARY = "sa AZPCANARY\r\nX-Injected: true"


def _sa_claims_with_unsafe_azp() -> VerifiedClaims:
    return VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={"azp": _AZP_CANARY, "realm_access": {"roles": ["platform_service"]}},
    )


async def test_sp_unsafe_azp_is_elided_from_the_audit_row(
    client_with_caller_jwks, monkeypatch, caplog
):
    """The azp itself fails the regex → the row names the header but must not
    record the rejected value as the caller."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    # Pinned empty rather than inherited: a non-empty allowlist would deny at
    # `not_allowlisted` and never reach the header-safety loop, so the test
    # would pass while measuring a different exit entirely.
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_ALLOWED_CLIENTS", frozenset())

    with caplog.at_level(logging.INFO, logger="main"):
        with patch("main.verify_jwt", return_value=_sa_claims_with_unsafe_azp()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    # Behaviour unchanged: still a bare 401, still fails closed.
    assert resp.status_code == 401
    assert resp.json()["reason"] == "unsafe_header_value"

    rows = [
        r for r in _audit_rows(caplog) if r["status"] == "denied_service_principal_unsafe_header"
    ]
    assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
    row = rows[0]
    # `x-caller-client-id` is what failed, and it is named; the value is not.
    assert row["reason"] == "caller=<unsafe-value-elided>,unsafe_header=x-caller-client-id"
    # ⛔ Load-bearing: serialised, so a leak into ANY field of the row fails it.
    assert "AZPCANARY" not in json.dumps(row), row

    # ⚠️ Scope, stated so no later reader over-reads the guarantee: the claim is
    # "not in the AUDIT ROW", NOT "not in any log". The pre-existing WARNING at
    # this site carries the raw azp verbatim and predates this branch — that is
    # deliberate (WARNING is what alerting watches, log_access is the audit
    # trail), and it is also the positive control proving the canary really was
    # present in this request and the absence above is a redaction, not an
    # empty run.
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("AZPCANARY" in m for m in warnings), warnings


async def test_sp_unsafe_role_leaves_the_verified_caller_recorded(
    client_with_caller_jwks, monkeypatch, caplog
):
    """Negative control against over-redaction: when it is a ROLE that fails,
    the azp did NOT fail and must still be named as the caller."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_ALLOWED_CLIENTS", frozenset())

    with caplog.at_level(logging.INFO, logger="main"):
        with patch("main.verify_jwt", return_value=_sa_claims_with_unsafe_role()):
            resp = await client_with_caller_jwks.post(
                "/api/action-intents/decide",
                headers={"X-Caller-Authorization": "Bearer sa-jwt"},
            )

    assert resp.status_code == 401
    rows = [
        r for r in _audit_rows(caplog) if r["status"] == "denied_service_principal_unsafe_header"
    ]
    assert len(rows) == 1
    # Elision is by VALUE: the azp is not the rejected string, so it survives.
    # A blanket "always elide the caller" mutant dies here.
    assert rows[0]["reason"] == "caller=marketing-service-account,unsafe_header=x-caller-roles"


async def test_service_principal_is_refused_on_a_strict_host(client_with_caller_jwks, monkeypatch):
    """A strict host (vllm.example) demands a `vllm` LICENCE; a service
    principal carries none, so it falls through to the standard 401."""
    monkeypatch.setattr(main, "SERVICE_PRINCIPAL_AUTH_ENABLED", True)
    with patch("main.verify_jwt", return_value=_sa_claims()):
        resp = await client_with_caller_jwks.post(
            "/v1/chat/completions",
            headers={
                "X-Caller-Authorization": "Bearer sa-jwt",
                "x-forwarded-host": "vllm.example",
            },
        )
    assert resp.status_code == 401
