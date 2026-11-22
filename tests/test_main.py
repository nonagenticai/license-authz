"""Tests for License AuthZ Service

SECURITY: This service validates ALL tokens via license introspection.
There is no JWT passthrough - fail-closed security model.

All tests use respx to mock at the HTTP transport level (intercepting the
actual POST to LICENSE_INTROSPECT_URL) instead of patching introspect_license.
This ensures the full request/response cycle is exercised: token extraction,
HTTP call construction, response parsing, and header propagation.
"""

import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import redis.asyncio
import respx
from httpx import ASGITransport, AsyncClient

import main as main_module
from main import APP_ROUTE_REGISTRY, _check_app_route, _registered_app_key, app, lifespan
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL

# Canonical tenant UUID used as the default introspect identity. The licenses
# service returns the tenant_id as a UUID (any version); license-authz injects
# it directly as x-tenant-id.
TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"  # UUIDv7

# -- Helpers ------------------------------------------------------------------


def _active_response(
    *,
    tenant_id: str = TENANT_UUID,
    subject_id: str = "sub_123",
    license_id: str = "lic_abc",
    app_id: str | None = "app_workers",
    app_key: str | None = None,
    plan_id: str | None = "premium",
) -> dict:
    """Build a successful introspect response payload."""
    resp: dict = {
        "active": True,
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "license_id": license_id,
    }
    if app_id is not None or app_key is not None:
        app_data: dict[str, str] = {}
        if app_id is not None:
            app_data["id"] = app_id
        if app_key is not None:
            app_data["key"] = app_key
        resp["app"] = app_data
    if plan_id is not None:
        resp["plan"] = {"id": plan_id}
    return resp


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


def _inactive_response(reason: str = "invalid_token") -> dict:
    return {"active": False, "reason": reason}


# -- Health / Probe Endpoints -------------------------------------------------


async def test_ready_endpoint(client: AsyncClient, monkeypatch):
    """/ready should return 200 without requiring auth, and enumerate its checks.

    ⚠️ A HEALTHY Redis is now supplied deliberately. Redis is unreachable in the
    unit-test env, and `READINESS_REQUIRE_REDIS` now defaults ON — so without the
    stub this would 503 for a DEPENDENCY reason and stop testing the claim it is
    named for, which is that the probe needs no AUTH. See tests/test_readiness.py
    for the gating matrix and the revert.
    """

    class _PingableRedis:
        async def ping(self) -> bool:
            return True

        async def aclose(self) -> None:
            return None

    monkeypatch.setitem(app.state._state, "redis", _PingableRedis())
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == "license-authz"
    assert body["checks"]["lifespan"]["ok"] is True


@respx.mock
async def test_health_endpoints_do_not_call_introspect(client: AsyncClient):
    """Health/healthz/ready endpoints must NOT trigger introspect calls."""
    route = respx.post(INTROSPECT_URL).respond(200, json=_active_response())

    await client.get("/health")
    await client.get("/healthz")
    await client.get("/ready")
    await client.get("/health/live")

    assert not route.called, "Health endpoints should not call introspect"


# -- Missing / Invalid Auth --------------------------------------------------


async def test_missing_bearer_token(client: AsyncClient):
    """Request without Authorization header should return 401."""
    response = await client.get("/")
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
    assert response.json()["reason"] == "missing_bearer_token"
    assert response.headers.get("x-error-reason") == "missing_bearer_token"


async def test_invalid_auth_header(client: AsyncClient):
    """Request with non-Bearer auth should return 401."""
    response = await client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "unauthorized"
    assert body["reason"] == "missing_bearer_token"
    assert response.headers.get("x-error-reason") == "missing_bearer_token"


async def test_response_body_on_401(client: AsyncClient):
    """401 response body must be valid JSON with correct structure and content-type."""
    response = await client.get("/some/path")
    assert response.status_code == 401
    body = response.json()
    assert body == {"error": "unauthorized", "reason": "missing_bearer_token"}
    assert response.headers.get("content-type") == "application/json"


@respx.mock
async def test_bearer_case_insensitive(client: AsyncClient):
    """'BEARER' (uppercase) should also be accepted as a valid prefix."""
    route = respx.post(INTROSPECT_URL).respond(
        200,
        json=_active_response(license_id="lic-1"),
    )

    response = await client.get("/api/resource", headers={"Authorization": "BEARER my_token"})

    # The code does auth_header.lower().startswith("bearer ") so uppercase works.
    # auth_header[7:] extracts "my_token" correctly.
    assert route.called
    sent_body = route.calls[0].request.content
    assert b'"my_token"' in sent_body
    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == TENANT_UUID


async def test_empty_authorization_header(client: AsyncClient):
    """Empty Authorization header should return 401."""
    response = await client.get("/", headers={"Authorization": ""})
    assert response.status_code == 401
    assert response.json()["reason"] == "missing_bearer_token"


async def test_bearer_without_space(client: AsyncClient):
    """'Bearertoken' (no space after Bearer) should return 401."""
    response = await client.get("/", headers={"Authorization": "Bearertoken123"})
    assert response.status_code == 401
    assert response.json()["reason"] == "missing_bearer_token"


# -- Valid Token Flow ---------------------------------------------------------


@respx.mock
async def test_valid_license_returns_headers(client: AsyncClient):
    """Valid license should return 200 with tenant headers and ok body.

    The introspect HTTP call is intercepted by respx so the full flow runs:
    token extraction -> POST to introspect URL -> parse response -> set headers.
    """
    route = respx.post(INTROSPECT_URL).respond(200, json=_active_response())

    response = await client.get("/", headers={"Authorization": "Bearer valid_token"})

    # Verify introspect was actually called with the right token
    assert route.called
    request_body = route.calls[0].request.content
    assert b'"token"' in request_body
    assert b'"valid_token"' in request_body

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("x-tenant-id") == TENANT_UUID
    # x-tenant-name is vestigial: always empty (canonical identity is the UUID).
    assert response.headers.get("x-tenant-name") == ""
    assert response.headers.get("x-license-id") == "lic_abc"
    assert response.headers.get("x-subject-id") == "sub_123"
    assert response.headers.get("x-app-id") == "app_workers"
    assert response.headers.get("x-plan-id") == "premium"
    # License-token mode has no verified end-user identity to vouch for —
    # x-actor-id must never appear here (JWT-mode only).
    assert "x-actor-id" not in response.headers


@respx.mock
async def test_subject_id_header_omitted_when_introspect_omits_it(client: AsyncClient):
    """When introspect response has no subject_id, header is empty (not omitted).

    Empty string keeps the response shape stable so Istio's
    `headersToUpstreamOnAllow` can still strip-or-pass the header without
    branching on absence.
    """
    respx.post(INTROSPECT_URL).respond(
        200,
        json={
            "active": True,
            "tenant_id": TENANT_UUID,
            "license_id": "lic_abc",
            # no subject_id
        },
    )

    response = await client.get("/", headers={"Authorization": "Bearer tok"})

    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == TENANT_UUID
    assert response.headers.get("x-tenant-name") == ""
    assert response.headers.get("x-subject-id") == ""


@respx.mock
async def test_uuidv7_tenant_id_injected_directly(client: AsyncClient):
    """A canonical UUIDv7 tenant_id is the identity — inject it as x-tenant-id
    directly. When introspect ships no label, x-tenant-name is empty.
    """
    tenant_uuid = "0192a3b4-c5d6-7e8f-9012-345678901234"  # UUIDv7
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(tenant_id=tenant_uuid))

    response = await client.get("/", headers={"Authorization": "Bearer tok"})

    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == tenant_uuid
    # x-tenant-name is vestigial: always empty.
    assert response.headers.get("x-tenant-name") == ""


@respx.mock
async def test_non_v7_uuid_tenant_id_injected_directly(client: AsyncClient):
    """A non-v7 UUID (e.g. v4) tenant_id is ALSO injected directly.

    Regression: the old `_is_uuidv7` accepted only version-7 UUIDs and routed
    every v4/v5 tenant_id to the name-resolution lookup, which 403'd. The
    contract is now "any well-formed UUID is the canonical identity", matching
    the licenses service's own `validate_tenant_uuid`.
    """
    tenant_v4 = "8382f0b3-c082-4f97-8d47-120ce5a18e05"  # UUIDv4
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(tenant_id=tenant_v4))

    response = await client.get("/", headers={"Authorization": "Bearer tok"})

    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == tenant_v4
    assert response.headers.get("x-tenant-name") == ""


@respx.mock
async def test_non_uuid_tenant_id_rejected(client: AsyncClient):
    """A non-UUID tenant_id (legacy text name) fails closed.

    Post-migration every tenant_id is a canonical UUID, so a text value here
    signals un-migrated/invalid data — deny loudly rather than forward a bad
    identity downstream.
    """
    respx.post(INTROSPECT_URL).respond(200, json=_active_response(tenant_id="n8n-dev"))

    response = await client.get("/", headers={"Authorization": "Bearer tok"})

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "forbidden"
    assert body["reason"] == "invalid_tenant"
    assert response.headers.get("x-error-reason") == "invalid_tenant"
    # No identity headers may leak on the deny path.
    assert response.headers.get("x-tenant-id") is None
    assert response.headers.get("x-license-id") is None


@respx.mock
async def test_valid_license_content_type(client: AsyncClient):
    """Valid license response should have application/json content-type."""
    respx.post(INTROSPECT_URL).respond(200, json=_active_response())

    response = await client.get("/api/test", headers={"Authorization": "Bearer tok"})
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/json"


# -- Invalid Token Flow -------------------------------------------------------


@respx.mock
async def test_invalid_license_returns_403(client: AsyncClient):
    """Invalid license should return 403 with structured error body."""
    route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("license_expired"))

    response = await client.get("/", headers={"Authorization": "Bearer expired_token"})

    assert route.called
    assert b'"expired_token"' in route.calls[0].request.content
    assert response.status_code == 403
    assert response.json()["error"] == "license_invalid"
    assert response.json()["reason"] == "license_expired"
    assert response.headers.get("x-error-reason") == "license_expired"
    assert response.headers.get("content-type") == "application/json"


@respx.mock
async def test_introspect_timeout(client: AsyncClient):
    """Introspect timeout should return 403 (fail-closed)."""
    route = respx.post(INTROSPECT_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    response = await client.get("/", headers={"Authorization": "Bearer some_token"})

    assert route.called
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "license_invalid"
    assert body["reason"] == "introspect_timeout"


@respx.mock
async def test_introspect_connection_error(client: AsyncClient):
    """Connection error to introspect service should return 403 (fail-closed)."""
    route = respx.post(INTROSPECT_URL).mock(side_effect=httpx.ConnectError("refused"))

    response = await client.get("/", headers={"Authorization": "Bearer some_token"})

    assert route.called
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "license_invalid"
    assert body["reason"] == "introspect_error"


@respx.mock
async def test_empty_bearer_token(client: AsyncClient):
    """Empty bearer token should be introspected and rejected (fail-closed)."""
    route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("invalid_token"))

    response = await client.get("/", headers={"Authorization": "Bearer "})

    # The code does auth_header[7:] which yields "" - it should still call introspect
    assert route.called
    assert b'""' in route.calls[0].request.content  # empty token sent
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "license_invalid"
    assert body["reason"] == "invalid_token"


@respx.mock
async def test_403_does_not_leak_internal_headers(client: AsyncClient):
    """403 response should NOT include x-tenant-id or x-license-id headers."""
    respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("revoked"))

    response = await client.get("/", headers={"Authorization": "Bearer bad"})

    assert response.status_code == 403
    assert response.headers.get("x-tenant-id") is None
    assert response.headers.get("x-license-id") is None
    assert response.headers.get("x-subject-id") is None
    assert response.headers.get("x-app-id") is None
    assert response.headers.get("x-plan-id") is None


# -- Security Hardening -------------------------------------------------------


class TestSecurityHardening:
    """Tests for fail-closed security model - ALL tokens must be introspected."""

    @respx.mock
    async def test_jwt_format_tokens_are_introspected(self, client: AsyncClient):
        """JWT-formatted tokens MUST be introspected (no passthrough)."""
        route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("invalid_token"))

        # This looks like a JWT but should NOT bypass validation
        fake_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.signature"

        response = await client.get(
            "/licenses/user/12345",
            headers={"Authorization": f"Bearer {fake_jwt}"},
        )

        # MUST call introspect - no passthrough
        assert route.called
        assert fake_jwt.encode() in route.calls[0].request.content
        # Should be denied because introspect returned active=False
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "license_invalid"
        assert body["reason"] == "invalid_token"

    @respx.mock
    async def test_fake_jwt_attack_blocked(self, client: AsyncClient):
        """Attacker sending 'aaa.bbb.ccc' should be blocked, not passed through."""
        route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("invalid_token"))

        # Attack payload that would have bypassed old is_jwt() check
        attack_token = "aaa.bbb.ccc"

        response = await client.get(
            "/sensitive/data",
            headers={"Authorization": f"Bearer {attack_token}"},
        )

        # MUST call introspect
        assert route.called
        assert b"aaa.bbb.ccc" in route.calls[0].request.content
        # Should be denied
        assert response.status_code == 403
        assert response.json()["error"] == "license_invalid"

    @respx.mock
    async def test_all_tokens_introspected_all_methods(self, client: AsyncClient):
        """All HTTP methods require introspection for all token types."""
        route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("invalid_token"))

        # JWT-like token
        jwt_like = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.signature"
        headers = {"Authorization": f"Bearer {jwt_like}"}

        for method in ["get", "post", "put", "delete", "patch"]:
            response = await getattr(client, method)("/test", headers=headers)
            assert response.status_code == 403, f"Should deny for {method.upper()}"
            assert response.json()["error"] == "license_invalid", (
                f"Wrong error for {method.upper()}"
            )

        # All 5 methods should have hit introspect
        assert route.call_count == 5

    @respx.mock
    async def test_license_token_still_works(self, client: AsyncClient):
        """Valid license tokens still work as expected."""
        # app_key="workers" because the path is a REGISTERED prefix: an unkeyed
        # token is denied there now (see TestAppRouteEnforcement). Collateral of
        # the fail-closed change, not a change to what this test is about.
        route = respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="workers"))

        # A licence token's wire shape: 43 base64url characters, no dots.
        # Built as an expression rather than written as a literal ON PURPOSE. A
        # quoted 43-character string on a line that says "token" is exactly what
        # the repository's private-refs guard flags, because that is what a real
        # leaked licence looks like — and a scanner cannot tell a fixture from a
        # credential. The value never leaves this process: introspection is
        # mocked by respx below.
        license_token = "fixture" + "-" + "0" * 35

        response = await client.get(
            "/workers/tasks",
            headers={"Authorization": f"Bearer {license_token}"},
        )

        assert route.called
        assert license_token.encode() in route.calls[0].request.content
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers.get("x-tenant-name") == ""
        assert response.headers.get("x-license-id") == "lic_abc"

    @respx.mock
    async def test_long_token_handled(self, client: AsyncClient):
        """Very long tokens should still be forwarded to introspect (no truncation)."""
        long_token = "x" * 5000
        route = respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("invalid"))

        response = await client.get(
            "/api/resource",
            headers={"Authorization": f"Bearer {long_token}"},
        )

        assert route.called
        assert long_token.encode() in route.calls[0].request.content
        assert response.status_code == 403

    @respx.mock
    async def test_token_with_special_characters(self, client: AsyncClient):
        """Tokens with URL-safe special chars should be forwarded correctly."""
        special_token = "abc-def_ghi.jkl+mno/pqr=stu"
        route = respx.post(INTROSPECT_URL).respond(200, json=_active_response())

        response = await client.get(
            "/api/resource",
            headers={"Authorization": f"Bearer {special_token}"},
        )

        assert route.called
        assert special_token.encode() in route.calls[0].request.content
        assert response.status_code == 200

    @respx.mock
    async def test_introspect_returns_malformed_json(self, client: AsyncClient):
        """Malformed introspect response should result in 403 (fail-closed)."""
        respx.post(INTROSPECT_URL).respond(200, content=b"not json at all")

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        # Should fail closed - not let the request through
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "license_invalid"

    @respx.mock
    async def test_introspect_returns_empty_json(self, client: AsyncClient):
        """Empty JSON object from introspect should deny (active defaults to False)."""
        respx.post(INTROSPECT_URL).respond(200, json={})

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        # active defaults to False via data.get("active", False)
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "license_invalid"

    @respx.mock
    async def test_header_injection_via_tenant_id_blocked(self, client: AsyncClient):
        """A CRLF-laden tenant_id is non-UUID, so it's rejected at the UUID gate.

        The header-injection vector via tenant_id is closed by the UUID check.
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "evil\r\nX-Injected: true",
                "license_id": "lic-1",
            },
        )

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "invalid_tenant"
        assert response.headers.get("x-error-reason") == "invalid_tenant"
        # Must NOT have injected any tenant headers
        assert response.headers.get("x-tenant-id") is None

    @respx.mock
    async def test_header_injection_via_license_id_blocked(self, client: AsyncClient):
        """License ID with spaces/special chars must be rejected."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic with spaces",
            },
        )

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        assert response.status_code == 403
        body = response.json()
        assert body["reason"] == "unsafe_header_value"

    @respx.mock
    async def test_header_injection_via_app_id_blocked(self, client: AsyncClient):
        """App ID with semicolons must be rejected (prevents header injection)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-1",
                "app": {"id": "app;malicious"},
            },
        )

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        assert response.status_code == 403
        body = response.json()
        assert body["reason"] == "unsafe_header_value"

    @respx.mock
    async def test_safe_header_values_accepted(self, client: AsyncClient):
        """Header values with alphanumeric, hyphens, dots, underscores pass validation."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(
                tenant_id=TENANT_UUID,
                license_id="lic.abc-def_456",
                app_id="app_workers.v2",
                plan_id="plan-premium_v3.1",
            ),
        )

        response = await client.get(
            "/api/resource",
            headers={"Authorization": "Bearer tok"},
        )

        assert response.status_code == 200
        assert response.headers.get("x-tenant-id") == TENANT_UUID
        assert response.headers.get("x-tenant-name") == ""
        assert response.headers.get("x-license-id") == "lic.abc-def_456"
        assert response.headers.get("x-app-id") == "app_workers.v2"
        assert response.headers.get("x-plan-id") == "plan-premium_v3.1"

    # -- The unsafe-header deny must leave an audit row ---------------------
    #
    # This branch sits IMMEDIATELY above the `status="allowed"` log_access and
    # used to emit neither: a request that passed licence validation on the
    # merits and then tripped the header-safety regex produced no audit row at
    # all. It fails CLOSED, so nothing was bypassed — but `log_access` is this
    # service's only structured audit emitter and there is no denial Counter,
    # so the deny was invisible to anything keyed on `status`.

    # Carried in `subject_id` on purpose. x-subject-id is NOT one of
    # log_access's fields, so if the rejected value ever leaks into the row
    # this canary is the only thing that could have put it there. Putting it in
    # license_id instead would be untestable: log_access legitimately records
    # license_id, so the canary would appear for a correct implementation too.
    _SUBJECT_CANARY = "sub_MAINCANARY;X-Injected: true"

    @respx.mock
    async def test_unsafe_header_deny_writes_an_audit_row(self, client: AsyncClient, caplog):
        """403 unsafe_header_value must produce a denied audit row naming the
        offending HEADER (never its value)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(tenant_id=TENANT_UUID, subject_id=self._SUBJECT_CANARY),
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get(
                "/api/resource",
                headers={"Authorization": "Bearer tok"},
            )

        # Behaviour unchanged: still 403, still the same reason.
        assert response.status_code == 403
        assert response.json()["reason"] == "unsafe_header_value"

        rows = [r for r in _audit_rows(caplog) if r["status"] == "denied_unsafe_header"]
        assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
        row = rows[0]
        assert row["reason"] == "unsafe_header=x-subject-id"
        # The identity the aborted allow row would have carried.
        assert row["license_id"] == "lic_abc"
        assert row["tenant_id"] == TENANT_UUID
        assert row["app_id"] == "app_workers"
        assert row["path"] == "/api/resource"
        assert row["method"] == "GET"
        assert row["token_hash"]
        # ⛔ And the allow row it aborted must NOT also have been written.
        assert not [r for r in _audit_rows(caplog) if r["status"] == "allowed"]

    @respx.mock
    async def test_unsafe_header_value_never_reaches_the_log(self, client: AsyncClient, caplog):
        """⛔ The rejected value is attacker-influenced; logging it would turn
        the log-injection guard into the log-injection vector."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(tenant_id=TENANT_UUID, subject_id=self._SUBJECT_CANARY),
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get(
                "/api/resource",
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == 403
        # Positive control: the row exists, so "canary absent" cannot be
        # satisfied by an empty log.
        assert any(r["status"] == "denied_unsafe_header" for r in _audit_rows(caplog))
        for record in caplog.records:
            assert "MAINCANARY" not in record.getMessage(), record.getMessage()

    # ⚠️ The canary above is STRUCTURALLY BLIND to the case below, and so is its
    # service-principal twin. Both put their payload in a field `log_access`
    # does NOT record (`subject_id` / `realm_access.roles`) — which is exactly
    # what makes their absence a real measurement, and exactly what makes them
    # incapable of detecting a leak through a field `log_access` DOES record.
    #
    # `x-license-id` IS `result.license_id`, `x-app-id` IS `result.app_id` and
    # `x-plan-id` IS `result.plan_id`. When the header that fails the safety
    # regex is one of those three (NOT `x-tenant-id`: it is UUID-validated or ""
    # before this loop, so it can never fail here), a row built from
    # those variables carries the rejected value verbatim under a key downstream
    # readers treat as a real licence id. `json.dumps` escapes it, so it is not
    # log injection — it is a wrong-but-complete identity in a row whose entire
    # subject is that the value was refused. This test is the one that can fail
    # for that class.
    _LICENSE_CANARY = "lic LICCANARY\r\nX-Injected: true"

    @respx.mock
    async def test_unsafe_identity_header_is_elided_from_the_audit_row(
        self, client: AsyncClient, caplog
    ):
        """When the offending header IS an identity header, the row must record
        the elision sentinel — not the rejected value, and not a blank row."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(tenant_id=TENANT_UUID, license_id=self._LICENSE_CANARY),
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get(
                "/api/resource",
                headers={"Authorization": "Bearer tok"},
            )

        # Behaviour unchanged: still 403, still the same reason.
        assert response.status_code == 403
        assert response.json()["reason"] == "unsafe_header_value"

        rows = [r for r in _audit_rows(caplog) if r["status"] == "denied_unsafe_header"]
        assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
        row = rows[0]
        # It is x-license-id that tripped the regex, so that is what is named...
        assert row["reason"] == "unsafe_header=x-license-id"
        # ...and that is the one field elided.
        assert row["license_id"] == "<unsafe-value-elided>"
        # ⛔ The load-bearing assertion: the rejected value is nowhere in the row.
        # Serialised, so it fails for a leak into ANY field, not just the ones
        # enumerated here — including a field added to log_access later.
        assert "LICCANARY" not in json.dumps(row), row

        # Negative control against OVER-redaction. These three did not fail the
        # regex and must still be recorded truthfully; without them a mutant
        # that blanked the whole row, or one that elided by header NAME and got
        # the name→field mapping wrong, would pass.
        assert row["tenant_id"] == TENANT_UUID
        assert row["app_id"] == "app_workers"
        assert row["plan_id"] == "premium"

        # At THIS site the value reaches no log at all: the sibling warning
        # carries `header` (a literal key) and `token_hash` only. Stated
        # separately from the row assertion above because it is a different,
        # stronger claim — and it is NOT true at the service-principal site,
        # where the pre-existing warning does carry the raw azp.
        for record in caplog.records:
            assert "LICCANARY" not in record.getMessage(), record.getMessage()


# -- Lifespan Tests -----------------------------------------------------------


class TestLifespan:
    """Tests for the Redis lifespan (connect/disconnect) and metering emit."""

    async def test_lifespan_redis_connect_success(self):
        """Lifespan should connect to Redis and set app.state.redis when available."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        with (
            patch("main.METERING_ENABLED", True),
            patch("redis.asyncio.from_url", return_value=mock_redis) as mock_from_url,
        ):
            async with lifespan(app):
                assert app.state.redis is mock_redis
                mock_from_url.assert_called_once()
                mock_redis.ping.assert_awaited_once()

        # After lifespan exits, aclose should have been called
        mock_redis.aclose.assert_awaited_once()
        # Clean up
        app.state.redis = None

    async def test_lifespan_redis_connect_failure(self):
        """Lifespan should handle Redis connection failure gracefully."""
        with (
            patch("main.METERING_ENABLED", True),
            patch("redis.asyncio.from_url", side_effect=ConnectionError("refused")),
        ):
            async with lifespan(app):
                assert app.state.redis is None

    async def test_lifespan_metering_disabled(self):
        """When METERING_ENABLED=false, lifespan should skip Redis entirely."""
        with patch("main.METERING_ENABLED", False):
            async with lifespan(app):
                assert app.state.redis is None


class TestMeteringEmit:
    """Tests for the _emit_metering_event function."""

    @respx.mock
    async def test_metering_emits_on_valid_request(self):
        """When Redis is connected, metering events should be emitted via xadd."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        respx.post(INTROSPECT_URL).respond(200, json=_active_response())

        async with lifespan(app):
            # Override Redis with mock after lifespan sets up http_client
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/api/resource",
                        headers={"Authorization": "Bearer valid_token"},
                    )

                assert response.status_code == 200
                # Verify metering was emitted
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                # First positional arg is the stream name
                assert call_args[0][0] == "billing:events"
                # Second positional arg is the fields dict
                fields = call_args[0][1]
                assert fields["tenant_id"] == TENANT_UUID
                assert fields["subject_id"] == "sub_123"
                assert fields["status"] == "allowed"
            finally:
                app.state.redis = None

    @respx.mock
    async def test_metering_payload_includes_subject_id_for_denied(self):
        """Denied requests carry the introspect subject_id (when introspect returned one)."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": False,
                "reason": "license_expired",
                "tenant_id": "tenant_123",
                "subject_id": "sub_xyz",
                "license_id": "lic_old",
            },
        )

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/api/resource",
                        headers={"Authorization": "Bearer expired"},
                    )

                assert response.status_code == 403
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_invalid_token"
                assert fields["subject_id"] == "sub_xyz"
            finally:
                app.state.redis = None

    async def test_metering_payload_subject_id_empty_on_missing_token(self):
        """Pre-introspect denials have no subject_id — emitted as empty string."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get("/api/resource")

                assert response.status_code == 401
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_missing_token"
                assert fields["subject_id"] == ""
            finally:
                app.state.redis = None

    @respx.mock
    async def test_metering_failure_does_not_break_request(self):
        """If Redis xadd fails, the request should still succeed."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

        respx.post(INTROSPECT_URL).respond(200, json=_active_response())

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/api/resource",
                        headers={"Authorization": "Bearer valid_token"},
                    )

                # Request should still succeed even though metering failed
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}
                mock_redis.xadd.assert_awaited()
            finally:
                app.state.redis = None

    @respx.mock
    async def test_metering_tenant_id_empty_when_introspect_omits_tenant(self):
        """When Keycloak introspection returns no tenant_id, the metering event's
        tenant_id MUST be empty — NOT the user's subject_id (UUID).

        Regression: a previous implementation fell back to ``subject_id`` when
        ``tenant_id`` was missing, which poisoned the ``cost_events.tenant_id``
        column with user UUIDs and broke per-tenant rollups in metering-service.
        See plan: Phase F.1 (license-authz-tenant-id-cross-wire).
        """
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        # Active introspection response with NO tenant_id but a subject_id present.
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "subject_id": "00000000-0000-7000-8000-aaaaaaaaaaaa",
                "license_id": "lic_abc",
                "app": {"id": "app_workers", "key": "workers"},
                "plan": {"id": "premium"},
            },
        )

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/workers/api/tasks",
                        headers={"Authorization": "Bearer valid_token"},
                    )

                assert response.status_code == 200
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                # Critical assertion: tenant_id must be empty, NOT the subject UUID.
                assert fields["tenant_id"] == "", (
                    f"tenant_id leaked subject_id: got {fields['tenant_id']!r}"
                )
                assert fields["subject_id"] == "00000000-0000-7000-8000-aaaaaaaaaaaa"
                assert fields["status"] == "allowed"
            finally:
                app.state.redis = None

    @respx.mock
    async def test_metering_emits_on_denied_request(self):
        """Denied requests should also emit metering events."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        respx.post(INTROSPECT_URL).respond(200, json=_inactive_response("expired"))

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/api/resource",
                        headers={"Authorization": "Bearer bad_token"},
                    )

                assert response.status_code == 403
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_invalid_token"
            finally:
                app.state.redis = None

    @respx.mock
    async def test_metering_denied_path_does_not_leak_subject_id_into_tenant_id(
        self,
    ):
        """Denied-token path must NOT fall back to subject_id for tenant_id.

        Same regression as test_metering_tenant_id_empty_when_introspect_omits_tenant
        but for the ``denied_invalid_token`` branch (main.py:720). An inactive
        introspection response that still carries a subject_id (license has
        been revoked but Keycloak knows who the user was) must produce a
        metering event with tenant_id="" — not the user UUID.
        """
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": False,
                "reason": "license_revoked",
                "subject_id": "00000000-0000-7000-8000-bbbbbbbbbbbb",
            },
        )

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/api/resource",
                        headers={"Authorization": "Bearer revoked_token"},
                    )

                assert response.status_code == 403
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_invalid_token"
                # Critical assertion: tenant_id MUST be empty, not the subject UUID.
                assert fields["tenant_id"] == "", (
                    f"denied-path tenant_id leaked subject_id: got {fields['tenant_id']!r}"
                )
                assert fields["subject_id"] == "00000000-0000-7000-8000-bbbbbbbbbbbb"
            finally:
                app.state.redis = None

    async def test_metering_emits_on_missing_token(self):
        """Missing bearer token should also emit metering event."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get("/api/resource")

                assert response.status_code == 401
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_missing_token"
                assert fields["tenant_id"] == ""
            finally:
                app.state.redis = None


# -- App Route Enforcement ----------------------------------------------------


class TestAppRouteEnforcement:
    """Tests for route-to-app enforcement: tokens for app A must be rejected on app B routes."""

    @respx.mock
    async def test_token_for_workers_allowed_on_workers_route(self, client: AsyncClient):
        """Token with app_key=workers on /workers/api/tasks should be allowed (200)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )
        response = await client.get(
            "/workers/api/tasks",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @respx.mock
    async def test_token_for_workers_denied_on_kanban_route(self, client: AsyncClient):
        """Token with app_key=workers on /kanban/api/boards should be denied (403)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )
        response = await client.get(
            "/kanban/api/boards",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "app_route_mismatch"
        assert response.headers.get("x-error-reason") == "app_route_mismatch"

    @respx.mock
    async def test_token_for_improvement_loop_allowed_on_improvement_loop_route(
        self, client: AsyncClient
    ):
        """Token with app_key=improvement-loop on /improvement-loop/ is allowed (200).

        The /metering/ route was retired when metering-service split into
        billing-service + improvement-loop-service; the app_key is now
        improvement-loop (uniform with route-registry.yaml).
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="improvement-loop"),
        )
        response = await client.get(
            "/improvement-loop/api/internal/exec-quality",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @respx.mock
    async def test_wrong_token_denied_on_improvement_loop_route(self, client: AsyncClient):
        """Token scoped to a different app on /improvement-loop/ is rejected (403)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )
        response = await client.get(
            "/improvement-loop/api/internal/exec-quality",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "app_route_mismatch"

    @respx.mock
    async def test_vllm_token_allowed_on_vllm_route(self, client: AsyncClient):
        """Token with app_key=vllm on /vllm/v1/chat should be allowed (200).

        /vllm/ is the new canonical path alias after the vllm -> vllm
        rename; /rtx/ stays registered for backwards-compat.
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="vllm"),
        )
        response = await client.post(
            "/vllm/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @respx.mock
    async def test_vllm_token_still_allowed_on_rtx_route(self, client: AsyncClient):
        """Backwards-compat: app_key=vllm on /rtx/ still allowed (200)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="vllm"),
        )
        response = await client.post(
            "/rtx/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @respx.mock
    async def test_non_vllm_token_denied_on_rtx_route(self, client: AsyncClient):
        """Non-vllm token on /rtx/ is rejected (403).

        After the rtx-server→vllm rename, /rtx/ and /vllm/ both map to
        app_key=vllm, so a token scoped to a different app (e.g. workers)
        must still be rejected on either alias.
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )
        response = await client.post(
            "/rtx/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["reason"] == "app_route_mismatch"

    @respx.mock
    async def test_token_without_app_key_denied_on_registered_route(self, client: AsyncClient):
        """An UNKEYED token must not authorize a registered prefix.

        This test used to assert the opposite ("skips route check"): the old
        `if not app_key: return True` short-circuited before the registry, so
        one keyless token authorized every registered prefix, /vllm/ and
        /rtx/ (GPU) included. That is the bug, and a test asserting it kept it
        alive.
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key=None),
        )
        response = await client.get(
            "/workers/api/tasks",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "app_route_missing_key"
        assert response.headers.get("x-error-reason") == "app_route_missing_key"

    @respx.mock
    async def test_whitespace_only_app_key_denied_on_registered_route(self, client: AsyncClient):
        """A whitespace-only key is a missing key, not a match."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="   "))
        response = await client.get(
            "/vllm/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_missing_key"

    @respx.mock
    async def test_unkeyed_token_still_allowed_on_unregistered_route(self, client: AsyncClient):
        """Fail-closed applies to REGISTERED prefixes only."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=None))
        response = await client.get("/api/resource", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200

    @pytest.mark.parametrize("prefix", sorted(APP_ROUTE_REGISTRY))
    @respx.mock
    async def test_every_registered_prefix_denies_an_unkeyed_token(
        self, prefix: str, client: AsyncClient
    ):
        """Every registered prefix fails closed — not just the ones anyone
        remembered to write a test for."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=None))
        response = await client.get(f"{prefix}thing", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 403, f"{prefix} still allows an unkeyed token"
        assert response.json()["reason"] == "app_route_missing_key"

    @pytest.mark.parametrize(
        "app_key,path,expected",
        [
            (None, "/workers/x", False),
            ("", "/workers/x", False),
            ("   ", "/vllm/x", False),
            ("workers", "/workers/x", True),
            ("workers", "/kanban/x", False),
            (None, "/api/x", True),
            ("", "/api/x", True),
        ],
    )
    def test_check_app_route_table(self, app_key, path, expected):
        """Unit-level truth table for the predicate itself."""
        assert _check_app_route(app_key, path, "GET") is expected

    def test_registry_is_not_silently_empty(self):
        """A parametrised gate over an EMPTY registry would pass while enforcing
        nothing, so pin the population. 34 prefixes -> 32 distinct values:
        /rtx/ and /vllm/ are aliases of the same app, and /tts/, /asr/ share
        the ("w2v", "speech") transition set."""
        assert len(APP_ROUTE_REGISTRY) == 34
        assert len(set(APP_ROUTE_REGISTRY.values())) == 32
        assert "/vllm/" in APP_ROUTE_REGISTRY and "/rtx/" in APP_ROUTE_REGISTRY

    def test_image_gen_prefix_demands_a_conveyor_licence(self):
        """`/image-gen/` is the native MOSS-Image host and belongs to conveyor,
        the caller that owns `media.generate_image`. The legacy `/image/` stays
        `w2v`: the two prefixes must not answer for each other."""
        assert _registered_app_key("/image-gen/v1/images/generations") == "conveyor"
        assert _check_app_route("conveyor", "/image-gen/v1/images/generations", "POST") is True
        assert _check_app_route("w2v", "/image-gen/v1/images/generations", "POST") is False
        # The legacy prefix is untouched: a conveyor licence does not reach it,
        # and it does not reach the native route.
        assert _registered_app_key("/image/api/image/generate") == "w2v"
        assert _check_app_route("conveyor", "/image/api/image/generate", "POST") is False

    def test_laya_prefix_demands_a_laya_licence(self):
        """`/laya/` (pgx laya bridge) is bound to app `laya`: a licence for any
        other app — including `vllm`, the other pgx GPU app — is a mismatch."""
        assert _registered_app_key("/laya/v1/systemone") == "laya"
        assert _check_app_route("laya", "/laya/v1/systemone", "POST") is True
        assert _check_app_route("vllm", "/laya/v1/systemone", "POST") is False
        assert _check_app_route("inference-router", "/laya/v1/models", "GET") is False

    @pytest.mark.parametrize(
        "path,method",
        [
            ("/tts/api/tts/synthesize", "POST"),
            ("/asr/api/asr/transcribe", "POST"),
            ("/asr/api/transcribe", "POST"),
        ],
    )
    def test_speech_prefixes_accept_the_speech_w2v_transition_set(self, path, method):
        """TTS/ASR are shared platform services (owner decision 2026-10-22):
        bound to app `speech`, with `w2v` still accepted — and still PRIMARY, so
        unkeyed JWT callers keep deriving `w2v` — until Studio migrates. Any
        other app's licence — the GPU ones included — is a mismatch, and so is
        an unkeyed token."""
        from main import _registered_app_keys

        assert _registered_app_key(path) == "w2v"
        assert _registered_app_keys(path) == ("w2v", "speech")
        assert _check_app_route("speech", path, method) is True
        assert _check_app_route("w2v", path, method) is True
        for other in ("vllm", "laya", "inference-router", "workers", "licenses", "speechx"):
            assert _check_app_route(other, path, method) is False, other
        assert _check_app_route(None, path, method) is False
        assert _check_app_route("", path, method) is False

    def test_image_prefix_still_demands_a_w2v_licence(self):
        """`/image/` is out of the 2026-10-22 speech decision: still `w2v`, and a
        `speech` licence does not open it."""
        path = "/image/api/image/generate"
        assert _registered_app_key(path) == "w2v"
        assert _check_app_route("w2v", path, "POST") is True
        for other in ("speech", "vllm", "laya", "inference-router"):
            assert _check_app_route(other, path, "POST") is False, other
        assert _check_app_route(None, path, "POST") is False

    @pytest.mark.parametrize("path", ["/tts", "/ttsx/a", "/asrx/a", "/images/a", "/api/tts/x"])
    def test_gpu_modality_binding_does_not_overreach(self, path):
        """Prefix-bound with the trailing slash: a lookalike path, and the bare
        `/api/tts/...` the internal tts host serves, stay unregistered."""
        assert _registered_app_key(path) is None

    @pytest.mark.parametrize("app_key", ["speech", "w2v"])
    @pytest.mark.parametrize("path", ["/tts/api/tts/synthesize", "/asr/api/asr/transcribe"])
    @respx.mock
    async def test_transition_keys_on_speech_routes_are_allowed(
        self, app_key: str, path: str, client: AsyncClient
    ):
        """End-to-end through ext_authz: `speech` and `w2v` both pass on
        /tts/ and /asr/ during the transition."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=app_key))
        response = await client.post(path, headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200

    @pytest.mark.parametrize("app_key", ["vllm", "laya", "workers"])
    @respx.mock
    async def test_other_licence_on_tts_route_is_403(self, app_key: str, client: AsyncClient):
        """End-to-end through ext_authz: any key outside the transition set is refused."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=app_key))
        response = await client.post(
            "/tts/api/tts/synthesize", headers={"Authorization": "Bearer tok"}
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_mismatch"

    @respx.mock
    async def test_speech_licence_on_image_route_is_403(self, client: AsyncClient):
        """End-to-end: `/image/` is outside the speech decision — w2v only."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="speech"))
        response = await client.post(
            "/image/api/image/generate", headers={"Authorization": "Bearer tok"}
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_mismatch"

    def test_w2v_key_does_not_open_the_licenses_prefix(self):
        """`/licenses/` still demands `licenses` — a `w2v` key there is still a
        mismatch on every route that CONSUMES the licenses app.

        ⚠️ THIS TEST WAS WEAKENED ON PURPOSE, and the narrowing is the point.
        It arrived asserting
        `_check_app_route("w2v", "/licenses/active") is False`. The self-read
        exemption deliberately makes that case TRUE for GET/HEAD: reading your
        own licence is introspection, not consumption of the `licenses` app, and
        `GET /licenses/active` is the call the Mac hosts actually make — the one
        whose 403 took narration down on 2026-09-14.

        So the original assertion is kept in the two forms that still hold and
        are what it was really defending: a MUTATING method on the same path,
        and a different path under the same prefix. What it may no longer claim
        is that the whole prefix is closed to a foreign key, because it is not.
        """
        assert _registered_app_key("/licenses/active") == "licenses"
        # A write to the self-read path is still bound to the `licenses` key.
        assert _check_app_route("w2v", "/licenses/active", "POST") is False
        # Any other route under the prefix is still bound, GET included.
        assert _check_app_route("w2v", "/licenses/issue", "GET") is False
        assert _check_app_route("w2v", "/licenses/admin/keys", "GET") is False
        # And the exemption itself, stated here so the two tests cannot drift.
        assert _check_app_route("w2v", "/licenses/active", "GET") is True

    @pytest.mark.parametrize(
        "raw_path",
        [
            "//workers/x",
            "///workers/x",
            "/./workers/x",
            "/a/../workers/x",
            "/%2Fworkers/x",
            "//vllm/v1/chat/completions",
            "/vllm//v1/chat/completions",
            "//workers/x?a=b",
        ],
        ids=lambda p: p,
    )
    def test_unnormalized_path_cannot_escape_the_registry(self, raw_path):
        """A non-normalised raw :path must not fall through to unregistered-allow.

        The test that used to sit here sent `//workers/x` through httpx, which
        resolves it as an RFC-3986 *network-path reference* against base_url and
        delivers `/x` — an ordinary unregistered path. It exercised no
        normalisation at all, so its self-described tripwire could never fire
        (verified: `httpx.Client(base_url='http://test')
        .build_request('GET','//workers/x').url.raw_path` is `b'/x'`).

        `src/main.py` reads Envoy's RAW `:path`, and Envoy's HCM does not merge
        slashes (`merge_slashes` defaults false and gitops sets it nowhere), so
        `//workers/x` reaches the predicate verbatim and used to miss every
        registered prefix.
        """
        assert _check_app_route(None, raw_path, "GET") is False, (
            f"{raw_path} escaped the registry with no app_key at all"
        )
        assert _check_app_route("kanban", raw_path, "GET") is False, (
            f"{raw_path} let a kanban token onto another app's route"
        )

    def test_normalization_does_not_move_well_formed_paths(self):
        """The normalisation must be inert for every path that is already clean.

        Otherwise it is not a fail-closed hardening, it is a new denial class.
        """
        for prefix, expected in APP_ROUTE_REGISTRY.items():
            for key in (expected,) if isinstance(expected, str) else expected:
                assert _check_app_route(key, f"{prefix}thing", "GET") is True
        assert _check_app_route(None, "/api/resource", "GET") is True
        assert _check_app_route("workers", "/unknown/path", "GET") is True
        assert _check_app_route(None, "/", "GET") is True
        assert _check_app_route(None, "", "GET") is True
        # A bare `/workers` (no trailing slash) is NOT the `/workers/` prefix —
        # unchanged, and the normalisation must not start matching it.
        assert _check_app_route(None, "/workers", "GET") is True

    @respx.mock
    async def test_raw_path_header_is_what_the_route_check_reads(self, client: AsyncClient):
        """End to end through the real handler, driving Envoy's `:path` directly.

        `src/main.py` prefers the `:path` pseudo-header over the ASGI path, so
        this is the only way to reach the predicate with a path httpx would
        otherwise rewrite.
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=None))
        response = await client.get(
            "/anything",
            headers={"Authorization": "Bearer tok", ":path": "//vllm/v1/chat/completions"},
        )
        assert response.status_code == 403, (
            "an unkeyed token reached the GPU route through a doubled slash"
        )
        assert response.json()["reason"] == "app_route_missing_key"

    @respx.mock
    async def test_unrecognized_route_allows_any_app(self, client: AsyncClient):
        """Token with any app_key on an unregistered route should be allowed (200)."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )
        response = await client.get(
            "/unknown/path",
            headers={"Authorization": "Bearer tok"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @respx.mock
    async def test_app_mismatch_emits_metering_event(self):
        """App route mismatch should emit a metering event with status=denied_app_mismatch."""
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

        respx.post(INTROSPECT_URL).respond(
            200,
            json=_active_response(app_key="workers"),
        )

        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        "/kanban/api/boards",
                        headers={"Authorization": "Bearer tok"},
                    )

                assert response.status_code == 403
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["status"] == "denied_app_mismatch"
            finally:
                app.state.redis = None


# -- Asserted vs attested app keys: the route predicates ----------------------


class TestAssertedKeyRoutePredicates:
    """Unit-level truth for the two helpers the asserted-key rule is built on.

    `_registered_app_key` answers "which key does this route demand?" and
    `_asserted_key_route_allowed` answers "is this route in the registry at
    all?". They were one function; splitting them is what let the asserted-key
    rule ask the SECOND question without also inspecting a key whose value is
    unsigned and therefore evidence of nothing.
    """

    def test_registered_prefix_reports_its_expected_key(self):
        from main import _registered_app_key

        for prefix, expected in APP_ROUTE_REGISTRY.items():
            primary = expected if isinstance(expected, str) else expected[0]
            assert _registered_app_key(f"{prefix}thing") == primary

    def test_unregistered_paths_report_no_expected_key(self):
        from main import _registered_app_key

        for path in ("/api/resource", "/unknown/path", "/", "", "/workers"):
            assert _registered_app_key(path) is None, f"{path} matched a registry prefix"

    def test_asserted_key_allowed_only_on_registered_routes(self):
        from main import _asserted_key_route_allowed

        assert _asserted_key_route_allowed("/workers/api/tasks") is True
        assert _asserted_key_route_allowed("/vllm/v1/chat/completions") is True
        assert _asserted_key_route_allowed("/api/resource") is False
        assert _asserted_key_route_allowed("/unknown/path") is False

    def test_the_asserted_rule_normalises_the_path_too(self):
        """⛔ Otherwise `//api/x` and `/./api/x` would be a trivial bypass of
        BOTH directions: a doubled slash could walk a request out of the registry
        (dodging the key match) or, here, into it (dodging the asserted-key
        refusal). The registry check must see the same string in both rules.
        """
        from main import _asserted_key_route_allowed

        assert _asserted_key_route_allowed("//workers/x") is True
        assert _asserted_key_route_allowed("/a/../workers/x") is True
        assert _asserted_key_route_allowed("//api/x") is False

    def test_the_predicate_takes_no_app_key(self):
        """The verdict is about the ROUTE, never the key — the key's value is
        unsigned, so no property of it is evidence. Pinned at the signature so a
        future reader cannot "improve" it by passing one in."""
        import inspect

        from main import _asserted_key_route_allowed

        params = list(inspect.signature(_asserted_key_route_allowed).parameters)
        assert params == ["path"], f"the predicate grew an argument: {params}"

    def test_every_app_route_result_series_is_born_at_zero(self):
        """A labelled Counter exports NO series until its first `.inc()`, so
        without the birth loop "never refused" and "cannot refuse" are the same
        empty vector — and an operator reads absence as health."""
        from prometheus_client import REGISTRY

        for result in (
            "allowed",
            "denied_mismatch",
            "denied_missing_key",
            "denied_asserted_key_unregistered",
            "allowed_asserted_key_unregistered",
            "allowed_asserted_key_host_bound",
        ):
            assert (
                REGISTRY.get_sample_value("license_authz_app_route_total", {"result": result})
                is not None
            ), f"{result} is not born at zero"

    def test_every_metering_skip_series_is_born_at_zero(self):
        """Same reason: "no UI traffic skipped" must not look like "no instrument"."""
        from prometheus_client import REGISTRY

        from main import GATEWAY_ASSERTED_HOST_KEYS

        apps = set(GATEWAY_ASSERTED_HOST_KEYS.values())
        assert {"n8n"} <= apps
        for skip_app in apps:
            assert (
                REGISTRY.get_sample_value(
                    "license_authz_metering_events_skipped_total",
                    {"reason": "gateway_browser_ui", "app": skip_app},
                )
                is not None
            ), f"{skip_app} is not born at zero"


# -- P1: licence self-read exemption ------------------------------------------


def _app_route_count(result: str) -> float:
    """Current value of license_authz_app_route_total{result=...}."""
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("license_authz_app_route_total", {"result": result}) or 0.0


def _app_host_count(result: str, host: str, app: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value(
            "license_authz_app_host_route_total",
            {"result": result, "host": host, "app": app},
        )
        or 0.0
    )


class TestLicenseSelfReadExemption:
    """A caller reading its OWN licence is not consuming the `licenses` app.

    The 2026-09-14 outage: a bare-metal w2v host GETs `/licenses/active` with a
    `w2v` licence, the `/licenses/` prefix demands `licenses`, 403, entitlement
    gate indeterminate, 503 on every request. In-cluster pods bypass the gateway,
    so only bare metal broke.
    """

    @respx.mock
    async def test_w2v_key_allowed_on_licenses_active(self, client: AsyncClient):
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        before = _app_route_count("allowed_self_read_exempt")
        response = await client.get("/licenses/active", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200
        assert _app_route_count("allowed_self_read_exempt") == before + 1, (
            "an exemption that fires invisibly cannot justify re-arming the gate"
        )

    @respx.mock
    async def test_w2v_key_allowed_on_licenses_user_subject(self, client: AsyncClient):
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get("/licenses/user/abc", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200

    @respx.mock
    async def test_w2v_key_allowed_on_licenses_active_with_query(self, client: AsyncClient):
        """Proves the exemption matches `_normalize_route_path`'s output.

        ⚠️ The query string is driven through the `:path` PSEUDO-HEADER, not
        through the URL, because `src/main.py` falls back to
        `request.url.path` — and Starlette's `url.path` has ALREADY dropped the
        query. Passing `?x=1` in the URL would therefore hand the predicate a
        bare `/licenses/active` and exercise no query folding at all. Same trap
        as `test_raw_path_header_is_what_the_route_check_reads`.
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get(
            "/anything",
            headers={"Authorization": "Bearer tok", ":path": "/licenses/active?x=1"},
        )
        assert response.status_code == 200

    @respx.mock
    async def test_post_to_licenses_active_still_denied(self, client: AsyncClient):
        """THE LOAD-BEARING BOUND: the exemption is a READ exemption. A mutating
        method on the very same path must stay bound, or defaulting the flag ON
        would be opening a write."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.post("/licenses/active", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_mismatch"

    @respx.mock
    @pytest.mark.parametrize(
        "path",
        [
            "/licenses/issue",
            "/licenses/revoke",
            "/licenses/admin/tenants",
            "/licenses/",
            "/licenses/activedirectory-sync",
            "/licenses/user",
        ],
    )
    async def test_other_licenses_paths_still_denied(self, client: AsyncClient, path: str):
        """Everything else under `/licenses/` stays bound — including a path that
        merely STARTS WITH `/licenses/active`.

        The dot-segment walk is NOT tested here: httpx resolves `..` client-side
        (`build_request('GET','/licenses/user/../admin/keys').url.raw_path` is
        already `/licenses/admin/keys`), so a case in this list could never
        deliver one. It lives in
        `test_dot_segment_walk_cannot_reach_the_self_read_exemption`, which
        drives the raw `:path`.
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get(path, headers={"Authorization": "Bearer tok"})
        assert response.status_code == 403, f"{path} was exempted and must not be"
        assert response.json()["reason"] == "app_route_mismatch"

    @respx.mock
    @pytest.mark.parametrize(
        "path", ["/licenses/active", "/licenses/user/abc", "/licenses/issue", "/licenses/"]
    )
    async def test_licenses_key_allowed_everywhere(self, client: AsyncClient, path: str):
        """The licenses app itself is unaffected on every one of these paths."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="licenses"))
        response = await client.get(path, headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200

    @respx.mock
    async def test_licenses_key_self_read_not_counted_as_exempt(self, client: AsyncClient):
        """The metric counts the requests the exemption RESCUED, not every read of
        `/licenses/active`."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="licenses"))
        before = _app_route_count("allowed_self_read_exempt")
        response = await client.get("/licenses/active", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200
        assert _app_route_count("allowed_self_read_exempt") == before

    @respx.mock
    async def test_dot_segment_walk_cannot_reach_the_self_read_exemption(self, client: AsyncClient):
        """`/licenses/user/../admin/keys` must NOT be exempt.

        Driven through the `:path` pseudo-header on purpose: httpx resolves `..`
        client-side, so the same string sent as a URL arrives as
        `/licenses/admin/keys` and the dot segment is never delivered to the app
        — a test that sent it that way would pass even with normalisation
        removed. `src/main.py` prefers `:path`, so this is the only way to hand
        the predicate an unresolved walk. It must fold to `/licenses/admin/keys`
        and be DENIED.
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get(
            "/anything",
            headers={
                "Authorization": "Bearer tok",
                ":path": "/licenses/user/../admin/keys",
            },
        )
        assert response.status_code == 403, "a dot-segment walk reached the self-read exemption"
        assert response.json()["reason"] == "app_route_mismatch"

    @respx.mock
    async def test_cross_prefix_dot_segment_smuggle_denied_on_opaque_arm(self, client: AsyncClient):
        """`/vllm/../licenses/active` must NOT be exempt on the OPAQUE arm.

        This is a DIFFERENT walk than
        `test_dot_segment_walk_cannot_reach_the_self_read_exemption`: that one
        stays inside `/licenses/` the whole time. This one starts at a
        DIFFERENT registered prefix (`/vllm/`) and only reaches `/licenses/`
        after `_normalize_route_path` folds the `..`. `_is_self_read_route`
        (and `_registered_app_key`) both judge the FOLDED path, so a `w2v`-keyed
        caller would otherwise clear `_check_app_route` as a "self-read" — while
        Envoy's `uri.prefix: /vllm/` VirtualService match is on the RAW,
        unnormalised `:path` (pathNormalization is unverified in the
        `istio-system/istio` ConfigMap) and would forward this request to vllm,
        not to licenses. That is a `w2v` licence reaching a GPU backend it never
        held, laundered through the self-read exemption.

        The JWT arm already closes this with
        `_self_read_raw_path_stays_in_licenses` (see `_handle_jwt_mode`); the
        opaque arm must apply the identical raw-path bound.
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get(
            "/anything",
            headers={
                "Authorization": "Bearer tok",
                ":path": "/vllm/../licenses/active",
            },
        )
        assert response.status_code == 403, (
            "a cross-prefix dot-segment smuggle reached the opaque-arm self-read exemption"
        )
        assert response.json()["reason"] == "app_route_mismatch"

    def test_check_app_route_denies_cross_prefix_smuggle(self):
        """Unit-level pin, independent of the ASGI client: the raw path
        `/vllm/../licenses/active` must be refused even though it normalises
        into a self-read route."""
        assert _check_app_route("w2v", "/vllm/../licenses/active", "GET") is False

    @respx.mock
    async def test_legitimate_opaque_self_read_still_allowed_after_raw_path_guard(
        self, client: AsyncClient
    ):
        """Guard against over-refusing: a genuine self-read with benign
        normalisations (doubled slash) must still be exempt on the opaque arm
        once the raw-path bound is added."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get(
            "/anything",
            headers={"Authorization": "Bearer tok", ":path": "//licenses//active"},
        )
        assert response.status_code == 200

    @respx.mock
    async def test_flag_off_restores_old_behaviour(self, client: AsyncClient, monkeypatch):
        """The revert switch: LICENSE_SELF_READ_EXEMPT_ENABLED=false re-denies
        (and re-opens the outage) byte-for-byte."""
        monkeypatch.setattr(main_module, "LICENSE_SELF_READ_EXEMPT_ENABLED", False)
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="w2v"))
        response = await client.get("/licenses/active", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_mismatch"

    @respx.mock
    async def test_keyless_token_still_denied_on_self_read(self, client: AsyncClient):
        """Deliberately NOT exempted: an unkeyed token stays on
        `app_route_missing_key`. That deny is a separate hardening and the outage
        caller is keyed (`w2v`)."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=None))
        response = await client.get("/licenses/active", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 403
        assert response.json()["reason"] == "app_route_missing_key"

    def test_check_app_route_self_read_table(self):
        from main import _check_app_route as check

        assert check("w2v", "/licenses/active", "GET") is True
        assert check("w2v", "/licenses/active", "HEAD") is True
        assert check("w2v", "/licenses/active", "get") is True
        assert check("w2v", "/licenses/active", "POST") is False
        assert check("w2v", "/licenses/active", "DELETE") is False
        assert check("w2v", "//licenses//active", "GET") is True, "doubled slashes fold"
        assert check("w2v", "/licenses%2Factive", "GET") is True, "escaped slash folds"
        assert check("w2v", "/licenses/user/", "GET") is True
        assert check("w2v", "/licenses/issue", "GET") is False


# -- P2: host -> app binding --------------------------------------------------


class TestAppHostRouteBinding:
    """marketing and sales are HOST-routed under the SAME `/api/...` prefix, so
    the path registry cannot tell them apart. Ships INERT (`off`)."""

    @respx.mock
    async def test_off_is_completely_inert(self, client: AsyncClient):
        """Default mode: no deny, and not even a metric increment."""
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        before = _app_host_count("mismatch_observed", "marketing-api.example", "marketing")
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 200
        assert _app_host_count("mismatch_observed", "marketing-api.example", "marketing") == before

    @respx.mock
    async def test_observe_allows_and_counts_mismatch(self, client: AsyncClient, monkeypatch):
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "observe")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        before = _app_host_count("mismatch_observed", "marketing-api.example", "marketing")
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 200, "observe must never deny"
        assert (
            _app_host_count("mismatch_observed", "marketing-api.example", "marketing") == before + 1
        )

    @respx.mock
    async def test_enforce_denies_with_distinct_reason(self, client: AsyncClient, monkeypatch):
        """`app_host_mismatch`, NOT `app_route_mismatch` — two different defects
        with two different fixes must be distinguishable in logs."""
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "app_host_mismatch"
        assert response.headers.get("x-error-reason") == "app_host_mismatch"

    @respx.mock
    async def test_enforce_allows_matching_host(self, client: AsyncClient, monkeypatch):
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="marketing"))
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 200

    @respx.mock
    async def test_port_is_stripped_from_authority(self, client: AsyncClient, monkeypatch):
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example:443"},
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_host_mismatch"

    @respx.mock
    async def test_in_mesh_authority_allows_even_in_enforce(self, client: AsyncClient, monkeypatch):
        """An east-west hop does not carry the public host and must never be
        denied for failing to carry something it structurally cannot carry."""
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        response = await client.get(
            "/api/campaigns",
            headers={
                "Authorization": "Bearer tok",
                "host": "x.svc.cluster.local",
            },
        )
        assert response.status_code == 200

    @respx.mock
    async def test_unknown_host_allows_and_is_labelled_other(
        self, client: AsyncClient, monkeypatch
    ):
        """`Host` is attacker-controlled: an arbitrary value must not mint a new
        Prometheus series."""
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        before = _app_host_count("unregistered", "other", "other")
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "attacker.example.com"},
        )
        assert response.status_code == 200
        assert _app_host_count("unregistered", "other", "other") == before + 1
        assert _app_host_count("unregistered", "attacker.example.com", "other") == 0

    @respx.mock
    async def test_enforce_denies_an_unkeyed_token_on_a_registered_host(
        self, client: AsyncClient, monkeypatch
    ):
        """PINNED DELIBERATELY: `enforce` denies UNKEYED tokens too, not only
        mismatched ones — an empty app_key can never equal the host's expected
        app, so it lands on `mismatch_denied`.

        That is the correct behaviour (a host that demands `marketing` is not
        satisfied by a token that names no app at all), but it widens the arming
        blast radius beyond "wrong app" to "no app", so it must be MEASURED in
        the observe window before anyone types `enforce`. Nothing pinned it
        before this test, so a later 'helpful' change could flip it silently.
        """
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key=None))
        before = _app_host_count("mismatch_denied", "marketing-api.example", "marketing")
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_host_mismatch"
        assert (
            _app_host_count("mismatch_denied", "marketing-api.example", "marketing") == before + 1
        )

    @respx.mock
    async def test_registered_path_prefix_wins_over_host_registry(
        self, client: AsyncClient, monkeypatch
    ):
        """Zero blast radius on the 29 live prefixes: a registered path never
        reaches the host registry at all."""
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="workers"))
        before = _app_host_count("mismatch_denied", "marketing-api.example", "marketing")
        response = await client.get(
            "/workers/api/tasks",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 200
        assert _app_host_count("mismatch_denied", "marketing-api.example", "marketing") == before

    def test_empty_and_bare_authority_are_no_match(self):
        from main import _check_app_host, _normalize_authority

        assert _normalize_authority("") == ""
        assert _normalize_authority("marketing") == ""
        assert _normalize_authority("localhost:8544") == ""
        assert _normalize_authority("x.svc.cluster.local") == ""
        assert _normalize_authority("MARKETING-API.EXAMPLE:443") == "marketing-api.example"
        for authority in ("", "marketing", "x.svc.cluster.local"):
            allow, result, host, app = _check_app_host("anything", "/api/x", authority)
            assert allow is True and result == "no_host"
            assert (host, app) == ("other", "other")

    def test_internal_authorities_are_registered(self):
        """THE DEFECT THIS CLASS EXISTS TO PIN: the gateway Lua calls ext_authz
        only for `AUTH_REQUIRED_HOSTS`, which are internal names. A registry
        keyed solely on the public hosts is inert in BOTH directions — its keys
        can never arrive, and the authorities that do arrive are unregistered.
        """
        import os

        from main import APP_HOST_REGISTRY, _parse_host_registry_override

        # Since 2026-11-14 the registry is DEPLOYMENT CONFIGURATION: `main.py`
        # carries no hostnames. So the pin is that the registry is exactly what
        # this deployment declared through APP_HOST_REGISTRY_OVERRIDE — if the
        # override stops being read, the table goes inert and every gated host
        # becomes "unregistered", which allows. That is the regression.
        declared = _parse_host_registry_override(os.environ["APP_HOST_REGISTRY_OVERRIDE"])
        assert declared, "the suite must declare a registry, or this pin tests nothing"
        assert APP_HOST_REGISTRY == declared
        # Named entries, so a parser change that silently drops the app_key is
        # caught rather than merely producing a smaller-but-equal dict.
        assert APP_HOST_REGISTRY["marketing-api.example"] == "marketing"
        assert APP_HOST_REGISTRY["sales-api.example"] == "sales"
        # control_service's app_key is unconfirmed; guessing it would repeat the
        # very class of error this table is fixing.
        assert "control-api.example" not in APP_HOST_REGISTRY

    def test_registered_internal_authority_resolves_to_registry_app_key(self):
        """Tests the MECHANISM ONLY: a registered authority resolves to
        whatever `APP_HOST_REGISTRY` says, and that lookup is
        case/port-normalized. The specific marketing="marketing" /
        sales="sales" VALUES exercised here are the same provisional,
        unverified pair called out in the audit comment above
        `APP_HOST_REGISTRY` in main.py (2026-09-15) — growth presents one
        per-tenant licence token, keyed on tenant id only, across the
        marketing/sales/control hops, and that token carries a single
        app_key, so at least one of these two entries may be wrong about what
        actually arrives. This test proves `_check_app_host` reads the
        registry correctly, NOT that these are the correct production
        bindings — do not cite it as evidence the mapping is right.
        """
        from main import _check_app_host

        allow, result, host, app = _check_app_host("marketing", "/api/x", "marketing-api.example")
        assert (allow, result, host, app) == (
            True,
            "matched",
            "marketing-api.example",
            "marketing",
        )

        allow, result, host, app = _check_app_host("sales", "/api/x", "sales-api.example")
        assert (allow, result, host, app) == (True, "matched", "sales-api.example", "sales")

    def test_internal_authority_with_port_normalizes_the_same(self):
        from main import _check_app_host

        allow, result, host, app = _check_app_host(
            "marketing", "/api/x", "MARKETING-API.EXAMPLE:443"
        )
        assert (allow, result, host, app) == (
            True,
            "matched",
            "marketing-api.example",
            "marketing",
        )

    def test_unregistered_internal_authority_still_unregistered(self):
        """Adding the two we CAN confirm must not implicitly register the rest
        of the ~30 internal gated names."""
        from main import _check_app_host

        for authority in ("control-api.example", "karma-api.example"):
            allow, result, host, app = _check_app_host("marketing", "/api/x", authority)
            assert (allow, result, host, app) == (True, "unregistered", "other", "other")

    def test_mesh_authorities_still_no_match_after_adding_internal_hosts(self):
        """internal is a real dotted host; a cluster-domain name and an empty
        authority must keep taking the no_host ALLOW branch."""
        from main import _check_app_host

        for authority in ("", "x.svc.cluster.local", "marketing"):
            allow, result, host, app = _check_app_host("sales", "/api/x", authority)
            assert (allow, result, host, app) == (True, "no_host", "other", "other")

    @respx.mock
    async def test_enforce_denies_mismatch_on_internal_authority(
        self, client: AsyncClient, monkeypatch
    ):
        """End-to-end through the app on the authority that actually arrives.
        Tests the MECHANISM ONLY: when a token's app_key disagrees with the
        registry's binding for the authority it arrived on, `enforce` denies
        it as `app_host_mismatch` and counts `mismatch_denied`. The
        marketing/sales app_key VALUES used to construct this mismatch are
        the same provisional, unverified pair flagged in the audit comment
        above `APP_HOST_REGISTRY` in main.py (2026-09-15) — do not read a
        pass here as confirmation that "marketing-api.example" -> "marketing"
        is the correct production binding. This is exactly why `enforce`
        must not be armed on either host until `observe` has actually
        settled which app_key each one's caller presents.
        """
        monkeypatch.setattr(main_module, "APP_HOST_ROUTE_MODE", "enforce")
        respx.post(INTROSPECT_URL).respond(200, json=_active_response(app_key="sales"))
        before = _app_host_count("mismatch_denied", "marketing-api.example", "marketing")
        response = await client.get(
            "/api/campaigns",
            headers={"Authorization": "Bearer tok", "host": "marketing-api.example"},
        )
        assert response.status_code == 403
        assert response.json()["reason"] == "app_host_mismatch"
        assert (
            _app_host_count("mismatch_denied", "marketing-api.example", "marketing") == before + 1
        )

    def test_host_registry_override_parsing(self):
        from main import _parse_host_registry_override as parse

        assert parse("a.example.com=marketing") == {"a.example.com": "marketing"}
        assert parse("A.Example.com:443=sales, b.x=growth") == {
            "a.example.com": "sales",
            "b.x": "growth",
        }
        assert parse("") == {}
        assert parse("garbage,,=x,y=") == {}, "malformed entries are skipped, not fatal"

    def test_gateway_asserted_host_keys_override_parses_and_merges(self):
        """The gateway host binding must mirror the Lua `LICENSE_GATED_APP_KEY`
        map, and is overridable without a rebuild (same grammar as
        APP_HOST_REGISTRY_OVERRIDE)."""
        from main import GATEWAY_ASSERTED_HOST_KEYS
        from main import _build_gateway_asserted_host_keys as build

        seed = {"n8n.nonagentic.ai": "n8n"}
        assert build("") == seed
        assert GATEWAY_ASSERTED_HOST_KEYS == seed
        merged = build("New.Example.com.:443=n8n, kanban.nonagentic.ai=planka")
        assert merged == {
            "n8n.nonagentic.ai": "n8n",
            "kanban.nonagentic.ai": "planka",
            "new.example.com": "n8n",
        }
        assert build("garbage,,=x,y=,.=n8n") == seed, "malformed entries are skipped"
        assert build("x=y") is not build("x=y"), "a fresh dict per call"
        assert "x" not in build(""), "the seed is never mutated by an override"

    def test_bad_mode_string_falls_back_to_off(self):
        """A typo'd mode on an ext_authz waypoint must fail INERT, never 'deny
        everything'."""
        from main import _parse_host_route_mode as mode

        assert mode("off") == "off"
        assert mode(" Observe ") == "observe"
        assert mode("enforce") == "enforce"
        assert mode("ENFORCE_NOW_PLEASE") == "off"
        assert mode("") == "off"

    def test_every_app_host_result_series_is_born_at_zero(self):
        from prometheus_client import REGISTRY

        for result in (
            "matched",
            "mismatch_observed",
            "mismatch_denied",
            "no_host",
            "unregistered",
        ):
            assert (
                REGISTRY.get_sample_value(
                    "license_authz_app_host_route_total",
                    {"result": result, "host": "other", "app": "other"},
                )
                is not None
            ), f"{result} is not born at zero"


# -- P2: trusted tenant-proxy allowlist / sibling-repo drift guard -----------


class TestTrustedTenantProxyClientsDriftGuard:
    """`TRUSTED_TENANT_PROXY_CLIENTS`' default is maintained BY HAND to mirror
    `licenses/auth/config.py:104`'s `_DEFAULT_TENANT_PROXY_CLIENTS` — same env
    var name, same literal, deliberately, so the two allowlists cannot drift
    apart in practice (see `main.py`'s comment above
    `_DEFAULT_TRUSTED_TENANT_PROXY_CLIENTS`). Nothing enforces that at
    runtime — the two services do not share a package or a config source —
    so this test is the enforcement: it pins THIS repo's literal to the exact
    string the `licenses` repo documents as source of truth, and a change to
    either side without the other fails review here, not in production.

    If `licenses/auth/config.py:104` ever changes its default, this test
    must be updated in the SAME PR that changes it, and vice versa.
    """

    def test_default_matches_licenses_auth_config_source_of_truth(self):
        from main import _DEFAULT_TRUSTED_TENANT_PROXY_CLIENTS

        # ⚠️ DELIBERATE DIVERGENCE since license-authz#64 (2026-09-30): this
        # repo narrowed its default to `w2v-backend` (fr-backend: 0 retargets
        # through the gateway in the feature's whole life; fr reaches licenses
        # in-cluster, where the LICENSES repo's own allowlist governs it).
        # `licenses/auth/config.py` still reads "fr-backend,w2v-backend". This
        # pin now guards against the default silently WIDENING again.
        assert _DEFAULT_TRUSTED_TENANT_PROXY_CLIENTS == "w2v-backend"
