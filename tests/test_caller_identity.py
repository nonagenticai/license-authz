"""Tests for caller-identity (X-Caller-Authorization) extraction in ext_authz_check.

Header parsing + secure-apps SA JWT verification + injection of x-caller-client-id
+ x-caller-roles upstream headers. Opt-in semantics during M1–M7 rollout: absent
header = empty injected values; present-but-invalid = 401 fail-loud.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
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
    """Client with caller_jwks stubbed in app.state — used by tests that pass an
    X-Caller-Authorization header. Tests that omit the header don't need this.

    Lifespan starts with secure_apps_issuer empty (so it skips real JWKS fetch),
    then we inject the stub JwksCache + the test issuer so verify_jwt is called
    with a deterministic `issuer=` arg.
    """
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


# -- Opt-in: absent header path ---------------------------------------------


@respx.mock
async def test_absent_caller_header_injects_empty_values(client):
    """No X-Caller-Authorization → 200 with empty x-caller-* (opt-in semantics)."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    resp = await client.get("/api/x", headers={"Authorization": "Bearer license-tok"})
    assert resp.status_code == 200
    # Empty injected headers — Envoy will still forward them upstream as empty.
    assert resp.headers.get("x-caller-client-id", "") == ""
    assert resp.headers.get("x-caller-roles", "") == ""


# -- Valid SA JWT happy path ------------------------------------------------


@respx.mock
async def test_valid_caller_jwt_injects_client_id_and_roles(client_with_caller_jwks):
    """Valid SA JWT → azp + comma-joined roles injected upstream."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    claims = VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={
            "azp": "worker-management-sa",
            "realm_access": {"roles": ["secrets-reader", "secrets-writer"]},
        },
    )
    with patch("main.verify_jwt", return_value=claims) as v:
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer sa-jwt",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["x-caller-client-id"] == "worker-management-sa"
    assert resp.headers["x-caller-roles"] == "secrets-reader,secrets-writer"
    # Verify we used the secure-apps issuer.
    kwargs = v.call_args.kwargs
    assert kwargs["issuer"] == "https://auth.nonagentic.ai/realms/secure-apps"
    assert kwargs["audience"] is None


# -- Invalid signature → 401 ------------------------------------------------


@respx.mock
async def test_invalid_caller_signature_returns_401(client_with_caller_jwks):
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    with patch("main.verify_jwt", side_effect=InvalidJWT("bad sig")):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer bad-jwt",
            },
        )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "caller_token_invalid"
    assert resp.headers["x-error-reason"] == "caller_token_invalid"


# -- Wrong issuer → 401 (verify_jwt raises InvalidJWT for issuer mismatch) ---


@respx.mock
async def test_wrong_issuer_returns_401(client_with_caller_jwks):
    """verify_jwt raises InvalidJWT for issuer mismatch — same path as bad sig."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    with patch("main.verify_jwt", side_effect=InvalidJWT("wrong issuer")):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer wrong-iss-jwt",
            },
        )
    assert resp.status_code == 401
    assert resp.headers["x-error-reason"] == "caller_token_invalid"


# -- Expired token → 401 ----------------------------------------------------


@respx.mock
async def test_expired_caller_jwt_returns_401(client_with_caller_jwks):
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    with patch("main.verify_jwt", side_effect=InvalidJWT("expired")):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer expired",
            },
        )
    assert resp.status_code == 401
    assert resp.headers["x-error-reason"] == "caller_token_invalid"


# -- realm_access.roles missing → 200 with empty roles header ---------------


@respx.mock
async def test_missing_realm_access_roles_returns_200_with_empty_roles(
    client_with_caller_jwks,
):
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    claims = VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={"azp": "some-sa"},
    )
    with patch("main.verify_jwt", return_value=claims):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer ok",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["x-caller-client-id"] == "some-sa"
    assert resp.headers.get("x-caller-roles", "") == ""


# -- Unsafe roles content → 403 unsafe_header_value -------------------------


@respx.mock
async def test_unsafe_role_chars_rejected_403(client_with_caller_jwks):
    """Roles with chars outside [\\w\\-\\.,] → reject (existing safety guard)."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    claims = VerifiedClaims(
        tenant_id="ignored",
        sub="sa-uuid",
        raw={
            "azp": "some-sa",
            # Newline / space → fails _SAFE_LIST_HEADER_RE.
            "realm_access": {"roles": ["good-role", "bad role with space"]},
        },
    )
    with patch("main.verify_jwt", return_value=claims):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "Bearer ok",
            },
        )
    assert resp.status_code == 403
    assert resp.json()["reason"] == "unsafe_header_value"


# -- Bearer prefix missing on caller header → 401 ---------------------------


@respx.mock
async def test_caller_header_without_bearer_returns_401(client_with_caller_jwks):
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    resp = await client_with_caller_jwks.get(
        "/api/x",
        headers={
            "Authorization": "Bearer license-tok",
            "X-Caller-Authorization": "raw-token-no-bearer",
        },
    )
    assert resp.status_code == 401
    assert resp.headers["x-error-reason"] == "caller_token_invalid"


# -- Header set but caller_jwks not configured → 401 ------------------------


@respx.mock
async def test_caller_header_with_no_jwks_configured_returns_401(monkeypatch):
    """Service has no secure-apps JWKS but caller sent a token → refuse, never silent-pass."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.caller_jwks = None
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/x",
                headers={
                    "Authorization": "Bearer license-tok",
                    "X-Caller-Authorization": "Bearer something",
                },
            )
    assert resp.status_code == 401
    assert resp.headers["x-error-reason"] == "caller_identity_not_configured"


# -- Every caller-identity 401 must leave an audit row ----------------------
#
# ⭐ All three 401 exits in `_extract_caller_identity` (non-bearer header, no
# secure-apps JWKS configured, token fails signature/issuer/expiry) returned a
# bare 401 with NO `log_access` row, NO `logger.warning` and NO Counter. A
# FORGED service-account JWT presented at the authz boundary was therefore
# answered correctly and recorded NOWHERE — the single most interesting event
# this service can see, and it was silent. `log_access` is the only structured
# audit emitter here, so anything keyed on `status` could not see these denies
# at all.
#
# The row is emitted at the ONE call site in `ext_authz_check`, so it covers all
# three exits and any fourth added later. These tests pin two of the three at
# the ASGI boundary, and the distinct `reason` proves the row is built from the
# response the helper actually returned rather than from a constant.


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


@respx.mock
async def test_caller_token_invalid_writes_an_audit_row(client_with_caller_jwks, caplog):
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    with caplog.at_level(logging.INFO, logger="main"):
        resp = await client_with_caller_jwks.get(
            "/api/x",
            headers={
                "Authorization": "Bearer license-tok",
                "X-Caller-Authorization": "raw-token-no-bearer",
            },
        )

    # Behaviour unchanged.
    assert resp.status_code == 401
    assert resp.headers["x-error-reason"] == "caller_token_invalid"

    rows = [r for r in _audit_rows(caplog) if r["status"] == "denied_caller_identity"]
    assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
    row = rows[0]
    assert row["reason"] == "caller_identity=caller_token_invalid"
    assert row["path"] == "/api/x"
    assert row["method"] == "GET"
    # Nothing is introspected before this exit, so no licence identity exists
    # and none is invented. `None`, not "" — "" would read as "looked it up and
    # it was blank".
    assert row["license_id"] is None
    assert row["tenant_id"] is None
    assert row["app_id"] is None
    assert row["plan_id"] is None
    # There is no licence token on this path to hash; the file's sentinel.
    assert row["token_hash"] == "caller_token"
    # ⛔ And the request must NOT also have been audited as allowed.
    assert not [r for r in _audit_rows(caplog) if r["status"] == "allowed"]


@respx.mock
async def test_caller_identity_not_configured_writes_a_distinct_audit_row(monkeypatch, caplog):
    """The second exit. A DIFFERENT `reason` is what proves the row is derived
    from the helper's own response; a hardcoded string would pass the test
    above and die here."""
    respx.post(INTROSPECT_URL).respond(200, json=_active())
    transport = ASGITransport(app=app)
    async with lifespan(app):
        app.state.caller_jwks = None
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            with caplog.at_level(logging.INFO, logger="main"):
                resp = await ac.get(
                    "/api/x",
                    headers={
                        "Authorization": "Bearer license-tok",
                        "X-Caller-Authorization": "Bearer something",
                    },
                )

    assert resp.status_code == 401
    rows = [r for r in _audit_rows(caplog) if r["status"] == "denied_caller_identity"]
    assert len(rows) == 1, f"expected exactly one audit row, got {_audit_rows(caplog)}"
    assert rows[0]["reason"] == "caller_identity=caller_identity_not_configured"
