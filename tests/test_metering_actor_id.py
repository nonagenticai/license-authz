"""Metering events carry `actor_id` — a UUID Keycloak `sub` — and licence-token
mode can vouch for one downstream as `x-actor-id`.

JWT mode: the verified token's `sub`. Licence-token mode: the introspected
licence subject's `subject_keycloak_sub`. A missing or non-UUID actor is OMITTED
from the event (never "") and counted on `license_authz_actor_missing_total`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan

TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"
KC_SUB = "7d1c0e2a-4b5f-4c6d-8e9f-0a1b2c3d4e5f"
FORWARDED_ACTOR = "11111111-2222-4333-8444-555555555555"


def _missing(reason: str) -> float:
    return REGISTRY.get_sample_value("license_authz_actor_missing_total", {"reason": reason}) or 0.0


def _introspect(**overrides) -> dict:
    body = {
        "active": True,
        "tenant_id": TENANT_UUID,
        "subject_id": "sub_123",
        "license_id": "lic_abc",
        "app": {"id": "app_workers", "key": "workers"},
        "plan": {"id": "premium", "code": "PRM"},
    }
    body.update(overrides)
    return body


async def _licence_request(introspect_body: dict, **headers: str):
    """One licence-token request; returns (response, XADD fields)."""
    mock_redis = AsyncMock(spec=redis.asyncio.Redis)
    mock_redis.xadd = AsyncMock()
    respx.post(INTROSPECT_URL).respond(200, json=introspect_body)
    async with lifespan(app):
        app.state.redis = mock_redis
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/api/resource",
                    headers={"Authorization": "Bearer valid_token", **headers},
                )
        finally:
            app.state.redis = None
    mock_redis.xadd.assert_awaited_once()
    return response, mock_redis.xadd.await_args[0][1]


class TestLicenceTokenActorOnMeteringEvent:
    @respx.mock
    async def test_subject_keycloak_sub_becomes_actor_id(self):
        """⛔ FAILS WHEN REVERTED: origin/main emitted no actor_id at all."""
        before = _missing("licence_no_subject_sub")
        response, fields = await _licence_request(_introspect(subject_keycloak_sub=KC_SUB))
        assert response.status_code == 200
        assert fields["actor_id"] == KC_SUB
        # The licences row id stays where it was — the two are not conflated.
        assert fields["subject_id"] == "sub_123"
        assert _missing("licence_no_subject_sub") == before

    @respx.mock
    async def test_uppercase_uuid_is_canonicalised(self):
        _, fields = await _licence_request(_introspect(subject_keycloak_sub=KC_SUB.upper()))
        assert fields["actor_id"] == KC_SUB

    @respx.mock
    async def test_absent_subject_sub_omits_field_and_counts(self):
        before = _missing("licence_no_subject_sub")
        response, fields = await _licence_request(_introspect())
        assert response.status_code == 200
        assert "actor_id" not in fields
        assert _missing("licence_no_subject_sub") == before + 1

    @respx.mock
    async def test_non_uuid_subject_sub_omits_field_and_counts(self):
        before = _missing("not_uuid")
        _, fields = await _licence_request(_introspect(subject_keycloak_sub="svc-not-a-uuid"))
        assert "actor_id" not in fields
        assert _missing("not_uuid") == before + 1


class TestLicenceTokenSubjectActorSub:
    """licenses#254: `subject_actor_sub` names the actor for a service subject whose
    `subject_keycloak_sub` is null (a `svc-*` label) — its own SA sub."""

    @respx.mock
    async def test_subject_actor_sub_becomes_actor_id_when_keycloak_sub_is_null(self):
        """⛔ FAILS WHEN REVERTED: origin/main read only subject_keycloak_sub."""
        before = _missing("licence_no_subject_sub")
        _, fields = await _licence_request(
            _introspect(subject_keycloak_sub=None, subject_actor_sub=FORWARDED_ACTOR)
        )
        assert fields["actor_id"] == FORWARDED_ACTOR
        assert _missing("licence_no_subject_sub") == before

    @respx.mock
    async def test_subject_actor_sub_is_preferred(self):
        _, fields = await _licence_request(
            _introspect(subject_keycloak_sub=KC_SUB, subject_actor_sub=KC_SUB)
        )
        assert fields["actor_id"] == KC_SUB

    @respx.mock
    async def test_older_licenses_without_the_field_still_uses_keycloak_sub(self):
        _, fields = await _licence_request(_introspect(subject_keycloak_sub=KC_SUB))
        assert fields["actor_id"] == KC_SUB

    @respx.mock
    async def test_neither_field_still_counts_licence_no_subject_sub(self):
        before = _missing("licence_no_subject_sub")
        _, fields = await _licence_request(
            _introspect(subject_keycloak_sub=None, subject_actor_sub=None)
        )
        assert "actor_id" not in fields
        assert _missing("licence_no_subject_sub") == before + 1


class TestLicenceTokenActorHeader:
    @respx.mock
    async def test_flag_off_injects_nothing(self, monkeypatch):
        """Default: no x-actor-id in licence-token mode (see the flag's comment)."""
        monkeypatch.setattr(main, "ACTOR_INJECT_LICENSE_TOKEN_MODE", False)
        response, _ = await _licence_request(_introspect(subject_keycloak_sub=KC_SUB))
        assert "x-actor-id" not in response.headers

    @respx.mock
    async def test_flag_on_injects_licence_subject_when_request_has_none(self, monkeypatch):
        monkeypatch.setattr(main, "ACTOR_INJECT_LICENSE_TOKEN_MODE", True)
        response, _ = await _licence_request(_introspect(subject_keycloak_sub=KC_SUB))
        assert response.headers.get("x-actor-id") == KC_SUB

    @respx.mock
    async def test_flag_on_never_overwrites_a_forwarded_actor(self, monkeypatch):
        """A service forwarding the human it acts for keeps that human."""
        monkeypatch.setattr(main, "ACTOR_INJECT_LICENSE_TOKEN_MODE", True)
        response, _ = await _licence_request(
            _introspect(subject_keycloak_sub=KC_SUB), **{"x-actor-id": FORWARDED_ACTOR}
        )
        assert "x-actor-id" not in response.headers

    @respx.mock
    async def test_flag_on_non_uuid_injects_nothing(self, monkeypatch):
        monkeypatch.setattr(main, "ACTOR_INJECT_LICENSE_TOKEN_MODE", True)
        response, _ = await _licence_request(_introspect(subject_keycloak_sub="svc-not-a-uuid"))
        assert "x-actor-id" not in response.headers


class TestJwtModeActorOnMeteringEvent:
    @pytest.mark.parametrize(
        "sub,expect_actor,missing_reason",
        [
            (KC_SUB, KC_SUB, None),
            ("kc-sub-raw", None, "not_uuid"),
            (None, None, "jwt_no_sub"),
        ],
    )
    async def test_verified_sub_becomes_actor_id(
        self, monkeypatch, sub, expect_actor, missing_reason
    ):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": sub or ""})
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub=sub, raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "subject_id": "subject-uuid-from-licenses",
                "license_id": "lic_abc",
                "app": {"id": "app_workers", "key": "workers"},
                "plan": {"id": "premium", "code": "PRM"},
            }
        )
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        before = _missing(missing_reason) if missing_reason else None

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            app.state.redis = mock_redis
            try:
                with patch("main.verify_jwt", return_value=claims):
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as ac:
                        resp = await ac.get(
                            "/workers/api/tasks",
                            headers={"Authorization": f"Bearer {token}", "x-app-key": "workers"},
                        )
            finally:
                app.state.redis = None

        assert resp.status_code == 200
        mock_redis.xadd.assert_awaited_once()
        fields = mock_redis.xadd.await_args[0][1]
        if expect_actor:
            assert fields["actor_id"] == expect_actor
        else:
            assert "actor_id" not in fields
            assert _missing(missing_reason) == before + 1


def test_every_reason_is_born_at_zero():
    for reason in ("no_identity", "jwt_no_sub", "licence_no_subject_sub", "not_uuid"):
        assert (
            REGISTRY.get_sample_value("license_authz_actor_missing_total", {"reason": reason})
            is not None
        )
