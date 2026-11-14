"""Integration tests for JWT-mode and metering-service HTTP fallback in ext_authz_check."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis.asyncio
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan

# Canonical tenant UUID — both token-mode and JWT-mode now carry the tenant
# identity as a UUID; license-authz injects it directly and rejects non-UUIDs.
TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"
TENANT_UUID_2 = "0192a3b4-c5d6-7e8f-9012-aaaaaaaaaaaa"


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


# -- JWT-mode integration ----------------------------------------------------


class TestJwtModeIntegration:
    """Tests for the _handle_jwt_mode branch through ext_authz_check."""

    async def test_jwt_mode_not_initialized_returns_503(self, monkeypatch):
        """JWT-mode flag enabled but app.state.jwks not set up → 503."""
        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)

        # Build a JWT-shaped token recognized by is_jwt_bearer.
        from jwt_mode import KEYCLOAK_ISSUER
        from tests.test_jwt_mode import _make_jwt_like

        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        async with lifespan(app):
            # Force jwks/resolver to None to simulate flag-flipped-without-restart
            app.state.jwks = None
            app.state.active_resolver = None
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    "/api/anything",
                    headers={"Authorization": f"Bearer {token}"},
                )
        assert resp.status_code == 503
        assert resp.json()["error"] == "jwt_mode_not_initialized"
        assert resp.headers.get("x-error-reason") == "jwt_mode_not_initialized"

    # -- Every JWT-mode error exit must leave an audit row -------------------
    #
    # `_handle_jwt_mode` has six error exits (jwt_mode_not_initialized,
    # invalid_token, anonymous, missing_app_key, licenses_unavailable,
    # license_required) and not one wrote a `log_access` row. They are less
    # blind than the caller-identity exit — `_jwt_error` increments
    # `metric_jwt_mode` — but a Counter is not an audit trail: it carries no
    # path, no source_ip, no token_hash, and cannot be joined to a request.
    # The row is emitted at the single call site, covering all six.

    @staticmethod
    def _audit_rows(caplog) -> list[dict]:
        rows = []
        for record in caplog.records:
            try:
                entry = json.loads(record.getMessage())
            except ValueError, TypeError:
                continue
            if isinstance(entry, dict) and "status" in entry and "source_ip" in entry:
                rows.append(entry)
        return rows

    async def test_jwt_mode_not_initialized_writes_an_audit_row(self, monkeypatch, caplog):
        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        from jwt_mode import KEYCLOAK_ISSUER
        from tests.test_jwt_mode import _make_jwt_like

        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})
        async with lifespan(app):
            app.state.jwks = None
            app.state.active_resolver = None
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with caplog.at_level(logging.INFO, logger="main"):
                    resp = await ac.get(
                        "/api/anything",
                        headers={"Authorization": f"Bearer {token}"},
                    )

        # Behaviour unchanged.
        assert resp.status_code == 503

        rows = [r for r in self._audit_rows(caplog) if r["status"] == "denied_jwt_mode"]
        assert len(rows) == 1, f"expected exactly one audit row, got {self._audit_rows(caplog)}"
        row = rows[0]
        assert row["reason"] == "jwt_mode=jwt_mode_not_initialized"
        assert row["path"] == "/api/anything"
        assert row["method"] == "GET"
        # Every error exit here IS a failure to resolve a licence, so there is
        # none to name. `None`, not a guess.
        assert row["license_id"] is None
        assert row["tenant_id"] is None
        assert row["app_id"] is None
        assert row["plan_id"] is None
        # Unlike the caller-identity row, the bearer token is real and hashed.
        assert row["token_hash"].startswith("sha256:")
        assert not [r for r in self._audit_rows(caplog) if r["status"] == "allowed"]

    async def test_jwt_mode_invalid_token_writes_a_distinct_audit_row(self, monkeypatch, caplog):
        """A DIFFERENT `reason` at a second exit is what proves the row is built
        from the response the helper returned; a hardcoded string passes the
        test above and dies here."""
        from jwt_mode import KEYCLOAK_ISSUER, InvalidJWT
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = MagicMock()
            with patch("main.verify_jwt", side_effect=InvalidJWT("bad sig")):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    with caplog.at_level(logging.INFO, logger="main"):
                        resp = await ac.get(
                            "/api/anything",
                            headers={"Authorization": f"Bearer {token}"},
                        )

        assert resp.status_code == 401
        rows = [r for r in self._audit_rows(caplog) if r["status"] == "denied_jwt_mode"]
        assert len(rows) == 1, f"expected exactly one audit row, got {self._audit_rows(caplog)}"
        assert rows[0]["reason"] == "jwt_mode=invalid_token"

    async def test_jwt_mode_invalid_token_returns_401(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, InvalidJWT
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = MagicMock()
            with patch("main.verify_jwt", side_effect=InvalidJWT("bad sig")):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={"Authorization": f"Bearer {token}"},
                    )
        assert resp.status_code == 401
        assert resp.json()["error"] == "invalid_token"

    async def test_jwt_mode_anonymous_returns_401(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="anonymous", sub=None, raw={})

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = MagicMock()
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 401
        assert resp.json()["error"] == "anonymous"

    async def test_jwt_mode_missing_app_key_returns_400(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="tenant-x", sub="u", raw={})

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = MagicMock()
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={"Authorization": f"Bearer {token}"},
                    )
        assert resp.status_code == 400
        assert resp.json()["error"] == "missing_app_key"

    # -- App key DERIVED from the route registry ----------------------------
    #
    # A caller that reaches a service through ingress with only `Authorization`
    # + `X-Tenant-Id` — no `x-app-key` — used to die at the 400 above, BEFORE
    # any route check, even on a path whose registered prefix already NAMES the
    # app. Where the registry names the key, it is derived instead. These tests
    # pin both directions of that: a registered prefix opens, and nothing else
    # does.

    async def test_jwt_mode_derives_app_key_on_registered_route(self, monkeypatch):
        """No `x-app-key` on a REGISTERED prefix → key derived from the registry
        and the request continues to the normal entitlement check (200)."""
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-coding",
                "app": {"id": "app-coding", "key": "coding"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/coding/api/graph",
                        headers={"Authorization": f"Bearer {token}"},
                    )

        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-app-key") == "coding"
        # The derived key is what the licence was actually looked up by — a
        # derivation that did not reach the resolver would leave this unset and
        # the test would still 200 on a lenient route check.
        assert resolver.resolve.await_args.kwargs["app_key"] == "coding"

    @pytest.mark.parametrize("path", ["/tts/api/tts/synthesize", "/asr/api/asr/transcribe"])
    async def test_jwt_mode_speech_routes_derive_w2v_during_transition(self, monkeypatch, path):
        """/tts/ and /asr/ accept ("w2v", "speech") (owner decision 2026-10-22).
        A JWT caller with NO `x-app-key` must keep today's behaviour: the
        derived key is the primary, `w2v` — not `speech` — until Studio migrates."""
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-w2v",
                "app": {"id": "app-w2v", "key": "w2v"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(path, headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200, resp.text
        assert resolver.resolve.await_args.kwargs["app_key"] == "w2v"

    async def test_jwt_mode_asserted_key_contradicting_registry_still_403(self, monkeypatch):
        """Derivation must not launder a WRONG asserted key into a right one.
        An asserted key on a registered prefix that demands another is still a
        403 `app_route_mismatch`."""
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-w2v",
                "app": {"id": "app-w2v", "key": "w2v"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/workers/api/tasks",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "w2v",
                        },
                    )

        assert resp.status_code == 403
        assert resp.json()["reason"] == "app_route_mismatch"
        # The asserted key was used verbatim — NOT replaced by the route's.
        assert resolver.resolve.await_args.kwargs["app_key"] == "w2v"

    async def test_jwt_mode_unregistered_path_without_app_key_still_400(self, monkeypatch):
        """An UNREGISTERED path derives nothing, so the original 400 stands.
        This is the test that fails if derivation ever invents a default."""
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/anything",
                        headers={"Authorization": f"Bearer {token}"},
                    )

        assert resp.status_code == 400
        assert resp.json()["error"] == "missing_app_key"
        # Refused BEFORE any licence lookup — no resolver call was made.
        resolver.resolve.assert_not_awaited()

    async def test_jwt_mode_w2v_key_on_licenses_prefix_still_403(self, monkeypatch):
        """A foreign (`w2v`) key must not open the `/licenses/` prefix.

        ⚠️ NARROWED DELIBERATELY, from GET to POST. This arrived asserting that
        a `w2v` key GETting `/licenses/active` is 403. The self-read exemption
        makes that GET allowed on purpose — it is exactly the call the bare-metal
        w2v hosts make, and its 403 is what took narration down on 2026-09-14.

        The claim this test exists to defend is that the `/licenses/` prefix is
        not widened by a foreign key, and a MUTATING method proves that without
        colliding with the exemption. The GET case is asserted in its allowed
        form at the end, so the two facts live together and cannot drift.
        """
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-w2v",
                "app": {"id": "app-w2v", "key": "w2v"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/licenses/active",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "w2v",
                        },
                    )
                    allowed = await ac.get(
                        "/licenses/active",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "w2v",
                        },
                    )

        assert resp.status_code == 403
        assert resp.json()["reason"] == "app_route_mismatch"
        # The other half of the same fact: the self-read exemption is what the
        # bare-metal hosts depend on, and it is method-scoped, not a hole.
        assert allowed.status_code == 200

    async def test_jwt_mode_license_required_returns_402(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="tenant-x", sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 402
        body = resp.json()
        assert body["error"] == "license_required"
        assert body["tenant_id"] == "tenant-x"
        assert body["app_key"] == "workers"

    async def test_jwt_mode_licenses_unavailable_returns_503(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="tenant-x", sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(side_effect=httpx.ConnectError("down"))

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 503
        assert resp.json()["error"] == "licenses_unavailable"

    async def test_jwt_mode_success_falls_through_to_200(self, monkeypatch):
        """Valid JWT + resolved license → falls into normal allow path → 200 with headers."""
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-1",
                "app": {"id": "app-w", "key": "workers"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/workers/api/tasks",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 200
        assert resp.headers.get("x-tenant-id") == TENANT_UUID
        assert resp.headers.get("x-tenant-name") == ""
        assert resp.headers.get("x-license-id") == "lic-1"
        assert resp.headers.get("x-app-key") == "workers"

    async def test_jwt_mode_propagates_subject_id_from_active_license(self, monkeypatch):
        """JWT-mode resolves subject_id via /licenses/active response, not the JWT.

        The licenses service returns the License row's subject_id (FK on
        Subject). license-authz threads that into the x-subject-id upstream
        header and into the metering event payload — JWT-mode and opaque-mode
        end up identical to downstream consumers.
        """
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "kc-sub-raw"})

        # extract_tenant_id discards `sub` and returns the tenant_id claim;
        # subject_id is meant to come from the licenses lookup, not the JWT.
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="kc-sub-raw", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "subject_id": "subject-uuid-from-licenses",
                "license_id": "lic-1",
                "app": {"id": "app-w", "key": "workers"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()

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
                            headers={
                                "Authorization": f"Bearer {token}",
                                "x-app-key": "workers",
                            },
                        )
                assert resp.status_code == 200
                assert resp.headers.get("x-subject-id") == "subject-uuid-from-licenses"

                # Metering payload must carry the same subject_id, not the
                # raw keycloak `sub` claim.
                mock_redis.xadd.assert_awaited()
                call_args = mock_redis.xadd.await_args
                assert call_args is not None
                fields = call_args[0][1]
                assert fields["subject_id"] == "subject-uuid-from-licenses"
                assert fields["tenant_id"] == TENANT_UUID
            finally:
                app.state.redis = None

    async def test_jwt_mode_emits_x_actor_id_from_verified_sub(self, monkeypatch):
        """JWT mode emits X-Actor-Id = the verified token's `sub`, and it is a
        DIFFERENT value from x-subject-id (the licence holder) — this is the
        acting human, not the billable party.
        """
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "kc-sub-raw"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="kc-sub-raw", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=_active(subject_id="subject-uuid-from-licenses"))

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/workers/api/tasks",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 200
        assert resp.headers.get("x-actor-id") == "kc-sub-raw"
        assert resp.headers.get("x-subject-id") == "subject-uuid-from-licenses"
        assert resp.headers.get("x-actor-id") != resp.headers.get("x-subject-id")

    async def test_jwt_mode_no_sub_omits_x_actor_id_entirely(self, monkeypatch):
        """A verified token with no `sub` claim must OMIT X-Actor-Id — not send
        an empty string, not fall back to anything else.
        """
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER})

        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub=None, raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=_active())

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/workers/api/tasks",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 200
        assert "x-actor-id" not in resp.headers


def _jwt_mode_counter(result: str):
    return REGISTRY.get_sample_value(
        "license_authz_jwt_mode_total",
        {"result": result},
    )


def _jwt_mode_by_tenant_counter(result: str, tenant: str, app: str):
    return REGISTRY.get_sample_value(
        "license_authz_jwt_mode_by_tenant_total",
        {"result": result, "tenant": tenant, "app": app},
    )


class TestJwtModeByTenantCounter:
    """The license_authz_jwt_mode_by_tenant_total counter is populated only for
    paths where (tenant, app) are both known.
    """

    async def test_license_required_increments_both_counters(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="tenant-counter-1", sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)

        coarse_before = _jwt_mode_counter("license_required") or 0.0
        tenant_before = (
            _jwt_mode_by_tenant_counter("license_required", "tenant-counter-1", "workers") or 0.0
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 402

        coarse_after = _jwt_mode_counter("license_required") or 0.0
        tenant_after = (
            _jwt_mode_by_tenant_counter("license_required", "tenant-counter-1", "workers") or 0.0
        )
        assert coarse_after == coarse_before + 1.0
        assert tenant_after == tenant_before + 1.0

    async def test_allowed_increments_both_counters(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id=TENANT_UUID_2, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID_2,
                "license_id": "lic-2",
                "app": {"id": "app-w", "key": "workers"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )

        coarse_before = _jwt_mode_counter("allowed") or 0.0
        tenant_before = _jwt_mode_by_tenant_counter("allowed", TENANT_UUID_2, "workers") or 0.0

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/workers/api/tasks",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": "workers",
                        },
                    )
        assert resp.status_code == 200

        coarse_after = _jwt_mode_counter("allowed") or 0.0
        tenant_after = _jwt_mode_by_tenant_counter("allowed", TENANT_UUID_2, "workers") or 0.0
        assert coarse_after == coarse_before + 1.0
        assert tenant_after == tenant_before + 1.0

    async def test_missing_app_key_does_not_increment_tenant_counter(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})

        claims = VerifiedClaims(tenant_id="tenant-counter-3", sub="u", raw={})

        coarse_before = _jwt_mode_counter("missing_app_key") or 0.0
        # Sample any prospective tenant series; should remain absent/unchanged.
        tenant_before = (
            _jwt_mode_by_tenant_counter("missing_app_key", "tenant-counter-3", "") or 0.0
        )

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = MagicMock()
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/api/x",
                        headers={"Authorization": f"Bearer {token}"},
                    )
        assert resp.status_code == 400

        coarse_after = _jwt_mode_counter("missing_app_key") or 0.0
        tenant_after = _jwt_mode_by_tenant_counter("missing_app_key", "tenant-counter-3", "") or 0.0
        assert coarse_after == coarse_before + 1.0
        assert tenant_after == tenant_before  # unchanged


# -- ASSERTED app key on an UNREGISTERED route --------------------------------
#
# THE DEFECT: on the JWT path the app key comes from the caller's UNSIGNED
# `x-app-key` header, and that header is what `/licenses/active` is queried by —
# so the caller picks which of its tenant's licences funds the call. The verified
# JWT establishes the TENANT (which is why this is a mis-charging defect, not a
# cross-tenant one) but NOT the APP: no claim in either realm names one.
#
# The only binding available for an asserted key is therefore the ROUTE. A
# REGISTERED prefix supplies it. An UNREGISTERED path supplies nothing, and used
# to `return True` for ANY key — a free choice of licence. That arm is now
# refused.
#
# ⛔ The tests that matter most here are the NEGATIVE CONTROLS: the ATTESTED
# (introspected) key population is untouched, and a registered prefix still
# allows. If those go red, this is not a narrowing — it is a new denial class
# over live traffic.


def _route_sample(result: str) -> float:
    return REGISTRY.get_sample_value("license_authz_app_route_total", {"result": result}) or 0.0


def _skipped_sample(app_key: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "license_authz_metering_events_skipped_total",
            {"reason": "gateway_browser_ui", "app": app_key},
        )
        or 0.0
    )


class TestAssertedAppKeyUnregisteredRoute:
    """JWT path (ASSERTED key) vs licence-token path (ATTESTED key)."""

    def _jwt_env(self, monkeypatch):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})
        resolver = MagicMock()
        resolver.resolve = AsyncMock(
            return_value={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": "lic-1",
                "app": {"id": "app-w", "key": "workers"},
                "plan": {"id": "p1", "code": "PRM"},
            }
        )
        return token, claims, resolver

    async def _jwt_request(self, monkeypatch, path: str, app_key: str = "workers"):
        token, claims, resolver = self._jwt_env(monkeypatch)
        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    return await ac.get(
                        path,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-app-key": app_key,
                        },
                    )

    async def test_asserted_key_on_an_unregistered_route_is_denied(self, monkeypatch):
        """⛔ FAILS WHEN REVERTED: origin/main returned 200 here.

        `/api/x` is in no registry prefix, so nothing binds the caller's chosen
        `x-app-key` to anything — the caller had a free choice of which licence
        to charge.
        """
        resp = await self._jwt_request(monkeypatch, "/api/x")
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"] == "forbidden"
        assert body["reason"] == "app_route_unregistered_asserted_key"
        assert resp.headers.get("x-error-reason") == "app_route_unregistered_asserted_key"

    async def test_asserted_key_on_a_registered_route_is_still_allowed(self, monkeypatch):
        """NEGATIVE CONTROL: the route supplies the binding, so the key is usable.

        If this reddens, the change is not "an asserted key needs a binding", it
        is "the JWT path is dead".
        """
        resp = await self._jwt_request(monkeypatch, "/workers/api/tasks")
        assert resp.status_code == 200
        assert resp.headers.get("x-app-key") == "workers"

    @respx.mock
    async def test_attested_key_on_an_unregistered_route_is_still_allowed(self):
        """NEGATIVE CONTROL, AND THE IMPORTANT ONE: the ATTESTED path is untouched.

        Here the key is returned BY `/license/introspect` as part of the
        credential's contents — the caller chose nothing — so there is no defect
        to close and denying it would be a brand-new denial class over the entire
        live licence-token population (which is 100% of real traffic:
        `license_authz_jwt_mode_total` measured 0 over 24h).
        """
        respx.post(INTROSPECT_URL).respond(200, json=_active())
        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/resource", headers={"Authorization": "Bearer tok"})
        assert resp.status_code == 200, "the attested-key path was narrowed — it must not be"

    async def test_the_revert_flag_restores_the_permissive_behaviour(self, monkeypatch):
        """The env var is the REVERT, not the arming switch. Setting it must give
        back the previous behaviour exactly."""
        monkeypatch.setattr(main, "ASSERTED_APP_KEY_ALLOW_UNREGISTERED", True)
        resp = await self._jwt_request(monkeypatch, "/api/x")
        assert resp.status_code == 200

    async def test_the_deny_is_counted(self, monkeypatch):
        before = _route_sample("denied_asserted_key_unregistered")
        resp = await self._jwt_request(monkeypatch, "/api/x")
        assert resp.status_code == 403
        assert _route_sample("denied_asserted_key_unregistered") == before + 1

    async def test_the_revert_is_counted_too_so_the_census_does_not_go_dark(self, monkeypatch):
        """⭐ THE POINT OF THE SECOND LABEL. If only the deny were counted, an
        operator who reverted to permissive would see the series fall to zero and
        read "the defect stopped happening" — when what stopped was the
        measurement, at exactly the moment it mattered most.

        The allow must also NOT land on the generic `allowed` series, or the
        reverted population becomes invisible inside ordinary traffic.
        """
        monkeypatch.setattr(main, "ASSERTED_APP_KEY_ALLOW_UNREGISTERED", True)
        before_revert = _route_sample("allowed_asserted_key_unregistered")
        before_allowed = _route_sample("allowed")

        resp = await self._jwt_request(monkeypatch, "/api/x")

        assert resp.status_code == 200
        assert _route_sample("allowed_asserted_key_unregistered") == before_revert + 1
        assert _route_sample("allowed") == before_allowed, (
            "the reverted allow was folded into the generic series and is now invisible"
        )

    async def test_a_normal_allow_still_counts_exactly_once_as_allowed(self, monkeypatch):
        """Guard against the restructuring double-counting: the registered-prefix
        allow must move `allowed` by exactly 1 and move neither new series."""
        before_allowed = _route_sample("allowed")
        before_denied = _route_sample("denied_asserted_key_unregistered")
        before_revert = _route_sample("allowed_asserted_key_unregistered")

        resp = await self._jwt_request(monkeypatch, "/workers/api/tasks")

        assert resp.status_code == 200
        assert _route_sample("allowed") == before_allowed + 1
        assert _route_sample("denied_asserted_key_unregistered") == before_denied
        assert _route_sample("allowed_asserted_key_unregistered") == before_revert


class TestAssertedAppKeyGatewayHostBinding:
    """The gateway's own key on an n8n/kanban UI path (unregistered) must pass.

    THE OUTAGE (prod, 2026-10-05): the ingress Lua sets `x-app-key` from a static
    host map (`LICENSE_GATED_APP_KEY`) after stripping the client's copy, and sets
    `x-forwarded-host`. `/workflow/<id>` is in no registry prefix, so the
    asserted-key deny refused every licensed browser navigation to n8n.
    """

    async def _req(self, monkeypatch, path, app_key, forwarded_host=None):
        helper = TestAssertedAppKeyUnregisteredRoute()
        token, claims, resolver = helper._jwt_env(monkeypatch)
        headers = {"Authorization": f"Bearer {token}", "x-app-key": app_key}
        if forwarded_host is not None:
            headers["x-forwarded-host"] = forwarded_host
        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    return await ac.get(path, headers=headers)

    def _assert_unregistered_deny(self, resp):
        assert resp.status_code == 403
        assert resp.json()["reason"] == "app_route_unregistered_asserted_key"

    async def test_n8n_host_with_n8n_key_on_ui_path_is_allowed_and_counted(self, monkeypatch):
        """⛔ FAILS WHEN REVERTED: origin/main returned 403 here."""
        before_bound = _route_sample("allowed_asserted_key_host_bound")
        before_allowed = _route_sample("allowed")
        before_denied = _route_sample("denied_asserted_key_unregistered")
        resp = await self._req(monkeypatch, "/workflow/abc", "n8n", "n8n.nonagentic.ai")
        assert resp.status_code == 200
        assert _route_sample("allowed_asserted_key_host_bound") == before_bound + 1
        assert _route_sample("allowed") == before_allowed, "folded into generic `allowed`"
        assert _route_sample("denied_asserted_key_unregistered") == before_denied

    async def test_kanban_host_with_kanban_key_is_allowed(self, monkeypatch):
        before = _route_sample("allowed_asserted_key_host_bound")
        resp = await self._req(monkeypatch, "/boards/123", "kanban", "kanban.nonagentic.ai")
        assert resp.status_code == 200
        assert _route_sample("allowed_asserted_key_host_bound") == before + 1

    async def test_n8n_host_with_a_different_key_is_still_denied(self, monkeypatch):
        """The binding is host -> EXACTLY one key, not "any key on a known host"."""
        before_bound = _route_sample("allowed_asserted_key_host_bound")
        before_denied = _route_sample("denied_asserted_key_unregistered")
        resp = await self._req(monkeypatch, "/workflow/abc", "kanban", "n8n.nonagentic.ai")
        self._assert_unregistered_deny(resp)
        assert _route_sample("allowed_asserted_key_host_bound") == before_bound
        assert _route_sample("denied_asserted_key_unregistered") == before_denied + 1

    async def test_unknown_host_with_n8n_key_is_denied(self, monkeypatch):
        resp = await self._req(monkeypatch, "/workflow/abc", "n8n", "evil.example.com")
        self._assert_unregistered_deny(resp)

    async def test_missing_forwarded_host_is_denied(self, monkeypatch):
        """No `:authority`/`Host` fallback: those name license-authz itself."""
        resp = await self._req(monkeypatch, "/workflow/abc", "n8n")
        self._assert_unregistered_deny(resp)

    async def test_host_header_alone_does_not_bind(self, monkeypatch):
        helper = TestAssertedAppKeyUnregisteredRoute()
        token, claims, resolver = helper._jwt_env(monkeypatch)
        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://n8n.nonagentic.ai"
                ) as ac:
                    resp = await ac.get(
                        "/workflow/abc",
                        headers={"Authorization": f"Bearer {token}", "x-app-key": "n8n"},
                    )
        self._assert_unregistered_deny(resp)

    @pytest.mark.parametrize(
        "forwarded_host",
        [
            "N8N.NonAgentic.AI",
            "n8n.nonagentic.ai:443",
            "n8n.nonagentic.ai.",
            "N8N.nonagentic.ai.:443",
        ],
    )
    async def test_host_normalises(self, monkeypatch, forwarded_host):
        resp = await self._req(monkeypatch, "/workflow/abc", "n8n", forwarded_host)
        assert resp.status_code == 200

    async def test_revert_flag_still_counts_its_own_label_for_unbound_requests(self, monkeypatch):
        """The flag arm is untouched for a request the host does not bind."""
        monkeypatch.setattr(main, "ASSERTED_APP_KEY_ALLOW_UNREGISTERED", True)
        before = _route_sample("allowed_asserted_key_unregistered")
        resp = await self._req(monkeypatch, "/workflow/abc", "kanban", "n8n.nonagentic.ai")
        assert resp.status_code == 200
        assert _route_sample("allowed_asserted_key_unregistered") == before + 1

    # -- metering: editor UI traffic is audited, not metered -------------------

    async def _req_metered(self, monkeypatch, path, app_key, forwarded_host, resolved_key=None):
        """`_req` with a mock Redis as the metering stream; returns (resp, xadd)."""
        helper = TestAssertedAppKeyUnregisteredRoute()
        token, claims, resolver = helper._jwt_env(monkeypatch)
        if resolved_key is not None:
            resolver.resolve.return_value["app"] = {
                "id": f"app-{resolved_key}",
                "key": resolved_key,
            }
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            app.state.redis = mock_redis
            try:
                with patch("main.verify_jwt", return_value=claims):
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as ac:
                        resp = await ac.get(
                            path,
                            headers={
                                "Authorization": f"Bearer {token}",
                                "x-app-key": app_key,
                                "x-forwarded-host": forwarded_host,
                            },
                        )
            finally:
                app.state.redis = None
        return resp, mock_redis.xadd

    @pytest.mark.parametrize(
        ("path", "app_key", "forwarded_host"),
        [
            ("/rest/workflows", "n8n", "n8n.nonagentic.ai"),
            ("/types/nodes.json", "n8n", "n8n.nonagentic.ai"),
            ("/workflow/abc", "n8n", "n8n.nonagentic.ai"),
            ("/boards/1", "kanban", "kanban.nonagentic.ai"),
        ],
    )
    async def test_gateway_ui_path_emits_no_metering_event(
        self, monkeypatch, path, app_key, forwarded_host
    ):
        """⛔ FAILS WHEN REVERTED: origin/main XADDed every SPA asset and editor
        REST call, which billing then filed under `unmapped:<first segment>`."""
        before = {k: _skipped_sample(k) for k in ("n8n", "kanban")}
        resp, xadd = await self._req_metered(monkeypatch, path, app_key, forwarded_host)
        assert resp.status_code == 200
        xadd.assert_not_awaited()
        after = {k: _skipped_sample(k) for k in ("n8n", "kanban")}
        assert after == {**before, app_key: before[app_key] + 1}

    async def test_registered_path_on_gateway_host_still_emits(self, monkeypatch):
        """NEGATIVE CONTROL: the skip is the unregistered-path arm, not the host.
        A registered `/kanban/` API path on the same host stays billable."""
        before = _skipped_sample("kanban")
        resp, xadd = await self._req_metered(
            monkeypatch,
            "/kanban/api/boards",
            "kanban",
            "kanban.nonagentic.ai",
            resolved_key="kanban",
        )
        assert resp.status_code == 200
        xadd.assert_awaited_once()
        assert xadd.await_args[0][1]["status"] == "allowed"
        assert _skipped_sample("kanban") == before

    @respx.mock
    async def test_forged_host_pair_on_licence_token_still_emits(self):
        """Both headers are unsigned. A licence-token caller that forges the
        gateway's pair never enters the host-bound arm (its key is attested by
        introspect, not asserted), so it must stay metered."""
        respx.post(INTROSPECT_URL).respond(200, json=_active())
        before = {k: _skipped_sample(k) for k in ("n8n", "kanban")}
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/rest/workflows",
                        headers={
                            "Authorization": "Bearer valid_token",
                            "x-app-key": "n8n",
                            "x-forwarded-host": "n8n.nonagentic.ai",
                        },
                    )
            finally:
                app.state.redis = None
        assert resp.status_code == 200
        mock_redis.xadd.assert_awaited_once()
        assert mock_redis.xadd.await_args[0][1]["status"] == "allowed"
        assert {k: _skipped_sample(k) for k in ("n8n", "kanban")} == before


class TestJwtModeSelfReadExemption:
    """JWT mode must ask the SAME self-read question the opaque path asks.

    THE REGRESSION (prod, 2026-09-16T09:01:26Z): mini1/mini2 `tts_backend`
    stopped presenting an opaque licence token and started minting a
    `w2v-backend` client-credentials JWT. It sends `X-Tenant-Id` and no
    `x-app-key`, so the key is DERIVED from the route — `/licenses/` names
    `licenses` — and the gate demanded the LICENSES app for a caller that was
    only reading its own licence:

        path:/licenses/active status:"denied_jwt_mode"
        reason:"jwt_mode=license_required"

    -> `GET /licenses/active` 402 -> entitlement `unknown` -> narration 503.
    The opaque path had already been exempted for exactly this call
    (`TestLicenseSelfReadExemption` in test_main.py); JWT mode never reached it
    because it dies at the 402 BEFORE `_check_app_route` runs.
    """

    @staticmethod
    def _resolver_returning_nothing() -> MagicMock:
        """No licence for `licenses` — the tenant is not one of the two that
        hold it. This is what makes these tests the regression rather than a
        restatement of the happy path."""
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)
        return resolver

    async def _get(self, monkeypatch, path: str, resolver, method: str = "GET", **headers):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={})

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    return await ac.request(
                        method,
                        path,
                        headers={"Authorization": f"Bearer {token}", **headers},
                    )

    async def test_licenses_active_allowed_without_a_licenses_licence(self, monkeypatch):
        """THE REGRESSION TEST. Keyless JWT caller, GET /licenses/active, tenant
        holds NO `licenses` licence -> allowed. Before the fix this is 402."""
        resolver = self._resolver_returning_nothing()
        before = _jwt_mode_counter("allowed_self_read_exempt") or 0.0
        tenant_before = (
            _jwt_mode_by_tenant_counter("allowed_self_read_exempt", TENANT_UUID, "licenses") or 0.0
        )

        resp = await self._get(monkeypatch, "/licenses/active", resolver)

        assert resp.status_code == 200, resp.text
        # The lookup really happened and really came back empty — an exemption
        # that short-circuited BEFORE the resolve would leave this un-awaited
        # and the test would pass while measuring nothing.
        assert resolver.resolve.await_args.kwargs["app_key"] == "licenses"
        # The licence it allows with is EMPTY, not fabricated. `x-license-id`
        # non-empty here would mean a licence was invented for billing to find.
        assert resp.headers.get("x-license-id") == ""
        assert resp.headers.get("x-subject-id") == ""
        assert resp.headers.get("x-plan-id") == ""
        # ...but the VERIFIED tenant is propagated, which is what scopes the read.
        assert resp.headers.get("x-tenant-id") == TENANT_UUID
        # Counted under its own label: an exemption folded into `allowed` cannot
        # be proven alive, and this one is the difference between narration and 503.
        assert (_jwt_mode_counter("allowed_self_read_exempt") or 0.0) == before + 1.0
        assert (
            _jwt_mode_by_tenant_counter("allowed_self_read_exempt", TENANT_UUID, "licenses") or 0.0
        ) == tenant_before + 1.0

    async def test_licenses_user_subject_allowed(self, monkeypatch):
        """The second entry in APP_ROUTE_SELF_READ is a PREFIX (path parameter)."""
        resp = await self._get(
            monkeypatch, "/licenses/user/abc", self._resolver_returning_nothing()
        )
        assert resp.status_code == 200, resp.text

    async def test_head_allowed(self, monkeypatch):
        """HEAD rides along with GET — the same read with the body suppressed."""
        resp = await self._get(
            monkeypatch, "/licenses/active", self._resolver_returning_nothing(), method="HEAD"
        )
        assert resp.status_code == 200

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
    async def test_non_exempt_licenses_paths_still_402(self, monkeypatch, path: str):
        """MANDATORY NEGATIVE CONTROL. Everything else under `/licenses/` still
        needs the licence — including a path that merely STARTS WITH
        `/licenses/active`, and the bare `/licenses/user` with no subject."""
        resp = await self._get(monkeypatch, path, self._resolver_returning_nothing())
        assert resp.status_code == 402, f"{path} was exempted and must not be"
        assert resp.json()["error"] == "license_required"

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    async def test_mutating_method_on_the_same_path_still_402(self, monkeypatch, method: str):
        """THE LOAD-BEARING BOUND, mirrored from the opaque path: this is a READ
        exemption. A default-ON exemption that opened a write would be a hole."""
        resp = await self._get(
            monkeypatch, "/licenses/active", self._resolver_returning_nothing(), method=method
        )
        assert resp.status_code == 402
        assert resp.json()["error"] == "license_required"

    async def test_dot_segment_walk_cannot_reach_the_exemption(self, monkeypatch):
        """`/licenses/user/../admin/keys` folds to `/licenses/admin/keys` and is
        NOT exempt. Driven through the `:path` pseudo-header because httpx
        resolves `..` client-side — sent as a URL the walk never arrives."""
        resp = await self._get(
            monkeypatch,
            "/anything",
            self._resolver_returning_nothing(),
            **{":path": "/licenses/user/../admin/keys"},
        )
        assert resp.status_code == 402, "a dot-segment walk reached the JWT self-read exemption"

    @pytest.mark.parametrize(
        "raw_path",
        [
            "/vllm/../licenses/active",
            "/admin/../licenses/user/abc",
            "/npm/../../licenses/active",
        ],
    )
    async def test_cross_route_smuggle_is_not_exempt(self, monkeypatch, raw_path: str):
        """THE RESIDUAL. These normalise INTO `/licenses/…` and so satisfy
        `_is_self_read_route`, but the RAW `:path` is what a router matches:
        `uri.prefix: /vllm/` would forward `/vllm/../licenses/active` to vllm,
        and we would have allowed an unlicensed caller on a request that never
        reaches the licences app. Whether Istio normalises before ext_authz is
        UNVERIFIED — no `pathNormalization` in the `istio-system/istio`
        ConfigMap — so the verdict must not depend on it.

        Driven through the `:path` pseudo-header because httpx resolves `..`
        client-side; sent as a URL the smuggle never arrives."""
        resp = await self._get(
            monkeypatch,
            "/anything",
            self._resolver_returning_nothing(),
            **{":path": raw_path},
        )
        assert resp.status_code == 402, f"{raw_path} reached the JWT self-read exemption"
        assert resp.json()["error"] == "license_required"

    @pytest.mark.parametrize(
        "raw_path",
        [
            "//licenses/active",
            "/licenses/active/",
            "/licenses%2Factive",
        ],
    )
    async def test_benign_rewrites_of_the_same_read_stay_exempt(self, monkeypatch, raw_path: str):
        """REGRESSION GUARD on the raw-path tightening. A duplicate leading
        slash, a trailing slash and an escaped separator are the SAME read —
        the audit confirmed all three exempt, and the fix for the smuggle above
        must not take them away. Any of these turning 402 means the raw-path
        check stopped being "benign normalisations only"."""
        resp = await self._get(
            monkeypatch,
            "/anything",
            self._resolver_returning_nothing(),
            **{":path": raw_path},
        )
        assert resp.status_code == 200, f"{raw_path} is the same read and must stay exempt"

    async def test_flag_off_restores_the_402(self, monkeypatch):
        """The revert switch is the SAME flag the opaque path uses."""
        monkeypatch.setattr(main, "LICENSE_SELF_READ_EXEMPT_ENABLED", False)
        resp = await self._get(monkeypatch, "/licenses/active", self._resolver_returning_nothing())
        assert resp.status_code == 402
        assert resp.json()["error"] == "license_required"

    async def test_asserted_key_is_not_exempted_from_holding_its_own_licence(self, monkeypatch):
        """Bound 1: the rescue is for the DERIVED key only. A caller that ASSERTS
        `x-app-key` has claimed a licence and must hold it; if it does, the
        OPAQUE-path exemption downstream in `_check_app_route` already allows the
        read (pinned by `test_jwt_mode_w2v_key_on_licenses_prefix_still_403`)."""
        resp = await self._get(
            monkeypatch,
            "/licenses/active",
            self._resolver_returning_nothing(),
            **{"x-app-key": "w2v"},
        )
        assert resp.status_code == 402
        assert resp.json()["app_key"] == "w2v"

    async def test_unregistered_path_is_untouched_by_the_exemption(self, monkeypatch):
        """No derivation, no exemption: the pre-existing 400 still stands."""
        resolver = self._resolver_returning_nothing()
        resp = await self._get(monkeypatch, "/api/anything", resolver)
        assert resp.status_code == 400
        assert resp.json()["error"] == "missing_app_key"


# -- Trusted tenant-proxy X-Tenant-Id passthrough ----------------------------
#
# THE DEFECT: `src/main.py`'s response-header build UNCONDITIONALLY overwrote
# a client-supplied `X-Tenant-Id` with the caller's own tenant, so the
# licenses service's own trusted-proxy retarget
# (`licenses/auth/middleware.py:404`, keyed on the SAME
# `TRUSTED_TENANT_PROXY_CLIENTS` allowlist by name/default) could never fire —
# the header it inspects already equalled the claim by the time it arrived.
# `w2v-backend` reading a bare-metal host's licence on a tenant it does not
# itself hold 402'd every time. These tests drive `GET /licenses/active`
# (the self-read-exempt route from `TestJwtModeSelfReadExemption` above) so
# the ONLY thing under test is which tenant lands in the outgoing
# `x-tenant-id` header — licence resolution never runs on this route.
class TestTrustedTenantProxyHeader:
    OTHER_TENANT = TENANT_UUID_2

    async def _get(
        self,
        monkeypatch,
        *,
        azp: str,
        tenant_header: str | None,
        asserted_header: str | None = None,
        forwarded_host: str | None = "licenses.example",
        clients: frozenset[str] | None = frozenset({"fr-backend", "w2v-backend"}),
    ):
        from jwt_mode import KEYCLOAK_ISSUER, VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        # `clients=None` leaves the MODULE DEFAULT in force — that is what the
        # default-config tests below exercise; do not patch around it.
        if clients is not None:
            monkeypatch.setattr(main, "TRUSTED_TENANT_PROXY_CLIENTS", clients)
        token = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})
        # `azp` lives on `claims.raw` — the SAME field `jwt_azp` reads in
        # `_handle_jwt_mode` — so this exercises the real extraction path, not
        # a shortcut around it.
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="u", raw={"azp": azp} if azp else {})

        headers = {"Authorization": f"Bearer {token}"}
        if tenant_header is not None:
            headers["x-tenant-id"] = tenant_header
        if asserted_header is not None:
            headers["x-asserted-tenant-id"] = asserted_header
        # The gateway's httpCall carries the original target host here (its
        # `:authority` is license-authz's own Service name). Default is the one
        # host a retarget is allowed on (license-authz#64).
        if forwarded_host is not None:
            headers["x-forwarded-host"] = forwarded_host

        # No `licenses` licence for this tenant — `/licenses/active` is the
        # self-read-exempt route, so this resolves with an EMPTY licence
        # (200, `self_read_exempt`) rather than exercising licence resolution,
        # keeping the test isolated to the x-tenant-id header decision.
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)

        async with lifespan(app):
            app.state.jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    return await ac.get("/licenses/active", headers=headers)

    async def test_trusted_client_asserted_tenant_is_preserved(self, monkeypatch):
        """`w2v-backend` (VERIFIED azp) sending X-Tenant-Id for a DIFFERENT
        tenant than its own claim -> the asserted tenant survives downstream.
        This is the fix: mini1's licence gate no longer 402s."""
        resp = await self._get(monkeypatch, azp="w2v-backend", tenant_header=self.OTHER_TENANT)
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT

    async def test_non_trusted_client_header_is_overwritten(self, monkeypatch):
        """MANDATORY NEGATIVE CONTROL. A caller whose azp is NOT on the
        allowlist gets TODAY'S behavior exactly: its own tenant, never the
        header it sent. Anything else would let any caller spoof a tenant."""
        resp = await self._get(
            monkeypatch, azp="some-other-service", tenant_header=self.OTHER_TENANT
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID
        assert resp.headers.get("x-tenant-id") != self.OTHER_TENANT

    async def test_trusted_client_malformed_header_falls_back_to_own_tenant(self, monkeypatch):
        """A trusted proxy cannot pass an arbitrary string through as a
        tenant — a non-UUID header is refused and the caller's own (already
        UUID-validated) tenant is forwarded instead."""
        resp = await self._get(
            monkeypatch, azp="w2v-backend", tenant_header="not-a-uuid; drop table tenants"
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID

    async def test_trusted_client_no_header_is_unchanged(self, monkeypatch):
        """No X-Tenant-Id at all -> nothing to preserve; own tenant as always."""
        resp = await self._get(monkeypatch, azp="w2v-backend", tenant_header=None)
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID

    # -- Gateway-forwarded assertion header ---------------------------------
    #
    # THE SECOND DEFECT: the branch above reads `x-tenant-id`, and on the
    # north-south path the istio ingressgateway STRIPS every client-supplied
    # identity header and calls ext_authz with an explicitly built header table
    # that contains no tenant header — so the retarget was structurally
    # unreachable through the gateway. The gateway now stamps its own
    # `x-asserted-tenant-id` (after stripping any client copy) and it is
    # preferred over `x-tenant-id`. The `azp` allowlist stays the security
    # control: an attacker-settable header name changes nothing about who is
    # permitted to retarget.

    async def test_asserted_header_trusted_client_is_retargeted(self, monkeypatch):
        """1. The gateway-stamped header + a trusted, VERIFIED azp + a valid
        UUID -> the asserted tenant is forwarded. This is the whole point: the
        north-south path can finally reach this branch."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT

    async def test_asserted_header_untrusted_client_is_not_retargeted(self, monkeypatch):
        """2. MANDATORY NEGATIVE CONTROL. The new header name does NOT widen
        who may retarget: an azp off the allowlist gets its own tenant, exactly
        as with `x-tenant-id`."""
        resp = await self._get(
            monkeypatch,
            azp="some-other-service",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID
        assert resp.headers.get("x-tenant-id") != self.OTHER_TENANT

    async def test_asserted_header_malformed_is_not_retargeted(self, monkeypatch):
        """3. The UUID guard applies to the new source too — an arbitrary
        string never passes through as a tenant."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header="not-a-uuid; drop table tenants",
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID

    async def test_tenant_header_only_preserves_existing_behaviour(self, monkeypatch):
        """4. No asserted header -> the in-mesh Service-to-Service path that
        PR #55 shipped is byte-for-byte unchanged."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=self.OTHER_TENANT,
            asserted_header=None,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT

    # -- Host bound + default narrowing + dual-tenant audit (#64) -----------
    #
    # Measured before this change: a listed azp could retarget on EVERY gated
    # internal host, although 66/66 real retargets were w2v-backend on
    # licenses.example. The retarget is now bounded to
    # `TENANT_PROXY_RETARGET_HOSTS` (default licenses.example) and refused
    # with a 403 elsewhere; the default allowlist is w2v-backend only; and the
    # audit row + metering event carry the chosen tenant next to the own one.

    async def test_retarget_on_licenses_internal_is_allowed(self, monkeypatch):
        """The one real use (w2v-backend -> licenses.example) keeps working,
        exactly as before — with the gateway's port-bearing, mixed-case host
        spelling normalised rather than refused."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
            forwarded_host="Licenses.Example:443",
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT

    async def test_retarget_on_other_gated_host_is_refused_with_log(self, monkeypatch, caplog):
        """A trusted azp asking for another tenant on any host but
        licenses.example -> 403 plus `trusted_tenant_proxy_retarget_refused`,
        never a silent downgrade to its own tenant."""

        caplog.set_level(logging.INFO)
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
            forwarded_host="secrets.example",
        )
        assert resp.status_code == 403, resp.text
        assert resp.headers.get("x-error-reason") == "tenant_proxy_retarget_host_not_allowed"
        assert resp.headers.get("x-tenant-id") is None
        refused = [
            json.loads(r.getMessage())
            for r in caplog.records
            if "trusted_tenant_proxy_retarget_refused" in r.getMessage()
        ]
        assert refused == [
            {
                "event": "trusted_tenant_proxy_retarget_refused",
                "azp": "w2v-backend",
                "host": "secrets.example",
                "reason": "host_not_allowed",
            }
        ]
        # And it did NOT fire the success line.
        assert not any('"trusted_tenant_proxy_retarget"' in r.getMessage() for r in caplog.records)

    async def test_retarget_with_no_identifiable_host_is_refused(self, monkeypatch, caplog):
        """No x-forwarded-host and a Host that is not licenses.example (the
        ASGI test client sends `test`) -> refused, fail closed."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=self.OTHER_TENANT,
            forwarded_host=None,
        )
        assert resp.status_code == 403, resp.text

    async def test_asserting_own_tenant_on_other_host_is_not_a_retarget(self, monkeypatch):
        """Asserting the tenant you already hold widens nothing, so the host
        bound must not 403 it."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header=TENANT_UUID,
            forwarded_host="kag.example",
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID

    async def test_default_config_fr_backend_does_not_retarget(self, monkeypatch):
        """With the MODULE DEFAULT allowlist, fr-backend (0 uses in 30d) is no
        longer a trusted proxy: its assertion is overwritten with its own
        tenant, exactly like any other unlisted caller."""
        resp = await self._get(
            monkeypatch,
            azp="fr-backend",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
            clients=None,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID

    async def test_default_config_w2v_backend_still_retargets(self, monkeypatch):
        """Positive control for the test above: the SAME request shape under
        the SAME default config does retarget for w2v-backend, so the fr test
        is red for the right reason."""
        resp = await self._get(
            monkeypatch,
            azp="w2v-backend",
            tenant_header=None,
            asserted_header=self.OTHER_TENANT,
            clients=None,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT

    async def test_retarget_audit_and_metering_carry_both_tenants(self, monkeypatch):
        """The allowed audit row and the metering event keep `tenant_id` = the
        caller's OWN tenant and add `retarget_tenant_id` = the chosen one."""
        audit = MagicMock(wraps=main.log_access)
        meter = AsyncMock()
        with patch("main.log_access", audit), patch("main._emit_metering_event", meter):
            resp = await self._get(
                monkeypatch,
                azp="w2v-backend",
                tenant_header=None,
                asserted_header=self.OTHER_TENANT,
            )
        assert resp.status_code == 200, resp.text
        allowed = [c for c in audit.call_args_list if c.kwargs.get("status") == "allowed"]
        assert len(allowed) == 1
        assert allowed[0].kwargs["tenant_id"] == TENANT_UUID
        assert allowed[0].kwargs["retarget_tenant_id"] == self.OTHER_TENANT
        assert meter.await_count == 1
        assert meter.await_args.kwargs["tenant_id"] == TENANT_UUID
        assert meter.await_args.kwargs["retarget_tenant_id"] == self.OTHER_TENANT

    async def test_non_retarget_audit_and_metering_unchanged(self, monkeypatch):
        """No retarget -> no chosen tenant anywhere; own tenant as always."""
        audit = MagicMock(wraps=main.log_access)
        meter = AsyncMock()
        with patch("main.log_access", audit), patch("main._emit_metering_event", meter):
            resp = await self._get(
                monkeypatch,
                azp="some-other-service",
                tenant_header=None,
                asserted_header=self.OTHER_TENANT,
                forwarded_host="secrets.example",
            )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == TENANT_UUID
        allowed = [c for c in audit.call_args_list if c.kwargs.get("status") == "allowed"]
        assert len(allowed) == 1
        assert allowed[0].kwargs["tenant_id"] == TENANT_UUID
        assert allowed[0].kwargs["retarget_tenant_id"] is None
        assert meter.await_args.kwargs["retarget_tenant_id"] is None


class TestRetargetTenantFieldEmission:
    """The additive field reaches the actual audit JSON and stream entry, and
    is ABSENT (not empty) when there is no retarget."""

    _BASE = dict(
        license_id="l",
        tenant_id=TENANT_UUID,
        app_id="a",
        plan_id="p",
        method="GET",
        path="/x",
        status="allowed",
        token_hash="h",
        source_ip="1.2.3.4",
        user_agent="ua",
    )

    def test_log_access_adds_field_only_when_set(self, caplog):

        caplog.set_level(logging.INFO)
        main.log_access(**self._BASE, retarget_tenant_id=TENANT_UUID_2)
        main.log_access(**self._BASE)
        rows = [
            json.loads(r.getMessage()) for r in caplog.records if '"token_hash"' in r.getMessage()
        ]
        assert rows[0]["tenant_id"] == TENANT_UUID
        assert rows[0]["retarget_tenant_id"] == TENANT_UUID_2
        assert "retarget_tenant_id" not in rows[1]

    async def test_metering_event_adds_field_only_when_set(self):
        fake = MagicMock()
        fake.xadd = AsyncMock()
        kw = dict(
            tenant_id=TENANT_UUID,
            license_id="l",
            app_id="a",
            plan_id="p",
            method="GET",
            path="/x",
            status="allowed",
            source_ip="1.2.3.4",
        )
        with patch("main._get_redis", AsyncMock(return_value=fake)):
            await main._emit_metering_event(MagicMock(), **kw, retarget_tenant_id=TENANT_UUID_2)
            await main._emit_metering_event(MagicMock(), **kw)
        first, second = (c.args[1] for c in fake.xadd.await_args_list)
        assert first["tenant_id"] == TENANT_UUID
        assert first["retarget_tenant_id"] == TENANT_UUID_2
        assert "retarget_tenant_id" not in second


# -- secure-apps S2S tenant proxy (realm-per-app P7) -------------------------
#
# Realm `portal` becomes staff-only; tts-backend (mini1/mini2) must present a
# SECURE-APPS `w2v-backend` client-credentials JWT on `licenses.example`.
# `secure-apps` can never join the user-JWT allowlist (it holds human
# accounts), so this arm is pinned by issuer + verified azp + verified role.
SA_ISSUER = "https://auth.nonagentic.ai/realms/secure-apps"


class TestSecureAppsTenantProxy:
    OTHER_TENANT = TENANT_UUID_2

    async def _get(
        self,
        monkeypatch,
        *,
        token_azp: str | None = "w2v-backend",
        verified_raw: dict | None = None,
        token_iss: str = SA_ISSUER,
        caller_jwks_present: bool = True,
        sa_issuer_configured: str = SA_ISSUER,
        path: str = "/licenses/active",
    ):
        from jwt_mode import VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        payload = {"iss": token_iss, "sub": "sa-sub"}
        if token_azp is not None:
            payload["azp"] = token_azp
        token = _make_jwt_like({"alg": "RS256"}, payload)
        if verified_raw is None:
            verified_raw = {
                "azp": "w2v-backend",
                "realm_access": {"roles": ["license-reader", "license-manager"]},
            }
        claims = VerifiedClaims(tenant_id=TENANT_UUID, sub="sa-sub", raw=verified_raw)

        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)
        headers = {
            "Authorization": f"Bearer {token}",
            "x-asserted-tenant-id": self.OTHER_TENANT,
            "x-forwarded-host": "licenses.example",
        }
        verify = MagicMock(return_value=claims)
        async with lifespan(app):
            monkeypatch.setattr(main._settings, "keycloak_secure_apps_issuer", sa_issuer_configured)
            app.state.jwks = MagicMock(name="portal_jwks")
            app.state.caller_jwks = MagicMock(name="sa_jwks") if caller_jwks_present else None
            app.state.active_resolver = resolver
            with (
                patch("main.verify_jwt", verify),
                respx.mock(assert_all_called=False) as rmock,
            ):
                rmock.post(INTROSPECT_URL).mock(
                    return_value=httpx.Response(200, json={"active": False})
                )
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(path, headers=headers)
            sa_jwks = app.state.caller_jwks
        return resp, verify, resolver, sa_jwks

    async def test_w2v_backend_secure_apps_token_retargets_on_licenses(self, monkeypatch):
        """The P7 path: verified against the SECURE-APPS key set with the
        issuer pinned, and the asserted tenant survives (same as portal twin)."""
        resp, verify, resolver, sa_jwks = await self._get(monkeypatch)
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-tenant-id") == self.OTHER_TENANT
        assert resp.headers.get("x-auth-realm") == "secure-apps"
        verify.assert_called_once()
        assert verify.call_args.args[1] is sa_jwks
        assert verify.call_args.kwargs["issuer"] == SA_ISSUER
        # Not an end user: no `realm` narrowing is forwarded to /licenses/active.
        assert "realm" not in resolver.resolve.call_args.kwargs

    async def test_verified_azp_off_allowlist_is_401(self, monkeypatch):
        """Unverified azp only SELECTS the arm; the VERIFIED azp decides."""
        resp, *_ = await self._get(
            monkeypatch,
            verified_raw={
                "azp": "billing-dashboard",
                "realm_access": {"roles": ["license-reader"]},
            },
        )
        assert resp.status_code == 401
        assert resp.headers.get("x-error-reason") == "invalid_token"

    async def test_missing_required_role_is_401(self, monkeypatch):
        resp, *_ = await self._get(
            monkeypatch,
            verified_raw={"azp": "w2v-backend", "realm_access": {"roles": ["offline_access"]}},
        )
        assert resp.status_code == 401

    async def test_role_check_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(main, "SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE", "")
        resp, *_ = await self._get(monkeypatch, verified_raw={"azp": "w2v-backend"})
        assert resp.status_code == 200, resp.text

    async def test_other_secure_apps_client_never_takes_jwt_mode(self, monkeypatch):
        """A secure-apps human/other client is NOT routed into JWT mode at all —
        it falls to opaque introspection (inactive -> 403) and is never verified
        as a user JWT."""
        resp, verify, *_ = await self._get(monkeypatch, token_azp="billing-dashboard")
        assert resp.status_code == 403  # opaque introspection: inactive
        verify.assert_not_called()

    async def test_empty_allowlist_disables_the_arm(self, monkeypatch):
        monkeypatch.setattr(main, "SECURE_APPS_TENANT_PROXY_CLIENTS", frozenset())
        resp, verify, *_ = await self._get(monkeypatch)
        assert resp.status_code == 403  # opaque introspection: inactive
        verify.assert_not_called()

    async def test_secure_apps_not_configured_disables_the_arm(self, monkeypatch):
        resp, verify, *_ = await self._get(monkeypatch, sa_issuer_configured="")
        assert resp.status_code == 403  # opaque introspection: inactive
        verify.assert_not_called()

    async def test_no_secure_apps_jwks_fails_closed(self, monkeypatch):
        resp, verify, *_ = await self._get(monkeypatch, caller_jwks_present=False)
        assert resp.status_code == 401
        assert resp.headers.get("x-error-reason") == "invalid_token"
        verify.assert_not_called()

    async def test_retarget_still_bounded_to_licenses_host(self, monkeypatch):
        """The host bound (license-authz#64) applies to this arm unchanged."""
        from jwt_mode import VerifiedClaims
        from tests.test_jwt_mode import _make_jwt_like

        monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
        token = _make_jwt_like(
            {"alg": "RS256"}, {"iss": SA_ISSUER, "sub": "sa-sub", "azp": "w2v-backend"}
        )
        claims = VerifiedClaims(
            tenant_id=TENANT_UUID,
            sub="sa-sub",
            raw={"azp": "w2v-backend", "realm_access": {"roles": ["license-reader"]}},
        )
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=None)
        async with lifespan(app):
            monkeypatch.setattr(main._settings, "keycloak_secure_apps_issuer", SA_ISSUER)
            app.state.jwks = MagicMock()
            app.state.caller_jwks = MagicMock()
            app.state.active_resolver = resolver
            with patch("main.verify_jwt", return_value=claims):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get(
                        "/licenses/active",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "x-asserted-tenant-id": self.OTHER_TENANT,
                            "x-forwarded-host": "secrets.example",
                        },
                    )
        assert resp.status_code == 403
        assert resp.headers.get("x-error-reason") == "tenant_proxy_retarget_host_not_allowed"

    async def test_portal_token_path_is_unchanged(self, monkeypatch):
        """Staff hosts: a portal token still verifies against the PORTAL key set."""
        from jwt_mode import KEYCLOAK_ISSUER

        resp, verify, _, sa_jwks = await self._get(
            monkeypatch,
            token_iss=KEYCLOAK_ISSUER,
            verified_raw={"azp": "w2v-backend"},
        )
        assert resp.status_code == 200, resp.text
        assert verify.call_args.args[1] is not sa_jwks
        assert verify.call_args.kwargs["issuer"] == KEYCLOAK_ISSUER
