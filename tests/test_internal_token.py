"""Tests for the Lane F1 internal-token issuer (RS256 + JWKS).

Two layers:
- Pure-function tests on `internal_token` (build_jwks / issue_internal_token) with
  an ephemeral RSA keypair — no settings/app needed.
- Route tests for GET /jwks + POST /internal-token, mutating the cached settings
  singleton (`main._settings`) and mocking the Keycloak SA caller-token verify.

The final roundtrip test locks the verify contract PR 2 (`kag_common`) implements:
signature + iss + aud + exp checked against the served JWKS.
"""

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from httpx import ASGITransport, AsyncClient

import internal_token
import main
from jwt_mode import VerifiedClaims
from settings import Settings

ISSUER = "license-authz"
AUDIENCE = "internal"
KID = "internal-test-1"


class TestIssuerSettingsGate:
    def test_enabled_without_key_fails_fast(self):
        with pytest.raises(RuntimeError, match="internal-token issuer is enabled"):
            Settings(
                environment="test",
                internal_token_issuer_enabled=True,
                internal_token_signing_key="",
                internal_token_kid="",
            )

    def test_disabled_does_not_require_key(self):
        s = Settings(environment="test", internal_token_issuer_enabled=False)
        assert s.internal_token_issuer_enabled is False

    def test_enabled_with_key_and_kid_ok(self):
        s = Settings(
            environment="test",
            internal_token_issuer_enabled=True,
            internal_token_signing_key="test-signing-key-not-a-real-pem",
            internal_token_kid="k1",
        )
        assert s.internal_token_kid == "k1"


@pytest.fixture(scope="module")
def keypair():
    """Ephemeral RS256 keypair: returns (private_pem_str, public_key_obj)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    # Clear the module-level key cache so a stale key from another test can't leak.
    internal_token._key_cache.clear()
    return pem, key.public_key()


# --------------------------------------------------------------------------
# Pure-function layer
# --------------------------------------------------------------------------
class TestBuildJwks:
    def test_public_jwk_shape(self, keypair):
        pem, _ = keypair
        doc = internal_token.build_jwks(pem=pem, kid=KID)
        assert list(doc.keys()) == ["keys"]
        (jwk,) = doc["keys"]
        assert jwk["kid"] == KID
        assert jwk["kty"] == "RSA"
        assert jwk["use"] == "sig"
        assert jwk["alg"] == "RS256"
        assert jwk["n"] and jwk["e"]

    def test_never_exposes_private_fields(self, keypair):
        pem, _ = keypair
        (jwk,) = internal_token.build_jwks(pem=pem, kid=KID)["keys"]
        for private_field in ("d", "p", "q", "dp", "dq", "qi"):
            assert private_field not in jwk

    def test_rejects_non_rsa_pem(self):
        internal_token._key_cache.clear()
        with pytest.raises(ValueError):
            internal_token.build_jwks(
                pem="-----BEGIN PUBLIC KEY-----\nbogus\n",
                kid=KID,
            )


class TestIssueInternalToken:
    def test_claims_and_kid(self, keypair):
        pem, public_key = keypair
        token, ttl = internal_token.issue_internal_token(
            pem=pem,
            kid=KID,
            issuer=ISSUER,
            audience=AUDIENCE,
            ttl_seconds=300,
            sub="alloy-sa",
            tenant_id="tenant-x",
        )
        assert ttl == 300
        assert pyjwt.get_unverified_header(token)["kid"] == KID
        claims = pyjwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["exp", "iss"]},
        )
        assert claims["iss"] == ISSUER
        assert claims["aud"] == AUDIENCE
        assert claims["sub"] == "alloy-sa"
        assert claims["tenant_id"] == "tenant-x"
        assert claims["exp"] - claims["iat"] == 300

    def test_tenant_id_omitted_when_none(self, keypair):
        pem, public_key = keypair
        token, _ = internal_token.issue_internal_token(
            pem=pem,
            kid=KID,
            issuer=ISSUER,
            audience=AUDIENCE,
            ttl_seconds=60,
            sub="alloy-sa",
            tenant_id=None,
        )
        claims = pyjwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
        )
        assert "tenant_id" not in claims

    def test_expired_token_rejected(self, keypair):
        pem, public_key = keypair
        token, _ = internal_token.issue_internal_token(
            pem=pem,
            kid=KID,
            issuer=ISSUER,
            audience=AUDIENCE,
            ttl_seconds=300,
            sub="alloy-sa",
            now=int(time.time()) - 1000,
        )
        with pytest.raises(pyjwt.ExpiredSignatureError):
            pyjwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                issuer=ISSUER,
                audience=AUDIENCE,
            )

    def test_verify_contract_roundtrip(self, keypair):
        """The exact checks kag_common.verify_signed_internal_token will run."""
        pem, _ = keypair
        token, _ = internal_token.issue_internal_token(
            pem=pem,
            kid=KID,
            issuer=ISSUER,
            audience=AUDIENCE,
            ttl_seconds=300,
            sub="alloy-sa",
        )
        # Verifier path: fetch JWKS → pick key by kid → verify sig + iss + aud + exp.
        (jwk,) = internal_token.build_jwks(pem=pem, kid=KID)["keys"]
        signing_key = pyjwt.PyJWK.from_dict(jwk).key
        claims = pyjwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["exp", "iss"]},
        )
        assert claims["sub"] == "alloy-sa"
        # wrong issuer rejected
        with pytest.raises(pyjwt.InvalidIssuerError):
            pyjwt.decode(
                token, signing_key, algorithms=["RS256"], issuer="someone-else", audience=AUDIENCE
            )


# --------------------------------------------------------------------------
# Route layer
# --------------------------------------------------------------------------
@pytest.fixture
def enable_issuer(keypair, monkeypatch):
    """Turn the issuer on by mutating the cached settings singleton."""
    pem, public_key = keypair
    monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", True)
    monkeypatch.setattr(main._settings, "internal_token_signing_key", pem)
    monkeypatch.setattr(main._settings, "internal_token_kid", KID)
    monkeypatch.setattr(main._settings, "internal_token_issuer", ISSUER)
    monkeypatch.setattr(main._settings, "internal_token_audience", AUDIENCE)
    monkeypatch.setattr(main._settings, "internal_token_ttl_seconds", 300)
    monkeypatch.setattr(
        main._settings,
        "keycloak_secure_apps_issuer",
        "https://auth.nonagentic.ai/realms/secure-apps",
    )
    # internal_token wrappers read get_settings() fresh — point them at the same instance.
    monkeypatch.setattr(internal_token, "get_settings", lambda: main._settings)
    internal_token._key_cache.clear()
    return pem, public_key


@pytest.fixture
async def raw_client():
    """AsyncClient without running lifespan (we set app.state manually per test)."""
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestJwksRoute:
    async def test_disabled_returns_404(self, raw_client, monkeypatch):
        monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", False)
        resp = await raw_client.get("/jwks")
        assert resp.status_code == 404
        assert resp.json()["error"] == "issuer_disabled"

    async def test_enabled_serves_public_keys(self, raw_client, enable_issuer):
        resp = await raw_client.get("/jwks")
        assert resp.status_code == 200
        (jwk,) = resp.json()["keys"]
        assert jwk["kid"] == KID
        assert jwk["kty"] == "RSA"
        assert "d" not in jwk


class TestInternalTokenRoute:
    async def test_disabled_returns_404(self, raw_client, monkeypatch):
        monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", False)
        resp = await raw_client.post("/internal-token")
        assert resp.status_code == 404

    async def test_missing_bearer(self, raw_client, enable_issuer):
        main.app.state.caller_jwks = object()
        resp = await raw_client.post("/internal-token")
        assert resp.status_code == 401
        assert resp.headers["x-error-reason"] == "missing_bearer"

    async def test_caller_verify_unavailable(self, raw_client, enable_issuer):
        main.app.state.caller_jwks = None
        resp = await raw_client.post(
            "/internal-token", headers={"Authorization": "Bearer sa-token"}
        )
        assert resp.status_code == 401
        assert resp.headers["x-error-reason"] == "caller_verify_unavailable"

    async def test_invalid_caller_token(self, raw_client, enable_issuer, monkeypatch):
        main.app.state.caller_jwks = object()

        def _raise(*a, **k):
            from jwt_mode import InvalidJWT

            raise InvalidJWT("bad sig")

        monkeypatch.setattr(main, "verify_jwt", _raise)
        resp = await raw_client.post(
            "/internal-token", headers={"Authorization": "Bearer sa-token"}
        )
        assert resp.status_code == 401
        assert resp.headers["x-error-reason"] == "invalid_caller_token"

    async def test_valid_caller_gets_decodable_token(self, raw_client, enable_issuer, monkeypatch):
        _, public_key = enable_issuer
        main.app.state.caller_jwks = object()
        monkeypatch.setattr(
            main,
            "verify_jwt",
            lambda *a, **k: VerifiedClaims(
                tenant_id="tenant-x", sub="svc", raw={"azp": "alloy-sa"}
            ),
        )
        resp = await raw_client.post(
            "/internal-token", headers={"Authorization": "Bearer sa-token"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 300
        claims = pyjwt.decode(
            body["access_token"],
            public_key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
        )
        assert claims["sub"] == "alloy-sa"  # azp wins over sub
        assert claims["tenant_id"] == "tenant-x"
