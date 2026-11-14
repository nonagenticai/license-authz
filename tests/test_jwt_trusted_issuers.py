"""Realm-per-app P0: user-JWT issuer allowlist (KEYCLOAK_TRUSTED_ISSUERS).

Covers the four properties the change must hold:
  * DEFAULT UNCHANGED - with the env unset the allowlist is (KEYCLOAK_ISSUER,),
    a portal token is verified exactly as before and licenses gets no ``realm``;
  * an ALLOWLISTED second issuer is routed to JWT mode, verified against ITS OWN
    JWKS with ``issuer=`` pinned to it, and its realm flows downstream;
  * a NON-allowlisted issuer is never routed to JWT mode (opaque path, 401);
  * one JWKS cache PER issuer - keys of one realm never verify another's token.

The end-to-end cases drive the real ``main.lifespan`` / ``JwksCache`` /
``verify_jwt`` with pyjwt's urllib fetch as the only seam (same seam as
test_jwt_real_jwks_roundtrip), serving a different key set per realm URL.
"""

import io
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt as pyjwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import RSAAlgorithm

import jwt_mode
import main
from jwt_mode import KEYCLOAK_ISSUER, ActiveLicenseResolver, IssuerJwks, JwksCache
from settings import Settings

PORTAL = KEYCLOAK_ISSUER
CP = "https://auth.nonagentic.ai/realms/commonpractice"
EVIL = "https://auth.nonagentic.ai/realms/not-allowlisted"
INCLUSTER = "http://keycloak.keycloak.example:8080"
TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"
PATH = "/coding/api/graph"


# -- settings: allowlist parsing + JWKS derivation ----------------------------


def _settings(**env) -> Settings:
    base = {"environment": "test", "keycloak_issuer": PORTAL}
    base.update(env)
    return Settings(**base)


def test_default_allowlist_is_exactly_the_primary_issuer():
    assert _settings().trusted_issuers() == (PORTAL,)
    # And the module the request path reads was built the same way.
    assert jwt_mode.TRUSTED_ISSUERS == (PORTAL,)


def test_default_allowlist_is_empty_without_any_issuer():
    assert _settings(keycloak_issuer="").trusted_issuers() == ()


def test_csv_allowlist_strips_slashes_and_spaces_and_dedups():
    s = _settings(keycloak_trusted_issuers=f" {PORTAL}/ , {CP},,{PORTAL}")
    assert s.trusted_issuers() == (PORTAL, CP)


def test_secure_apps_issuer_in_the_user_allowlist_refuses_to_boot():
    sa = "https://auth.nonagentic.ai/realms/secure-apps"
    with pytest.raises(RuntimeError, match="secure-apps"):
        _settings(keycloak_secure_apps_issuer=sa, keycloak_trusted_issuers=f"{PORTAL},{sa}/")


def test_jwks_url_primary_keeps_configured_url():
    url = f"{INCLUSTER}/realms/portal/protocol/openid-connect/certs"
    assert _settings(jwks_url=url).jwks_url_for_issuer(PORTAL) == url


def test_jwks_url_second_realm_same_host_stays_in_cluster():
    url = f"{INCLUSTER}/realms/portal/protocol/openid-connect/certs"
    got = _settings(jwks_url=url).jwks_url_for_issuer(CP)
    assert got == f"{INCLUSTER}/realms/commonpractice/protocol/openid-connect/certs"


def test_jwks_url_default_derivation_when_jwks_url_unset():
    s = _settings()  # jwks_url defaults to {issuer}/protocol/openid-connect/certs
    assert s.jwks_url_for_issuer(CP) == f"{CP}/protocol/openid-connect/certs"


def test_jwks_url_foreign_host_falls_back_to_issuer_certs():
    other = "https://other.example/realms/x"
    url = f"{INCLUSTER}/realms/portal/protocol/openid-connect/certs"
    assert (
        _settings(jwks_url=url).jwks_url_for_issuer(other)
        == f"{other}/protocol/openid-connect/certs"
    )


# -- routing predicate --------------------------------------------------------


def _unsigned(iss: str) -> str:
    return pyjwt.encode({"iss": iss, "sub": "u"}, "k" * 32, algorithm="HS256")


def test_is_jwt_bearer_default_routes_only_the_primary():
    assert jwt_mode.is_jwt_bearer(_unsigned(PORTAL)) is True
    assert jwt_mode.is_jwt_bearer(_unsigned(CP)) is False


def test_is_jwt_bearer_routes_an_allowlisted_second_issuer(monkeypatch):
    monkeypatch.setattr(jwt_mode, "TRUSTED_ISSUERS", (PORTAL, CP))
    assert jwt_mode.is_jwt_bearer(_unsigned(CP)) is True
    assert jwt_mode.is_jwt_bearer(_unsigned(PORTAL)) is True
    assert jwt_mode.is_jwt_bearer(_unsigned(EVIL)) is False
    # Token-side iss is compared EXACTLY: a trailing slash is not the same issuer.
    assert jwt_mode.is_jwt_bearer(_unsigned(CP + "/")) is False


def test_is_jwt_bearer_explicit_expected_iss_still_works():
    assert jwt_mode.is_jwt_bearer(_unsigned(CP), expected_iss=CP) is True
    assert jwt_mode.is_jwt_bearer(_unsigned(PORTAL), expected_iss=CP) is False


def test_realm_of_issuer():
    assert jwt_mode.realm_of_issuer(CP) == "commonpractice"
    assert jwt_mode.realm_of_issuer(PORTAL + "/") == "portal"
    assert jwt_mode.realm_of_issuer("license-authz") == ""
    assert jwt_mode.realm_of_issuer(None) == ""


# -- per-issuer JWKS cache ----------------------------------------------------


def test_issuer_jwks_one_cache_per_issuer_and_none_for_unlisted():
    primary = JwksCache(jwks_url="http://kc/realms/portal/protocol/openid-connect/certs")
    reg = IssuerJwks(
        primary,
        primary_issuer=PORTAL,
        issuers=(PORTAL, CP, "https://auth.nonagentic.ai/realms/fr"),
        jwks_url_for=lambda iss: f"{iss}/certs",
    )
    assert reg.get(PORTAL) is primary
    cp = reg.get(CP)
    fr = reg.get("https://auth.nonagentic.ai/realms/fr")
    assert isinstance(cp, JwksCache) and isinstance(fr, JwksCache)
    assert cp is not fr and cp is not primary
    assert cp._jwks_url == f"{CP}/certs"
    # Same object on every lookup: the cache is per issuer, not per request.
    assert reg.get(CP) is cp
    assert reg.get(EVIL) is None
    assert set(reg.extra) == {CP, "https://auth.nonagentic.ai/realms/fr"}


def test_issuer_jwks_default_has_no_extra_caches():
    reg = IssuerJwks(JwksCache(jwks_url="http://kc/certs"))
    assert reg.extra == {}
    assert reg.get(CP) is None


# -- resolver: realm forwarded to licenses only when given --------------------


@respx.mock
async def test_resolver_sends_no_realm_by_default_and_realm_when_given():
    url = "http://licenses.test/licenses/active"
    route = respx.get(url).mock(return_value=httpx.Response(404))
    resolver = ActiveLicenseResolver(url=url)
    async with httpx.AsyncClient() as client:
        await resolver.resolve(client, tenant_id="t", app_key="a", auth_headers={})
        await resolver.resolve(client, tenant_id="t", app_key="a", auth_headers={}, realm="cp")
        # Cached separately per realm: a second call for each issues no request.
        await resolver.resolve(client, tenant_id="t", app_key="a", auth_headers={})
        await resolver.resolve(client, tenant_id="t", app_key="a", auth_headers={}, realm="cp")
    assert route.call_count == 2
    first, second = (c.request.url.params for c in route.calls)
    assert dict(first) == {"tenant_id": "t", "app_key": "a"}
    assert dict(second) == {"tenant_id": "t", "app_key": "a", "realm": "cp"}


# -- end to end through the REAL verify path ----------------------------------


def _key() -> dict:
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = k.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    jwk = json.loads(RSAAlgorithm.to_jwk(k.public_key()))
    return {"pem": pem, "jwk": jwk}


@pytest.fixture(scope="module")
def realm_keys() -> dict:
    return {"portal": _key(), "cp": _key()}


def _sign(key: dict, kid: str, iss: str, **over) -> str:
    claims = {
        "iss": iss,
        "exp": int(time.time()) + 600,
        "iat": int(time.time()),
        "sub": "user-1",
        "tenant_id": TENANT_UUID,
        "azp": "commonpractice-frontend",
    }
    claims.update(over)
    return pyjwt.encode(claims, key["pem"], algorithm="RS256", headers={"kid": kid})


class _MultiRealmJwks:
    """pyjwt's urllib seam answering a DIFFERENT key set per JWKS URL."""

    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs
        self.fetches: list[str] = []
        self.stray: list[str] = []

    def build_opener(self, *_handlers):
        return self

    def open(self, req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        if url not in self.docs:
            self.stray.append(url)
            raise urllib.error.URLError(f"hermetic test: no route to {url}")
        self.fetches.append(url)
        return io.BytesIO(json.dumps(self.docs[url]).encode())


@pytest.fixture
def endpoints(monkeypatch, realm_keys) -> _MultiRealmJwks:
    portal_jwk = dict(realm_keys["portal"]["jwk"], kid="portal-1", use="sig", alg="RS256")
    cp_jwk = dict(realm_keys["cp"]["jwk"], kid="cp-1", use="sig", alg="RS256")
    fake = _MultiRealmJwks(
        {
            jwt_mode.JWKS_URL: {"keys": [portal_jwk]},
            main._settings.jwks_url_for_issuer(CP): {"keys": [cp_jwk]},
        }
    )
    monkeypatch.setattr(urllib.request, "build_opener", fake.build_opener)
    return fake


async def _boot(monkeypatch, endpoints, trusted: tuple[str, ...]):
    monkeypatch.delenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(jwt_mode, "TRUSTED_ISSUERS", trusted)
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
    return resolver


@pytest.fixture
async def two_realms(monkeypatch, endpoints):
    resolver = await _boot(monkeypatch, endpoints, (PORTAL, CP))
    try:
        async with main.lifespan(main.app):
            main.app.state.active_resolver = resolver
            async with AsyncClient(
                transport=ASGITransport(app=main.app), base_url="http://test"
            ) as ac:
                yield SimpleNamespace(client=ac, resolver=resolver, endpoints=endpoints)
    finally:
        main.app.state.jwks = None
        main.app.state.issuer_jwks = None
        main.app.state.active_resolver = None


@pytest.fixture
async def portal_only(monkeypatch, endpoints):
    resolver = await _boot(monkeypatch, endpoints, (PORTAL,))
    try:
        async with main.lifespan(main.app):
            main.app.state.active_resolver = resolver
            async with AsyncClient(
                transport=ASGITransport(app=main.app), base_url="http://test"
            ) as ac:
                yield SimpleNamespace(client=ac, resolver=resolver, endpoints=endpoints)
    finally:
        main.app.state.jwks = None
        main.app.state.issuer_jwks = None
        main.app.state.active_resolver = None


async def _check(ctx, token: str):
    return await ctx.client.get(PATH, headers={"Authorization": f"Bearer {token}"})


async def test_allowlisted_second_issuer_is_accepted_with_its_own_keys(two_realms, realm_keys):
    token = _sign(realm_keys["cp"], "cp-1", CP)
    resp = await _check(two_realms, token)
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("x-auth-realm") == "commonpractice"
    kwargs = two_realms.resolver.resolve.await_args.kwargs
    assert kwargs["tenant_id"] == TENANT_UUID
    # Non-primary realm is handed to licenses for its realm_owner check.
    assert kwargs["realm"] == "commonpractice"
    # Both JWKS were fetched at boot, from the in-process seam only.
    assert set(two_realms.endpoints.fetches) == {
        jwt_mode.JWKS_URL,
        main._settings.jwks_url_for_issuer(CP),
    }
    assert two_realms.endpoints.stray == []


async def test_primary_issuer_unchanged_with_two_realms(two_realms, realm_keys):
    resp = await _check(two_realms, _sign(realm_keys["portal"], "portal-1", PORTAL))
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("x-auth-realm") == "portal"
    # Primary realm: the licenses request carries no realm (byte-identical).
    assert "realm" not in two_realms.resolver.resolve.await_args.kwargs


async def test_keys_of_one_realm_never_verify_another_realms_token(two_realms, realm_keys):
    """Per-issuer pinning: a token claiming CP but signed by a PORTAL key (kid
    absent from CP's set) must not verify — nor the reverse."""
    forged_cp = _sign(realm_keys["portal"], "portal-1", CP)
    forged_portal = _sign(realm_keys["cp"], "cp-1", PORTAL)
    for tok in (forged_cp, forged_portal):
        resp = await _check(two_realms, tok)
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"] == "invalid_token"
    two_realms.resolver.resolve.assert_not_awaited()


async def test_non_allowlisted_issuer_is_not_routed_to_jwt_mode(two_realms, realm_keys, respx_mock):
    """Signed by a key a realm DOES publish, but iss is not allowlisted: it takes
    the opaque licence-token path (introspect -> 403), never JWT mode."""
    introspect = respx_mock.post(main.LICENSE_INTROSPECT_URL).mock(
        return_value=httpx.Response(200, json={"active": False})
    )
    resp = await _check(two_realms, _sign(realm_keys["cp"], "cp-1", EVIL))
    # Inactive introspect answer -> the opaque arm's 403, never JWT mode's 200/401.
    assert resp.status_code == 403
    assert introspect.called
    two_realms.resolver.resolve.assert_not_awaited()


async def test_default_config_second_realm_token_is_not_routed(portal_only, realm_keys, respx_mock):
    """Env unset: a CP-realm token behaves exactly as before the allowlist —
    opaque path, no CP JWKS fetched, and the portal path still works."""
    introspect = respx_mock.post(main.LICENSE_INTROSPECT_URL).mock(
        return_value=httpx.Response(200, json={"active": False})
    )
    resp = await _check(portal_only, _sign(realm_keys["cp"], "cp-1", CP))
    assert resp.status_code == 403
    assert introspect.called
    portal_only.resolver.resolve.assert_not_awaited()
    assert portal_only.endpoints.fetches == [jwt_mode.JWKS_URL]
    assert main.app.state.issuer_jwks.extra == {}

    ok = await _check(portal_only, _sign(realm_keys["portal"], "portal-1", PORTAL))
    assert ok.status_code == 200, ok.text
