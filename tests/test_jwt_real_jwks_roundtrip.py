"""Real JWKS round-trip through the REAL verify path (#61 regression guard).

WHY THIS MODULE EXISTS. On 2026-09-29 (#61) pyjwt 2.15 shipped and ext_authz
rejected every valid JWT: the code read PyJWKClient's ``jwk_set_cache`` assuming
the 2.14 dict shape, and 2.15 stores a parsed ``PyJWKSet``. No test caught it,
because every JWT test in this repo replaces the layer that broke:
  * ``TestVerifyJwt`` hands ``verify_jwt`` a MagicMock signing key;
  * ``TestJwksCacheRefresh`` patches ``PyJWKClient``;
  * ``test_jwt_integration`` patches ``main.verify_jwt``.

Here NOTHING on that path is replaced. The ONLY seam is pyjwt's own network
call, ``urllib.request.build_opener`` (``jwt/jwks_client.py`` ``fetch_data``),
which is answered from an in-memory JWKS. Everything above it is production
code: ``main.lifespan`` builds the real ``JwksCache`` and does the eager boot
fetch, the real ``PyJWKClient`` fills its real JWK-set cache, ``verify_jwt``
resolves keys from it, ``ext_authz_check`` turns the verdict into a status,
and ``/ready`` reads the real latches. No network: any URL other than the
configured JWKS URL is refused and recorded.

The licences lookup that follows a SUCCESSFUL verification is the only other
stand-in (``app.state.active_resolver``), exactly as in test_jwt_integration;
it sits after the code under test and has no bearing on the JWT verdict.
"""

import io
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import RSAAlgorithm

import jwt_mode
import main
from jwt_mode import KEYCLOAK_ISSUER, InvalidJWT, JwksCache, verify_jwt

TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"
KID_1 = "portal-rs256-1"
KID_2 = "portal-rs256-2"
# A registered prefix, so the app key is DERIVED from the route and no
# caller-asserted x-app-key is needed (see _handle_jwt_mode).
PATH = "/coding/api/graph"


# -- keys and JWKS ------------------------------------------------------------


def _new_key() -> dict:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return {"private_pem": pem, "public_key": key.public_key()}


def _jwk(public_key, kid: str) -> dict:
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return jwk


@pytest.fixture(scope="module")
def keys() -> dict:
    """Two PUBLISHED keys (two kids in the JWKS) and one never published."""
    return {"k1": _new_key(), "k2": _new_key(), "unpublished": _new_key()}


@pytest.fixture(scope="module")
def jwks_document(keys) -> dict:
    return {
        "keys": [
            _jwk(keys["k1"]["public_key"], KID_1),
            _jwk(keys["k2"]["public_key"], KID_2),
        ]
    }


def _sign(private_pem: bytes, kid: str, **overrides) -> str:
    claims = {
        "iss": KEYCLOAK_ISSUER,
        "exp": int(time.time()) + 600,
        "iat": int(time.time()),
        "sub": "user-1",
        "tenant_id": TENANT_UUID,
        "azp": "portal-frontend",
    }
    claims.update(overrides)
    return pyjwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": kid})


# -- the ONE seam: pyjwt's urllib fetch ---------------------------------------


class _FakeJwksEndpoint:
    """Stands in for Keycloak's certs endpoint at pyjwt's own fetch seam.

    ``fetches`` records one entry (the timeout passed) per real JWKS fetch, so
    a test can say exactly how many network round-trips the code made.
    ``stray`` records any URL that is NOT the configured JWKS URL: those are
    refused, never dialled.
    """

    def __init__(self, url: str, document: dict) -> None:
        self.url = url
        self.document = document
        self.fail = False
        self.fetches: list[float | None] = []
        self.stray: list[str] = []

    def build_opener(self, *_handlers):
        return _FakeOpener(self)


class _FakeOpener:
    def __init__(self, endpoint: _FakeJwksEndpoint) -> None:
        self._endpoint = endpoint

    def open(self, req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        if url != self._endpoint.url:
            self._endpoint.stray.append(url)
            raise urllib.error.URLError(f"hermetic test: no route to {url}")
        if self._endpoint.fail:
            raise urllib.error.URLError("hermetic test: keycloak down")
        self._endpoint.fetches.append(timeout)
        return io.BytesIO(json.dumps(self._endpoint.document).encode())


@pytest.fixture
def jwks_endpoint(monkeypatch, jwks_document) -> _FakeJwksEndpoint:
    # The URL production fetches: settings.jwks_url, defaulted from
    # KEYCLOAK_ISSUER (conftest) to {issuer}/protocol/openid-connect/certs and
    # bound as JwksCache's default at import.
    assert jwt_mode.JWKS_URL.startswith("http"), jwt_mode.JWKS_URL
    endpoint = _FakeJwksEndpoint(jwt_mode.JWKS_URL, jwks_document)
    monkeypatch.setattr(urllib.request, "build_opener", endpoint.build_opener)
    return endpoint


@pytest.fixture
async def portal(monkeypatch, jwks_endpoint):
    """The app booted in JWT mode exactly as production boots it.

    ``LICENSE_AUTHZ_JWT_MODE_ENABLED`` is read into ``main.JWT_MODE_ENABLED`` at
    import, so it is flipped on the module (the established pattern in
    test_jwt_integration). ``main.lifespan`` then constructs the real
    ``JwksCache()`` with production defaults and does its eager boot fetch
    through the seam above.

    Singletons reset: ``app.state.jwks`` / ``app.state.active_resolver`` live on
    the module-level ``main.app`` and survive lifespan exit, so they are cleared
    on teardown. The readiness transition latch lives on the JwksCache instance,
    which is fresh per test. Redis state is reset by conftest's autouse fixture.
    The SA client secret is removed so ServiceTokenProvider stays `unconfigured`
    and never dials Keycloak's token endpoint (no network).
    """
    monkeypatch.delenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
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
    try:
        async with main.lifespan(main.app):
            jwks = main.app.state.jwks
            assert type(jwks) is JwksCache, "lifespan must build the REAL JwksCache"
            main.app.state.active_resolver = resolver
            transport = ASGITransport(app=main.app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield SimpleNamespace(
                    client=ac, jwks=jwks, resolver=resolver, endpoint=jwks_endpoint
                )
    finally:
        main.app.state.jwks = None
        main.app.state.active_resolver = None


async def _check(portal, token: str):
    return await portal.client.get(PATH, headers={"Authorization": f"Bearer {token}"})


# -- positive path + the #61 regression ---------------------------------------


async def test_boot_fetch_makes_the_pod_ready(portal):
    """#61 part 2: readiness must read the boot fetch as loaded key material."""
    assert len(portal.endpoint.fetches) == 1, "lifespan's eager boot fetch"
    # Production bounds the fetch (JWKS_HTTP_TIMEOUT), pyjwt's 30s default is not used.
    assert portal.endpoint.fetches[0] == jwt_mode.JWKS_HTTP_TIMEOUT
    assert portal.endpoint.stray == [], "something tried to leave the process"
    assert portal.jwks.ever_loaded_ok is True
    assert portal.jwks.last_load_ok is True

    resp = await portal.client.get("/ready")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["checks"]["jwks_portal"] == {
        "ok": True,
        "gating": True,
        "detail": "signing keys loaded",
    }
    assert "jwks_portal" not in body["degraded"]
    # Inside JWKS_REVALIDATE_INTERVAL the probe must not re-fetch.
    assert len(portal.endpoint.fetches) == 1


async def test_valid_tokens_for_both_kids_are_accepted_from_one_fetch(portal, keys):
    """THE #61 PATH: kid lookup served from the cached JWK set after the boot fetch.

    Same token twice, then a token under the SECOND kid: all three must be
    allowed with the boot fetch as the only network round-trip. Under #61 every
    one of these was a 401 ``invalid_token``.
    """
    token_1 = _sign(keys["k1"]["private_pem"], KID_1)
    token_2 = _sign(keys["k2"]["private_pem"], KID_2, sub="user-2")

    first = await _check(portal, token_1)
    again = await _check(portal, token_1)
    second_kid = await _check(portal, token_2)

    for resp in (first, again, second_kid):
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-app-key") == "coding"
    # The VERIFIED tenant claim is what reached the licence lookup.
    assert portal.resolver.resolve.await_count == 3
    for call in portal.resolver.resolve.await_args_list:
        assert call.kwargs["tenant_id"] == TENANT_UUID
        assert call.kwargs["app_key"] == "coding"
    assert len(portal.endpoint.fetches) == 1, "kid lookups must be served from the cached set"


async def test_verify_jwt_returns_the_signed_claims(portal, keys):
    """Same real cache, called at the function the three call sites share."""
    verified = verify_jwt(_sign(keys["k2"]["private_pem"], KID_2, sub="user-9"), portal.jwks)

    assert verified.tenant_id == TENANT_UUID
    assert verified.sub == "user-9"
    assert verified.raw["iss"] == KEYCLOAK_ISSUER
    assert len(portal.endpoint.fetches) == 1


# -- negatives -----------------------------------------------------------------


def _assert_invalid_token(resp) -> None:
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"] == "invalid_token"
    assert resp.headers.get("x-error-reason") == "invalid_token"


async def test_token_signed_by_an_unpublished_key_is_rejected(portal, keys):
    """Published kid, wrong private key: the kid resolves, the signature does not.
    No refresh: the kid was found, so nothing triggers a fetch."""
    forged = _sign(keys["unpublished"]["private_pem"], KID_1)

    _assert_invalid_token(await _check(portal, forged))
    assert len(portal.endpoint.fetches) == 1
    portal.resolver.resolve.assert_not_awaited()


async def test_expired_token_is_rejected(portal, keys):
    expired = _sign(keys["k1"]["private_pem"], KID_1, exp=int(time.time()) - 120)

    _assert_invalid_token(await _check(portal, expired))
    assert len(portal.endpoint.fetches) == 1
    portal.resolver.resolve.assert_not_awaited()


async def test_unknown_kid_is_rejected_after_exactly_one_refresh(portal, keys):
    """Refresh semantics on a kid miss, as the code stands:

    1. pyjwt's own forced refresh is suppressed by its cooldown
       (``cooldown_duration`` 30s, not overridden by JwksCache) because the
       boot fetch just succeeded, so PyJWKClient raises PyJWKClientError.
    2. ``JwksCache.get_signing_key`` catches that and calls ``_refresh()``
       ONCE: a new client plus one real fetch_data(), so +1 fetch.
    3. The retry on the fresh client is a JWK-set cache hit, the kid is still
       absent and the new client is cooling down, so it raises and
       verify_jwt converts that to InvalidJWT, giving a 401.
    The refreshed client must still serve the published kids from cache.
    """
    unknown = _sign(keys["unpublished"]["private_pem"], "rotated-away-kid")

    _assert_invalid_token(await _check(portal, unknown))
    assert len(portal.endpoint.fetches) == 2, "one boot fetch + exactly one kid-miss refresh"
    assert portal.jwks.last_load_ok is True, "the refresh fetch itself succeeded"
    portal.resolver.resolve.assert_not_awaited()

    still_valid = await _check(portal, _sign(keys["k1"]["private_pem"], KID_1))
    assert still_valid.status_code == 200, still_valid.text
    assert len(portal.endpoint.fetches) == 2, "a valid kid after a refresh needs no fetch"


async def test_wrong_audience_is_rejected_when_audience_verification_is_armed(
    portal, keys, monkeypatch
):
    """Armed the way production would arm it (JWT_VERIFY_AUDIENCE +
    JWT_EXPECTED_AUDIENCES, read into module globals at import). Positive
    control first, so the 401 is attributable to the audience alone."""
    monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
    monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("portal-aud",))

    ok = await _check(portal, _sign(keys["k1"]["private_pem"], KID_1, aud="portal-aud"))
    assert ok.status_code == 200, ok.text

    wrong = _sign(keys["k1"]["private_pem"], KID_1, aud="some-other-client")
    _assert_invalid_token(await _check(portal, wrong))
    assert len(portal.endpoint.fetches) == 1


async def test_wrong_issuer_is_rejected(portal, keys):
    """Driven at verify_jwt, not through ext_authz: ``is_jwt_bearer`` only
    routes a bearer into JWT mode when its ``iss`` equals KEYCLOAK_ISSUER, so a
    foreign-issuer token never reaches this verifier from the HTTP path. The
    real cache resolves the published kid; the issuer check must still refuse.
    """
    foreign = _sign(keys["k1"]["private_pem"], KID_1, iss="https://auth.nonagentic.ai/realms/other")

    with pytest.raises(InvalidJWT):
        verify_jwt(foreign, portal.jwks)
    # And the converse: a portal token checked against another realm's issuer
    # (the shape of the secure-apps call sites) is refused too.
    with pytest.raises(InvalidJWT):
        verify_jwt(
            _sign(keys["k1"]["private_pem"], KID_1),
            portal.jwks,
            issuer="https://auth.nonagentic.ai/realms/secure-apps",
        )
    assert len(portal.endpoint.fetches) == 1


# -- negative control for the readiness assertion ------------------------------


async def test_failed_boot_fetch_is_not_ready_and_rejects_tokens(monkeypatch, jwks_endpoint, keys):
    """Proves the readiness assertion above can fail: with Keycloak down at
    boot the pod holds no key material, /ready is a gating 503, and a token
    that is otherwise valid is a 401."""
    jwks_endpoint.fail = True
    monkeypatch.delenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    try:
        async with main.lifespan(main.app):
            jwks = main.app.state.jwks
            assert type(jwks) is JwksCache
            assert jwks.ever_loaded_ok is False
            main.app.state.active_resolver = MagicMock()
            transport = ASGITransport(app=main.app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                ready = await ac.get("/ready")
                denied = await ac.get(
                    PATH,
                    headers={"Authorization": f"Bearer {_sign(keys['k1']['private_pem'], KID_1)}"},
                )
    finally:
        main.app.state.jwks = None
        main.app.state.active_resolver = None

    assert ready.status_code == 503, ready.text
    assert "jwks_portal" in ready.json()["failed"]
    _assert_invalid_token(denied)
    assert jwks_endpoint.fetches == [], "no fetch can have succeeded"
