"""Unit tests for JWT-mode detection, claim extraction, and license resolution."""

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import jwt_mode
from jwt_mode import (
    KEYCLOAK_ISSUER,
    ActiveLicenseResolver,
    InvalidJWT,
    JwksCache,
    _b64url_decodable,
    _jwt_iss,
    extract_tenant_id,
    is_jwt_bearer,
    verify_jwt,
)


def _make_jwt_like(header: dict, payload: dict) -> str:
    """Build a well-formed JWT string (no real signature — only structure tested)."""

    def b64(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{b64(header)}.{b64(payload)}.fake-signature"


class TestIsJwtBearer:
    def test_keycloak_jwt_detected(self):
        tok = _make_jwt_like({"alg": "RS256"}, {"iss": KEYCLOAK_ISSUER, "sub": "u"})
        assert is_jwt_bearer(tok) is True

    def test_wrong_issuer_rejected(self):
        tok = _make_jwt_like({"alg": "RS256"}, {"iss": "https://other.example", "sub": "u"})
        assert is_jwt_bearer(tok) is False

    def test_opaque_license_token_rejected(self):
        # License tokens have no dots.
        assert is_jwt_bearer("opaque-license-token-without-dots") is False

    def test_two_parts_rejected(self):
        assert is_jwt_bearer("a.b") is False

    def test_malformed_header_rejected(self):
        assert is_jwt_bearer("$$$.payload.sig") is False

    def test_missing_iss_rejected(self):
        tok = _make_jwt_like({"alg": "RS256"}, {"sub": "u"})
        assert is_jwt_bearer(tok) is False


class TestExtractTenantId:
    def test_prefers_tenant_id_claim(self):
        assert extract_tenant_id({"tenant_id": "gerard-dev", "sub": "u"}) == "gerard-dev"

    def test_falls_back_to_sub(self):
        assert extract_tenant_id({"sub": "uuid-xyz"}) == "uuid-xyz"

    def test_anonymous_when_nothing(self):
        assert extract_tenant_id({}) == "anonymous"

    def test_empty_tenant_id_falls_back(self):
        assert extract_tenant_id({"tenant_id": "", "sub": "uuid"}) == "uuid"


@pytest.mark.asyncio
class TestActiveLicenseResolver:
    async def _mock_client(self, status_code: int, body: dict | None = None):
        client = AsyncMock(spec=httpx.AsyncClient)
        resp = httpx.Response(
            status_code,
            json=body or {},
            request=httpx.Request("GET", "http://test/licenses/active"),
        )
        client.get.return_value = resp
        return client

    async def test_resolve_returns_license_on_200(self):
        resolver = ActiveLicenseResolver(url="http://test/licenses/active")
        client = await self._mock_client(
            200, {"active": True, "tenant_id": "t1", "license_id": "L1"}
        )
        out = await resolver.resolve(client, "t1", "n8n", {})
        assert out is not None
        assert out["license_id"] == "L1"

    async def test_resolve_returns_none_on_404(self):
        resolver = ActiveLicenseResolver()
        client = await self._mock_client(404, {"detail": "no_active_license"})
        out = await resolver.resolve(client, "t1", "n8n", {})
        assert out is None

    async def test_cache_hit_skips_http(self):
        resolver = ActiveLicenseResolver()
        client = await self._mock_client(200, {"active": True, "tenant_id": "t1"})
        await resolver.resolve(client, "t1", "n8n", {})
        await resolver.resolve(client, "t1", "n8n", {})
        assert client.get.call_count == 1

    async def test_404_is_cached(self):
        resolver = ActiveLicenseResolver()
        client = await self._mock_client(404, {})
        await resolver.resolve(client, "t1", "n8n", {})
        await resolver.resolve(client, "t1", "n8n", {})
        assert client.get.call_count == 1

    async def test_cache_expires(self):
        resolver = ActiveLicenseResolver(ttl=0)
        client = await self._mock_client(200, {"active": True})
        await resolver.resolve(client, "t1", "n8n", {})
        time.sleep(0.01)
        await resolver.resolve(client, "t1", "n8n", {})
        assert client.get.call_count == 2

    async def test_409_raises(self):
        resolver = ActiveLicenseResolver()
        client = await self._mock_client(409, {"detail": "multiple"})
        with pytest.raises(httpx.HTTPStatusError):
            await resolver.resolve(client, "t1", "n8n", {})

    async def test_cache_max_entries_evicts_oldest(self):
        resolver = ActiveLicenseResolver(max_entries=1000)
        for i in range(1001):
            resolver._put((f"t{i}", "app"), {"license_id": f"L{i}"})
        assert len(resolver._cache) == 1000
        assert ("t0", "app") not in resolver._cache
        assert ("t1000", "app") in resolver._cache


# -- Internal helpers --------------------------------------------------------


class TestB64UrlDecodable:
    def test_valid_b64url(self):
        # "{}" → "e30"
        assert _b64url_decodable("e30") is True

    def test_invalid_returns_false(self):
        assert _b64url_decodable("$$$not-base64$$$") is False


class TestJwtIss:
    def test_returns_iss_string(self):
        payload = base64.urlsafe_b64encode(json.dumps({"iss": "x"}).encode()).rstrip(b"=").decode()
        assert _jwt_iss(payload) == "x"

    def test_returns_none_on_garbage(self):
        assert _jwt_iss("not-base64-$$$") is None

    def test_returns_none_when_iss_not_string(self):
        payload = base64.urlsafe_b64encode(json.dumps({"iss": 123}).encode()).rstrip(b"=").decode()
        assert _jwt_iss(payload) is None

    def test_returns_none_when_iss_missing(self):
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "u"}).encode()).rstrip(b"=").decode()
        assert _jwt_iss(payload) is None


# -- Real RS256 signing for verify_jwt tests ---------------------------------


@pytest.fixture(scope="module")
def rsa_key_pair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_key = key.public_key()
    return {"private_pem": private_pem, "public_key": public_key, "private_key": key}


def _sign(claims: dict, private_pem: bytes) -> str:
    return pyjwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": "test-kid"})


class TestVerifyJwt:
    def _stub_jwks(self, public_key) -> JwksCache:
        cache = JwksCache.__new__(JwksCache)
        signing_key = MagicMock()
        signing_key.key = public_key
        cache.get_signing_key = MagicMock(return_value=signing_key)
        return cache

    def test_happy_path(self, rsa_key_pair):
        claims = {
            "iss": KEYCLOAK_ISSUER,
            "exp": int(time.time()) + 600,
            "tenant_id": "tenant-x",
            "sub": "user-1",
        }
        token = _sign(claims, rsa_key_pair["private_pem"])
        jwks = self._stub_jwks(rsa_key_pair["public_key"])

        verified = verify_jwt(token, jwks)
        assert verified.tenant_id == "tenant-x"
        assert verified.sub == "user-1"
        assert verified.raw["iss"] == KEYCLOAK_ISSUER

    def test_expired_rejected(self, rsa_key_pair):
        claims = {
            "iss": KEYCLOAK_ISSUER,
            "exp": int(time.time()) - 60,
            "sub": "u",
        }
        token = _sign(claims, rsa_key_pair["private_pem"])
        jwks = self._stub_jwks(rsa_key_pair["public_key"])
        with pytest.raises(InvalidJWT):
            verify_jwt(token, jwks)

    def test_wrong_issuer_rejected(self, rsa_key_pair):
        claims = {
            "iss": "https://wrong.example",
            "exp": int(time.time()) + 600,
            "sub": "u",
        }
        token = _sign(claims, rsa_key_pair["private_pem"])
        jwks = self._stub_jwks(rsa_key_pair["public_key"])
        with pytest.raises(InvalidJWT):
            verify_jwt(token, jwks)

    def test_audience_mismatch_rejected(self, rsa_key_pair):
        claims = {
            "iss": KEYCLOAK_ISSUER,
            "exp": int(time.time()) + 600,
            "sub": "u",
            "aud": "right-aud",
        }
        token = _sign(claims, rsa_key_pair["private_pem"])
        jwks = self._stub_jwks(rsa_key_pair["public_key"])
        with pytest.raises(InvalidJWT):
            verify_jwt(token, jwks, audience=["wrong-aud"])

    def test_audience_match_accepted(self, rsa_key_pair):
        claims = {
            "iss": KEYCLOAK_ISSUER,
            "exp": int(time.time()) + 600,
            "sub": "u",
            "aud": "right-aud",
        }
        token = _sign(claims, rsa_key_pair["private_pem"])
        jwks = self._stub_jwks(rsa_key_pair["public_key"])
        verified = verify_jwt(token, jwks, audience=["right-aud"])
        assert verified.sub == "u"

    def test_signing_key_lookup_failure_propagates(self, rsa_key_pair):
        token = _sign(
            {"iss": KEYCLOAK_ISSUER, "exp": int(time.time()) + 60, "sub": "u"},
            rsa_key_pair["private_pem"],
        )
        jwks = JwksCache.__new__(JwksCache)
        jwks.get_signing_key = MagicMock(side_effect=pyjwt.exceptions.PyJWTError("no kid"))
        with pytest.raises(InvalidJWT):
            verify_jwt(token, jwks)


class TestJwksCacheRefresh:
    def test_get_signing_key_refreshes_on_kid_miss(self):
        """First lookup raises PyJWKClientError → _refresh() called → second succeeds."""
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        signing_key = MagicMock()

        # First call raises (kid miss), second succeeds.
        first_client = MagicMock()
        first_client.get_signing_key_from_jwt.side_effect = pyjwt.exceptions.PyJWKClientError(
            "kid not found"
        )
        second_client = MagicMock()
        second_client.get_signing_key_from_jwt.return_value = signing_key

        cache._client = first_client

        with patch("jwt_mode.PyJWKClient", return_value=second_client) as mock_ctor:
            out = cache.get_signing_key("token-here")

        assert out is signing_key
        # The refreshed client carries a BOUNDED http timeout — PyJWKClient
        # defaults to 30s, long enough for an unresponsive Keycloak to pin a
        # worker thread past the kubelet's probe timeout.
        mock_ctor.assert_called_once_with(
            "http://test/jwks",
            cache_keys=True,
            lifespan=600,
            timeout=jwt_mode.JWKS_HTTP_TIMEOUT,
        )
        assert jwt_mode.JWKS_HTTP_TIMEOUT < 5.0, "JWKS fetch must be bounded below probe timeouts"
        # After refresh, the cache's client should be the new one.
        assert cache._client is second_client
        second_client.get_signing_key_from_jwt.assert_called_once_with("token-here")
        # Successfully resolving a signing key latches the readiness signal.
        assert cache.last_load_ok is True

    def test_refresh_increments_jwks_counter(self):
        """JwksCache._refresh bumps the success counter via main's hook."""
        # Importing main wires the counter hooks into jwt_mode.
        from prometheus_client import REGISTRY

        import main  # noqa: F401

        before = (
            REGISTRY.get_sample_value("license_authz_jwks_refresh_total", {"result": "success"})
            or 0.0
        )

        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        new_client = MagicMock()
        with patch("jwt_mode.PyJWKClient", return_value=new_client):
            cache._refresh()

        after = (
            REGISTRY.get_sample_value("license_authz_jwks_refresh_total", {"result": "success"})
            or 0.0
        )
        assert after == before + 1.0
        # The success it counted must correspond to a fetch it actually made.
        new_client.fetch_data.assert_called_once_with()

    def test_refresh_failure_increments_failure_counter(self):
        from prometheus_client import REGISTRY

        import main  # noqa: F401

        before = (
            REGISTRY.get_sample_value("license_authz_jwks_refresh_total", {"result": "failure"})
            or 0.0
        )

        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        # OSError simulates a network/socket failure during JWKS fetch — one of
        # the narrow exception types JwksCache._refresh now catches (others:
        # jwt.PyJWKClientError, ValueError).
        with patch("jwt_mode.PyJWKClient", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                cache._refresh()

        after = (
            REGISTRY.get_sample_value("license_authz_jwks_refresh_total", {"result": "failure"})
            or 0.0
        )
        assert after == before + 1.0

    def test_get_signing_key_happy_path_no_refresh(self):
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        signing_key = MagicMock()
        client = MagicMock()
        client.get_signing_key_from_jwt.return_value = signing_key
        cache._client = client

        with patch("jwt_mode.PyJWKClient") as mock_ctor:
            out = cache.get_signing_key("tok")

        assert out is signing_key
        mock_ctor.assert_not_called()


class TestJwksMetricReflectsRealFetches:
    """F4b: the refresh counter must describe fetches that actually happened.

    ``PyJWKClient.__init__`` performs ZERO network I/O — it validates the URI
    scheme and allocates caches, nothing more (verified against the installed
    pyjwt 2.13.0). The first version of ``_refresh`` counted a ``success`` for
    merely constructing one, so during a total Keycloak outage the failure
    counter never moved and any alert built on it was blind.
    """

    @staticmethod
    def _hook():
        """Replace the metrics hook with a recorder; returns the recorded list."""
        recorded: list[str] = []
        jwt_mode.set_metrics_hooks(on_jwks_refresh=recorded.append)
        return recorded

    def teardown_method(self):
        # Restore main's real counter hook for the rest of the suite.
        import main

        jwt_mode.set_metrics_hooks(on_jwks_refresh=main._bump_jwks_refresh)

    def test_constructing_a_client_alone_counts_nothing(self):
        """The pyjwt fact this bug rested on, asserted directly."""
        with patch("urllib.request.urlopen") as urlopen:
            jwt_mode.PyJWKClient("http://test/jwks", cache_keys=True, lifespan=600, timeout=3.0)
        urlopen.assert_not_called()

    def test_refresh_counts_success_only_for_a_real_fetch(self):
        recorded = self._hook()
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)

        new_client = MagicMock()
        with patch("jwt_mode.PyJWKClient", return_value=new_client):
            cache._refresh()

        assert new_client.fetch_data.call_count == 1, (
            "a 'success' was counted without any fetch being performed — "
            "PyJWKClient.__init__ does no network I/O"
        )
        assert recorded == ["success"]

    def test_refresh_counts_failure_when_the_fetch_fails(self):
        """During a Keycloak outage the FAILURE counter must move, not success."""
        recorded = self._hook()
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)

        broken = MagicMock()
        broken.fetch_data.side_effect = pyjwt.PyJWKClientConnectionError("connection refused")
        with patch("jwt_mode.PyJWKClient", return_value=broken):
            with pytest.raises(pyjwt.PyJWKClientError):
                cache._refresh()

        assert broken.fetch_data.call_count == 1
        assert recorded == ["failure"], (
            "an unreachable Keycloak was reported as a successful refresh — "
            "this is exactly what made the failure counter read zero during an outage"
        )
        assert cache.last_load_ok is False

    def test_failed_refresh_keeps_the_working_client_and_its_cached_keys(self):
        """A blip must not destroy the key material we are still serving from.

        Installing the replacement before proving it can fetch threw away the
        populated JWK-set cache and per-kid LRU on the FIRST kid miss during an
        outage, permanently breaking a pod that had been answering correctly.
        """
        recorded = self._hook()
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        working = MagicMock(name="working-client-with-cached-keys")
        cache._client = working

        broken = MagicMock()
        broken.fetch_data.side_effect = OSError("connection refused")
        with patch("jwt_mode.PyJWKClient", return_value=broken):
            with pytest.raises(OSError):
                cache._refresh()

        assert cache._client is working
        assert recorded == ["failure"]

    def test_cached_kid_resolution_does_not_assert_fetch_health(self):
        """A cache hit does no network I/O, so it proves nothing about Keycloak.

        Same fabrication as F4b: if ordinary traffic could clear the live
        signal, any amount of JWT traffic would paper over an ongoing outage.
        """
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        cache.last_load_ok = False  # a real fetch just failed
        cache.ever_loaded_ok = True

        client = MagicMock()
        client.get_signing_key_from_jwt.return_value = MagicMock()
        cache._client = client
        cache.get_signing_key("tok")

        assert cache.last_load_ok is False, (
            "an LRU/JWK-set cache hit performed no fetch — it must not claim the "
            "JWKS endpoint is reachable"
        )
        assert cache.ever_loaded_ok is True, "but it does prove key material is present"

    def test_successful_fetch_sets_both_latches(self):
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        cache._client = MagicMock()
        cache.fetch()
        assert cache.last_load_ok is True
        assert cache.ever_loaded_ok is True

    def test_failed_boot_fetch_leaves_ever_loaded_false(self):
        cache = JwksCache(jwks_url="http://test/jwks", ttl=600)
        client = MagicMock()
        client.fetch_data.side_effect = pyjwt.PyJWKClientConnectionError("refused")
        cache._client = client
        with pytest.raises(pyjwt.PyJWKClientError):
            cache.fetch()
        assert cache.last_load_ok is False
        assert cache.ever_loaded_ok is False, "no key material was ever loaded — readiness gates"


# === Round 8: audience verification, behind a default-off flag ===============
#
# `verify_jwt` hard-disabled `aud` verification (`options={"verify_aud": False}`).
# JWT_VERIFY_AUDIENCE arms it. It DEFAULTS OFF because nothing in this repo
# records which audience real tokens carry — the only `aud` values here are a
# synthetic test fixture and the audience license-authz mints on its OWN internal
# tokens — so arming it blind 401s every JWT caller at once. The flag-off tests
# are the important ones: they pin that merging this changes no live decision.


class TestJwtVerifyAudienceFlag:
    def _stub_jwks(self, public_key) -> JwksCache:
        cache = JwksCache.__new__(JwksCache)
        signing_key = MagicMock()
        signing_key.key = public_key
        cache.get_signing_key = MagicMock(return_value=signing_key)
        return cache

    def _token(self, rsa_key_pair, **extra) -> str:
        claims = {"iss": KEYCLOAK_ISSUER, "exp": int(time.time()) + 600, "sub": "u"}
        claims.update(extra)
        return _sign(claims, rsa_key_pair["private_pem"])

    def test_defaults_off(self):
        """Default-on here is a fleet-wide 401. Pin the default itself."""
        assert jwt_mode.JWT_VERIFY_AUDIENCE is False
        assert jwt_mode.JWT_EXPECTED_AUDIENCES == ()

    def test_flag_off_ignores_a_mismatched_audience(self, rsa_key_pair, monkeypatch):
        """FLAG OFF: today's behaviour — `aud` is not looked at, whatever it says."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", False)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        token = self._token(rsa_key_pair, aud="something-else-entirely")

        verified = verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"]))

        assert verified.sub == "u"

    def test_flag_off_accepts_a_token_with_no_audience_at_all(self, rsa_key_pair, monkeypatch):
        """FLAG OFF: a token carrying no `aud` must still verify (the live shape)."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", False)
        token = self._token(rsa_key_pair)

        assert verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"])).sub == "u"

    def test_flag_on_rejects_a_mismatched_audience(self, rsa_key_pair, monkeypatch):
        """FLAG ON: an `aud` outside the configured list is now InvalidJWT → 401."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        token = self._token(rsa_key_pair, aud="something-else-entirely")

        with pytest.raises(InvalidJWT):
            verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"]))

    def test_flag_on_accepts_a_matching_audience(self, rsa_key_pair, monkeypatch):
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud", "other-aud"))
        token = self._token(rsa_key_pair, aud="other-aud")

        assert verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"])).sub == "u"

    def test_flag_on_rejects_a_token_with_no_audience(self, rsa_key_pair, monkeypatch):
        """⚠️ THE LOCKOUT SHAPE, pinned so nobody arms this without measuring.

        If real portal tokens carry no `aud` at all, arming the flag rejects every
        one of them. This test is the demonstration, not a wish.
        """
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        token = self._token(rsa_key_pair)

        with pytest.raises(InvalidJWT):
            verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"]))

    def test_an_explicit_caller_audience_still_wins_over_the_flag(self, rsa_key_pair, monkeypatch):
        """`_extract_caller_identity` passes audience=None DELIBERATELY, and a call
        site that passes a list must keep controlling its own verification."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("flag-aud",))
        token = self._token(rsa_key_pair, aud="call-site-aud")

        verified = verify_jwt(
            token, self._stub_jwks(rsa_key_pair["public_key"]), audience=["call-site-aud"]
        )
        assert verified.sub == "u"

    def test_arming_the_flag_also_affects_the_caller_identity_path(self, rsa_key_pair, monkeypatch):
        """⚠️ `audience=None` IS NOT AN OPT-OUT — pinned, because it looks like one.

        `_extract_caller_identity` passes `audience=None` explicitly, which is
        indistinguishable from passing nothing, so arming JWT_VERIFY_AUDIENCE arms
        audience verification on the SECURE-APPS paths too. Those tokens come from a
        different realm and may carry a different `aud`, so one list may not satisfy
        both. This test documents the coupling rather than hiding it.
        """
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", True)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("portal-aud",))
        token = self._token(rsa_key_pair, aud="secure-apps-aud")

        with pytest.raises(InvalidJWT):
            verify_jwt(token, self._stub_jwks(rsa_key_pair["public_key"]), audience=None)

    def test_arming_with_an_empty_list_is_refused_at_boot(self):
        """An armed flag with no audiences rejects EVERY token (PyJWT: audience=[]).

        That must fail the pod at boot, not silently 401 the fleet. Exercises the
        real guard function the module calls at import — deliberately NOT via
        `importlib.reload(jwt_mode)`, which would reset this module's metrics hooks
        to their no-ops and unwire main.py's counters for the rest of the session.
        """
        with pytest.raises(RuntimeError, match="JWT_EXPECTED_AUDIENCES is empty"):
            jwt_mode._validate_audience_config(True, ())

    def test_the_guard_permits_every_other_combination(self):
        """The guard must refuse ONLY armed-and-empty; anything else boots."""
        assert jwt_mode._validate_audience_config(False, ()) is None
        assert jwt_mode._validate_audience_config(False, ("a",)) is None
        assert jwt_mode._validate_audience_config(True, ("a",)) is None

    def test_the_guard_actually_runs_at_import(self):
        """A guard nobody calls is not a guard. Pin the call site's existence."""
        import inspect

        source = inspect.getsource(jwt_mode)
        assert "\n_validate_audience_config(JWT_VERIFY_AUDIENCE, JWT_EXPECTED_AUDIENCES)" in source


# === Round 8: the tenant claim is configurable, and the `sub` fallback counts ==
#
# `extract_tenant_id` read a hardcoded `tenant_id` and fell back to `sub`. The
# fallback silently turns a per-USER Keycloak principal into a per-TENANT billing
# identity (two users at one customer bill as two tenants) and is the firing
# LicenseTenantIdIsKeycloakPrincipal alert. ⛔ It is KEPT: neither realm emits a
# tenant claim, so removing it 401s every caller. What is added is the
# measurement that makes removal decidable.


class TestTenantClaimName:
    def test_default_claim_name_is_tenant_id(self):
        assert jwt_mode.TENANT_CLAIM == "tenant_id"

    def test_default_behaviour_is_unchanged(self):
        """FLAG OFF (default claim name): identical to the pre-change function."""
        assert extract_tenant_id({"tenant_id": "gerard-dev", "sub": "u"}) == "gerard-dev"
        assert extract_tenant_id({"sub": "uuid-xyz"}) == "uuid-xyz"
        assert extract_tenant_id({}) == "anonymous"
        assert extract_tenant_id({"tenant_id": "", "sub": "uuid"}) == "uuid"

    def test_a_configured_claim_name_is_read(self, monkeypatch):
        monkeypatch.setattr(jwt_mode, "TENANT_CLAIM", "org_id")
        assert extract_tenant_id({"org_id": "acme", "sub": "u"}) == "acme"

    def test_a_configured_claim_name_does_not_read_the_old_one(self, monkeypatch):
        """Once TENANT_CLAIM moves, `tenant_id` is just another claim — it must not
        keep being honoured, or the rename silently does nothing."""
        monkeypatch.setattr(jwt_mode, "TENANT_CLAIM", "org_id")
        assert extract_tenant_id({"tenant_id": "stale", "sub": "u"}) == "u"

    def test_sub_fallback_survives_a_claim_rename(self, monkeypatch):
        """⛔ The fallback must stay reachable under any claim name — removing it
        returns "anonymous", which the caller turns into a 401 for everyone."""
        monkeypatch.setattr(jwt_mode, "TENANT_CLAIM", "org_id")
        assert extract_tenant_id({"sub": "uuid-xyz"}) == "uuid-xyz"

    def test_a_non_string_claim_falls_back(self, monkeypatch):
        """A claim present but not a usable string is an ABSENT claim, not a tenant."""
        monkeypatch.setattr(jwt_mode, "TENANT_CLAIM", "tenant_id")
        assert extract_tenant_id({"tenant_id": {"nested": "x"}, "sub": "u"}) == "u"
        assert extract_tenant_id({"tenant_id": 12345, "sub": "u"}) == "u"


class TestTenantClaimCounter:
    """All THREE outcomes are counted, not just the fallback.

    A fallback count with no denominator cannot answer "has the claim started
    arriving?" — which is the only question that decides when the fallback can be
    removed. So `tenant_claim` is the denominator and `absent` is the third state.
    """

    def _record(self, monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(jwt_mode, "_on_tenant_claim", seen.append)
        return seen

    def test_the_claim_path_is_counted_as_tenant_claim(self, monkeypatch):
        seen = self._record(monkeypatch)
        extract_tenant_id({"tenant_id": "acme", "sub": "u"})
        assert seen == ["tenant_claim"]

    def test_the_sub_fallback_is_counted_as_sub(self, monkeypatch):
        seen = self._record(monkeypatch)
        extract_tenant_id({"sub": "uuid-xyz"})
        assert seen == ["sub"], "the fallback fired and nothing measured it"

    def test_neither_present_is_counted_as_absent(self, monkeypatch):
        seen = self._record(monkeypatch)
        extract_tenant_id({})
        assert seen == ["absent"]

    def test_exactly_one_count_per_call(self, monkeypatch):
        """Double-counting would make the claim:sub ratio — the whole point —
        unreadable."""
        seen = self._record(monkeypatch)
        extract_tenant_id({"tenant_id": "acme"})
        extract_tenant_id({"sub": "u"})
        extract_tenant_id({})
        assert seen == ["tenant_claim", "sub", "absent"]

    def test_the_default_hook_is_a_noop_so_the_module_stands_alone(self):
        """jwt_mode must stay importable and unit-testable without Prometheus."""
        assert jwt_mode._noop_tenant_claim("sub") is None

    def test_verify_jwt_counts_through_extract(self, rsa_key_pair, monkeypatch):
        """The counter has to fire on the REAL request path, not only in a unit
        call of extract_tenant_id."""
        seen = self._record(monkeypatch)
        cache = JwksCache.__new__(JwksCache)
        signing_key = MagicMock()
        signing_key.key = rsa_key_pair["public_key"]
        cache.get_signing_key = MagicMock(return_value=signing_key)
        token = _sign(
            {"iss": KEYCLOAK_ISSUER, "exp": int(time.time()) + 600, "sub": "user-1"},
            rsa_key_pair["private_pem"],
        )

        verified = verify_jwt(token, cache)

        assert verified.tenant_id == "user-1"
        assert seen == ["sub"]


# === Audience shadow evaluation ==============================================
#
# The brief asked whether JWT_VERIFY_AUDIENCE's default could flip. It cannot:
# the only audience this repo records is INTERNAL_TOKEN_AUDIENCE (live value
# `internal`), which is the audience license-authz MINTS onto tokens it ISSUES
# from /internal-token — this service's OUTPUT. `verify_jwt`'s INPUT is a
# Keycloak portal JWT or a secure-apps SA token from a different issuer entirely,
# so `internal` is the wrong answer rather than a near miss, and arming on it
# would 401 every caller on all three call sites at once.
#
# So the default stays OFF and this counter measures what arming WOULD cost, per
# already-verified token. The flag becomes armable when `mismatch` and `absent`
# are both zero under real traffic across BOTH realms.


class TestAudienceShadowVerdict:
    def test_the_default_is_still_off(self):
        """⛔ The whole reason the shadow counter exists. If this ever passes
        while `JWT_EXPECTED_AUDIENCES` is empty, every JWT caller is 401ing."""
        assert jwt_mode.JWT_VERIFY_AUDIENCE is False
        assert jwt_mode.JWT_EXPECTED_AUDIENCES == ()

    def test_no_expected_list_reports_unconfigured(self):
        """THE LIVE STATE, and it must be its own verdict.

        Folding it into `mismatch` would read as "every token would break" and
        folding it into `match` as "every token is fine" — both are inventions.
        `unconfigured` is the instrument saying it has no hypothesis to test yet.
        """
        assert jwt_mode.audience_shadow_verdict({"aud": "anything"}) == "unconfigured"
        assert jwt_mode.audience_shadow_verdict({}) == "unconfigured"

    def test_absent_aud_is_its_own_verdict(self, monkeypatch):
        """A token with no `aud` WOULD be rejected once an audience is passed
        (PyJWT raises MissingRequiredClaim), so it cannot share with `mismatch`:
        the fixes are different — publish an aud vs. add one to the list."""
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        assert jwt_mode.audience_shadow_verdict({"sub": "u"}) == "absent"

    def test_a_matching_audience_reports_match(self, monkeypatch):
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud", "other-aud"))
        assert jwt_mode.audience_shadow_verdict({"aud": "other-aud"}) == "match"

    def test_a_list_aud_matches_on_any_element(self, monkeypatch):
        """PyJWT accepts on a NON-EMPTY INTERSECTION, so the shadow must too —
        an `all()` here would predict rejections that would not happen."""
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        assert jwt_mode.audience_shadow_verdict({"aud": ["nope", "expected-aud"]}) == "match"

    def test_a_non_matching_audience_reports_mismatch(self, monkeypatch):
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        assert jwt_mode.audience_shadow_verdict({"aud": "something-else"}) == "mismatch"
        assert jwt_mode.audience_shadow_verdict({"aud": ["a", "b"]}) == "mismatch"

    def test_a_malformed_aud_reports_mismatch(self, monkeypatch):
        """PyJWT rejects a non-string element too, so for the question being asked
        — "would arming reject this?" — malformed belongs with mismatch. ⛔ It must
        NOT read as `match`: that would predict a pass for a token that fails."""
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        assert jwt_mode.audience_shadow_verdict({"aud": [123, {"x": 1}]}) == "mismatch"
        assert jwt_mode.audience_shadow_verdict({"aud": 123}) == "mismatch"


class TestAudienceShadowIsRecordedByVerifyJwt:
    def _stub_jwks(self, public_key) -> JwksCache:
        cache = JwksCache.__new__(JwksCache)
        signing_key = MagicMock()
        signing_key.key = public_key
        cache.get_signing_key = MagicMock(return_value=signing_key)
        return cache

    def _token(self, rsa_key_pair, **extra) -> str:
        claims = {"iss": KEYCLOAK_ISSUER, "exp": int(time.time()) + 600, "sub": "u"}
        claims.update(extra)
        return _sign(claims, rsa_key_pair["private_pem"])

    def _record(self, monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(jwt_mode, "_on_audience_shadow", seen.append)
        return seen

    def test_an_unchecked_token_is_shadow_evaluated(self, rsa_key_pair, monkeypatch):
        """⛔ FAILS WHEN THE HOOK CALL IS REVERTED: `seen` stays empty.

        The flag is off, so this token's `aud` was NOT verified — and that is
        exactly the population arming the flag would start judging.
        """
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", False)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        seen = self._record(monkeypatch)

        verify_jwt(
            self._token(rsa_key_pair, aud="something-else"),
            self._stub_jwks(rsa_key_pair["public_key"]),
        )

        assert seen == ["mismatch"], "the token that arming would reject was not counted"

    def test_a_token_that_really_was_audience_checked_is_not_shadowed(
        self, rsa_key_pair, monkeypatch
    ):
        """⚠️ NEGATIVE CONTROL, and the one that keeps the number meaningful.

        A token whose audience WAS enforced already got a real verdict. Counting
        it here would mix a shadow with an enforcement, and the answer to "how
        many live tokens would arming break?" would be inflated by tokens that
        were never at risk.
        """
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        seen = self._record(monkeypatch)

        verify_jwt(
            self._token(rsa_key_pair, aud="call-site-aud"),
            self._stub_jwks(rsa_key_pair["public_key"]),
            audience=["call-site-aud"],
        )

        assert seen == [], "an enforced token was also counted as a shadow"

    def test_the_shadow_never_changes_the_decision(self, rsa_key_pair, monkeypatch):
        """NEGATIVE CONTROL: measuring must not start rejecting. A token the
        shadow calls `mismatch` must still verify while the flag is off."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", False)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))

        verified = verify_jwt(
            self._token(rsa_key_pair, aud="something-else"),
            self._stub_jwks(rsa_key_pair["public_key"]),
        )

        assert verified.sub == "u"

    def test_exactly_one_verdict_per_verification(self, rsa_key_pair, monkeypatch):
        """Double-counting would make the mismatch:match ratio — the thing the
        arming decision is read off — wrong in the unsafe direction."""
        monkeypatch.setattr(jwt_mode, "JWT_VERIFY_AUDIENCE", False)
        monkeypatch.setattr(jwt_mode, "JWT_EXPECTED_AUDIENCES", ("expected-aud",))
        seen = self._record(monkeypatch)
        jwks = self._stub_jwks(rsa_key_pair["public_key"])

        verify_jwt(self._token(rsa_key_pair, aud="expected-aud"), jwks)
        verify_jwt(self._token(rsa_key_pair), jwks)

        assert seen == ["match", "absent"]

    def test_the_default_hook_is_a_noop_so_the_module_stands_alone(self):
        """jwt_mode must stay importable and unit-testable without Prometheus."""
        assert jwt_mode._noop_audience_shadow("match") is None


class TestUnverifiedAzp:
    def test_reads_azp(self):
        from jwt_mode import unverified_azp

        tok = _make_jwt_like({"alg": "RS256"}, {"iss": "x", "azp": "w2v-backend"})
        assert unverified_azp(tok) == "w2v-backend"

    def test_absent_or_non_string_is_none(self):
        from jwt_mode import unverified_azp

        assert unverified_azp(_make_jwt_like({"alg": "RS256"}, {"iss": "x"})) is None
        assert unverified_azp(_make_jwt_like({"alg": "RS256"}, {"azp": 7})) is None

    def test_garbage_is_none(self):
        from jwt_mode import unverified_azp

        assert unverified_azp("opaque-licence-token") is None
        assert unverified_azp("a.b.c") is None
        # Non-dict JSON payload (a list): built at runtime, not a literal token.
        hdr = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(b'["list"]').rstrip(b"=").decode()
        assert unverified_azp(f"{hdr}.{body}.sig") is None
