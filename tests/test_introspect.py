"""Tests for introspect_license function, utility functions, and probe endpoints.

Covers the actual HTTP call to the licenses service including
success, non-200, timeout, and exception paths.

All external HTTP mocking uses respx at the transport level.
"""

import hashlib
import json
import logging
import time

import httpx
import pytest
import respx
from httpx import AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import IntrospectResult, hash_token, introspect_license, log_access

# Canonical tenant UUID — license-authz injects the introspected tenant_id
# directly as x-tenant-id; a non-UUID value is rejected at the boundary.
TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"

# -- Probe Endpoints ----------------------------------------------------------


class _PingableRedis:
    """Healthy Redis stub for the probe tests.

    ⚠️ Needed because `READINESS_REQUIRE_REDIS` now defaults ON: Redis is
    unreachable in the unit-test env, so without this `/ready` would 503 for a
    DEPENDENCY reason and these tests would stop testing what they are named for
    — that the probes need no AUTH. Substituting a 503-tolerant assertion instead
    would have made both tests unable to fail for their actual subject.
    """

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


async def test_probe_endpoints_never_return_auth_errors(client: AsyncClient, monkeypatch):
    """All probe endpoints must return 200 even without Authorization header.

    This verifies they're handled by dedicated FastAPI routes, not the
    catch-all ext_authz_check that would return 401 for missing tokens.
    """
    monkeypatch.setitem(main.app.state._state, "redis", _PingableRedis())
    for path in ["/health", "/healthz", "/ready", "/health/live"]:
        response = await client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code} without auth"
        body = response.json()
        assert "error" not in body, f"{path} returned error body without auth: {body}"


async def test_probe_endpoints_ignore_auth_header_if_present(client: AsyncClient, monkeypatch):
    """Probe endpoints must not validate tokens even if Authorization is present.

    If a probe request happens to carry an Authorization header (e.g., from
    Envoy injecting it), the probe must still return 200 -- it must NOT
    attempt introspection.
    """
    monkeypatch.setitem(main.app.state._state, "redis", _PingableRedis())
    for path in ["/health", "/healthz", "/ready", "/health/live"]:
        response = await client.get(path, headers={"Authorization": "Bearer garbage"})
        assert response.status_code == 200, (
            f"{path} returned {response.status_code} when auth header present"
        )


# -- introspect_license function tests ----------------------------------------


class TestIntrospectLicense:
    """Tests for the introspect_license function.

    Uses respx to intercept the actual httpx.AsyncClient.post() call
    that introspect_license makes, instead of patching httpx.AsyncClient.
    """

    @respx.mock
    async def test_successful_introspect(self, http_client):
        """Successful introspect returns active result with all fields."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "tenant-abc",
                "subject_id": "sub-123",
                "license_id": "lic-456",
                "app": {"id": "app-789"},
                "plan": {"id": "plan-012"},
                "reason": None,
            },
        )

        result = await introspect_license(http_client, "some-token")

        assert result.active is True
        assert result.tenant_id == "tenant-abc"
        assert result.subject_id == "sub-123"
        assert result.license_id == "lic-456"
        assert result.app_id == "app-789"
        assert result.plan_id == "plan-012"

        # Verify the correct token was sent in the POST body
        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "some-token"}

    @respx.mock
    async def test_successful_introspect_without_app_and_plan(self, http_client):
        """Successful introspect with no app or plan returns None for those fields."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "tenant-abc",
                "subject_id": "sub-123",
                "license_id": "lic-456",
            },
        )

        result = await introspect_license(http_client, "some-token")

        assert result.active is True
        assert result.app_id is None
        assert result.plan_id is None

        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "some-token"}

    @respx.mock
    async def test_introspect_inactive_token(self, http_client):
        """Introspect returning active=False is handled correctly."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": False,
                "reason": "token_revoked",
            },
        )

        result = await introspect_license(http_client, "revoked-token")

        assert result.active is False
        assert result.reason == "token_revoked"

        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "revoked-token"}

    @respx.mock
    async def test_introspect_non_200_status(self, http_client):
        """Non-200 response from introspect returns inactive result."""
        route = respx.post(INTROSPECT_URL).respond(500)

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_failed:500"
        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "some-token"}

    @respx.mock
    async def test_introspect_timeout(self, http_client):
        """Timeout during introspect returns inactive with timeout reason."""
        route = respx.post(INTROSPECT_URL).mock(side_effect=httpx.TimeoutException("timed out"))

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_timeout"
        assert route.called

    @respx.mock
    async def test_introspect_generic_exception(self, http_client):
        """Generic exception during introspect returns inactive with error reason."""
        route = respx.post(INTROSPECT_URL).mock(side_effect=httpx.ConnectError("refused"))

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_error"
        assert route.called

    @respx.mock
    async def test_timeout_after_success_serves_stale(self, http_client):
        """A timeout AFTER a prior successful introspect serves the last-known-good
        result instead of denying — the stale-while-error fix for the 401 flake.
        """
        good = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-stale", "license_id": "lic-1"}
        )
        first = await introspect_license(http_client, "stale-tok")
        assert first.active is True
        good.mock(side_effect=httpx.TimeoutException("timed out"))

        served = await introspect_license(http_client, "stale-tok")
        assert served.active is True  # stale-served, NOT a 403
        assert served.tenant_id == "t-stale"

    @respx.mock
    async def test_non_200_after_success_serves_stale(self, http_client):
        """A non-200 (e.g. 503) after a prior success also serves stale."""
        route = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-2", "license_id": "lic-2"}
        )
        assert (await introspect_license(http_client, "tok2")).active is True
        route.respond(503)

        served = await introspect_license(http_client, "tok2")
        assert served.active is True
        assert served.tenant_id == "t-2"

    @respx.mock
    async def test_definitive_inactive_not_served_stale(self, http_client):
        """A definitive 200 active=false denies even with a stale good present —
        a genuine revocation is honored, never masked by stale-while-error.
        """
        route = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-3", "license_id": "lic-3"}
        )
        assert (await introspect_license(http_client, "tok3")).active is True
        route.respond(200, json={"active": False, "reason": "revoked"})

        result = await introspect_license(http_client, "tok3")
        assert result.active is False  # denies, not stale
        assert result.reason == "revoked"

        # ...and the revocation must SURVIVE the next blip. Honoring it only on
        # the revocation request itself is worthless: the stale entry recorded
        # before the revocation is still there, so the very next transient
        # failure would serve the pre-revocation active=True and the revoked
        # credential works again for the rest of the stale TTL.
        route.mock(side_effect=httpx.TimeoutException("x"))
        after_blip = await introspect_license(http_client, "tok3")
        assert after_blip.active is False, (
            "revoked credential resurrected from the stale store by a transient failure"
        )
        assert after_blip.reason == "introspect_timeout"

    @respx.mock
    async def test_revocation_survives_repeated_transient_failures(self, http_client):
        """A revoked token stays denied across MANY consecutive blips.

        The definitive active=false must purge the last-known-good outright, not
        merely be preferred for one request — otherwise every subsequent licenses
        hiccup within the 15-minute stale TTL re-admits the revoked credential.
        """
        route = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-rev", "license_id": "lic-rev"}
        )
        assert (await introspect_license(http_client, "rev-tok")).active is True

        route.respond(200, json={"active": False, "reason": "revoked"})
        assert (await introspect_license(http_client, "rev-tok")).active is False

        # Three different transient failure shapes, all must stay denied.
        route.mock(side_effect=httpx.TimeoutException("x"))
        assert (await introspect_license(http_client, "rev-tok")).active is False
        route.mock(side_effect=httpx.ConnectError("refused"))
        assert (await introspect_license(http_client, "rev-tok")).active is False
        route.mock(side_effect=None, return_value=httpx.Response(503))
        assert (await introspect_license(http_client, "rev-tok")).active is False

    @respx.mock
    async def test_definitive_inactive_does_not_evict_other_tokens(self, http_client):
        """Revoking token A must not clear token B's last-known-good.

        The fix must be a precise per-key drop, not a cache-wide flush — a flush
        would reintroduce the spurious-401 flake the stale store exists to fix.
        """
        route = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-a", "license_id": "lic-a"}
        )
        assert (await introspect_license(http_client, "tok-a")).active is True
        route.respond(200, json={"active": True, "tenant_id": "t-b", "license_id": "lic-b"})
        assert (await introspect_license(http_client, "tok-b")).active is True

        # A is revoked (definitive), B is untouched.
        route.respond(200, json={"active": False, "reason": "revoked"})
        assert (await introspect_license(http_client, "tok-a")).active is False

        # B still gets stale-served through a blip...
        route.mock(side_effect=httpx.TimeoutException("x"))
        served = await introspect_license(http_client, "tok-b")
        assert served.active is True, "revoking A wrongly cleared B's last-known-good"
        assert served.tenant_id == "t-b"
        # ...while A stays denied.
        assert (await introspect_license(http_client, "tok-a")).active is False

    @respx.mock
    async def test_cold_cache_timeout_fails_closed(self, http_client):
        """A timeout with NO prior success still returns the transient deny."""
        respx.post(INTROSPECT_URL).mock(side_effect=httpx.TimeoutException("timed out"))
        result = await introspect_license(http_client, "never-seen")
        assert result.active is False
        assert result.reason == "introspect_timeout"

    @respx.mock
    async def test_introspect_404_status(self, http_client):
        """404 response from introspect returns inactive result."""
        route = respx.post(INTROSPECT_URL).respond(404)

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_failed:404"
        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "some-token"}

    @respx.mock
    async def test_introspect_with_app_none_explicitly(self, http_client):
        """Introspect with app=None explicitly in response."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "t",
                "app": None,
                "plan": None,
            },
        )

        result = await introspect_license(http_client, "tok")

        assert result.active is True
        assert result.app_id is None
        assert result.plan_id is None
        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "tok"}

    @respx.mock
    async def test_introspect_401_status(self, http_client):
        """401 from introspect service returns inactive result (fail-closed)."""
        route = respx.post(INTROSPECT_URL).respond(401)

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_failed:401"
        assert route.called

    @respx.mock
    async def test_introspect_empty_body_200(self, http_client):
        """200 with empty JSON body defaults active to False."""
        route = respx.post(INTROSPECT_URL).respond(200, json={})

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.tenant_id is None
        assert result.license_id is None
        assert route.called

    @respx.mock
    async def test_introspect_malformed_json(self, http_client):
        """Malformed JSON from introspect should return inactive (fail-closed)."""
        route = respx.post(INTROSPECT_URL).respond(200, content=b"not json")

        result = await introspect_license(http_client, "some-token")

        assert result.active is False
        assert result.reason == "introspect_error"
        assert route.called

    @respx.mock
    async def test_introspect_with_extra_fields_ignored(self, http_client):
        """Extra fields in introspect response should be silently ignored."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "t1",
                "license_id": "lic-1",
                "extra_field": "should_be_ignored",
                "another": 42,
            },
        )

        result = await introspect_license(http_client, "tok")

        assert result.active is True
        assert result.tenant_id == "t1"
        assert result.license_id == "lic-1"
        assert route.called

    @respx.mock
    async def test_introspect_with_nested_app_fields(self, http_client):
        """App object with extra nested fields should only extract id."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": "t1",
                "app": {"id": "app-1", "name": "MyApp", "version": "2.0"},
                "plan": {"id": "plan-1", "name": "Enterprise", "max_seats": 100},
            },
        )

        result = await introspect_license(http_client, "tok")

        assert result.active is True
        assert result.app_id == "app-1"
        assert result.plan_id == "plan-1"
        assert route.called


# -- stale-while-error store bounds -------------------------------------------


class TestStaleStoreBounded:
    """The stale store must not grow without limit.

    license-authz is the ext_authz callout on EVERY mesh request and callers
    mint fresh tokens (regenerate-token), so an unbounded dict keyed on token
    hash retains every one-shot token forever — TTL entries are only dropped
    lazily on a _stale_get of that same key, which never comes.
    """

    def test_stale_store_never_exceeds_max_entries(self):
        cap = main._INTROSPECT_STALE_MAX_ENTRIES
        for i in range(cap + 25):
            main._stale_put(f"key-{i}", IntrospectResult(active=True, tenant_id=f"t-{i}"))
            assert len(main._introspect_stale) <= cap, f"store exceeded cap at insert {i}"
        assert len(main._introspect_stale) == cap

    def test_stale_store_evicts_oldest_first(self):
        cap = main._INTROSPECT_STALE_MAX_ENTRIES
        for i in range(cap):
            main._stale_put(f"key-{i}", IntrospectResult(active=True, tenant_id=f"t-{i}"))
        assert "key-0" in main._introspect_stale

        main._stale_put("overflow", IntrospectResult(active=True, tenant_id="t-overflow"))

        assert "key-0" not in main._introspect_stale, "oldest entry was not evicted"
        assert "key-1" in main._introspect_stale
        assert "overflow" in main._introspect_stale
        assert len(main._introspect_stale) == cap

    def test_stale_put_touches_lru_so_refreshed_keys_survive(self):
        """A re-put moves the key to the end, so eviction is LRU-ish not FIFO."""
        cap = main._INTROSPECT_STALE_MAX_ENTRIES
        for i in range(cap):
            main._stale_put(f"key-{i}", IntrospectResult(active=True, tenant_id=f"t-{i}"))

        # key-0 is the oldest; refresh it, so key-1 becomes the eviction victim.
        main._stale_put("key-0", IntrospectResult(active=True, tenant_id="t-0-again"))
        main._stale_put("overflow", IntrospectResult(active=True, tenant_id="t-overflow"))

        assert "key-0" in main._introspect_stale, "refreshed key was evicted as if never touched"
        assert "key-1" not in main._introspect_stale
        assert main._stale_get("key-0").tenant_id == "t-0-again"

    def test_eviction_bumps_the_shared_cache_eviction_metric(self):
        cap = main._INTROSPECT_STALE_MAX_ENTRIES
        labels = {"cache": "introspect_stale"}
        before = REGISTRY.get_sample_value("license_authz_cache_evictions_total", labels) or 0.0

        for i in range(cap + 3):
            main._stale_put(f"metric-key-{i}", IntrospectResult(active=True))

        after = REGISTRY.get_sample_value("license_authz_cache_evictions_total", labels)
        assert after is not None, "introspect_stale never registered on the eviction counter"
        assert after == before + 3

    def test_stale_get_touch_keeps_a_token_alive_during_its_own_outage(self):
        """_stale_get's LRU touch is load-bearing, not decorative.

        While one token's introspect is failing, _stale_put never runs for it —
        it runs on SUCCESS only — so the stale read is the entry's only touch.
        Without it the token ages to the front of the eviction order and is
        dropped precisely while in active use, and the next blip denies it:
        exactly the spurious 401 the stale store exists to prevent.
        """
        cap = main._INTROSPECT_STALE_MAX_ENTRIES
        for i in range(cap):
            main._stale_put(f"key-{i}", IntrospectResult(active=True, tenant_id=f"t-{i}"))

        # key-0 is the oldest; being stale-served (a READ) must renew it.
        assert main._stale_get("key-0") is not None
        main._stale_put("overflow", IntrospectResult(active=True, tenant_id="t-overflow"))

        assert "key-0" in main._introspect_stale, (
            "stale-served token evicted during its own outage; _stale_get must LRU-touch"
        )
        assert "key-1" not in main._introspect_stale, "the untouched oldest entry should have gone"

    def test_stale_get_drops_the_entry_once_the_ttl_expires(self):
        """An expired entry is removed, not merely hidden from the caller."""
        main._stale_put("ttl-key", IntrospectResult(active=True, tenant_id="t-ttl"))
        assert main._stale_get("ttl-key").tenant_id == "t-ttl"

        # Age it past its TTL in place — deterministic, no global clock patching.
        _expiry, result = main._introspect_stale["ttl-key"]
        main._introspect_stale["ttl-key"] = (time.monotonic() - 1.0, result)

        assert main._stale_get("ttl-key") is None, "expired entry was still served"
        assert "ttl-key" not in main._introspect_stale, (
            "expired entry was hidden but left in the store, so it never frees its slot"
        )


# -- stale-while-error kill-switch (cap <= 0) ----------------------------------


class TestStaleStoreDisabled:
    """A cap of 0 or negative disables the store entirely.

    Setting a max-entries knob to 0 is the natural way an operator turns the
    stale-while-error behaviour off. It must be a clean disable, not a crash:
    an unguarded ``popitem(last=False)`` on an empty OrderedDict raises
    KeyError, which is NOT in introspect_license's except tuple, so it escapes
    and 500s every SUCCESSFUL introspect — a fleet-wide deny-all, because the
    gateway Lua fails closed.
    """

    @pytest.mark.parametrize("cap", [0, -5])
    @respx.mock
    async def test_non_positive_cap_does_not_crash_the_success_path(
        self, http_client, monkeypatch, cap
    ):
        monkeypatch.setattr(main, "_INTROSPECT_STALE_MAX_ENTRIES", cap)
        route = respx.post(INTROSPECT_URL).respond(
            200, json={"active": True, "tenant_id": "t-off", "license_id": "lic-off"}
        )

        # The success path must survive: this is where the KeyError fired.
        first = await introspect_license(http_client, "off-tok")
        assert first.active is True
        assert first.tenant_id == "t-off"
        assert len(main._introspect_stale) == 0, "store disabled but still recorded an entry"

        # Disabled => no fallback => a transient failure fails CLOSED, exactly
        # as it did before the stale store existed.
        route.mock(side_effect=httpx.TimeoutException("x"))
        after = await introspect_license(http_client, "off-tok")
        assert after.active is False
        assert after.reason == "introspect_timeout"

    @pytest.mark.parametrize("cap", [0, -5])
    def test_non_positive_cap_stops_serving_existing_entries_immediately(self, monkeypatch, cap):
        """The kill-switch takes effect at once, not after the TTL drains."""
        main._stale_put("live-key", IntrospectResult(active=True, tenant_id="t-live"))
        assert main._stale_get("live-key") is not None

        monkeypatch.setattr(main, "_INTROSPECT_STALE_MAX_ENTRIES", cap)
        assert main._stale_get("live-key") is None, "disabled store still served a stale entry"


# -- hash_token tests ---------------------------------------------------------


class TestHashToken:
    """Tests for the hash_token utility function."""

    def test_hash_token_returns_sha256_prefix(self):
        """hash_token should return sha256:<first 16 chars of hex digest>."""
        result = hash_token("test-token")
        assert result.startswith("sha256:")
        hex_part = result.split(":")[1]
        assert len(hex_part) == 16

    def test_hash_token_deterministic(self):
        """Same input produces same hash."""
        assert hash_token("abc") == hash_token("abc")

    def test_hash_token_different_inputs(self):
        """Different inputs produce different hashes."""
        assert hash_token("token-a") != hash_token("token-b")

    def test_hash_token_known_value(self):
        """Verify hash_token produces the correct SHA256 truncated output."""
        token = "known-test-token"
        expected_full_hex = hashlib.sha256(token.encode()).hexdigest()
        expected = f"sha256:{expected_full_hex[:16]}"
        assert hash_token(token) == expected

    def test_hash_token_empty_string(self):
        """Empty string should still produce a valid hash."""
        result = hash_token("")
        assert result.startswith("sha256:")
        expected_full_hex = hashlib.sha256(b"").hexdigest()
        assert result == f"sha256:{expected_full_hex[:16]}"

    def test_hash_token_never_contains_raw_token(self):
        """Hash output must never contain the raw token value."""
        token = "secret-license-token-value"
        result = hash_token(token)
        assert token not in result


# -- log_access tests ---------------------------------------------------------


class TestLogAccess:
    """Tests for the log_access function."""

    def test_log_access_with_reason(self, caplog):
        """log_access with reason should log a JSON entry containing the reason field."""
        with caplog.at_level(logging.INFO, logger="main"):
            log_access(
                license_id="lic-1",
                tenant_id="tenant-1",
                app_id="app-1",
                plan_id="plan-1",
                method="GET",
                path="/test",
                status="denied",
                token_hash="sha256:abc",
                source_ip="1.2.3.4",
                user_agent="test-agent",
                reason="expired",
            )

        # Find the log record that contains our JSON entry
        log_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                log_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(log_entries) == 1, f"Expected 1 JSON log entry, got {len(log_entries)}"
        entry = log_entries[0]

        assert entry["license_id"] == "lic-1"
        assert entry["tenant_id"] == "tenant-1"
        assert entry["app_id"] == "app-1"
        assert entry["plan_id"] == "plan-1"
        assert entry["method"] == "GET"
        assert entry["path"] == "/test"
        assert entry["status"] == "denied"
        assert entry["token_hash"] == "sha256:abc"
        assert entry["source_ip"] == "1.2.3.4"
        assert entry["user_agent"] == "test-agent"
        assert entry["reason"] == "expired"
        assert "timestamp" in entry

    def test_log_access_without_reason(self, caplog):
        """log_access without reason should log a JSON entry WITHOUT the reason field."""
        with caplog.at_level(logging.INFO, logger="main"):
            log_access(
                license_id=None,
                tenant_id=None,
                app_id=None,
                plan_id=None,
                method="POST",
                path="/check",
                status="allowed",
                token_hash="sha256:def",
                source_ip="5.6.7.8",
                user_agent="other-agent",
            )

        log_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                log_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(log_entries) == 1, f"Expected 1 JSON log entry, got {len(log_entries)}"
        entry = log_entries[0]

        assert entry["license_id"] is None
        assert entry["tenant_id"] is None
        assert entry["app_id"] is None
        assert entry["plan_id"] is None
        assert entry["method"] == "POST"
        assert entry["path"] == "/check"
        assert entry["status"] == "allowed"
        assert entry["token_hash"] == "sha256:def"
        assert entry["source_ip"] == "5.6.7.8"
        assert entry["user_agent"] == "other-agent"
        assert "reason" not in entry
        assert "timestamp" in entry


# -- End-to-end ext_authz tests with transport-level mocking -------------------


class TestExtAuthzWithIntrospect:
    """End-to-end tests for the ext_authz endpoint with respx transport-level mocking.

    These tests exercise the full request cycle: HTTP request with Bearer token ->
    token extraction -> actual introspect_license call (intercepted by respx) ->
    response parsing -> header propagation.
    """

    @respx.mock
    async def test_valid_token_with_tenant_id(self, client: AsyncClient):
        """Valid token with tenant_id returns correct headers."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": TENANT_UUID,
                "subject_id": "sub-1",
                "license_id": "lic-1",
                "app": {"id": "app-1"},
                "plan": {"id": "plan-1"},
            },
        )

        response = await client.get("/some/path", headers={"Authorization": "Bearer tok"})

        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"token": "tok"}

        assert response.status_code == 200
        assert response.headers.get("x-tenant-id") == TENANT_UUID
        assert response.headers.get("x-tenant-name") == ""
        assert response.headers.get("x-license-id") == "lic-1"
        assert response.headers.get("x-app-id") == "app-1"
        assert response.headers.get("x-plan-id") == "plan-1"

    @respx.mock
    async def test_valid_token_no_tenant_id_does_not_leak_subject_id(self, client: AsyncClient):
        """When tenant_id is None, the downstream x-tenant-name header MUST be
        empty/absent — it must NOT fall back to subject_id.

        Falling back to subject_id used to poison cost_events.tenant_id with
        user UUIDs and break per-tenant rollups (Phase F.1 fix).
        """
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": None,
                "subject_id": "subject-fallback",
                "license_id": "lic-2",
            },
        )

        response = await client.get("/test", headers={"Authorization": "Bearer tok"})

        assert response.status_code == 200
        # Empty string or absent is acceptable; the user UUID must NOT leak through.
        tenant_header = response.headers.get("x-tenant-name") or ""
        assert tenant_header == "", f"x-tenant-name leaked subject_id: got {tenant_header!r}"

    @respx.mock
    async def test_inactive_token_with_no_reason(self, client: AsyncClient):
        """Inactive token with no reason returns 'license_invalid' as error reason header."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": False,
                "reason": None,
            },
        )

        response = await client.get("/test", headers={"Authorization": "Bearer bad"})

        assert response.status_code == 403
        assert response.headers.get("x-error-reason") == "license_invalid"
        # Also verify the body contains null reason
        body = response.json()
        assert body["error"] == "license_invalid"
        assert body["reason"] is None

    @respx.mock
    async def test_request_with_custom_envoy_headers(self, client: AsyncClient, caplog):
        """Envoy pseudo-headers (:method, :path) are used when present."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={"active": True, "tenant_id": TENANT_UUID},
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get(
                "/actual-path",
                headers={
                    "Authorization": "Bearer tok",
                    ":method": "POST",
                    ":path": "/original-path",
                    ":authority": "my-service.example",
                },
            )

        assert response.status_code == 200

        # Verify that the logged method and path come from the pseudo-headers,
        # not the actual HTTP request method/path
        access_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                if "method" in entry and "path" in entry and "status" in entry:
                    access_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(access_entries) >= 1, "Expected at least one access log entry"
        logged = access_entries[0]
        assert logged["method"] == "POST", (
            f"Expected logged method 'POST' from :method pseudo-header, got '{logged['method']}'"
        )
        assert logged["path"] == "/original-path", (
            f"Expected logged path '/original-path' from :path pseudo-header, got '{logged['path']}'"
        )

    @respx.mock
    async def test_source_ip_from_x_forwarded_for(self, client: AsyncClient, caplog):
        """Source IP extraction from x-forwarded-for header works."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={"active": True, "tenant_id": TENANT_UUID},
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get(
                "/path",
                headers={
                    "Authorization": "Bearer tok",
                    "x-forwarded-for": "10.0.0.1",
                },
            )

        assert response.status_code == 200

        # Verify that the logged source_ip comes from x-forwarded-for
        access_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                if "source_ip" in entry and "status" in entry:
                    access_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(access_entries) >= 1, "Expected at least one access log entry"
        logged = access_entries[0]
        assert logged["source_ip"] == "10.0.0.1", (
            f"Expected source_ip '10.0.0.1' from x-forwarded-for, got '{logged['source_ip']}'"
        )

    @respx.mock
    async def test_multiple_http_methods(self, client: AsyncClient):
        """Various HTTP methods go through the ext_authz check."""
        route = respx.post(INTROSPECT_URL).respond(
            200,
            json={"active": True, "tenant_id": TENANT_UUID},
        )

        for method in ["post", "put", "delete", "patch"]:
            response = await getattr(client, method)(
                "/resource", headers={"Authorization": "Bearer tok"}
            )
            assert response.status_code == 200, f"Failed for {method.upper()}"
            assert response.json() == {"status": "ok"}, f"Wrong response body for {method.upper()}"
            assert response.headers.get("x-tenant-id") == TENANT_UUID, (
                f"Missing x-tenant-id for {method.upper()}"
            )
            assert response.headers.get("x-tenant-name") == "", (
                f"x-tenant-name should be empty for {method.upper()}"
            )

        # 4 methods should have each triggered an introspect call
        assert route.call_count == 4

    @respx.mock
    async def test_both_tenant_and_subject_none(self, client: AsyncClient):
        """When both tenant_id and subject_id are None, x-tenant-id is empty string."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": None,
                "subject_id": None,
                "license_id": "lic-99",
            },
        )

        response = await client.get("/test", headers={"Authorization": "Bearer tok"})

        assert response.status_code == 200
        assert response.headers.get("x-tenant-id") == ""

    @respx.mock
    async def test_denied_request_logged_with_reason(self, client: AsyncClient, caplog):
        """Denied requests should include the denial reason in the access log."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={"active": False, "reason": "quota_exceeded"},
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get("/api/resource", headers={"Authorization": "Bearer tok"})

        assert response.status_code == 403

        access_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                if "status" in entry and entry.get("status") == "denied_invalid_token":
                    access_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(access_entries) >= 1
        logged = access_entries[0]
        assert logged["status"] == "denied_invalid_token"
        assert logged["reason"] == "quota_exceeded"

    @respx.mock
    async def test_allowed_request_logged_without_reason(self, client: AsyncClient, caplog):
        """Allowed requests should be logged with status=allowed and no reason."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={"active": True, "tenant_id": TENANT_UUID, "license_id": "lic-1"},
        )

        with caplog.at_level(logging.INFO, logger="main"):
            response = await client.get("/api/resource", headers={"Authorization": "Bearer tok"})

        assert response.status_code == 200

        access_entries = []
        for record in caplog.records:
            try:
                entry = json.loads(record.message)
                if "status" in entry and entry.get("status") == "allowed":
                    access_entries.append(entry)
            except json.JSONDecodeError, TypeError:
                continue

        assert len(access_entries) >= 1
        logged = access_entries[0]
        assert logged["status"] == "allowed"
        assert "reason" not in logged

    @respx.mock
    async def test_missing_optional_header_fields_default_to_empty(self, client: AsyncClient):
        """When introspect returns no app_id/plan_id, headers should be empty strings."""
        respx.post(INTROSPECT_URL).respond(
            200,
            json={
                "active": True,
                "tenant_id": TENANT_UUID,
                "license_id": None,
            },
        )

        response = await client.get("/test", headers={"Authorization": "Bearer tok"})

        assert response.status_code == 200
        assert response.headers.get("x-license-id") == ""
        assert response.headers.get("x-app-id") == ""
        assert response.headers.get("x-plan-id") == ""
