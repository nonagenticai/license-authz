"""Tests for the /ready readiness probe's REAL dependency checks.

Every test here must be able to FAIL against the old hardcoded
``return {"status": "ready"}``. A readiness test that passes against the
literal is vacuous — that literal is the bug being fixed.

Structure:
  * gating failures  -> 503 + a useful body naming the failed dependency
  * fail-open deps   -> 200 "degraded", never gating (Redis)
  * liveness/health  -> still dependency-free literals (restart-storm guard)
  * cost guards      -> healthy probe does no JWKS I/O; unhealthy self-heal is
                        rate-limited; the whole check is time-bounded
"""

import asyncio
import json
import time
from unittest.mock import patch

import jwt
import pytest
import redis.exceptions
from httpx import ASGITransport, AsyncClient

import main
import readiness
from main import app, lifespan

# -- Fakes --------------------------------------------------------------------


class FakeJwks:
    """Stands in for jwt_mode.JwksCache with controllable load latches.

    Mirrors the real two-latch split: ``ever_loaded_ok`` (has this pod ever held
    key material — GATING) and ``last_load_ok`` (outcome of the most recent real
    fetch — REPORTED). ``ever_loaded_ok`` defaults to whatever ``last_load_ok``
    implies so callers that only care about the boot state stay readable.
    """

    def __init__(
        self,
        *,
        last_load_ok: bool | None = None,
        ever_loaded_ok: bool | None = None,
        fetch_succeeds: bool = False,
    ):
        self.last_load_ok = last_load_ok
        self.ever_loaded_ok = bool(last_load_ok) if ever_loaded_ok is None else ever_loaded_ok
        self.last_attempt_at: float | None = None
        self.fetch_calls = 0
        self._fetch_succeeds = fetch_succeeds

    def fetch(self) -> None:
        self.fetch_calls += 1
        self.last_attempt_at = time.monotonic()
        if not self._fetch_succeeds:
            self.last_load_ok = False
            raise jwt.PyJWKClientError("simulated JWKS fetch failure")
        self.last_load_ok = True
        self.ever_loaded_ok = True


class FakeRedis:
    """Stands in for redis.asyncio.Redis — PING only."""

    def __init__(self, *, exc: Exception | None = None, delay: float = 0.0):
        self._exc = exc
        self._delay = delay
        self.ping_calls = 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return True

    async def aclose(self) -> None:
        """The ASGI lifespan teardown closes state.redis."""
        return None


@pytest.fixture
async def healthy_client(monkeypatch):
    """Client whose gating dependencies are all satisfied.

    The default test env leaves JWT mode / secure-apps / the internal-token
    issuer switched off, so every optional gate is skipped; Redis is stubbed
    healthy so the steady state is a clean "ready".
    """
    transport = ASGITransport(app=app)
    async with lifespan(app):
        monkeypatch.setitem(app.state._state, "redis", FakeRedis())
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# -- Steady state -------------------------------------------------------------


async def test_ready_reports_individual_checks(healthy_client: AsyncClient):
    """Healthy /ready is 200 and enumerates the checks it actually ran.

    Fails against the old literal: that body had no `checks` key at all.
    """
    response = await healthy_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == "license-authz"
    assert "lifespan" in body["checks"]
    assert "redis" in body["checks"]
    assert body["checks"]["lifespan"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is True
    assert body["degraded"] == []


# -- Gating failures ----------------------------------------------------------


async def test_ready_503_when_lifespan_incomplete(healthy_client: AsyncClient, monkeypatch):
    """A pod whose lifespan never produced the shared clients must not serve."""
    monkeypatch.setitem(app.state._state, "service_auth", None)
    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "lifespan" in body["failed"]
    assert "service_auth" in body["checks"]["lifespan"]["detail"]


def test_check_lifespan_flags_missing_http_client():
    """Unit-level: the http_client arm of the same gate.

    Nulling http_client on the live app would break the ASGI lifespan teardown
    (it awaits http_client.aclose()), so that arm is asserted here instead.
    """

    class _State:
        http_client = None
        service_auth = object()

    result = readiness.check_lifespan(_State())
    assert result.ok is False
    assert result.gating is True
    assert "http_client" in result.detail


async def test_ready_503_when_portal_jwks_never_loaded(healthy_client: AsyncClient, monkeypatch):
    """JWT mode on + JWKS never loaded ⇒ every browser JWT 401s ⇒ shed this pod."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    fake = FakeJwks(last_load_ok=None)
    monkeypatch.setitem(app.state._state, "jwks", fake)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "jwks_portal" in body["failed"]
    assert body["checks"]["jwks_portal"]["gating"] is True
    assert "cannot be verified" in body["checks"]["jwks_portal"]["detail"]
    assert fake.fetch_calls == 1, "a never-loaded JWKS must attempt one self-heal fetch"


async def test_check_jwks_reports_never_loaded_while_rate_limited():
    """Unit-level: the never-attempted wording, with the re-fetch suppressed."""
    fake = FakeJwks(last_load_ok=None)
    fake.last_attempt_at = time.monotonic()  # inside the recheck window

    result = await readiness.check_jwks(fake, name="jwks_portal", required=True)
    assert result is not None
    assert result.ok is False
    assert result.gating is True
    assert "never loaded" in result.detail
    assert fake.fetch_calls == 0


async def test_ready_503_when_portal_jwks_last_fetch_failed(
    healthy_client: AsyncClient, monkeypatch
):
    """A failed BOOT fetch (attempted, not merely 'never tried') is also gating.

    ``ever_loaded_ok`` is still False here — the pod attempted a load and got
    nothing — so this is the zero-key-material state, not the post-boot
    degradation. Those are separated in the POST-BOOT section above.
    """
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    fake = FakeJwks(last_load_ok=False)
    fake.last_attempt_at = time.monotonic()  # inside the rate-limit window
    monkeypatch.setitem(app.state._state, "jwks", fake)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    assert "jwks_portal" in response.json()["failed"]
    assert fake.fetch_calls == 0, "re-fetch must be rate-limited, not fired every probe"


async def test_ready_503_when_jwt_mode_on_but_no_jwks_client(
    healthy_client: AsyncClient, monkeypatch
):
    """Flag on with no JWKS client at all is a misconfigured boundary."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setitem(app.state._state, "jwks", None)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    assert "jwks_portal" in response.json()["failed"]


async def test_ready_503_when_secure_apps_jwks_never_loaded(
    healthy_client: AsyncClient, monkeypatch
):
    """Without secure-apps JWKS every allowlisted service principal is denied."""
    monkeypatch.setattr(
        main._settings,
        "keycloak_secure_apps_issuer",
        "https://auth.nonagentic.ai/realms/secure-apps",
    )
    fake = FakeJwks(last_load_ok=None)
    monkeypatch.setitem(app.state._state, "caller_jwks", fake)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert "jwks_secure_apps" in body["failed"]


async def test_ready_503_when_internal_token_key_unparseable(
    healthy_client: AsyncClient, monkeypatch
):
    """Settings only assert the PEM is non-empty; a garbage PEM can never self-heal."""
    monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", True)

    def _boom():
        raise ValueError("could not deserialize key data")

    monkeypatch.setattr(main.internal_token, "jwks_document", _boom)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert "internal_token_key" in body["failed"]
    assert "ValueError" in body["checks"]["internal_token_key"]["detail"]


async def test_ready_200_when_internal_token_key_parses(healthy_client: AsyncClient, monkeypatch):
    """The internal-token gate must pass on a genuinely usable key."""
    monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", True)
    monkeypatch.setattr(
        main.internal_token,
        "jwks_document",
        lambda: {"keys": [{"kid": "internal-test", "kty": "RSA"}]},
    )

    response = await healthy_client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["internal_token_key"]["ok"] is True


async def test_ready_503_when_internal_token_key_yields_empty_jwks(
    healthy_client: AsyncClient, monkeypatch
):
    monkeypatch.setattr(main._settings, "internal_token_issuer_enabled", True)
    monkeypatch.setattr(main.internal_token, "jwks_document", lambda: {"keys": []})

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    assert "internal_token_key" in response.json()["failed"]


async def test_ready_body_names_every_failed_gate(healthy_client: AsyncClient, monkeypatch):
    """Multiple simultaneous gating failures are all reported, not just the first."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setitem(app.state._state, "jwks", FakeJwks(last_load_ok=False))
    monkeypatch.setitem(app.state._state, "service_auth", None)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    failed = set(response.json()["failed"])
    assert {"lifespan", "jwks_portal"} <= failed


# -- POST-BOOT JWKS failure: detected, reported, never gating -----------------
#
# The regression these cover: /ready only ever observed the BOOT fetch. After a
# healthy boot nothing could move the latch back, so if Keycloak died at T+1h
# the probe answered 200 "ready" forever while every browser JWT 401'd. Each
# test below fails against that behaviour, and the pair of controls at the end
# pins the two states that must NOT change.


@pytest.fixture
def booted_healthy_then(monkeypatch):
    """A pod that booted with a good JWKS; returns its (fake, flip-to-broken)."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    # 0.0 == "re-validate on every probe", so the test does not sleep 300s.
    monkeypatch.setattr(readiness, "JWKS_REVALIDATE_INTERVAL", 0.0)
    monkeypatch.setattr(readiness, "JWKS_RECHECK_INTERVAL", 0.0)
    fake = FakeJwks(last_load_ok=True, ever_loaded_ok=True, fetch_succeeds=True)
    monkeypatch.setitem(app.state._state, "jwks", fake)
    return fake


async def test_keycloak_dying_after_healthy_boot_is_detected(
    healthy_client: AsyncClient, booted_healthy_then: FakeJwks
):
    """THE REGRESSION. Healthy boot → Keycloak dies → /ready must stop saying "ready".

    Against the pre-fix code every one of the N probes below returned
    status "ready" with jwks_portal.ok True, because the latch could only move
    at boot. The pod is reported degraded instead — and stays 200, see the
    fail-shut assertion.
    """
    healthy = await healthy_client.get("/ready")
    assert healthy.status_code == 200
    assert healthy.json()["status"] == "ready"

    booted_healthy_then._fetch_succeeds = False  # Keycloak dies

    for probe in range(5):
        response = await healthy_client.get("/ready")
        body = response.json()
        assert response.status_code == 200, (
            f"probe {probe}: a SHARED Keycloak outage must not evict the ext_authz "
            "boundary — every replica would go NotReady at once and statusOnError 403 "
            "would take the whole mesh down"
        )
        assert body["status"] == "degraded", f"probe {probe}: /ready still claims to be healthy"
        assert body["checks"]["jwks_portal"]["ok"] is False
        assert body["checks"]["jwks_portal"]["gating"] is False
        assert body["degraded"] == ["jwks_portal"]
        assert "unreachable" in body["checks"]["jwks_portal"]["detail"]
    assert booted_healthy_then.fetch_calls == 6, (
        "the probe itself must keep re-validating; nothing else in the process "
        "re-fetches without JWT traffic, so no signal would exist otherwise"
    )


async def test_post_boot_jwks_failure_keeps_the_pod_in_the_endpoints(
    healthy_client: AsyncClient, booted_healthy_then: FakeJwks
):
    """Constraint check, stated on its own: degraded is 200, never 503.

    Inverting the latch instead of splitting it would pass the test above's
    "not ready" wording but fail this one, and would be a fleet-wide outage.
    """
    booted_healthy_then._fetch_succeeds = False
    response = await healthy_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert "failed" not in body, "a shared-dependency outage must never name a failed GATE"
    assert body["checks"]["jwks_portal"]["gating"] is False


async def test_post_boot_jwks_failure_recovers_without_a_restart(
    healthy_client: AsyncClient, booted_healthy_then: FakeJwks
):
    """Constraint 4: the probe drives recovery, with or without traffic."""
    booted_healthy_then._fetch_succeeds = False
    assert (await healthy_client.get("/ready")).json()["status"] == "degraded"

    booted_healthy_then._fetch_succeeds = True  # Keycloak returns
    recovered = await healthy_client.get("/ready")
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "ready"
    assert recovered.json()["checks"]["jwks_portal"]["ok"] is True


async def test_post_boot_jwks_failure_on_secure_apps_is_also_reported(
    healthy_client: AsyncClient, monkeypatch
):
    """The caller-identity JWKS gets the same split, not just the portal one."""
    monkeypatch.setattr(
        main._settings,
        "keycloak_secure_apps_issuer",
        "https://auth.nonagentic.ai/realms/secure-apps",
    )
    monkeypatch.setattr(readiness, "JWKS_REVALIDATE_INTERVAL", 0.0)
    fake = FakeJwks(last_load_ok=True, ever_loaded_ok=True, fetch_succeeds=False)
    monkeypatch.setitem(app.state._state, "caller_jwks", fake)

    response = await healthy_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["degraded"] == ["jwks_secure_apps"]
    assert body["checks"]["jwks_secure_apps"]["gating"] is False


async def test_degraded_transition_is_logged_because_metrics_are_unscrapable(
    healthy_client: AsyncClient, booted_healthy_then: FakeJwks, caplog
):
    """license-authz :9001 cannot be scraped (root-ns ambient STRICT), so the
    log line is the only channel that reaches an operator today."""
    booted_healthy_then._fetch_succeeds = False
    with caplog.at_level("INFO", logger="readiness"):
        await healthy_client.get("/ready")
        assert any("jwks_degraded" in r.message for r in caplog.records)
        assert any(r.levelname == "ERROR" for r in caplog.records)

        caplog.clear()
        booted_healthy_then._fetch_succeeds = True
        await healthy_client.get("/ready")
        assert any("jwks_recovered" in r.message for r in caplog.records)


# -- Controls: the two states that must NOT change ----------------------------


async def test_control_healthy_steady_state_stays_ready(
    healthy_client: AsyncClient, booted_healthy_then: FakeJwks
):
    """A reachable Keycloak stays plain 200 "ready" across many probes."""
    for _ in range(5):
        response = await healthy_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["degraded"] == []


async def test_control_boot_failure_is_still_503(healthy_client: AsyncClient, monkeypatch):
    """Never-loaded is a different failure and must keep gating.

    Zero key material means every JWT-mode answer is WRONG, and a sibling that
    booted while Keycloak was up is unaffected — so shedding this pod helps.
    """
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(readiness, "JWKS_RECHECK_INTERVAL", 0.0)
    fake = FakeJwks(last_load_ok=None, ever_loaded_ok=False, fetch_succeeds=False)
    monkeypatch.setitem(app.state._state, "jwks", fake)

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failed"] == ["jwks_portal"]
    assert body["checks"]["jwks_portal"]["gating"] is True


# -- Redis: GATING by default (direction changed) -----------------------------
#
# ⚠️⚠️ THESE TESTS USED TO ASSERT THE OPPOSITE, and the old text is kept in each
# docstring so the flip is visible in the diff rather than silently rewritten. A
# pod that cannot reach Redis cannot enforce a budget, cannot enforce a quota and
# cannot record a metering event; it now stops claiming Ready while that is true.
#
# ⛔ THE COST, which the revert exists for: Redis is SHARED, so an estate-wide
# outage takes EVERY replica NotReady at once, the ext_authz cluster drops to
# zero endpoints and the mesh fails shut. `READINESS_REQUIRE_REDIS=false`
# restores the previous behaviour byte-for-byte and is pinned below.


async def test_redis_down_is_unready(healthy_client: AsyncClient, monkeypatch):
    """WAS `test_redis_down_is_degraded_not_unready` (200 / gating False).

    ⛔ FAILS WHEN THE DEFAULT IS REVERTED, which is the point: flipping
    READINESS_REQUIRE_REDIS back cannot happen silently.
    """
    # ⛔ ARM IT EXPLICITLY. The module default is OFF (see
    # `test_require_redis_defaults_off`), so inheriting the default here would
    # make this test pass for the wrong reason the day someone flips it.
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    import redis.exceptions

    monkeypatch.setitem(
        app.state._state, "redis", FakeRedis(exc=redis.exceptions.ConnectionError("refused"))
    )
    response = await healthy_client.get("/ready")
    assert response.status_code == 503, "a pod that can enforce nothing still claimed Ready"
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["ok"] is False
    assert body["checks"]["redis"]["gating"] is True
    assert body["failed"] == ["redis"]


async def test_redis_absent_is_unready(healthy_client: AsyncClient, monkeypatch):
    """WAS `test_redis_absent_is_degraded_not_unready` (200 / report, don't gate).

    A pod that booted while Redis was down is the SINGLE-pod fault this default
    is for: shedding it lets a healthy sibling serve.
    """
    # ⛔ ARM IT EXPLICITLY. The module default is OFF (see
    # `test_require_redis_defaults_off`), so inheriting the default here would
    # make this test pass for the wrong reason the day someone flips it.
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    monkeypatch.setitem(app.state._state, "redis", None)
    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["ok"] is False
    assert "fail-open" in body["checks"]["redis"]["detail"]


async def test_the_old_non_gating_behaviour_is_exactly_what_the_revert_restores(
    healthy_client: AsyncClient, monkeypatch
):
    """THE REVERT, pinned as behaviour rather than as a flag read.

    This is byte-for-byte the assertion `test_redis_down_is_degraded_not_unready`
    used to make. If setting READINESS_REQUIRE_REDIS=false stops producing this,
    the escape hatch is broken and the estate-wide Redis outage has no way out
    short of a code change.
    """
    import redis.exceptions

    monkeypatch.setattr(readiness, "REQUIRE_REDIS", False)
    monkeypatch.setitem(
        app.state._state, "redis", FakeRedis(exc=redis.exceptions.ConnectionError("refused"))
    )
    response = await healthy_client.get("/ready")
    assert response.status_code == 200, "the revert no longer restores the fail-open behaviour"
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"]["gating"] is False
    assert body["degraded"] == ["redis"]
    assert "failed" not in body


async def test_ready_drives_the_redis_reconnect(healthy_client: AsyncClient, monkeypatch):
    """THE production-relevant test: /ready is what re-establishes Redis.

    Traffic is currently 100% allowed_service_principal, and that branch returns
    before any metering emit — so nothing on the request path touches Redis.
    /ready (every 10s) is the only driver there is; if it reads app.state
    directly instead of going through the accessor, a pod that booted while
    Redis was down never reconnects.
    """
    monkeypatch.setitem(app.state._state, "redis", None)
    app.state.redis_failures = 1
    app.state.redis_next_attempt_at = 0.0  # backoff window has elapsed
    reconnected = FakeRedis()

    with patch("redis.asyncio.from_url", return_value=reconnected) as from_url:
        response = await healthy_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["redis"]["ok"] is True, "the probe did not reconnect"
    assert body["status"] == "ready"
    assert from_url.call_count == 1
    assert app.state.redis is reconnected, "the reconnected client was not latched onto app.state"
    assert app.state.redis_failures == 0


async def test_ready_with_healthy_redis_makes_no_connect_attempt(healthy_client: AsyncClient):
    """Probe-cost guard: a connected pod must not rebuild a pool every 10s."""
    app.state.redis_next_attempt_at = 0.0  # window open — only the early return protects us

    with patch("redis.asyncio.from_url") as from_url:
        response = await healthy_client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["redis"]["ok"] is True
    from_url.assert_not_called()


async def test_ready_reconnect_failure_is_unready_but_keeps_retrying(
    healthy_client: AsyncClient, monkeypatch
):
    """WAS `test_ready_reconnect_failure_is_degraded_not_unready` (200).

    ⭐ AND IT PINS THE ARGUMENT THAT DIED WITH THE FLIP. The old default was
    defended as "gating is self-defeating: the probe IS the reconnect driver, so
    a gating failure evicts the pod that was healing itself". The eviction premise
    is wrong — a NotReady pod is removed from the Service ENDPOINTS, it is not
    killed and the kubelet keeps probing it. So the reconnect attempt must still
    happen on a gating failure, and that is what `from_url.call_count` measures
    here. If it ever stops, gating really would block recovery and the default
    must go back.
    """
    # ⛔ ARM IT EXPLICITLY. The module default is OFF (see
    # `test_require_redis_defaults_off`), so inheriting the default here would
    # make this test pass for the wrong reason the day someone flips it.
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    monkeypatch.setitem(app.state._state, "redis", None)
    app.state.redis_next_attempt_at = 0.0

    with patch(
        "redis.asyncio.from_url", side_effect=redis.exceptions.ConnectionError("refused")
    ) as from_url:
        response = await healthy_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["ok"] is False
    assert body["checks"]["redis"]["gating"] is True
    assert app.state.redis is None
    assert from_url.call_count == 1, "gating suppressed the reconnect — recovery is now blocked"


async def test_redis_detail_distinguishes_kill_switch_from_a_blip(monkeypatch):
    """A deliberate METERING_ENABLED=false must not read as a broken dependency.

    ⚠️ The gating assertion flipped with the default: a transient absence now
    gates, the kill switch still never does. That asymmetry is the carve-out, and
    it got MORE load-bearing with the flip — reaching it used to need two
    deliberate opt-ins, and now METERING_ENABLED=false alone is enough.
    """
    # ⛔ ARM IT EXPLICITLY. The module default is OFF (see
    # `test_require_redis_defaults_off`), so inheriting the default here would
    # make this test pass for the wrong reason the day someone flips it.
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    transient = await readiness.check_redis(None, metering_enabled=True)
    disabled = await readiness.check_redis(None, metering_enabled=False)

    assert transient.gating is True
    assert disabled.gating is False, "the kill switch became gating — every replica goes NotReady"
    assert "fail-open" in transient.detail and "fail-open" in disabled.detail
    assert "retries" in transient.detail
    assert "kill switch" in disabled.detail
    assert transient.detail != disabled.detail


async def test_redis_hang_is_bounded(healthy_client: AsyncClient, monkeypatch):
    """A hung Redis must time out inside our budget, not at the kubelet.

    WAS `test_redis_hang_is_bounded_and_non_gating`. The BOUND is the claim being
    tested and it is unchanged; only the status code moved with the default.
    """
    # ⛔ ARM IT EXPLICITLY. The module default is OFF (see
    # `test_require_redis_defaults_off`), so inheriting the default here would
    # make this test pass for the wrong reason the day someone flips it.
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    monkeypatch.setattr(readiness, "DEP_TIMEOUT", 0.05)
    monkeypatch.setitem(app.state._state, "redis", FakeRedis(delay=5.0))

    started = time.monotonic()
    response = await healthy_client.get("/ready")
    elapsed = time.monotonic() - started

    assert response.status_code == 503
    assert elapsed < 1.0, f"readiness took {elapsed:.2f}s — probe budget not honoured"
    body = response.json()
    assert body["checks"]["redis"]["ok"] is False
    assert "exceeded" in body["checks"]["redis"]["detail"]


async def test_ready_total_budget_below_probe_timeout():
    """Guard the invariant from the k8s probe config (timeoutSeconds: 5)."""
    assert readiness.TOTAL_BUDGET < 5.0
    assert readiness.DEP_TIMEOUT <= readiness.TOTAL_BUDGET


async def test_ready_returns_503_when_checks_exceed_total_budget(
    healthy_client: AsyncClient, monkeypatch
):
    """Overrunning the total budget yields a clean 503, never a kubelet timeout."""
    monkeypatch.setattr(readiness, "TOTAL_BUDGET", 0.05)
    monkeypatch.setattr(readiness, "DEP_TIMEOUT", 5.0)
    monkeypatch.setitem(app.state._state, "redis", FakeRedis(delay=5.0))

    response = await healthy_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["failed"] == ["timeout"]
    assert "exceeded" in body["detail"]


# -- Cost guards --------------------------------------------------------------


async def test_healthy_probe_jwks_io_is_capped_at_one_per_revalidate_window(
    healthy_client: AsyncClient, monkeypatch
):
    """Probes fire every 10s per replica — the healthy path must stay ~free.

    The healthy path is no longer ZERO network I/O: a post-boot Keycloak outage
    is undetectable on a replica with no JWT traffic unless the probe itself
    re-validates. The cost invariant is therefore "at most one fetch per
    JWKS_REVALIDATE_INTERVAL", not "never".
    """
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(readiness, "JWKS_REVALIDATE_INTERVAL", 300.0)
    fake = FakeJwks(last_load_ok=True, fetch_succeeds=True)
    fake.last_attempt_at = time.monotonic()  # booted moments ago
    monkeypatch.setitem(app.state._state, "jwks", fake)

    for _ in range(30):  # 30 probes == 5 minutes at periodSeconds: 10
        response = await healthy_client.get("/ready")
        assert response.status_code == 200
    assert fake.fetch_calls == 0, "healthy readiness must not re-fetch inside the window"


async def test_revalidate_interval_dwarfs_the_probe_period():
    """Guard the cost invariant against a careless default change.

    The live probe is periodSeconds: 10 across 2 replicas. Anything near that
    would turn the readiness probe into a Keycloak load generator.
    """
    assert readiness.JWKS_REVALIDATE_INTERVAL >= 60.0
    assert readiness.JWKS_REVALIDATE_INTERVAL > readiness.JWKS_RECHECK_INTERVAL


async def test_healthy_probe_refetches_once_the_window_elapses(
    healthy_client: AsyncClient, monkeypatch
):
    """The window is a rate limit, not an off switch — otherwise no detection."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(readiness, "JWKS_REVALIDATE_INTERVAL", 0.05)
    fake = FakeJwks(last_load_ok=True, fetch_succeeds=True)
    fake.last_attempt_at = time.monotonic()
    monkeypatch.setitem(app.state._state, "jwks", fake)

    assert (await healthy_client.get("/ready")).status_code == 200
    assert fake.fetch_calls == 0, "still inside the window"

    await asyncio.sleep(0.06)
    assert (await healthy_client.get("/ready")).status_code == 200
    assert fake.fetch_calls == 1, "window elapsed — the probe must re-validate"


async def test_unhealthy_jwks_self_heals_then_rate_limits(healthy_client: AsyncClient, monkeypatch):
    """NotReady means no traffic, so the probe itself must drive recovery — once per window."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(readiness, "JWKS_RECHECK_INTERVAL", 30.0)
    fake = FakeJwks(last_load_ok=None, fetch_succeeds=False)
    monkeypatch.setitem(app.state._state, "jwks", fake)

    first = await healthy_client.get("/ready")
    assert first.status_code == 503
    assert fake.fetch_calls == 1, "a never-loaded JWKS must trigger exactly one re-fetch"

    for _ in range(4):
        assert (await healthy_client.get("/ready")).status_code == 503
    assert fake.fetch_calls == 1, "re-fetch must be rate-limited to one per recheck window"


async def test_jwks_recovery_flips_pod_back_to_ready(healthy_client: AsyncClient, monkeypatch):
    """Once Keycloak returns, the pod rejoins the endpoints without a restart."""
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setattr(readiness, "JWKS_RECHECK_INTERVAL", 0.0)
    fake = FakeJwks(last_load_ok=None, fetch_succeeds=False)
    monkeypatch.setitem(app.state._state, "jwks", fake)

    assert (await healthy_client.get("/ready")).status_code == 503

    fake._fetch_succeeds = True
    recovered = await healthy_client.get("/ready")
    assert recovered.status_code == 200
    assert recovered.json()["checks"]["jwks_portal"]["ok"] is True


# -- Liveness / health must stay dumb ----------------------------------------


async def test_liveness_is_dependency_free(healthy_client: AsyncClient, monkeypatch):
    """Liveness must NOT check dependencies — that is a fleet-wide restart storm.

    Every gating dependency is broken here; /health/live must still be 200.
    """
    import redis.exceptions

    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setitem(app.state._state, "jwks", FakeJwks(last_load_ok=False))
    monkeypatch.setitem(app.state._state, "service_auth", None)
    monkeypatch.setitem(
        app.state._state, "redis", FakeRedis(exc=redis.exceptions.ConnectionError("x"))
    )

    response = await healthy_client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_health_and_healthz_stay_literal_when_dependencies_are_down(
    healthy_client: AsyncClient, monkeypatch
):
    """/health + /healthz are ext_authz-exempt paths other callers rely on.

    They must keep returning 200 regardless of dependency state; the readiness
    signal lives on /ready alone.
    """
    monkeypatch.setattr(main, "JWT_MODE_ENABLED", True)
    monkeypatch.setitem(app.state._state, "jwks", FakeJwks(last_load_ok=False))
    monkeypatch.setitem(app.state._state, "redis", None)

    health = await healthy_client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "service": "license-authz"}

    healthz = await healthy_client.get("/healthz")
    assert healthz.status_code == 200
    assert healthz.json() == {"status": "healthy"}


async def test_ready_needs_no_auth_even_when_unready(healthy_client: AsyncClient, monkeypatch):
    """/ready must stay on its own route, never fall through to ext_authz.

    A 503 from the readiness gate must not be mistaken for an auth failure: the
    body says not_ready, and an Authorization header changes nothing.
    """
    monkeypatch.setitem(app.state._state, "service_auth", None)
    response = await healthy_client.get("/ready", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "error" not in body, "a readiness failure must not look like an authz denial"


# -- Unit-level checks on the readiness helpers -------------------------------


async def test_check_jwks_skipped_when_feature_disabled():
    """A disabled feature contributes no check at all (no false gating)."""
    assert await readiness.check_jwks(None, name="jwks_portal", required=False) is None


def test_check_internal_token_key_skipped_when_issuer_disabled():
    assert readiness.check_internal_token_key(enabled=False, jwks_document=dict) is None


def test_build_report_gating_vs_soft():
    """build_report folds gating and non-gating failures differently."""
    gating_fail = readiness.CheckResult(name="g", ok=False, gating=True, detail="d")
    soft_fail = readiness.CheckResult(name="s", ok=False, gating=False, detail="d")
    ok = readiness.CheckResult(name="o", ok=True, gating=True, detail="d")

    status, body = readiness.build_report([ok, soft_fail])
    assert status == 200
    assert body["status"] == "degraded"
    assert body["degraded"] == ["s"]

    status, body = readiness.build_report([ok, gating_fail, soft_fail])
    assert status == 503
    assert body["failed"] == ["g"]
    assert body["degraded"] == ["s"]

    status, body = readiness.build_report([ok])
    assert status == 200
    assert body["status"] == "ready"


def test_ready_body_is_json_serialisable():
    """The endpoint json.dumps() the report — every field must survive it."""
    _, body = readiness.build_report(
        [readiness.CheckResult(name="x", ok=False, gating=True, detail="boom")]
    )
    assert json.loads(json.dumps(body))["failed"] == ["x"]


# -- READINESS_REQUIRE_REDIS --------------------------------------------------
#
# check_redis was unconditionally NON-GATING: a pod that could not reach Redis —
# and therefore could not enforce a single quota — kept reporting Ready and kept
# taking traffic. That is the defect, and it is now closed by DEFAULT.
#
# ⚠️⚠️ THE DEFAULT FLIPPED. `test_require_redis_defaults_off` below used to assert
# `is False` with the docstring "a default-on flip here is an outage". That
# warning is not wrong and it is not deleted — it is the cost, and the env var is
# now the REVERT rather than the arming switch. ⛔ Running this default in
# production requires the estate-wide case to be covered elsewhere (a
# PodDisruptionBudget / minAvailable that refuses to take the last replica out),
# because a SHARED Redis outage now takes every replica NotReady at once.
#
# Both directions are pinned below: the default gates, and the revert restores
# the old behaviour exactly.


async def test_require_redis_defaults_off():
    """The gating machinery exists and is unarmed, and BOTH halves are pinned.

    ⛔ FAILS IF THE DEFAULT IS ARMED IN SOURCE. Arming is a real decision with a
    measured outage mode — the live mesh sets `statusOnError: "403"` with no
    `failOpen`, so every replica going NotReady on a SHARED Redis outage
    hard-403s all four enforced namespaces — and it must not be possible to make
    it by editing one string.

    ⚠️ The PodDisruptionBudget does NOT make arming safe, and it exists
    (`license-authz-pdb`, minAvailable=1): a PDB governs VOLUNTARY disruption,
    not endpoint removal driven by a readiness probe.

    `test_redis_failure_is_gating_when_armed` pins the other direction, so this
    is not a test that merely freezes the status quo — the behaviour it gates is
    fully built and proven, just not switched on.
    """
    assert readiness.REQUIRE_REDIS is False


async def test_redis_failure_is_non_gating_with_flag_off(monkeypatch):
    """THE REVERT: every failure shape goes back to non-gating, byte-for-byte."""
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", False)

    absent = await readiness.check_redis(None, metering_enabled=True)
    exploding = await readiness.check_redis(
        FakeRedis(exc=redis.exceptions.ConnectionError("down")), metering_enabled=True
    )
    monkeypatch.setattr(readiness, "DEP_TIMEOUT", 0.05)
    hung = await readiness.check_redis(FakeRedis(delay=5.0), metering_enabled=True)

    for result in (absent, exploding, hung):
        assert result.ok is False
        assert result.gating is False, "flag off must not gate — this is the mesh-shut risk"


async def test_ready_stays_200_degraded_on_redis_failure_with_flag_off(
    healthy_client: AsyncClient, monkeypatch
):
    """FLAG OFF, end to end: a dead Redis is `degraded` at 200, never a 503."""
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", False)
    monkeypatch.setitem(
        app.state._state, "redis", FakeRedis(exc=redis.exceptions.ConnectionError("down"))
    )

    response = await healthy_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["degraded"] == ["redis"]
    assert body["checks"]["redis"]["gating"] is False


async def test_redis_failure_is_gating_when_armed(monkeypatch):
    """FLAG ON: each failure shape gates, so the pod can be shed."""
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)

    absent = await readiness.check_redis(None, metering_enabled=True)
    exploding = await readiness.check_redis(
        FakeRedis(exc=redis.exceptions.ConnectionError("down")), metering_enabled=True
    )
    monkeypatch.setattr(readiness, "DEP_TIMEOUT", 0.05)
    hung = await readiness.check_redis(FakeRedis(delay=5.0), metering_enabled=True)

    for result in (absent, exploding, hung):
        assert result.ok is False
        assert result.gating is True


async def test_ready_returns_503_on_redis_failure_when_armed(
    healthy_client: AsyncClient, monkeypatch
):
    """FLAG ON, end to end: the pod stops claiming Ready while it cannot enforce."""
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    monkeypatch.setitem(
        app.state._state, "redis", FakeRedis(exc=redis.exceptions.ConnectionError("down"))
    )

    response = await healthy_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failed"] == ["redis"]


async def test_healthy_redis_is_never_gating_in_either_mode(monkeypatch):
    """A PASSING check's `gating` flag must not depend on the flag.

    build_report only reads `gating` on failures, so flipping it on success is
    inert for the status code and only makes the /ready body's shape flag-
    dependent. Pinned so a later "consistency" tidy-up cannot introduce that.
    """
    for armed in (False, True):
        monkeypatch.setattr(readiness, "REQUIRE_REDIS", armed)
        result = await readiness.check_redis(FakeRedis(), metering_enabled=True)
        assert (result.ok, result.gating) == (True, False)


async def test_metering_kill_switch_is_never_gating_even_when_armed(monkeypatch):
    """⛔ The one combination that must not brick the auth boundary.

    METERING_ENABLED=false means there is no client BY INSTRUCTION and no retry
    will ever be attempted, so gating it would take every replica NotReady
    permanently — a config combination silently bricking ext_authz with no fault
    anywhere. Arming READINESS_REQUIRE_REDIS must not be able to do that.
    """
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)

    result = await readiness.check_redis(None, metering_enabled=False)

    assert result.ok is False
    assert result.gating is False, "the kill switch became gating — every replica goes NotReady"
    assert "kill switch" in result.detail


async def test_require_redis_is_read_per_call_not_captured_at_import(monkeypatch):
    """The flag is the switch an operator reaches for DURING an incident.

    If check_redis captured it at import, changing it would need a restart of
    every replica while the mesh is already failing shut.
    """
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", True)
    armed = await readiness.check_redis(None, metering_enabled=True)
    monkeypatch.setattr(readiness, "REQUIRE_REDIS", False)
    disarmed = await readiness.check_redis(None, metering_enabled=True)

    assert (armed.gating, disarmed.gating) == (True, False)
