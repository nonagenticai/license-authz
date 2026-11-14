"""Every dropped billable event, and every budget/quota fail-open, leaves a record.

WHY THIS FILE EXISTS
--------------------
Three paths in ``src/main.py`` discarded billable work or disengaged the
enforcement gate with **no counter and no alertable log**, which makes the loss
undetectable by construction rather than merely unnoticed:

  1. ``_emit_metering_event`` began ``if redis_client is None: return`` — a bare
     return on all five call sites. Every billable event during a Redis outage
     or reconnect-backoff window vanished with no signal whatsoever.
  2. The ``RedisError``/``OSError`` arm of the same function logged a WARNING
     but incremented nothing, so the loss could not be alerted on or counted,
     despite this module already registering ten Prometheus counters.
  3. The budget/quota check failed OPEN with no counter, so a gate that had
     stopped gating was indistinguishable from a gate finding every request
     under its limit.

⚠️ EVERY test here must FAIL against origin/main. Both new counters are absent
there, so ``REGISTRY.get_sample_value`` returns ``None`` and each
``+ 1`` assertion evaluates ``0 == 1``. A presence-check that passes before and
after the change would prove nothing.

⛔ SCOPE. These tests assert MEASUREMENT ONLY: failing closed denies live
customer traffic on an infrastructure fault and is an operator decision. Any
test added here that asserts a DENY on an infra fault is a policy change
wearing a test.
"""

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio
import redis.exceptions
from prometheus_client import REGISTRY

import main

DROPPED_METRIC = "license_authz_metering_events_dropped_total"
FAIL_OPEN_METRIC = "license_authz_budget_fail_open_total"

TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"

# The live fleet rules MeteringEventsDroppedNoTenant / MeteringEventsDroppedUnconfigured
# select `{__name__=~".*metering_events_dropped_total", reason="..."}` — see
# gitops-production clusters/strato/services/prometheus/rules-b.yaml. Renaming
# this metric out of that suffix unhooks it from the alert silently.
FLEET_DROP_SELECTOR = re.compile(r".*metering_events_dropped_total")


def _sample(metric: str, reason: str) -> float:
    return REGISTRY.get_sample_value(metric, {"reason": reason}) or 0.0


def _mock_redis(store: dict[str, str] | None = None) -> AsyncMock:
    mock = AsyncMock(spec=redis.asyncio.Redis)
    data = dict(store or {})

    async def _get(key: str) -> str | None:
        return data.get(key)

    mock.get = AsyncMock(side_effect=_get)
    mock.xadd = AsyncMock(return_value="stream-id")
    return mock


async def _emit(request_app, **overrides) -> None:
    kwargs = {
        "tenant_id": TENANT_UUID,
        "license_id": "lic_abc",
        "subject_id": "sub_123",
        "app_id": "app_workers",
        "plan_id": "premium",
        "method": "GET",
        "path": "/api/resource",
        "status": "allowed",
        "source_ip": "10.42.0.1",
    }
    kwargs.update(overrides)
    await main._emit_metering_event(SimpleNamespace(app=request_app, headers={}), **kwargs)


def _events(caplog, name: str) -> list[dict]:
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if name not in message:
            continue
        try:
            out.append(json.loads(message))
        except json.JSONDecodeError:
            continue
    return [e for e in out if e.get("event") == name]


# -- Defect 1: a null Redis client swallowed every event ----------------------


async def test_null_redis_client_counts_the_dropped_event(caplog):
    """`if redis_client is None: return` dropped a billable event with no signal.

    Fails on origin/main: the counter does not exist, so get_sample_value
    returns None and 0.0 != 1.0.
    """
    before = _sample(DROPPED_METRIC, "redis_unavailable")

    with patch("main._get_redis", AsyncMock(return_value=None)):
        with caplog.at_level("WARNING"):
            await _emit(main.app)

    assert _sample(DROPPED_METRIC, "redis_unavailable") == before + 1, (
        "a billable event was discarded because Redis was unavailable and "
        "nothing counted it — the loss is undetectable by construction"
    )


async def test_null_redis_client_logs_the_lost_event_at_warning(caplog):
    """The log line is the only surviving record of the event, so it must carry it.

    Fails on origin/main: that branch emitted no log record at all.
    """
    with patch("main._get_redis", AsyncMock(return_value=None)):
        with caplog.at_level("WARNING"):
            await _emit(main.app, status="denied_invalid_token")

    events = _events(caplog, "metering_event_dropped")
    assert events, "the dropped event produced no WARNING — nothing records it existed"
    event = events[-1]
    assert event["reason"] == "redis_unavailable"
    # Reconstructability: without these the line proves a loss but cannot say
    # whose, which is exactly the state the counter alone would leave us in.
    assert event["tenant_id"] == TENANT_UUID
    assert event["license_id"] == "lic_abc"
    assert event["app_id"] == "app_workers"
    assert event["path"] == "/api/resource"
    assert event["status"] == "denied_invalid_token"


async def test_warning_is_at_warning_not_debug(caplog):
    """`logger.debug` here would be a record nothing collects. Pin the level."""
    with patch("main._get_redis", AsyncMock(return_value=None)):
        with caplog.at_level("DEBUG"):
            await _emit(main.app)

    levels = {r.levelname for r in caplog.records if "metering_event_dropped" in r.getMessage()}
    assert levels == {"WARNING"}, f"expected WARNING only, got {levels or 'no record'}"


async def test_kill_switch_gets_its_own_reason_and_no_per_request_warning(caplog):
    """METERING_ENABLED=false is benign and permanently non-zero.

    It must NOT share a series with redis_unavailable, or the alert on the real
    defect has to be silenced wherever the kill switch is set — the same split
    karma_service records for `metering_unconfigured` vs `no_tenant`. And it
    must not log per request: it is a config state, not an incident.

    Fails on origin/main: neither reason series exists.
    """
    before_unconfigured = _sample(DROPPED_METRIC, "metering_unconfigured")
    before_unavailable = _sample(DROPPED_METRIC, "redis_unavailable")

    with patch("main.METERING_ENABLED", False):
        with patch("main._get_redis", AsyncMock(return_value=None)):
            with caplog.at_level("DEBUG"):
                await _emit(main.app)

    assert _sample(DROPPED_METRIC, "metering_unconfigured") == before_unconfigured + 1
    assert _sample(DROPPED_METRIC, "redis_unavailable") == before_unavailable, (
        "the deliberate kill switch was counted as a Redis outage — that series "
        "sits permanently non-zero and takes the real signal with it"
    )
    assert not _events(caplog, "metering_event_dropped"), (
        "the kill switch logged per request; that is one WARNING per request for "
        "the life of the deployment"
    )


# -- Defect 2: RedisError warned but never counted ----------------------------


@pytest.mark.parametrize(
    "exc",
    [
        redis.exceptions.ConnectionError("connection refused"),
        redis.exceptions.TimeoutError("timed out"),
        OSError("socket died"),
    ],
    ids=["connection_error", "timeout", "oserror"],
)
async def test_xadd_failure_is_counted_not_only_logged(exc, caplog):
    """origin/main logged `metering_emit_failed` and incremented nothing.

    Fails on origin/main: the counter does not exist. The pre-existing WARNING
    is deliberately still asserted so this test also pins that the log was not
    traded away for the counter.
    """
    before = _sample(DROPPED_METRIC, "redis_error")

    failing = _mock_redis()
    failing.xadd = AsyncMock(side_effect=exc)

    with patch("main._get_redis", AsyncMock(return_value=failing)):
        with caplog.at_level("WARNING"):
            await _emit(main.app)

    assert _sample(DROPPED_METRIC, "redis_error") == before + 1, (
        f"{type(exc).__name__} on XADD lost a billable event with no counter"
    )
    events = _events(caplog, "metering_emit_failed")
    assert events, "the pre-existing WARNING was lost"
    assert events[-1]["tenant_id"] == TENANT_UUID


async def test_emit_failure_never_raises_into_the_authz_path():
    """Instrumentation must not turn a metering fault into an ext_authz 500.

    Envoy converts a 500 from ext_authz into a DENY for every request, so this
    is the invariant the whole `never raises` docstring exists to protect.
    """
    failing = _mock_redis()
    failing.xadd = AsyncMock(side_effect=redis.exceptions.ConnectionError("boom"))

    with patch("main._get_redis", AsyncMock(return_value=failing)):
        await _emit(main.app)  # must not raise


# -- The wiring to the fleet alert -------------------------------------------


def test_drop_metric_name_matches_the_live_fleet_alert_selector():
    """Renaming out of the `.*metering_events_dropped_total` suffix unhooks the alert.

    The rules in gitops-production select by permissive `__name__` regex, not by
    a bare metric name, precisely so a new emitter needs no second rule. A
    rename would keep every test here green while making the series unalertable.
    """
    assert FLEET_DROP_SELECTOR.fullmatch(DROPPED_METRIC)


def test_every_reason_series_is_born_at_zero():
    """A labelled Counter exports NO series until its first .inc().

    Without birth-at-import, "nothing was dropped" and "this build has no
    instrument" are the same empty vector, and an operator querying the series
    reads absence as health.

    Fails on origin/main: neither metric exists, so every lookup returns None.
    """
    for reason in ("redis_unavailable", "redis_error", "metering_unconfigured"):
        assert REGISTRY.get_sample_value(DROPPED_METRIC, {"reason": reason}) is not None, (
            f"{DROPPED_METRIC}{{reason={reason}}} has no series until it first fires"
        )
    for reason in ("no_license_id", "no_tenant_id", "metering_http_error"):
        assert REGISTRY.get_sample_value(FAIL_OPEN_METRIC, {"reason": reason}) is not None, (
            f"{FAIL_OPEN_METRIC}{{reason={reason}}} has no series until it first fires"
        )


# === Round 8: the fail-open gaps this instrument still had ====================
#
# `license_authz_budget_fail_open_total` reads 0 in production. Round 8 asked
# whether that is health or a gap. BOTH, in different places:
#
#   * It read 0 because the check was never CALLED: 100% of traffic takes the
#     service-principal fast path, and BUDGET_ENFORCEMENT_ENABLED is false
#     besides. Neither of those skips emitted anything at all — the first gap.
#   * The billing HTTP checks fail open on transport AND shape errors
#     while counting only `metering_lookup{result="error"|"contract_error"}` —
#     which names the transport defect, not the ALLOW it produced. Second gap.

SKIPPED_METRIC = "license_authz_budget_enforcement_skipped_total"


def _skipped(reason: str) -> float:
    return REGISTRY.get_sample_value(SKIPPED_METRIC, {"reason": reason}) or 0.0


def test_the_skipped_metric_is_separate_from_the_fail_open_family():
    """⛔ A PERMANENT config state must not share a series with the defect reasons.

    BUDGET_ENFORCEMENT_ENABLED=false is the DEFAULT, so folding it into
    `budget_fail_open` would pin any `> 0` alert on that metric firing forever and
    force it to be silenced — taking the real defect reasons down with it.
    Same reasoning `metric_metering_events_dropped` records for keeping
    `metering_unconfigured` off its two loss reasons.
    """
    assert SKIPPED_METRIC != FAIL_OPEN_METRIC
    for reason in ("enforcement_disabled", "no_license_id"):
        assert REGISTRY.get_sample_value(SKIPPED_METRIC, {"reason": reason}) is not None, (
            f"{SKIPPED_METRIC}{{reason={reason}}} has no series until it first fires"
        )


# ⚠️ The two skip arms are exercised END TO END in tests/test_budget.py
# (test_a_disabled_gate_is_counted / test_an_active_licence_with_no_id_is_counted),
# driving the real handler. A test that re-implemented the call-site `if` here
# would be grading a copy of the predicate and could not fail for a change to the
# real one.


def test_the_metering_http_fail_open_reason_exists_and_is_born_at_zero():
    """The billing HTTP check's fail-open allow has a name on the right instrument.

    It is on `budget_fail_open` rather than a new metric because it IS a defect
    class fail-open — the gate disengaged on a fault — unlike the two permanent
    config states above.
    """
    assert (
        REGISTRY.get_sample_value(FAIL_OPEN_METRIC, {"reason": "metering_http_error"}) is not None
    )


# === Round 8: the tenant-claim source counter ================================


TENANT_CLAIM_METRIC = "license_authz_tenant_claim_total"


def test_every_tenant_claim_source_is_born_at_zero():
    """All three sources, so the claim:sub RATIO is readable from the first scrape.

    A fallback count with no denominator cannot answer "has the tenant claim
    started arriving?", which is the only question that decides when the `sub`
    fallback can be removed.
    """
    for source in ("tenant_claim", "sub", "absent"):
        assert REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": source}) is not None, (
            f"{TENANT_CLAIM_METRIC}{{source={source}}} has no series until it first fires"
        )


def test_the_tenant_claim_hook_is_actually_wired_into_jwt_mode():
    """⛔ A counter that exists but is never incremented reads 0 and looks healthy.

    jwt_mode's hook defaults to a NO-OP, so a missing `set_metrics_hooks` wiring
    would leave this metric permanently at 0 with every unit test still green —
    the exact failure this whole file exists to prevent. Drive the real extractor
    and require the real counter to move.
    """
    import jwt_mode
    import main  # noqa: F401  — importing main is what performs the wiring

    before = REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "sub"}) or 0.0

    assert jwt_mode.extract_tenant_id({"sub": "uuid-xyz"}) == "uuid-xyz"

    after = REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "sub"})
    assert after == before + 1, "jwt_mode's tenant-claim hook is not wired to main's counter"


def test_the_claim_path_and_the_fallback_are_separate_series():
    """Folding them would destroy the only signal that says the fallback is removable."""
    import jwt_mode
    import main  # noqa: F401

    before_claim = REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "tenant_claim"}) or 0.0
    before_sub = REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "sub"}) or 0.0

    assert jwt_mode.extract_tenant_id({"tenant_id": "acme", "sub": "u"}) == "acme"

    assert REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "tenant_claim"}) == (
        before_claim + 1
    )
    assert REGISTRY.get_sample_value(TENANT_CLAIM_METRIC, {"source": "sub"}) == before_sub


# === Audience verification, measured BEFORE it is armed ======================

AUDIENCE_SHADOW_METRIC = "license_authz_jwt_audience_shadow_total"


def test_every_audience_verdict_is_born_at_zero():
    """⛔ THE FAILURE THIS PREVENTS IS AN ARMED FLAG, not a missing dashboard.

    The decision this counter feeds is "is it safe to turn JWT_VERIFY_AUDIENCE
    on?", and the answer is read off `mismatch` and `absent` being zero. Without
    the birth loop those series do not exist until they first fire, so "no token
    would be rejected" and "this build never measured" are the same empty vector
    — and the operator arms the flag on the strength of an absence. That is a
    fleet-wide 401.
    """
    for verdict in ("unconfigured", "absent", "match", "mismatch"):
        assert (
            REGISTRY.get_sample_value(AUDIENCE_SHADOW_METRIC, {"verdict": verdict}) is not None
        ), f"{AUDIENCE_SHADOW_METRIC}{{verdict={verdict}}} has no series until it first fires"


def test_the_audience_shadow_hook_is_actually_wired_into_jwt_mode():
    """A counter whose hook is not wired reads 0 forever and looks like good news.

    jwt_mode's hook defaults to a NO-OP, so a missing `set_metrics_hooks` entry
    would leave this metric pinned at zero — which is exactly the reading that
    says "safe to arm". Drive the real hook and require the real counter to move.
    """
    import jwt_mode
    import main  # noqa: F401  — importing main is what performs the wiring

    before = REGISTRY.get_sample_value(AUDIENCE_SHADOW_METRIC, {"verdict": "unconfigured"}) or 0.0

    jwt_mode._on_audience_shadow("unconfigured")

    after = REGISTRY.get_sample_value(AUDIENCE_SHADOW_METRIC, {"verdict": "unconfigured"})
    assert after == before + 1, "jwt_mode's audience-shadow hook is not wired to main's counter"
