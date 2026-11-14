"""Tests for the lazy Redis accessor (``main._get_redis``).

Boot used to be the ONLY Redis construction site. If the ping at lifespan
raised, ``app.state.redis = None`` was latched for the entire process lifetime:
``_emit_metering_event`` returned silently, the budget/quota check returned
``allowed=True``, and ``/ready`` reported redis non-gating so the pod kept
serving. At replicas 2 that is ~50% of mesh traffic unmetered and
budget-unenforced until someone restarts the pod.

Every test here must be able to FAIL against the old behaviour
(``return application.state.redis``) or against the removal of the specific
guard it names — a test that passes against the bug is vacuous.
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
import redis.exceptions

import main
from main import app, lifespan

# -- Fakes --------------------------------------------------------------------


class FakeRedis:
    """Stands in for redis.asyncio.Redis — PING + aclose only.

    ``gate`` lets a test hold a connect in flight (to observe state *during*
    the await); ``delay`` simulates a black-holed server; ``exc`` a refusal.
    """

    def __init__(
        self,
        *,
        exc: Exception | None = None,
        delay: float = 0.0,
        gate: asyncio.Event | None = None,
    ):
        self._exc = exc
        self._delay = delay
        self._gate = gate
        self.ping_calls = 0
        self.aclose_calls = 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self._gate is not None:
            await self._gate.wait()
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return True

    async def aclose(self) -> None:
        self.aclose_calls += 1


class ExplodingLock:
    """A lock that fails the test if it is ever entered."""

    async def __aenter__(self) -> "ExplodingLock":
        raise AssertionError(
            "the backoff window must be checked BEFORE the connect lock — the "
            "ext_authz hot path must never contend on a lock while Redis is down"
        )

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


# -- Hot path: an established client costs one attribute read -----------------


async def test_existing_client_returned_without_reconnecting():
    """Connected → early return. No clock, no lock, no construction.

    The window is left OPEN and the lock is booby-trapped on purpose: the
    under-lock re-check would also hand back the existing client, so asserting
    only "from_url was not called" cannot tell the hot-path early return from
    its absence (verified — that weaker assertion survives the mutation).
    """
    main._init_redis_state(app)
    existing = FakeRedis()
    app.state.redis = existing
    app.state.redis_next_attempt_at = 0.0
    app.state.redis_connect_lock = ExplodingLock()

    with patch("redis.asyncio.from_url") as from_url:
        got = await main._get_redis(app)

    assert got is existing
    from_url.assert_not_called()


# -- The D1 test: a boot-time failure is NOT latched for the process lifetime --


async def test_boot_failure_is_not_latched_second_call_connects():
    """A failed boot connect must be retried, not latched until pod restart."""
    healthy = FakeRedis()
    with (
        patch("main.METERING_ENABLED", True),
        patch(
            "redis.asyncio.from_url",
            side_effect=[redis.exceptions.ConnectionError("refused"), healthy],
        ) as from_url,
    ):
        async with lifespan(app):
            assert app.state.redis is None, "staged boot failure did not take effect"
            assert from_url.call_count == 1
            assert app.state.redis_failures == 1

            # Simulate the backoff window elapsing (the /ready probe drives this
            # every 10s in production).
            app.state.redis_next_attempt_at = 0.0
            got = await main._get_redis(app)

            assert got is healthy, "the accessor never retried — the boot latch is still there"
            assert app.state.redis is healthy
            assert from_url.call_count == 2
    app.state.redis = None


# -- Backoff window -----------------------------------------------------------


async def test_second_call_inside_backoff_window_does_not_reconnect():
    """A down Redis must not be dialled once per request."""
    main._init_redis_state(app)
    with patch(
        "redis.asyncio.from_url", side_effect=redis.exceptions.ConnectionError("refused")
    ) as from_url:
        assert await main._get_redis(app) is None
        assert from_url.call_count == 1
        assert app.state.redis_next_attempt_at > time.monotonic(), "window was not armed"

        assert await main._get_redis(app) is None
        assert from_url.call_count == 1, "a call inside the window must not reconnect"


async def test_window_check_happens_before_the_lock():
    """The window guard precedes ``async with`` — no lock contention when down."""
    main._init_redis_state(app)
    app.state.redis_next_attempt_at = time.monotonic() + 3600
    app.state.redis_connect_lock = ExplodingLock()

    with patch("redis.asyncio.from_url") as from_url:
        assert await main._get_redis(app) is None

    from_url.assert_not_called()


def test_backoff_is_non_decreasing_and_capped():
    """Exponential, monotonic, and capped at REDIS_RECONNECT_MAX_INTERVAL."""
    values = [main._redis_backoff_seconds(n) for n in range(1, 40)]

    assert values[0] == main.REDIS_RECONNECT_MIN_INTERVAL
    assert all(later >= earlier for earlier, later in zip(values, values[1:])), (
        f"backoff is not non-decreasing: {values}"
    )
    assert max(values) <= main.REDIS_RECONNECT_MAX_INTERVAL, (
        f"backoff exceeded the cap: max={max(values)}"
    )
    assert values[-1] == main.REDIS_RECONNECT_MAX_INTERVAL, "backoff never reaches the cap"


def test_retry_delay_is_jittered_to_desync_replicas():
    """Jitter de-syncs the two replicas; without it both dial in lockstep.

    Asserted on the delay itself, not on ``redis_next_attempt_at - t0``: that
    difference also carries the (always varying) execution time, so it looks
    jittered even when the jitter is gone — verified, the wall-clock form of
    this test survives the mutation.
    """
    base = main._redis_backoff_seconds(1)
    delays = {main._redis_retry_delay(1) for _ in range(25)}

    assert len(delays) > 1, "the retry delay is not jittered — replicas will retry in lockstep"
    assert min(delays) >= 0.8 * base
    assert max(delays) <= 1.2 * base


async def test_armed_window_uses_the_jittered_delay():
    """The window the accessor arms is the jittered delay, not the raw backoff.

    Asserted across repeated arms on purpose. A single sample's
    ``[0.8*base, 1.2*base]`` band CONTAINS base, so it cannot tell
    ``now + _redis_retry_delay(n)`` from ``now + _redis_backoff_seconds(n)`` —
    the one mutation this test exists to catch. Un-jittered, every arm lands on
    base plus sub-millisecond execution time, so the SPREAD is what discriminates.
    """
    base = main._redis_backoff_seconds(1)
    deltas: list[float] = []
    for _ in range(20):
        main._init_redis_state(app)
        started = time.monotonic()
        with patch(
            "redis.asyncio.from_url", side_effect=redis.exceptions.ConnectionError("refused")
        ):
            assert await main._get_redis(app) is None
        deltas.append(app.state.redis_next_attempt_at - started)

    assert all(0.8 * base <= d <= 1.2 * base + 0.5 for d in deltas), (
        f"an armed window fell outside the jitter band: {deltas}"
    )
    spread = max(deltas) - min(deltas)
    assert spread > 0.05 * base, (
        f"the armed window is not jittered — 20 arms spread only {spread:.6f}s over base={base}s"
    )


async def test_backoff_is_armed_before_the_connect_is_awaited():
    """Arming after the await lets every probe tick pile in behind a hung connect."""
    main._init_redis_state(app)
    gate = asyncio.Event()
    slow = FakeRedis(gate=gate)

    with patch("redis.asyncio.from_url", return_value=slow):
        task = asyncio.create_task(main._get_redis(app))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert slow.ping_calls == 1, "the connect is not in flight yet"
        assert app.state.redis_next_attempt_at > time.monotonic(), (
            "the retry window must be armed BEFORE the connect is awaited"
        )

        gate.set()
        assert await task is slow
    app.state.redis = None


# -- Failure accounting -------------------------------------------------------


async def test_success_resets_the_failure_count():
    """Otherwise the backoff keeps growing across a transient outage."""
    main._init_redis_state(app)
    healthy = FakeRedis()
    with patch(
        "redis.asyncio.from_url",
        side_effect=[redis.exceptions.ConnectionError("refused"), healthy],
    ):
        assert await main._get_redis(app) is None
        assert app.state.redis_failures == 1

        app.state.redis_next_attempt_at = 0.0
        assert await main._get_redis(app) is healthy

    assert app.state.redis_failures == 0, "a successful connect must reset the failure count"
    assert app.state.redis_next_attempt_at == 0.0
    app.state.redis = None


async def test_failed_connect_closes_the_discarded_client():
    """from_url builds a ConnectionPool — retrying forever would leak one per attempt."""
    main._init_redis_state(app)
    broken = FakeRedis(exc=redis.exceptions.ConnectionError("refused"))

    with patch("redis.asyncio.from_url", return_value=broken):
        assert await main._get_redis(app) is None

    assert broken.aclose_calls == 1, "the discarded client's connection pool was leaked"


async def test_cancelled_connect_closes_the_client_and_counts_the_attempt():
    """A /ready budget cancellation mid-PING must not leak the ConnectionPool.

    ``/ready`` wraps the whole probe in ``asyncio.wait_for(..., 2.0)``
    (``main.py`` readiness handler). ``CancelledError`` is a ``BaseException``,
    so it does NOT match ``except (RedisError, OSError, asyncio.TimeoutError)``:
    without an arm of its own the client ``from_url`` just built is dropped with
    its ConnectionPool + socket still open, and ``redis_failures`` is never
    incremented, so the backoff stays at its first window forever.

    The shape here is production's exactly: an outer ``wait_for`` shorter than
    the connect, around a PING that never returns.
    """
    main._init_redis_state(app)
    gate = asyncio.Event()  # never set — the PING hangs until we are cancelled
    hung = FakeRedis(gate=gate)

    with patch("redis.asyncio.from_url", return_value=hung):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(main._get_redis(app), timeout=0.05)

    assert hung.ping_calls == 1, "the connect never got in flight — the test proves nothing"
    assert hung.aclose_calls == 1, (
        "the cancelled connect leaked its ConnectionPool: aclose() never ran"
    )
    assert app.state.redis_failures == 1, (
        "a cancelled attempt was not counted, so the backoff never escalates past window 1"
    )
    assert app.state.redis is None


async def test_hung_ping_returns_none_within_the_timeout(monkeypatch):
    """socket_connect_timeout does not bound DNS; /ready's total budget is 2.0s."""
    monkeypatch.setattr(main, "REDIS_CONNECT_TIMEOUT", 0.05)
    main._init_redis_state(app)
    hung = FakeRedis(delay=30.0)

    started = time.monotonic()
    with patch("redis.asyncio.from_url", return_value=hung):
        # Outer bound so an unbounded ping fails the test instead of hanging it.
        got = await asyncio.wait_for(main._get_redis(app), timeout=5.0)
    elapsed = time.monotonic() - started

    assert got is None
    assert elapsed < 2.0, f"the connect was not bounded: {elapsed:.2f}s"
    assert app.state.redis is None
    assert hung.aclose_calls == 1


async def test_client_is_built_with_bounded_socket_timeouts():
    """An unbounded socket lets a black-holed Redis hang the ext_authz hot path."""
    main._init_redis_state(app)
    healthy = FakeRedis()

    with patch("redis.asyncio.from_url", return_value=healthy) as from_url:
        assert await main._get_redis(app) is healthy

    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == main.REDIS_CONNECT_TIMEOUT
    assert kwargs["socket_timeout"] == main.REDIS_SOCKET_TIMEOUT
    app.state.redis = None


# -- Concurrency --------------------------------------------------------------


async def test_concurrent_callers_construct_exactly_one_client(monkeypatch):
    """50 concurrent hot-path callers must not each build a ConnectionPool."""
    # Zero backoff so the window can never be what serialises the callers —
    # the lock and the under-lock re-check have to carry it on their own.
    monkeypatch.setattr(main, "REDIS_RECONNECT_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(main, "REDIS_RECONNECT_MAX_INTERVAL", 0.0)
    main._init_redis_state(app)

    gate = asyncio.Event()
    healthy = FakeRedis(gate=gate)

    with patch("redis.asyncio.from_url", return_value=healthy) as from_url:
        tasks = [asyncio.create_task(main._get_redis(app)) for _ in range(50)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(*tasks)

    assert from_url.call_count == 1, f"built {from_url.call_count} clients, expected 1"
    assert healthy.ping_calls == 1
    assert all(r is healthy for r in results)
    app.state.redis = None


# -- The conftest isolation guarantee, made falsifiable -----------------------


async def test_lifespan_never_opens_a_real_connection(monkeypatch):
    """conftest promises no live socket in a unit test — prove it.

    ``lifespan`` calls ``_init_redis_state`` (which zeroes the backoff window
    the conftest had parked at +inf) and then ``_get_redis``, with
    ``METERING_ENABLED`` defaulting true. So every fixture that enters lifespan
    really dialled ``redis://localhost:6379/2``; the +inf claim was false for
    most of the suite.

    Asserted at the event loop's connection layer, not on wall-clock time, so a
    merely *faster* failure cannot satisfy it.
    """
    loop = asyncio.get_running_loop()
    opened: list[Any] = []
    real_create_connection = loop.create_connection

    async def spy(*args: Any, **kwargs: Any) -> Any:
        opened.append((args, kwargs))
        return await real_create_connection(*args, **kwargs)

    monkeypatch.setattr(loop, "create_connection", spy)

    async with lifespan(app):
        pass

    assert opened == [], f"lifespan opened a real connection during a unit test: {opened}"


# -- Kill switch --------------------------------------------------------------


async def test_metering_disabled_never_connects():
    """METERING_ENABLED=false must not so much as arm a backoff."""
    main._init_redis_state(app)
    with patch("main.METERING_ENABLED", False), patch("redis.asyncio.from_url") as from_url:
        assert await main._get_redis(app) is None

    from_url.assert_not_called()
    assert app.state.redis_next_attempt_at == 0.0
