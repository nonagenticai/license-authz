"""Tests for the pushgateway exporter.

The properties under test are safety properties, not feature properties. This
module runs inside the ext_authz process, so the things that must hold are:
it is off unless configured, it never raises, and it never blocks the event
loop that serves auth decisions.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from prometheus_client import REGISTRY

import metrics_push
from metrics_push import MetricsPusher


def _push_count(outcome: str) -> float:
    return REGISTRY.get_sample_value("license_authz_pushgateway_total", {"outcome": outcome}) or 0.0


def test_disabled_when_url_unset():
    """Merging this must change nothing until gitops sets PUSHGATEWAY_URL."""
    p = MetricsPusher(url="", interval_seconds=1, timeout_seconds=1)
    assert p.enabled is False
    p.start()  # must not raise and must not create a task
    assert p._task is None


@pytest.mark.asyncio
async def test_grouping_key_is_per_pod(monkeypatch):
    """Both replicas must not share a pushgateway group.

    A shared group means the second pod's PUT replaces the first's, and the
    fleet reads one pod's counters as though they were the whole service.
    """
    seen = {}
    monkeypatch.setattr(
        metrics_push,
        "push_to_gateway",
        lambda url, job, registry, grouping_key, timeout: seen.update(grouping_key=grouping_key),
    )
    monkeypatch.setenv("HOSTNAME", "license-authz-pod-one")
    p = MetricsPusher(url="pg:9091", interval_seconds=1, timeout_seconds=1)
    await p.push_once()
    assert seen["grouping_key"] == {"instance": "license-authz-pod-one"}

    monkeypatch.setenv("HOSTNAME", "license-authz-pod-two")
    await p.push_once()
    assert seen["grouping_key"] == {"instance": "license-authz-pod-two"}


@pytest.mark.asyncio
async def test_push_failure_never_propagates(monkeypatch):
    """A dead pushgateway must not be able to raise into the auth process."""

    def boom(*a, **k):
        raise ConnectionRefusedError("pushgateway is down")

    monkeypatch.setattr(metrics_push, "push_to_gateway", boom)
    before = _push_count("failed")
    p = MetricsPusher(url="pg:9091", interval_seconds=1, timeout_seconds=1)
    assert await p.push_once() == "failed"  # returns, does not raise
    assert _push_count("failed") == before + 1


@pytest.mark.asyncio
async def test_push_timeout_is_bounded(monkeypatch):
    """A hung pushgateway must be abandoned, not waited on forever."""
    monkeypatch.setattr(metrics_push, "push_to_gateway", lambda *a, **k: time.sleep(5))
    before = _push_count("timeout")
    p = MetricsPusher(url="pg:9091", interval_seconds=1, timeout_seconds=0.1)
    started = time.monotonic()
    assert await p.push_once() == "timeout"
    elapsed = time.monotonic() - started
    assert elapsed < 3, f"push_once waited {elapsed:.1f}s despite a 0.1s timeout"
    assert _push_count("timeout") == before + 1


@pytest.mark.asyncio
async def test_push_does_not_block_the_event_loop(monkeypatch):
    """THE property that matters: a slow push must not stall ext_authz.

    push_to_gateway is synchronous urllib. If it were awaited directly rather
    than handed to a worker thread, a slow gateway would freeze every
    concurrent auth decision in this process. The control is the tick counter:
    a coroutine ticking every 10ms must keep ticking while the push is stuck.
    """
    monkeypatch.setattr(metrics_push, "push_to_gateway", lambda *a, **k: time.sleep(0.6))
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    p = MetricsPusher(url="pg:9091", interval_seconds=1, timeout_seconds=5)
    await p.push_once()
    t.cancel()
    assert ticks > 10, (
        f"event loop advanced only {ticks} ticks during a 0.6s push — "
        "the push is blocking the loop that serves ext_authz"
    )
