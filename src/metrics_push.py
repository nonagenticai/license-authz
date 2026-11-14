"""Push Prometheus metrics to the cluster pushgateway.

WHY THIS EXISTS
---------------
license-authz cannot be scraped. It runs pod-level ambient inside the ROOT
namespace ``istio-system``, where mTLS STRICT (mesh-wide ``mesh-default`` plus
the workload-scoped ``license-authz-strict``) refuses Prometheus's plaintext
scrape at ztunnel *before* RBAC is consulted. Relaxing it has been tried twice
and failed -- gitops #819 (``portLevelMtls {9001: PERMISSIVE}``) and #833
(workload ``mtls.mode: PERMISSIVE``) both stayed
``istio_converted_static_strict``, a documented root-namespace ambient
limitation. Every workload that *is* scraped through the mesh
(``ingestion:8555``, ``improvement-loop-service:8156``, ``kube-state-metrics``)
is namespace-level ambient, which this service is not.

Measured 2026-08-24: Prometheus holds 4224 distinct series and **zero** of them
are ``license_authz_*``. So ``license_authz_jwt_mode_total`` and
``license_authz_sa_token_total`` -- the only signals that say whether the
licence gate is allowing or denying anything -- are unreadable, and every
``LicenseAuthz*`` alert keys on kube-state-metrics endpoint counts, i.e. "is it
up", never "what did it decide".

The pushgateway is the one path that needs no mesh change: it is already
deployed (``pushgateway.example:9091``), already scraped by
Prometheus's annotation-driven ``kubernetes-pods`` job, already resolvable from
inside this pod, and already carries a live producer
(``push_time_seconds{exported_job="detector-scan-streams"}``).

SAFETY -- this module is on the auth hot path's process and must never affect it
--------------------------------------------------------------------------------
1. **Disabled by default.** ``PUSHGATEWAY_URL`` defaults to "", so merging this
   changes nothing. Arming it is a separate, revertible gitops env change.
2. **Never blocks the event loop.** ``prometheus_client.push_to_gateway`` is
   synchronous urllib, so it runs in a worker thread via ``asyncio.to_thread``
   and is additionally bounded by ``asyncio.wait_for``. A hung pushgateway
   costs one idle thread, not an ext_authz request.
3. **Never raises.** Every exception is caught and counted. A push failure must
   not be able to mark the pod unready or fail a check.
4. **Not coupled to the readiness probe.** An earlier design reused the ``/ready``
   tick, which is this repo's established "reuse an existing tick" pattern -- but
   that would put pushgateway availability on the path that decides whether this
   replica keeps serving auth. Deliberately rejected.
"""

from __future__ import annotations

import asyncio
import logging
import os

from prometheus_client import REGISTRY, Counter, push_to_gateway

logger = logging.getLogger(__name__)

# The push path's own health. Circular by nature (it can only be read once a
# push succeeds, or locally at /metrics), but it makes a partial outage
# visible: a pod whose pushes are all failing shows a flat push_time_seconds
# at the gateway, and this counter says why in its own /metrics.
metric_push = Counter(
    "license_authz_pushgateway_total",
    "Outcomes of pushing this pod's metrics to the pushgateway",
    ["outcome"],  # ok | failed | timeout
)

# Pushgateway groups are keyed by job + grouping_key. Both replicas must NOT
# share a group or the second push overwrites the first and the fleet reads one
# pod's counters as the whole. HOSTNAME is the pod name and is set by the
# kubelet for free -- no downward-API change in gitops is needed.
_JOB = "license-authz"


def _instance() -> str:
    return os.environ.get("HOSTNAME", "unknown")


class MetricsPusher:
    """Periodically push the default registry to a pushgateway."""

    def __init__(
        self,
        url: str,
        interval_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._url = url
        self._interval = interval_seconds
        self._timeout = timeout_seconds
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def start(self) -> None:
        """Start the background loop. No-op when unconfigured."""
        if not self.enabled:
            logger.info("pushgateway export disabled (PUSHGATEWAY_URL unset)")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "pushgateway export enabled url=%s interval=%ss instance=%s",
            self._url,
            self._interval,
            _instance(),
        )

    async def stop(self) -> None:
        """Cancel the loop and wait for it to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                raise
            await self.push_once()

    async def push_once(self) -> str:
        """Push the registry once. Returns the outcome; never raises."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    push_to_gateway,
                    self._url,
                    job=_JOB,
                    registry=REGISTRY,
                    grouping_key={"instance": _instance()},
                    timeout=self._timeout,
                ),
                timeout=self._timeout + 1.0,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            metric_push.labels(outcome="timeout").inc()
            logger.warning("pushgateway push timed out after %ss", self._timeout)
            return "timeout"
        except Exception as exc:  # noqa: BLE001 - must never propagate
            metric_push.labels(outcome="failed").inc()
            logger.warning("pushgateway push failed: %s", exc)
            return "failed"
        metric_push.labels(outcome="ok").inc()
        return "ok"
