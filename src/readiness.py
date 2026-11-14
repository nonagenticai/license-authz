"""Real dependency probing behind the ``/ready`` readiness endpoint.

Why this module exists
----------------------
``/health``, ``/healthz``, ``/ready`` and ``/health/live`` each returned a
hardcoded literal, so neither the Kubernetes readiness probe nor the liveness
probe could ever fail while the process was up. A pod that had never loaded a
JWKS -- and therefore 401s every JWT it is shown -- stayed in the Service
endpoints and blackholed real traffic.

What gates readiness, and what deliberately does not
----------------------------------------------------
license-authz is an ext_authz AUTH BOUNDARY. If every replica goes NotReady the
Envoy ext_authz cluster has zero endpoints and the mesh fails shut, so a
readiness check keyed off a SHARED REMOTE dependency converts "one dependency
is degraded" into "everything is down". The rule applied here:

  GATING (503)   per-pod, locally-latched initialisation state, where removing
                 *this* pod from the endpoints genuinely helps because a
                 sibling replica is unaffected.
  REPORTED (200) shared remote dependencies and paths the service is designed
                 to fail open around. Surfaced in the body for operators, never
                 reflected in the status code.

Per dependency:

* ``lifespan`` -- GATING. Proves the ASGI lifespan actually completed and the
  shared httpx client / service-token provider exist. Purely local.
* portal JWKS (when ``LICENSE_AUTHZ_JWT_MODE_ENABLED``) -- SPLIT, see below. A
  pod whose boot fetch failed 401s every oauth2-proxy browser JWT forever;
  nothing on the request path retries the fetch unless a kid miss happens, which
  needs traffic the pod may never get. That is per-pod (its sibling may be fine)
  and it means serving WRONG answers, not degrading. Self-heals: see
  ``_refresh_if_stale``.
* secure-apps JWKS (when a secure-apps issuer is configured) -- SPLIT, same
  shape. Without it every ``X-Caller-Authorization`` presented by an allowlisted
  service principal is rejected with ``caller_token_invalid``.
* internal-token signing key (when the issuer is enabled) -- GATING. Settings
  only assert the PEM is non-empty; a present-but-unparseable PEM 500s ``/jwks``
  and ``/internal-token`` on every call and can never self-heal. The parse is
  cached by PEM contents, so this is free after the first call.
* Redis -- REPORTED, NOT GATING, by default; ``READINESS_REQUIRE_REDIS=true``
  ARMS gating (it is not a revert -- the default is the unarmed state). ⚠️ This is the one entry that does NOT follow the GATING/REPORTED
  rule above, and the exception is deliberate: Redis is a shared remote
  dependency, so the rule would put it in REPORTED, but a pod that cannot reach
  it cannot enforce a quota and cannot record a metering
  event, while still reporting Ready and taking traffic. Shedding that pod is
  right for the SINGLE-pod fault; for an estate-wide Redis outage it empties the
  ext_authz endpoint list and fails the mesh shut, which is why the revert exists
  and why a PodDisruptionBudget / minAvailable is a precondition for running this
  default in production. Probed with a bounded PING. Full argument, and the cost,
  at the flag's declaration; the METERING_ENABLED=false kill switch is never
  gating regardless.
* licenses service and the Keycloak token endpoint -- NOT PROBED AT ALL. Shared
  remote dependencies on the request path: if they are down, every replica is
  equally affected, so shedding endpoints cannot help and only removes the
  paths that still work (``introspect_license`` already serves last-known-good
  on transient failure). Probing them would also mean a real query on every
  probe tick, per replica.

The JWKS split: "never loaded" gates, "lost Keycloak" only reports
------------------------------------------------------------------
The first version of this module keyed the JWKS check on a single latch,
``last_load_ok``, and ``_refresh_if_stale`` early-returned whenever that latch
was truthy. A healthy boot therefore made the check permanently unfalsifiable:
after boot NOTHING could move the latch back, so if Keycloak died at T+1h
``/ready`` answered 200 "ready" forever while every browser JWT 401'd -- the
precise blackholing the probe was written to catch. That is fixed here, but
NOT by simply letting the latch fall to False, because doing so would trade a
silent failure for a much louder one. Two distinct states are separated:

  ``ever_loaded_ok`` False -- this pod has NEVER held key material. Its
      jwk_set_cache and per-kid LRU are both empty, so it rejects every JWT it
      is shown. Its answers are WRONG, and a sibling that booted while Keycloak
      was up is entirely unaffected -- so removing THIS pod from the endpoints
      strictly helps. GATING (503).

  ``last_load_ok`` False while ``ever_loaded_ok`` is True -- this pod loaded
      keys successfully and has since lost contact with Keycloak. It still
      verifies every cached kid CORRECTLY (``PyJWKClient.fetch_data`` only
      writes the cache on success, so a failed re-validation cannot wipe the
      keys it is serving from). It is the best available server, and every
      sibling is in exactly the same position because Keycloak is SHARED.
      REPORTED as ``degraded`` at 200, never gating.

Why the second one must not gate, stated plainly: license-authz is the Envoy
ext_authz cluster with ``statusOnError: "403"``. A Keycloak outage hits all
replicas simultaneously, so latching readiness False on it would take every
endpoint out at the same instant, leave ext_authz with zero endpoints, and fail
the whole mesh SHUT -- converting "logins are broken" into "nothing in the
cluster can talk to anything". The pod would also be evicted from the endpoints
precisely when it is still able to answer correctly from cache. The general
rule this instantiates is the GATING/REPORTED rule above: gate on per-pod state,
report shared-dependency state.

Detection therefore lives in the body (``status: degraded``) and in a structured
transition log line, NOT in the status code. The log is the channel that
actually reaches an operator today: ``license_authz_jwks_refresh_total`` is
exported on :9001 but is NOT scraped -- the pod carries no ``prometheus.io/
scrape`` annotation, and the workload-scoped ``license-authz-strict``
PeerAuthentication (STRICT) makes ztunnel RST the unmeshed Prometheus pod's
plaintext scrape (verified live: ``wget`` from the prometheus pod to
this service's pod IP on :9001/metrics returns "Connection reset by peer", and
``license_authz_jwks_refresh_total`` returns an empty vector). An alert on that
metric would be permanently vacuous, so none is added here; pod logs ARE shipped
(alloy -> loki), so the transition log is a real signal. Making the metric
scrapable means relaxing mTLS on the auth boundary's port and is deliberately
left as a separate, explicit decision.

Timeouts
--------
The live readiness probe uses ``timeoutSeconds: 5``. ``TOTAL_BUDGET`` (2.0s) and
the per-dependency ``DEP_TIMEOUT`` (1.0s) both sit well below it so a slow
dependency yields a clean 503 with a diagnosable body instead of the kubelet
timing the probe out with no information.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger("readiness")

# Per-dependency I/O budget and the overall budget for one /ready call. Both
# MUST stay below the readinessProbe's timeoutSeconds (5s in gitops) so the
# service returns a clean 503 rather than being cut off by the kubelet.
DEP_TIMEOUT = float(os.getenv("READINESS_DEP_TIMEOUT", "1.0"))
TOTAL_BUDGET = float(os.getenv("READINESS_TOTAL_TIMEOUT", "2.0"))

# Minimum seconds between readiness-driven JWKS re-fetch attempts while the pod
# holds NO key material. This is the self-heal path: the pod is already out of
# the endpoints, so it gets no traffic and no kid miss, and the probe is the
# only thing left that can retry. Short, because recovery latency here is
# downtime for this replica.
JWKS_RECHECK_INTERVAL = float(os.getenv("READINESS_JWKS_RECHECK_INTERVAL", "30.0"))

# Minimum seconds between readiness-driven JWKS RE-VALIDATION attempts on a pod
# that already holds key material. This is the monitoring path that makes a
# post-boot Keycloak outage observable at all: a replica with no JWT traffic
# never re-fetches otherwise, which is exactly how /ready came to latch
# ready-but-broken. Much longer than JWKS_RECHECK_INTERVAL because the pod is
# serving correctly meanwhile -- this only costs detection latency, and the
# probe fires every 10s per replica, so a short window here would turn the
# readiness probe into a Keycloak load generator.
JWKS_REVALIDATE_INTERVAL = float(os.getenv("READINESS_JWKS_REVALIDATE_INTERVAL", "300.0"))

# Whether a failing Redis PING is allowed to take this pod out of the Service.
#
# ⛔ THE DEFAULT STAYS **OFF** (non-gating). The gating machinery below is built,
# tested and ready, and `READINESS_REQUIRE_REDIS=true` arms it — but it is NOT
# armed here, and the reason is measured rather than cautious.
#
# ⚠️ THE PRECONDITION ORIGINALLY WRITTEN HERE — "a PodDisruptionBudget /
# minAvailable that refuses to take the last replica out" — DOES NOT MITIGATE
# THIS, and shipping on the strength of it would have been the whole mistake.
# Measured 2026-09-14: `license-authz-pdb` EXISTS (minAvailable=1, 126d old,
# 2/2 replicas ready). A PDB governs VOLUNTARY disruption — drains, evictions.
# It has no bearing on a readiness probe: a NotReady pod is removed from the
# Service endpoints by the endpoints controller, which no PDB gates. So the
# estate-wide case is NOT covered, and the PDB's existence must not be read as
# covering it.
#
# ⛔ AND THE BLAST RADIUS IS CONFIRMED FAIL-SHUT, not merely suspected. The live
# mesh config sets `statusOnError: "403"` on the `license-authz` extension
# provider with no `failOpen`, so an empty ext_authz cluster returns 403 — an
# estate-wide Redis outage would hard-403 all four enforced namespaces
# (control-service, marketing, sales, training-data-generator).
#
# ⭐ WHAT ARMING WOULD BUY TODAY: nothing. Measured over 24h,
# `license_authz_budget_exceeded_total` = 0 and every one of ~1,532 checks took
# the service-principal fast path that returns before any budget is read. So the
# enforcement this probe protects does not currently deny anything, while the
# outage mode it creates is real. Arm it when enforcement actually bites, and
# decide `failOpen` in the same change.
#
# THE DEFECT THIS CLOSES: a pod whose Redis is unreachable cannot enforce a
# budget, cannot enforce a quota and cannot record a metering event — it can only
# fail open — and it went on reporting Ready and taking traffic while that was
# true. The one signal that state produced was a word in `/ready`'s BODY, which
# nothing reads: the Service annotation for scraping is deliberately absent (see
# main.metric_redis_connect), so "this replica enforces nothing" was visible to a
# human running curl and to nobody else.
#
# ⛔ THE COST OF THIS DEFAULT, WHICH IS REAL AND MUST NOT BE DISCOVERED LATER:
# Redis is SHARED by every replica, so it does not fail per-pod — it fails for
# all of them at the same instant. With this default, an estate-wide Redis
# outage takes EVERY replica NotReady, the Envoy ext_authz cluster drops to zero
# endpoints, and ext_authz fails shut: a mesh-wide outage, converted from what
# was previously a billing problem. That trade is the whole content of this
# flag. ⛔ BEFORE THIS REACHES PRODUCTION, the estate-wide case must be covered
# some other way — a PodDisruptionBudget / minAvailable that refuses to take the
# last replica out, or an operator who knows to set READINESS_REQUIRE_REDIS=false
# during a Redis incident. Neither is in this repo; both live in gitops.
#
# ✅ WHAT IT BUYS: a SINGLE-pod Redis fault (its own DNS, its own NetworkPolicy, a
# stale connection pool) now sheds that pod so a healthy sibling takes the
# traffic. That is a real class of fault and it was invisible to the scheduler.
#
# ⭐ ONE OLD ARGUMENT AGAINST GATING DOES NOT SURVIVE, and it is recorded here so
# it is not re-derived: this used to say gating is "self-defeating" because
# ``_readiness_checks`` resolves the client through ``main._get_redis`` and so
# THIS probe is what drives the reconnect — "a gating failure would evict the pod
# that was in the middle of healing itself". The eviction premise is wrong. The
# kubelet keeps running the readiness probe on a NotReady pod; a NotReady pod is
# removed from the Service ENDPOINTS, it is not killed and it is not stopped
# being probed. So the reconnect driver keeps driving, and the pod rejoins the
# endpoints on the first probe after Redis returns. What gating costs is traffic
# during the outage, not the recovery — and for an estate-wide outage that cost
# is the paragraph above, which is why the revert exists.
#
# Read as a module global inside ``check_redis`` (not captured at import) so it
# can be flipped in tests and, in production, by env on the Deployment.
REQUIRE_REDIS = os.getenv("READINESS_REQUIRE_REDIS", "false").lower() == "true"


class _JwksLike(Protocol):
    """Structural type for jwt_mode.JwksCache (avoids an import cycle)."""

    # Most recent REAL fetch outcome (live; reported). None = never attempted.
    last_load_ok: bool | None
    # Has this pod EVER held usable key material? (latched; gating).
    ever_loaded_ok: bool
    last_attempt_at: float | None

    def fetch(self) -> None: ...


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single dependency check.

    ``gating`` records whether this check's failure is allowed to influence the
    HTTP status code, so the body always explains *why* a failing dependency did
    or did not take the pod out of service.
    """

    name: str
    ok: bool
    gating: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "gating": self.gating, "detail": self.detail}


def check_lifespan(state: Any) -> CheckResult:
    """GATING: the ASGI lifespan completed and shared clients exist."""
    missing = [
        attr for attr in ("http_client", "service_auth") if getattr(state, attr, None) is None
    ]
    if missing:
        return CheckResult(
            name="lifespan",
            ok=False,
            gating=True,
            detail=f"app.state not initialised: {','.join(missing)}",
        )
    return CheckResult(name="lifespan", ok=True, gating=True, detail="app.state initialised")


async def _refresh_if_stale(jwks: _JwksLike) -> None:
    """Rate-limited JWKS fetch driven by the probe itself. Never raises.

    Serves two different purposes depending on what the pod holds, which is why
    there are two windows:

    * No key material yet (``ever_loaded_ok`` False) -- SELF-HEAL. The pod is
      NotReady, so it gets no traffic, so no kid miss, so nothing on the request
      path ever retries; without this the pod would stay NotReady forever.
      Retried every ``JWKS_RECHECK_INTERVAL`` (short: this is downtime).
    * Key material present -- RE-VALIDATION. The pod is serving fine, but a
      replica with no JWT traffic would otherwise never touch Keycloak again,
      which is how a post-boot outage stayed invisible. Retried every
      ``JWKS_REVALIDATE_INTERVAL`` (long: this is monitoring, not downtime).

    The previous version early-returned whenever ``last_load_ok`` was truthy,
    which removed the second case entirely and made the check unfalsifiable
    after a healthy boot.
    """
    now = time.monotonic()
    last = jwks.last_attempt_at
    interval = JWKS_RECHECK_INTERVAL if not jwks.ever_loaded_ok else JWKS_REVALIDATE_INTERVAL
    if last is not None and (now - last) < interval:
        return
    try:
        await asyncio.wait_for(asyncio.to_thread(jwks.fetch), timeout=DEP_TIMEOUT)
    except Exception:
        # Covers TimeoutError plus every fetch failure. fetch() has already
        # latched last_load_ok=False and bumped the failure metric; a readiness
        # probe must never raise, so the caller just reads the latches.
        return


def _log_jwks_transition(jwks: Any, name: str, *, healthy: bool, detail: str) -> None:
    """Emit a structured line only when the JWKS health of this pod CHANGES.

    Logged per-transition rather than per-probe because the probe fires every
    10s per replica; a line every tick would be noise an operator learns to
    ignore. Previous state is kept on the JWKS object itself, not in a
    module-level dict, so it cannot leak between the portal and secure-apps
    clients (or between tests).

    This is the operator-facing signal for a post-boot Keycloak outage: the
    refresh METRIC is not scrapable today (see the module docstring), while pod
    logs are shipped to loki, so this line is the channel that actually works.
    """
    previous = getattr(jwks, "_readiness_last_reported_ok", None)
    if previous == healthy:
        return
    try:
        jwks._readiness_last_reported_ok = healthy
    except AttributeError:
        # A slotted/frozen stand-in must not break the probe.
        return
    if previous is None and healthy:
        # First observation of an already-healthy pod is not an event.
        return
    payload = json.dumps(
        {
            "event": "jwks_recovered" if healthy else "jwks_degraded",
            "check": name,
            "detail": detail,
        }
    )
    if healthy:
        logger.info(payload)
    else:
        logger.error(payload)


async def check_jwks(jwks: _JwksLike | None, *, name: str, required: bool) -> CheckResult | None:
    """Check this pod's JWKS state. GATING only where shedding the pod helps.

    Gates on ``ever_loaded_ok`` (no key material at all -> every answer is
    wrong, and a sibling may be fine). Reports, without gating, a live fetch
    failure on a pod that still holds usable keys -- that is a SHARED Keycloak
    outage, and taking every replica out at once would fail the mesh shut. See
    the module docstring for the full argument.

    Returns None when the feature is switched off (nothing to check).
    """
    if not required:
        return None
    if jwks is None:
        return CheckResult(
            name=name,
            ok=False,
            gating=True,
            detail="enabled by config but no JWKS client was constructed",
        )
    await _refresh_if_stale(jwks)

    if not jwks.ever_loaded_ok:
        # No key material has ever been present on this pod.
        detail = (
            "JWKS never loaded (boot fetch failed); tokens signed by this issuer cannot be verified"
            if jwks.last_load_ok is None
            else "last JWKS fetch failed; tokens signed by this issuer cannot be verified"
        )
        _log_jwks_transition(jwks, name, healthy=False, detail=detail)
        return CheckResult(name=name, ok=False, gating=True, detail=detail)

    if jwks.last_load_ok is False:
        # Keys are loaded and still serving; Keycloak is currently unreachable.
        detail = (
            "JWKS endpoint unreachable; serving last-known-good keys. Cached kids still "
            "verify, but a token signed by a NEW key will be rejected until Keycloak returns"
        )
        _log_jwks_transition(jwks, name, healthy=False, detail=detail)
        return CheckResult(name=name, ok=False, gating=False, detail=detail)

    _log_jwks_transition(jwks, name, healthy=True, detail="signing keys loaded")
    return CheckResult(name=name, ok=True, gating=True, detail="signing keys loaded")


def check_internal_token_key(*, enabled: bool, jwks_document: Any) -> CheckResult | None:
    """GATING: the internal-token signing PEM actually parses.

    ``jwks_document`` is passed in (rather than imported) so this stays unit
    testable without touching module-level settings. Returns None when the
    issuer is disabled.
    """
    if not enabled:
        return None
    try:
        doc = jwks_document()
    except Exception as e:
        # Any parse/config failure is a hard fail — a broken PEM cannot self-heal.
        return CheckResult(
            name="internal_token_key",
            ok=False,
            gating=True,
            detail=f"signing key unusable: {type(e).__name__}",
        )
    keys = doc.get("keys") if isinstance(doc, dict) else None
    if not keys:
        return CheckResult(
            name="internal_token_key",
            ok=False,
            gating=True,
            detail="signing key produced an empty JWKS document",
        )
    return CheckResult(name="internal_token_key", ok=True, gating=True, detail="signing key parsed")


async def check_redis(redis_client: Any, *, metering_enabled: bool = True) -> CheckResult:
    """Bounded PING so operators can see Redis. GATING unless reverted.

    PING is O(1) and touches no keyspace, so probe traffic costs Redis nothing.
    Failure is reported at ``ok=False`` and, BY DEFAULT, ALSO takes this pod out
    of the Service: a pod that cannot reach Redis cannot
    enforce a quota and cannot record a metering event, and it must stop claiming
    Ready while that is true.

    ``READINESS_REQUIRE_REDIS`` (default TRUE) sets the ``gating`` flag on the
    failure results only. ⚠️ The env var is the REVERT, not the arming switch --
    ``READINESS_REQUIRE_REDIS=false`` restores the previous non-gating behaviour
    byte-for-byte, and it is the switch to reach for during an estate-wide Redis
    outage, where gating empties the ext_authz endpoint list on every replica at
    once. The cost of the default, and the PodDisruptionBudget precondition it
    implies, are recorded in full at the flag's declaration above.

    ⚠️ The flag is read from the module global on every call, deliberately: it is
    the switch an operator reaches for DURING an incident, and a value captured
    at import could only be changed by a restart -- of every replica, while the
    mesh is already failing shut.

    ⚠️ The SUCCESS result stays ``gating=False`` in both modes. ``gating`` on a
    passing check is inert for the status code (``build_report`` only reads it on
    failures) and flipping it would make the reported shape of a HEALTHY Redis
    depend on the flag, which is noise in every diff of a ``/ready`` body.

    ``metering_enabled`` separates the two ways a client can be absent: a
    permanent, deliberate one (METERING_ENABLED=false -- nothing is broken and
    no retry will ever happen) from a transient one (the connect failed and the
    accessor is backing off, so a later probe may well recover it).

    ⛔ The kill-switch case is NEVER gating, whatever the flag says. With
    METERING_ENABLED=false there is no client by operator instruction, no retry
    will ever be attempted, and no sibling is any better -- so gating it would
    take every replica NotReady permanently and forever. ⚠️ This carve-out got
    MORE load-bearing when REQUIRE_REDIS flipped to default-true: it used to need
    two deliberate opt-ins to reach, and now setting METERING_ENABLED=false alone
    is enough to reach it. It is the only thing standing between that single
    setting and a permanently bricked auth boundary.
    """
    gating = REQUIRE_REDIS
    if redis_client is None:
        if metering_enabled:
            detail = (
                "not connected: connect failed; the /ready probe retries it with backoff;"
                " quota fail-open"
            )
        else:
            detail = (
                "not connected: metering disabled by kill switch (METERING_ENABLED=false),"
                " no retry will be attempted; quota fail-open"
            )
            # See the docstring: a deliberate kill switch must not be able to
            # take every replica out of service permanently.
            gating = False
        return CheckResult(name="redis", ok=False, gating=gating, detail=detail)
    try:
        await asyncio.wait_for(redis_client.ping(), timeout=DEP_TIMEOUT)
    except asyncio.TimeoutError:
        return CheckResult(
            name="redis", ok=False, gating=gating, detail=f"PING exceeded {DEP_TIMEOUT}s"
        )
    except Exception as e:
        # RedisError / OSError / anything: a probe must never raise.
        return CheckResult(
            name="redis", ok=False, gating=gating, detail=f"PING failed: {type(e).__name__}"
        )
    return CheckResult(name="redis", ok=True, gating=False, detail="PING ok")


def build_report(results: list[CheckResult]) -> tuple[int, dict[str, Any]]:
    """Fold check results into an (http_status, body) pair.

    503 when any GATING check failed. 200 with ``status: degraded`` when only
    non-gating checks failed, so a Redis outage is visible without evicting the
    pod from the Service.
    """
    failed_gating = [r.name for r in results if r.gating and not r.ok]
    failed_soft = [r.name for r in results if not r.gating and not r.ok]
    checks = {r.name: r.as_dict() for r in results}
    if failed_gating:
        return 503, {
            "status": "not_ready",
            "service": "license-authz",
            "failed": failed_gating,
            "degraded": failed_soft,
            "checks": checks,
        }
    return 200, {
        "status": "degraded" if failed_soft else "ready",
        "service": "license-authz",
        "degraded": failed_soft,
        "checks": checks,
    }
