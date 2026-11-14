"""Keycloak service-account token acquisition for internal calls.

Used to authenticate license-authz → licenses `/licenses/active` and
license-authz → metering-service `/api/budget/check`, `/api/quota/check`.

Ported from metering_service/src/service_auth.py.

OBSERVABILITY CONTRACT (added 2026-08-11). `get_token()` returns ``None`` on
four structurally different failures — never configured, Keycloak refused the
client_credentials grant, the transport failed, and Keycloak answered 200 with
something that is not a token — and `auth_header()` folds all four into ``{}``.
Both call sites (`main.py:_handle_jwt_mode` and the budget HTTP fallback) then
make an *unauthenticated* call that is byte-for-byte indistinguishable from a
working one. That is why a missing production secret went unnoticed for months.
The return values are deliberately unchanged here; what changes is that each
outcome is now distinguishable from the outside.

⚠️ The fourth of those — the malformed 200 — did NOT resolve to ``None`` before
2026-08-11: it RAISED, all the way out of the ext_authz handler. See the arm in
``get_token`` for why that is a platform-wide DENY and not a local error.

⚠️ THE LOG LINE IS THE OPERATOR-VISIBLE SIGNAL, NOT THE COUNTER. This service's
``/metrics`` is served but **not scraped**: license-authz is per-pod ambient in
the ``istio-system`` root namespace under ``istio_converted_static_strict``, and
the plaintext scrape is reset by ztunnel, so the Service deliberately carries no
``prometheus.io/scrape`` annotation (full rationale, including the two live
PERMISSIVE attempts that did not work, is in gitops
``clusters/strato/services/license-authz/service.yaml``). Pod logs reach Loki
regardless of the mesh, so each WARNING below is written to be self-sufficient:
it names the outcome, the HTTP status when there is one, and the exception
class, and is diagnosable without any metric.
"""

import json
import logging
import math
import os
import time

import httpx
from prometheus_client import Counter

from settings import get_settings

logger = logging.getLogger("service_auth")

# Prometheus metric.
#
# NOT COLLECTED TODAY — see the module docstring. This is an in-pod verification
# instrument on the same footing as `metric_redis_connect` (main.py:203-215),
# not an alertable signal; it becomes visible the day the scrape can succeed.
# The operator channel is the WARNING log line below.
#
# Registered on prometheus_client's DEFAULT registry — the same global registry
# every ``license_authz_*`` counter in ``main.py`` uses, and the same one
# ``main.py``'s ``/metrics`` route serves via ``generate_latest()``. There is no
# second registry and no second endpoint.
#
# It is declared HERE rather than in main.py's counter inventory because main.py
# imports this module (`from service_auth import ServiceTokenProvider`), so a
# back-import would be circular. main.py's other cross-module counters solve
# that with an injected hook (`jwt_mode.set_metrics_hooks`); this one needs no
# indirection because it has exactly one emitter — this file.
#
# The five outcomes PARTITION every `get_token()` call: exactly one is
# incremented per call, so `sum by (outcome) (rate(...))` is a real ratio.
# `mint_failed` and `malformed_response` are kept apart on purpose — a 4xx/5xx
# means Keycloak rejected us (credential problem), a malformed 200 means
# something answered that is not Keycloak speaking OIDC (proxy, wrong URL,
# wrong grant). Those have different operators and different fixes.
metric_sa_token = Counter(
    "license_authz_sa_token_total",
    "Keycloak service-account token acquisition outcomes.",
    ["outcome"],  # ok | cached | unconfigured | mint_failed | malformed_response
)


class ServiceTokenProvider:
    """Acquires and caches a Keycloak service-account JWT via client_credentials."""

    def __init__(
        self,
        token_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        # Read env at __init__-time so tests can monkeypatch.setenv after import.
        # Falls back to settings (which reflects env at module load); no source-
        # side hardcoded URL — see config-externalization-policy.md.
        self._token_url = token_url or os.getenv(
            "KEYCLOAK_TOKEN_URL",
            get_settings().keycloak_token_url,
        )
        self._client_id = client_id or os.getenv("LICENSE_AUTHZ_SA_CLIENT_ID", "license-authz")
        self._client_secret = client_secret or os.getenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", "")
        self._token: str | None = None
        self._expires_at: float = 0.0
        # One-shot latch for the `unconfigured` WARNING. The counter still moves
        # on every call (that is the rate signal); the log does not, because
        # this runs on the ext_authz hot path and an unconfigured deployment
        # would otherwise emit one line per request forever.
        self._logged_unconfigured = False

    @property
    def configured(self) -> bool:
        return bool(self._token_url and self._client_secret)

    def _log_unconfigured_once(self) -> None:
        """WARN exactly once per provider that the SA credential is absent.

        Never emits the secret (there is none) nor the client secret env value.
        """
        if self._logged_unconfigured:
            return
        self._logged_unconfigured = True
        logger.warning(
            json.dumps(
                {
                    "event": "sa_token_unconfigured",
                    "client_id": self._client_id,
                    "token_url_set": bool(self._token_url),
                    "client_secret_set": bool(self._client_secret),
                    "consequence": "internal calls are made unauthenticated",
                }
            )
        )

    async def get_token(self, client: httpx.AsyncClient | None = None) -> str | None:
        """Return a cached or fresh SA JWT, or None if not configured."""
        if not self.configured:
            metric_sa_token.labels(outcome="unconfigured").inc()
            self._log_unconfigured_once()
            return None
        if self._token and time.monotonic() < self._expires_at:
            metric_sa_token.labels(outcome="cached").inc()
            return self._token

        http = client if client is not None else httpx.AsyncClient(timeout=5.0)
        owns_client = client is None
        try:
            resp = await http.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            if resp.status_code == 200:
                # A 200 is NOT proof of a token. Everything below parses an
                # untrusted body, and every name is bound to a LOCAL first: the
                # provider's cache is committed to only after the whole parse
                # succeeds, so a half-parsed response cannot leave `_token` set
                # against a stale `_expires_at`.
                try:
                    body = resp.json()
                    token = body["access_token"]
                    if not isinstance(token, str) or not token:
                        raise TypeError("access_token is empty or not a string")
                    expires_in = body.get("expires_in", 300)
                    # `expires_in` needs the SAME guard as `access_token` — same
                    # threat model, and NOTHING downstream would raise for it.
                    # Python's json.loads accepts the non-standard literals
                    # Infinity / -Infinity / NaN BY DEFAULT and returns real
                    # floats, so the arithmetic below succeeds and the except
                    # tuple never fires:
                    #   Infinity  -> _expires_at = inf, so the cache check
                    #                `monotonic() < _expires_at` is true FOREVER.
                    #                The token is pinned for the pod's lifetime;
                    #                once the real SA JWT expires, /licenses/active
                    #                starts 401ing, which surfaces as
                    #                licenses_unavailable 503 on every JWT-mode
                    #                request until the pod restarts.
                    #   NaN       -> every comparison with NaN is False, so the
                    #   -Infinity    cache NEVER hits and this mints a fresh
                    #   True         Keycloak token on EVERY ext_authz request.
                    # bool is checked first because True/False ARE ints in
                    # Python — the same trap `metering_client._as_int` guards, and
                    # the same one that shipped as finding N2 of PR #37.
                    # TypeError is raised on purpose: it lands in the arm below
                    # and resolves to the same `malformed_response` outcome.
                    # ⚠️ The `not isinstance(...)` half is a PROVEN EQUIVALENT
                    # MUTANT and is kept deliberately — do not "simplify" it
                    # away, and do not log it as an uncovered mutation. Removing
                    # it is behaviourally identical across all 13 JSON-derivable
                    # shapes (null/str/list/dict/bool/int/float/non-finite/
                    # overflow/absent, differentially probed 2026-08-11) because
                    # the arithmetic below raises TypeError for every one of
                    # them anyway. It stays because relying on the arithmetic to
                    # reject a type is fragile: change that line to
                    # `float(expires_in)` and a string "300" would silently
                    # start being COERCED instead of refused. The bool half is
                    # NOT equivalent — see P3.
                    if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)):
                        raise TypeError("expires_in is not a number")
                    if isinstance(expires_in, float) and not math.isfinite(expires_in):
                        raise TypeError("expires_in is not finite")
                    # An int too large for float still reaches this and raises
                    # OverflowError, deliberately — see the tuple below.
                    expires_at = time.monotonic() + expires_in - 30
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    # ⚠️ THIS ARM IS A CRASH FIX, NOT A STYLE PREFERENCE.
                    # None of these are httpx.HTTPError, so before this existed
                    # they escaped get_token() -> escaped auth_header() ->
                    # escaped _handle_jwt_mode (unwrapped at main.py:1569) ->
                    # escaped the ext_authz handler -> 500, and Envoy converts a
                    # 500 from ext_authz into a DENY for EVERY request on the
                    # platform. Same outage class as the load-bearing comment at
                    # main.py:1652-1670.
                    #
                    # ⚠️ READ THIS BEFORE ARMING THE CREDENTIAL. The JWT-mode
                    # flag is ON in production — `LICENSE_AUTHZ_JWT_MODE_ENABLED
                    # =true` on deploy/license-authz in ns istio-system, read
                    # live 2026-08-11. Do NOT believe any note that says JWT mode
                    # is "disabled"; an earlier version of THIS comment said so
                    # and was wrong. The path is unreached today for two
                    # INDEPENDENT reasons, and arming the secret removes one:
                    #   1. main.py:1568 gates on JWT_MODE_ENABLED *and*
                    #      is_jwt_bearer(token). The flag is true; no current
                    #      traffic carries a portal JWT bearer, so the second
                    #      conjunct is false. Evidence:
                    #      license_authz_jwt_mode_total has zero child series on
                    #      both replicas over 27h, against ~36k
                    #      service-principal events in the same scrape.
                    #   2. `configured` is False (the secretKeyRef for
                    #      LICENSE_AUTHZ_SA_CLIENT_SECRET is optional:true, so a
                    #      missing key resolves to ""), and get_token() returns
                    #      at the `unconfigured` branch before any HTTP call —
                    #      this arm cannot be reached at all.
                    # Arming LICENSE_AUTHZ_SA_CLIENT_SECRET removes guard 2,
                    # leaving only "no JWT-bearer traffic arrives" between this
                    # code and the ext_authz hot path.
                    #
                    # Each member of the tuple is reachable and necessary. This
                    # is the set reachable from parsing an untrusted JSON body;
                    # it is NOT a claim that nothing else can raise here (e.g.
                    # httpx.ResponseNotRead is a RuntimeError and is caught by
                    # neither arm — unreachable, because `await http.post(...)`
                    # has already read the body):
                    #   ValueError     json.JSONDecodeError (an HTML error page
                    #                  from a proxy) and UnicodeDecodeError
                    #                  (undecodable bytes) — both ValueError
                    #                  subclasses, which is why the base class is
                    #                  named rather than JSONDecodeError.
                    #   KeyError       a JSON object with no access_token (wrong
                    #                  client, wrong grant_type).
                    #   TypeError      body is a list/str/number/null and is not
                    #                  subscriptable; or expires_in is not a
                    #                  number; or access_token is not a string.
                    #   OverflowError  expires_in is a legal JSON integer too
                    #                  large for float. This is EXACTLY the
                    #                  family that escaped a (TypeError,
                    #                  ValueError) catch in metering_client and
                    #                  shipped as finding N1 of PR #37 — the same
                    #                  repo, the same mistake, two files apart.
                    #
                    # Deliberately NOT `except Exception`: that would swallow
                    # genuine programming errors and BaseException-adjacent
                    # control flow, turning a contract check into a bug hider.
                    metric_sa_token.labels(outcome="malformed_response").inc()
                    # SHAPE ONLY, never content: on the success path this body
                    # IS the token, and on the failure path it can still echo
                    # request material. Class name, byte length and the declared
                    # content-type — enough to tell "a proxy returned HTML" from
                    # "Keycloak returned the wrong JSON".
                    # The content-type is UNTRUSTED (it is whatever answered,
                    # possibly a proxy), so it is the one attacker-influenced
                    # field here and the [:64] cap is what bounds it — do not
                    # remove it. The claim is that nothing here can carry a
                    # CREDENTIAL, not that nothing here is untrusted.
                    logger.warning(
                        json.dumps(
                            {
                                "event": "sa_token_mint_failed",
                                "reason": "malformed_response",
                                "error_class": type(exc).__name__,
                                "body_bytes": len(resp.content),
                                "content_type": resp.headers.get("content-type", "")[:64],
                                "client_id": self._client_id,
                            }
                        )
                    )
                    return None
                self._token = token
                self._expires_at = expires_at
                metric_sa_token.labels(outcome="ok").inc()
                # DEBUG, and never the token itself: a successful mint is the
                # boring case and this is a hot path.
                #
                # This dump is reached ONLY after the validation above, so
                # `expires_in` is a finite int/float here and json.dumps cannot
                # emit the bare `Infinity`/`NaN` literals that a strict JSON
                # parser (Loki's) rejects. That validation is deliberately the
                # fix rather than `allow_nan=False` on this call: this line sits
                # on the SUCCESS path, outside the try, so a dump that can raise
                # would re-open the exact escape-to-500 this file just closed.
                # Fix the input, do not add a throw to the happy path.
                # `tests/test_service_auth.py` parses every emitted line with a
                # strict parser, so a regression here goes red.
                logger.debug(
                    json.dumps(
                        {
                            "event": "sa_token_minted",
                            "client_id": self._client_id,
                            "expires_in": expires_in,
                        }
                    )
                )
                return self._token
            metric_sa_token.labels(outcome="mint_failed").inc()
            # WARNING every time, not latched: a credential that has stopped
            # working is the alertable signal. Status code only — the Keycloak
            # error body is never logged, so no grant material can leak.
            logger.warning(
                json.dumps(
                    {
                        "event": "sa_token_mint_failed",
                        "reason": "http_status",
                        "status": resp.status_code,
                        "client_id": self._client_id,
                    }
                )
            )
        except httpx.HTTPError as exc:
            metric_sa_token.labels(outcome="mint_failed").inc()
            # Exception CLASS only, never str(exc): httpx renders the request
            # URL into some messages and the form body is where the secret is.
            logger.warning(
                json.dumps(
                    {
                        "event": "sa_token_mint_failed",
                        "reason": "transport_error",
                        "error_class": type(exc).__name__,
                        "client_id": self._client_id,
                    }
                )
            )
            return None
        finally:
            if owns_client:
                await http.aclose()
        return None

    async def auth_header(self, client: httpx.AsyncClient | None = None) -> dict[str, str]:
        """Build an Authorization header dict (empty if no token available)."""
        tok = await self.get_token(client)
        return {"Authorization": f"Bearer {tok}"} if tok else {}
