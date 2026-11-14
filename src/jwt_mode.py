"""JWT-mode: Keycloak JWT → (tenant_id, app_key) → license resolution.

Used by the ext_authz Check handler when the bearer is a Keycloak JWT (from
oauth2-proxy) instead of an opaque license token. Resolves the active license
via licenses `/licenses/active` and returns the same IntrospectResult shape
the token-mode path produces, so downstream budget/app-route logic is shared.
"""

import base64
import binascii
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import jwt
from jwt import PyJWKClient

from settings import get_settings

# Service URLs come from gitops Deployment env (no source-side fallback;
# see gitops-production/docs/config-externalization-policy.md).
_settings = get_settings()
KEYCLOAK_ISSUER = _settings.keycloak_issuer
JWKS_URL = _settings.jwks_url
# Realm-per-app P0: allowlist of issuers whose END-USER JWTs take JWT mode.
# Default (KEYCLOAK_TRUSTED_ISSUERS unset) is (KEYCLOAK_ISSUER,), i.e. exactly the
# single-issuer behaviour. See settings.trusted_issuers.
TRUSTED_ISSUERS: tuple[str, ...] = _settings.trusted_issuers()
JWKS_TTL_SECONDS = int(os.getenv("JWKS_TTL_SECONDS", "600"))
# HTTP timeout for the JWKS fetch itself. PyJWKClient defaults to 30s, which is
# long enough for an unresponsive Keycloak to pin a worker thread well past the
# kubelet's probe timeout (and past the ext_authz request budget on the hot
# path). Bounded here so both the request path and the readiness probe fail
# fast and deterministically instead of hanging.
JWKS_HTTP_TIMEOUT = float(os.getenv("JWKS_HTTP_TIMEOUT", "3.0"))
LICENSES_ACTIVE_URL = _settings.licenses_active_url
ACTIVE_LICENSE_CACHE_TTL = int(os.getenv("ACTIVE_LICENSE_CACHE_TTL", "30"))
MAX_CACHE_ENTRIES = int(os.getenv("LICENSE_AUTHZ_MAX_CACHE_ENTRIES", "1000"))

# === Audience verification (default OFF) ====================================
# Direction: FAILS OPEN. False (the default) keeps ``verify_aud: False``, i.e.
# exactly today's behaviour -- signature, issuer and exp are verified, the
# ``aud`` claim is not. True passes JWT_EXPECTED_AUDIENCES to ``jwt.decode``, so a
# token whose ``aud`` is not in that list is rejected as InvalidJWT -> 401.
#
# ⛔ WHICH AUDIENCE(S) REAL TOKENS CARRY: THIS REPO DOES NOT KNOW, AND THAT IS THE
# WHOLE REASON THE DEFAULT IS OFF. Grepped for ``aud`` across src/ and tests/ and
# every hit is one of two things, NEITHER of which is a portal browser token:
#   * synthetic test fixtures -- ``"aud": "right-aud"`` / ``["wrong-aud"]`` in
#     tests/test_jwt_mode.py, invented to exercise the match/mismatch branches.
#   * the audience license-authz MINTS on its own internal tokens --
#     ``settings.internal_token_audience``, default ``"internal"``
#     (src/internal_token.py). That is this service's OUTPUT, not the input
#     verified here.
# So there is no recorded, measured ``aud`` for a Keycloak portal token or a
# secure-apps SA token anywhere in the repo. A remembered default ("account", the
# client id, ...) is a guess, and a guess here is a fleet-wide 401.
#
# ⛔ ARMING THIS WITHOUT THAT LIST IS HOW YOU LOCK EVERYONE OUT: audience
# verification is all-or-nothing per token, it hits EVERY JWT caller
# simultaneously, and the failure presents as "all logins broken" with a generic
# ``invalid_token``. Measure first, from a live token:
#   kubectl -n istio-system logs deploy/license-authz | grep jwt_verify_failed
#   # or decode a real bearer: python -c 'import jwt,sys;
#   #   print(jwt.decode(sys.argv[1], options={"verify_signature": False})["aud"])'
# and note that the portal realm and the secure-apps realm may well carry
# DIFFERENT audiences -- both must be listed, or arming this breaks the realm you
# did not sample.
#
# ⭐ THERE IS NOW AN IN-PROCESS INSTRUMENT FOR THIS, so the measurement no longer
# needs a live bearer in a shell: set JWT_EXPECTED_AUDIENCES to a candidate list
# while LEAVING THIS FLAG OFF (safe -- `_validate_audience_config` only raises
# when the flag is armed AND the list is empty; a list with the flag off changes
# no decision), then read
# `license_authz_jwt_audience_shadow_total{verdict=...}`: it records, per already
# verified token, what arming WOULD have done. Arm this flag when `mismatch` and
# `absent` are both zero under real traffic across BOTH realms, and not before.
# See `audience_shadow_verdict` below. Because an empty list is the dangerous state, an armed flag with
# no audiences raises at import rather than starting a pod that 401s every
# request (see below) -- a pod that refuses to start is a rollout that halts; a
# pod that starts and rejects every token is an outage.
JWT_VERIFY_AUDIENCE = os.getenv("JWT_VERIFY_AUDIENCE", "false").lower() == "true"
JWT_EXPECTED_AUDIENCES: tuple[str, ...] = tuple(
    a.strip() for a in os.getenv("JWT_EXPECTED_AUDIENCES", "").split(",") if a.strip()
)


def _validate_audience_config(enabled: bool, audiences: tuple[str, ...]) -> None:
    """Refuse an armed flag with no audiences. Called at import; raises at boot.

    Same shape ``settings.model_post_init`` uses for the internal-token issuer's
    key+kid: a misconfigured opt-in must fail fast rather than become a 401 on
    every request. ``jwt.decode(..., audience=[])`` rejects every token, so the
    empty-list state is the lockout state — a pod that refuses to start halts a
    rollout, a pod that starts and rejects every token is an outage.

    A plain function rather than an inline ``if`` so the guard can be tested
    WITHOUT ``importlib.reload(jwt_mode)``: a reload resets this module's metrics
    hooks to their no-ops, silently unwiring main.py's counters for every test
    that runs afterwards in the same session.
    """
    if enabled and not audiences:
        raise RuntimeError(
            "JWT_VERIFY_AUDIENCE is enabled but JWT_EXPECTED_AUDIENCES is empty. "
            "Arming audience verification without the list of audiences real tokens "
            "carry rejects EVERY JWT (401 for every browser caller). Measure the `aud` "
            "claim on a live portal token and a live secure-apps SA token, then set "
            "JWT_EXPECTED_AUDIENCES to the comma-separated list of both."
        )


_validate_audience_config(JWT_VERIFY_AUDIENCE, JWT_EXPECTED_AUDIENCES)

# === Tenant claim name (task 6) =============================================
# Which claim carries the TENANT identity on a verified Keycloak JWT. Not a
# feature flag: it is a claim NAME, and it is configurable because the claim does
# not exist yet in either realm, so the name it eventually ships under is not
# this repo's to fix. Default ``tenant_id`` matches fr's ``deriveTenantId`` and
# the shape ``extract_tenant_id`` has always read.
TENANT_CLAIM = os.getenv("TENANT_CLAIM", "tenant_id")


# Hooks set by main.py to bump Prometheus counters without creating an import
# cycle. Default no-op so jwt_mode is independently importable in tests.
def _noop_jwks_refresh(_result: str) -> None:
    pass


def _noop_cache_eviction(_cache: str) -> None:
    pass


def _noop_tenant_claim(_source: str) -> None:
    pass


def _noop_audience_shadow(_verdict: str) -> None:
    pass


_on_jwks_refresh: Callable[[str], None] = _noop_jwks_refresh
_on_cache_eviction: Callable[[str], None] = _noop_cache_eviction
_on_tenant_claim: Callable[[str], None] = _noop_tenant_claim
_on_audience_shadow: Callable[[str], None] = _noop_audience_shadow


def set_metrics_hooks(
    on_jwks_refresh: Callable[[str], None] | None = None,
    on_cache_eviction: Callable[[str], None] | None = None,
    on_tenant_claim: Callable[[str], None] | None = None,
    on_audience_shadow: Callable[[str], None] | None = None,
) -> None:
    """Wire main.py's Prometheus counters into jwt_mode."""
    global _on_jwks_refresh, _on_cache_eviction, _on_tenant_claim, _on_audience_shadow
    if on_audience_shadow is not None:
        _on_audience_shadow = on_audience_shadow
    if on_jwks_refresh is not None:
        _on_jwks_refresh = on_jwks_refresh
    if on_cache_eviction is not None:
        _on_cache_eviction = on_cache_eviction
    if on_tenant_claim is not None:
        _on_tenant_claim = on_tenant_claim


def _b64url_decodable(part: str) -> bool:
    try:
        padded = part + "=" * (-len(part) % 4)
        base64.urlsafe_b64decode(padded.encode())
        return True
    except binascii.Error, ValueError:
        # Invalid base64 alphabet or padding. Cheap pre-branch check: opaque
        # license tokens routinely fail this — must NOT raise.
        return False


def _jwt_iss(payload_b64: str) -> str | None:
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except binascii.Error, ValueError, json.JSONDecodeError:
        # Malformed base64, non-JSON payload, or non-dict JSON. Pre-branch
        # check only — caller treats None as "not a Keycloak JWT".
        return None
    if not isinstance(claims, dict):
        return None
    iss = claims.get("iss")
    return iss if isinstance(iss, str) else None


def unverified_issuer(bearer: str) -> str | None:
    """``iss`` of a JWT-shaped bearer, UNVERIFIED. None for anything else.

    Only ever used to SELECT which allowlisted issuer's keys to verify with and
    which ``issuer=`` to pin — never trusted on its own. Compared EXACTLY (no
    normalisation of the token side), so a token is routed only when its ``iss``
    is byte-identical to an allowlist entry, as before.
    """
    parts = bearer.split(".")
    if len(parts) != 3:
        return None
    if not _b64url_decodable(parts[0]):
        return None
    return _jwt_iss(parts[1])


def unverified_azp(bearer: str) -> str | None:
    """``azp`` of a JWT-shaped bearer, UNVERIFIED. None for anything else.

    Same contract as :func:`unverified_issuer`: it only SELECTS a path (the
    secure-apps tenant-proxy arm in ``main.ext_authz_check``). The verified
    ``azp`` is re-checked after ``verify_jwt`` before anything is granted.
    """
    parts = bearer.split(".")
    if len(parts) != 3 or not _b64url_decodable(parts[0]):
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except binascii.Error, ValueError, json.JSONDecodeError:
        return None
    if not isinstance(claims, dict):
        return None
    azp = claims.get("azp")
    return azp if isinstance(azp, str) else None


def realm_of_issuer(issuer: str | None) -> str:
    """``https://host/realms/<realm>`` -> ``<realm>``; "" when not a realm issuer."""
    if not issuer or "/realms/" not in issuer:
        return ""
    return issuer.rpartition("/realms/")[2].strip("/")


def is_jwt_bearer(bearer: str, expected_iss: str | None = None) -> bool:
    """Detect Keycloak user JWTs without verifying — cheap pre-branch check.

    Opaque license tokens are URL-safe random strings with no dots; JWTs are
    three base64url segments separated by dots with a known issuer.

    ``expected_iss`` None (the default) = the ``TRUSTED_ISSUERS`` allowlist, which
    defaults to ``(KEYCLOAK_ISSUER,)``. Before the allowlist, a JWT from another
    realm failed this check and fell through to the OPAQUE licence-token path
    (an introspect miss); now an allowlisted realm is routed here and verified
    against its own JWKS with its own ``issuer=`` pin.
    """
    iss = unverified_issuer(bearer)
    if iss is None:
        return False
    if expected_iss is not None:
        return iss == expected_iss
    return iss in TRUSTED_ISSUERS


class JwksCache:
    """Thin wrapper around PyJWKClient with refresh on kid miss.

    PyJWKClient caches internally; we add a TTL and forced refresh path for
    when Keycloak rotates a key mid-session.
    """

    def __init__(
        self,
        jwks_url: str = JWKS_URL,
        ttl: int = JWKS_TTL_SECONDS,
        timeout: float = JWKS_HTTP_TIMEOUT,
    ) -> None:
        self._jwks_url = jwks_url
        self._ttl = ttl
        self._timeout = timeout
        self._client = self._new_client()
        self._loaded_at = time.monotonic()
        # TWO SEPARATE SIGNALS. Collapsing them into one is what made a
        # post-boot Keycloak outage undetectable (see readiness.py).
        #
        # last_load_ok -- outcome of the most recent REAL NETWORK FETCH. Moves
        # only where a fetch actually happened (eager boot fetch, readiness
        # re-validation, kid-miss refresh). None = never attempted. This is a
        # LIVE signal: it goes False the moment Keycloak stops answering and
        # True again when it comes back. It is REPORTED, never gating.
        #
        # Deliberately NOT derived from PyJWKClient's internal jwk_set_cache:
        # that cache EXPIRES every `ttl` seconds, so reading it would make a
        # perfectly healthy pod flap every JWKS_TTL_SECONDS. This flag only
        # moves on a real fetch outcome, so it is stable in steady state.
        self.last_load_ok: bool | None = None
        # ever_loaded_ok -- has this pod EVER held usable key material? A pod
        # whose boot fetch failed has an empty jwk_set_cache and an empty
        # per-kid LRU, so it 401s every JWT: its answers are WRONG and shedding
        # it genuinely helps, because a sibling that booted fine is unaffected.
        # A pod that loaded once and later lost Keycloak still verifies every
        # cached kid correctly, so it is the BEST AVAILABLE server and its
        # sibling is no better. This latch (never un-set) is what GATES
        # readiness; last_load_ok is not.
        self.ever_loaded_ok: bool = False
        # Monotonic timestamp of the last fetch attempt — used by the readiness
        # probe to rate-limit its re-validation so probes (every few seconds,
        # per replica) can never hammer Keycloak.
        self.last_attempt_at: float | None = None

    def _new_client(self) -> PyJWKClient:
        return PyJWKClient(
            self._jwks_url,
            cache_keys=True,
            lifespan=self._ttl,
            timeout=self._timeout,
        )

    def fetch(self) -> None:
        """Force a real JWKS fetch and latch the outcome. Re-raises on failure.

        Blocking (PyJWKClient uses urllib) — call via ``asyncio.to_thread``
        from async code. Bounded by ``JWKS_HTTP_TIMEOUT``.
        """
        self.last_attempt_at = time.monotonic()
        try:
            self._client.fetch_data()
        except jwt.PyJWKClientError, json.JSONDecodeError, OSError:
            # PyJWKClientConnectionError for network/HTTP, JSONDecodeError for a
            # malformed JWKS payload, OSError as a backstop for anything urllib
            # raises before pyjwt wraps it.
            self.last_load_ok = False
            _on_jwks_refresh("failure")
            raise
        self.last_load_ok = True
        self.ever_loaded_ok = True
        self._loaded_at = time.monotonic()
        _on_jwks_refresh("success")

    def _refresh(self) -> None:
        """Rebuild the client against a FRESH fetch. Re-raises on failure.

        Two things this must get right, both of which the first version did not:

        1. ``PyJWKClient.__init__`` performs ZERO network I/O — it only
           validates the URI scheme and allocates the caches. Counting a
           ``success`` for merely constructing one reported a fetch that never
           happened, so during a total Keycloak outage the failure counter
           stayed at zero and any alert built on it was blind. The metric is
           now bumped on the outcome of an ACTUAL ``fetch_data()``.
        2. The new client is only installed ON SUCCESS. Assigning it first
           threw away the working client — with its populated JWK-set cache and
           per-kid LRU — before finding out the replacement could not fetch, so
           a single kid miss during a Keycloak blip permanently destroyed the
           key material this pod was still serving correct answers from.
        """
        try:
            client = self._new_client()
            self.last_attempt_at = time.monotonic()
            client.fetch_data()
        except jwt.PyJWKClientError, json.JSONDecodeError, OSError, ValueError:
            # Network/HTTP failures fetching JWKS, a malformed JWKS payload, a
            # malformed URL, or socket errors. Latch the LIVE signal, bump the
            # failure metric, then re-raise so the caller (verify_jwt) converts
            # to InvalidJWT and we fail closed. Latching here is safe now that
            # readiness gates on ever_loaded_ok: it marks the pod degraded, it
            # cannot evict it. Note this only latches on a FETCH failure — a
            # stream of bogus kids fetches fine and raises "unable to find a
            # signing key" from the retry below, which is not a JWKS failure.
            self.last_load_ok = False
            _on_jwks_refresh("failure")
            raise
        self._client = client
        self._loaded_at = time.monotonic()
        self.last_load_ok = True
        self.ever_loaded_ok = True
        _on_jwks_refresh("success")

    def get_signing_key(self, token: str):
        try:
            key = self._client.get_signing_key_from_jwt(token)
        except jwt.exceptions.PyJWKClientError:
            # kid miss or stale cache — refresh once and retry.
            self._refresh()
            key = self._client.get_signing_key_from_jwt(token)
        # Resolving a signing key proves this pod HOLDS usable key material, so
        # it clears the gating latch. It does NOT set last_load_ok: the common
        # path here is an LRU/JWK-set-cache hit with zero network I/O, so it
        # observes nothing about whether Keycloak is currently reachable.
        # Asserting fetch health from a cache hit is the same fabrication as
        # counting a success for a fetch that never happened, and it would let
        # ordinary traffic paper over an ongoing JWKS outage.
        self.ever_loaded_ok = True
        return key


@dataclass
class VerifiedClaims:
    tenant_id: str
    sub: str | None
    raw: dict[str, Any]
    # The VERIFIED issuer (pinned via ``jwt.decode(issuer=...)``) and its realm.
    # Downstream keys app ownership on the realm (realm-per-app: the token's
    # realm names the app, ``tenant_id`` names the tenant).
    issuer: str = ""
    realm: str = ""


class IssuerJwks:
    """One ``JwksCache`` per allowlisted issuer (realm-per-app P0).

    The primary issuer (``KEYCLOAK_ISSUER``) uses the cache main.py already
    builds and the readiness probe gates on, so the single-issuer pod is
    unchanged. Extra issuers get their own cache, created up front from the
    allowlist — never from a token, so an attacker cannot grow this map.
    """

    def __init__(
        self,
        primary: JwksCache | None,
        primary_issuer: str | None = None,
        issuers: tuple[str, ...] | None = None,
        jwks_url_for: Callable[[str], str] | None = None,
    ) -> None:
        # Module globals read at CALL time (not bound as defaults at import) so
        # the allowlist in force is the one the module currently holds.
        self.primary = primary
        self.primary_issuer = KEYCLOAK_ISSUER if primary_issuer is None else primary_issuer
        self.issuers = TRUSTED_ISSUERS if issuers is None else issuers
        url_for = _settings.jwks_url_for_issuer if jwks_url_for is None else jwks_url_for
        self.extra: dict[str, JwksCache] = {
            iss: JwksCache(jwks_url=url_for(iss))
            for iss in self.issuers
            if iss != self.primary_issuer
        }

    def get(self, issuer: str) -> JwksCache | None:
        """Cache for an ALLOWLISTED issuer; None for anything else."""
        if issuer not in self.issuers:
            return None
        if issuer == self.primary_issuer:
            return self.primary
        return self.extra.get(issuer)


class InvalidJWT(Exception):
    """Signature invalid, expired, or otherwise unverifiable."""


def audience_shadow_verdict(claims: dict[str, Any]) -> str:
    """What arming JWT_VERIFY_AUDIENCE WOULD have done to this already-verified token.

    ⭐ WHY THIS EXISTS INSTEAD OF A DEFAULT FLIP. The brief asked for the default
    to flip if the correct audience could be established from the code. It cannot,
    and the one candidate on offer is a trap worth naming: ``INTERNAL_TOKEN_AUDIENCE``
    (live value ``internal``) is the audience license-authz MINTS onto tokens it
    ISSUES from ``/internal-token`` — this service's OUTPUT. ``verify_jwt``'s
    INPUT is a Keycloak portal browser JWT or a secure-apps SA token, minted by a
    different issuer entirely. Setting JWT_EXPECTED_AUDIENCES=internal would
    therefore 401 every caller on all three call sites at once. So the default
    stays OFF and this counter measures what arming would cost, which is the
    evidence the decision actually needs.

    Verdicts (four, bounded, no remote-controlled string ever becomes a label —
    the ``aud`` VALUES are deliberately not exported, only the verdict):

      unconfigured - JWT_EXPECTED_AUDIENCES is empty, so no verdict is possible.
                     ⚠️ THE LIVE STATE, and it is the instrument telling you it is
                     not yet loaded: an operator sets JWT_EXPECTED_AUDIENCES to a
                     candidate list (harmless while the flag is off —
                     ``_validate_audience_config`` only raises when the flag is
                     ARMED and the list empty) and the other three verdicts start
                     answering.
      absent       - the token carries no ``aud`` claim at all. WOULD BE REJECTED:
                     PyJWT raises MissingRequiredClaim once an audience is passed.
      match        - at least one of the token's ``aud`` values is in the expected
                     list. WOULD PASS (PyJWT accepts on a non-empty intersection).
      mismatch     - the token has an ``aud`` and none of it matches. WOULD BE
                     REJECTED. A malformed ``aud`` (a non-string element) lands
                     here too, which is correct for the question being asked:
                     PyJWT rejects that shape as well.

    ⚠️ Read this PER CALL SITE, not in aggregate. All three paths
    (``_handle_jwt_mode`` on the portal realm, ``_extract_caller_identity`` and
    ``/internal-token`` on secure-apps) feed one counter, and the two realms may
    carry different audiences — so a mixed ``match``/``mismatch`` reading is the
    expected shape of "one list cannot satisfy both", not noise.
    """
    if not JWT_EXPECTED_AUDIENCES:
        return "unconfigured"
    raw = claims.get("aud")
    if raw is None:
        return "absent"
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    if any(isinstance(v, str) and v in JWT_EXPECTED_AUDIENCES for v in values):
        return "match"
    return "mismatch"


def verify_jwt(
    token: str,
    jwks: JwksCache,
    issuer: str = KEYCLOAK_ISSUER,
    audience: list[str] | None = None,
) -> VerifiedClaims:
    """Verify signature + iss + exp (+ optional aud). Raise InvalidJWT on any failure.

    A NON-EMPTY ``audience`` list from the caller always wins. When the caller
    passes ``None``, ``JWT_VERIFY_AUDIENCE`` decides: off (default) keeps
    ``verify_aud: False`` — today's behaviour exactly — and on substitutes
    ``JWT_EXPECTED_AUDIENCES``. See that flag's declaration for which audiences real
    tokens carry (nothing in this repo records it) and why arming it blind is a
    fleet-wide 401.

    ⚠️ ``audience=None`` IS NOT AN OPT-OUT, and two call sites pass it. There is no
    way to distinguish "this call site deliberately wants no audience check" from
    "this call site did not think about it", so arming the flag arms audience
    verification on ALL THREE paths at once:
      * ``_handle_jwt_mode``      — portal realm browser JWTs (passes nothing)
      * ``_extract_caller_identity`` — secure-apps SA tokens (passes None explicitly)
      * the ``/internal-token`` mint endpoint — secure-apps SA tokens
    The last two verify against the SECURE-APPS issuer, so if that realm's tokens
    carry a different ``aud`` than the portal realm's — which is likely, they are
    different realms and different clients — one list cannot satisfy both and
    arming breaks whichever realm was not sampled. ⛔ Measure the ``aud`` on a live
    token from BOTH realms and list both, or add a per-call-site opt-out before
    arming. Pinned by
    ``test_arming_the_flag_also_affects_the_caller_identity_path``.
    """
    if audience is None and JWT_VERIFY_AUDIENCE:
        # Non-empty by construction: an armed flag with an empty list raises at
        # import, because `audience=[]` would reject every token.
        audience = list(JWT_EXPECTED_AUDIENCES)
    try:
        signing_key = jwks.get_signing_key(token)
        decode_kwargs: dict[str, Any] = {
            "algorithms": ["RS256", "ES256"],
            "issuer": issuer,
        }
        if audience is not None:
            decode_kwargs["audience"] = audience
            decode_kwargs["options"] = {"require": ["exp", "iss"]}
        else:
            decode_kwargs["options"] = {"verify_aud": False, "require": ["exp", "iss"]}
        claims = jwt.decode(token, signing_key.key, **decode_kwargs)
    except jwt.PyJWTError as e:
        raise InvalidJWT(str(e)) from e

    if audience is None:
        # SHADOW EVALUATION, and only on the population that would newly change.
        # `audience is None` means no audience check was applied to this token —
        # the flag is off, or the call site passed nothing. That is exactly the
        # set of tokens arming the flag would start judging, so it is the set to
        # measure. Tokens that DID carry an audience check are excluded: they
        # already got a real verdict, and counting them would mix a shadow with
        # an enforcement and make the "how many would break?" answer wrong.
        _on_audience_shadow(audience_shadow_verdict(claims))

    return VerifiedClaims(
        tenant_id=extract_tenant_id(claims),
        sub=claims.get("sub"),
        raw=claims,
        issuer=issuer,
        realm=realm_of_issuer(issuer),
    )


def extract_tenant_id(claims: dict[str, Any]) -> str:
    """Match fr's deriveTenantId order: TENANT_CLAIM > sub > "anonymous".

    "anonymous" is returned when neither is present; the caller MUST treat
    that as 401 (no per-(tenant,app) license can resolve for anonymous).

    ⚠️ THE ``sub`` FALLBACK IS A KNOWN DEFECT AND IT IS KEPT ON PURPOSE. ``sub`` is
    a per-USER Keycloak principal; ``tenant_id`` is a per-CUSTOMER billing
    identity. Promoting one into the other means two humans at the same customer
    resolve as two different tenants, so their licences, quotas and cost rollups
    split — which is the firing ``LicenseTenantIdIsKeycloakPrincipal`` alert, not a
    hypothetical.

    ⛔ Do NOT remove the fallback. MEASURED: neither Keycloak realm (``portal`` or
    secure-apps) emits a tenant claim at all today, so ``sub`` is the source for
    100% of JWT-mode traffic. Removing it returns "anonymous", which the caller
    turns into a 401 — i.e. it breaks every browser caller at once. The claim has
    to exist upstream FIRST.

    What this function adds instead is the MEASUREMENT that makes removal decidable
    later: every one of the three outcomes is counted through ``_on_tenant_claim``
    (``tenant_claim`` / ``sub`` / ``absent``). All three, not just the fallback,
    because a fallback count with no denominator cannot answer "has the claim
    started arriving?". The fallback becomes removable when ``sub`` reaches zero
    under real traffic — and the counter is the only thing that can say so.

    ⚠️ Counting happens HERE rather than at the call site because this is the only
    place the decision is made, and it is reached by both ``verify_jwt`` paths
    (portal JWT-mode and secure-apps caller identity). Hooked via
    ``set_metrics_hooks`` for the same no-import-cycle reason as the other two
    counters; the default hook is a no-op, so importing jwt_mode standalone (and
    every unit test of this function) stays free of Prometheus.
    """
    t = claims.get(TENANT_CLAIM)
    if isinstance(t, str) and t:
        _on_tenant_claim("tenant_claim")
        return t
    sub = claims.get("sub")
    if isinstance(sub, str) and sub:
        _on_tenant_claim("sub")
        return sub
    _on_tenant_claim("absent")
    return "anonymous"


class ActiveLicenseResolver:
    """Resolve (tenant_id, app_key) → license via licenses /licenses/active.

    In-memory TTL cache (30s) keyed on the pair. Small memory footprint: one
    entry per distinct (tenant, app) seen within the window.
    """

    _SENTINEL_404 = object()

    def __init__(
        self,
        url: str = LICENSES_ACTIVE_URL,
        ttl: int = ACTIVE_LICENSE_CACHE_TTL,
        max_entries: int = MAX_CACHE_ENTRIES,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[str, ...], tuple[float, Any]] = OrderedDict()

    def _cached(self, key: tuple[str, ...]) -> Any | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        # LRU touch
        self._cache.move_to_end(key)
        return value

    def _put(self, key: tuple[str, ...], value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
            _on_cache_eviction("active_license")
        self._cache[key] = (time.monotonic() + self._ttl, value)

    async def resolve(
        self,
        client: httpx.AsyncClient,
        tenant_id: str,
        app_key: str,
        auth_headers: dict[str, str],
        realm: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the introspect-shaped license dict, or None if 404.

        ``realm`` is the VERIFIED issuer realm of a NON-primary user token. It is
        sent to licenses as the ``realm`` query param so licenses can apply its
        ``realm_owner`` check (realm-per-app P0) — the owner table lives there,
        not here. None (every primary-issuer token) sends exactly today's
        request. The realm is part of the cache key so an answer for one realm
        is never served to another.
        """
        key = (tenant_id, app_key) if not realm else (tenant_id, app_key, realm)
        cached = self._cached(key)
        if cached is self._SENTINEL_404:
            return None
        if cached is not None:
            return cached

        params = {"tenant_id": tenant_id, "app_key": app_key}
        if realm:
            params["realm"] = realm
        resp = await client.get(
            self._url,
            params=params,
            headers=auth_headers,
        )
        if resp.status_code == 404:
            self._put(key, self._SENTINEL_404)
            return None
        if resp.status_code == 200:
            data = resp.json()
            self._put(key, data)
            return data
        # 409 multiple licenses, 5xx, etc — don't cache, let caller decide.
        resp.raise_for_status()
        return None
