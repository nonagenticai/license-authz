"""
License AuthZ Service - HTTP ext_authz for Envoy/Istio

This service validates license tokens and injects trusted tenant headers.
It's called by Envoy's ext_authz filter at the IngressGateway.

Flow:
1. Envoy sends request headers (including Authorization: Bearer <token>)
2. We call /license/introspect to validate the token
3. If valid: return 200 with headers to inject (x-tenant-id, x-license-id, etc)
4. If invalid: return 403 with error details

SECURITY: All tokens MUST be validated via license introspection.
No passthrough - fail-closed security model.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import httpx
import jwt
import redis.asyncio as aioredis
import redis.exceptions
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel
from redis.typing import EncodableT, FieldT

import internal_token
import jwt_mode
import metering_client as _metering_client_mod
import readiness
from internal_token import IssuerDisabled
from jwt_mode import (
    ActiveLicenseResolver,
    InvalidJWT,
    IssuerJwks,
    JwksCache,
    is_jwt_bearer,
    unverified_azp,
    unverified_issuer,
    verify_jwt,
)
from metering_client import BudgetDecision, MeteringClient
from metrics_push import MetricsPusher
from service_auth import ServiceTokenProvider
from settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# Configuration — service URLs come from gitops Deployment env (no source-side
# fallback; see gitops-production/docs/config-externalization-policy.md).
_settings = get_settings()
LICENSE_INTROSPECT_URL = _settings.license_introspect_url
LICENSES_INTERNAL_TENANTS_URL = _settings.licenses_internal_tenants_url
# Wall-clock budget for a single introspection POST. Default 10s (was 2s): under
# the concurrent e2e burst the licenses /introspect endpoint can exceed a tight
# 2s budget, and a timeout was conflated with an invalid token (see
# stale-while-error below). Aligns with the w2v_common in-process middleware.
INTROSPECT_TIMEOUT = float(os.getenv("INTROSPECT_TIMEOUT", "10.0"))

# How often the background refresher renews the Keycloak service-account token.
# ServiceTokenProvider retires its cache 30 s before Keycloak's expiry (≈270 s
# for the default 300 s lifetime), so 200 s keeps the token continuously valid.
# 0 disables the refresher; the request path still mints on demand.
SA_TOKEN_REFRESH_SECONDS = float(os.getenv("LICENSE_AUTHZ_SA_TOKEN_REFRESH_SECONDS", "200"))

# Tenant identity: the licenses service is the single owner of tenant identity.
# /license/introspect returns the canonical tenant UUID (any UUID version — the
# licenses boundary validator `validate_tenant_uuid` accepts any version).
# license-authz trusts the introspected UUID directly and injects it as
# x-tenant-id; it no longer re-resolves a name via /license/tenants/{name} (that
# lookup 403'd every non-v7-UUID tenant).


def _is_uuid(value: str) -> bool:
    """True if ``value`` is a well-formed UUID of any version.

    Matches the licenses service contract (`auth/tenant.py:validate_tenant_uuid`,
    which accepts any UUID version). A canonical tenant_id from introspect IS the
    authoritative identity — inject it directly.
    """
    try:
        UUID(value)
        return True
    except ValueError, AttributeError, TypeError:
        return False


# Metering configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")
METERING_STREAM = os.getenv("METERING_STREAM", "billing:events")
METERING_ENABLED = os.getenv("METERING_ENABLED", "true").lower() == "true"

# Lazy-reconnect policy for the metering/budget Redis. See _get_redis: the
# redis client already retries mid-life drops on its own (Retry with
# ExponentialWithJitterBackoff, 10 retries, wired into execute_command), so
# these knobs govern ONLY the case redis-py cannot cover — never having had a
# client at all, because the connect itself failed.
REDIS_RECONNECT_MIN_INTERVAL = float(os.getenv("REDIS_RECONNECT_MIN_INTERVAL", "5.0"))
REDIS_RECONNECT_MAX_INTERVAL = float(os.getenv("REDIS_RECONNECT_MAX_INTERVAL", "60.0"))
# Bounds the whole connect+PING, DNS included. Must stay well under
# readiness.TOTAL_BUDGET (2.0s) because /ready is what drives the reconnect.
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "0.5"))
# Per-command socket timeout on the established client. Without it a
# black-holed Redis hangs the ext_authz hot path (XADD in the metering emit,
# the only Redis call left) with no bound at all.
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "1.0"))

# Budget enforcement
BUDGET_ENFORCEMENT_ENABLED = os.getenv("BUDGET_ENFORCEMENT_ENABLED", "false").lower() == "true"

# JWT-mode (Keycloak JWTs from oauth2-proxy browser hosts)
JWT_MODE_ENABLED = os.getenv("LICENSE_AUTHZ_JWT_MODE_ENABLED", "false").lower() == "true"

# === ASSERTED app keys may only authorize a REGISTERED route =================
# ⚠️ Direction: this SHIPS ON. The env var is the REVERT, not the arming switch,
# which is the opposite of every other flag in this file — so read this before
# assuming the usual shape.
#
# THE DEFECT. There are two ways this service learns an app key, and they are not
# equally trustworthy:
#   ATTESTED  - the opaque-licence-token path. `/license/introspect` RETURNS the
#               key as part of the credential's own contents. The caller chose
#               nothing; the licences service stated it.
#   ASSERTED  - the JWT path. The caller sets the unsigned `x-app-key` header,
#               and that header is the key `/licenses/active` is QUERIED by, so
#               the caller picks which of its tenant's licences funds the call
#               (see the comment at the header read in `_handle_jwt_mode`).
#
# What the verified JWT establishes is the TENANT — which is why this is a
# mis-charging defect and not a cross-tenant one, `/licenses/active` cannot
# return another tenant's licence. What it does NOT establish is the APP: ⛔ no
# claim in EITHER realm names an app (the same measurement recorded at
# jwt_mode.JWT_VERIFY_AUDIENCE and TENANT_CLAIM — the repo knows of no such
# claim), so there is nothing in the credential to match the asserted key
# against, and inventing one would be a guess dressed as a check.
#
# So the only binding available for an asserted key is the ROUTE. On a REGISTERED
# prefix the registry supplies it and `_check_app_route` already enforces it. On
# an UNREGISTERED path there is nothing to check against at all, and the
# pre-existing `return True` handed the caller a free choice of licence. That arm
# is now REFUSED (403 `app_route_unregistered_asserted_key`).
#
# ⭐ WHY THIS IS SAFE TO SHIP ON: `license_authz_jwt_mode_total` measured 0 over
# 24h — the asserted-key path carries no live traffic, so this deny cannot reject
# a live request today. (⚠️ That is a MEASUREMENT WITH A SHELF LIFE. Re-read the
# counter before treating "ships on" as "costs nothing"; the moment JWT mode
# carries traffic, an unregistered path becomes a 403 for a real caller.)
#
# Setting ASSERTED_APP_KEY_ALLOW_UNREGISTERED=true restores the permissive
# behaviour byte-for-byte. The outcome is counted on `metric_app_route` EITHER
# WAY — see `allowed_asserted_key_unregistered` there.
#
# ⛔ This does NOT touch the ATTESTED path. An introspected key on an
# unregistered path still allows, exactly as before: the caller chose nothing, so
# there is no defect to close and denying it would be a new denial class over the
# entire live licence-token population.
ASSERTED_APP_KEY_ALLOW_UNREGISTERED = (
    os.getenv("ASSERTED_APP_KEY_ALLOW_UNREGISTERED", "false").lower() == "true"
)

# Service-principal auth (opt-in): when ON, a request that carries a *verified*
# secure-apps SA token in X-Caller-Authorization but NO primary license token in
# Authorization is PERMITted as a trusted first-party service principal. Identity
# only — the caller overlay (x-caller-client-id / x-caller-roles) is injected and
# tenant/license headers are empty; downstream services enforce their own tenant
# and policy scope. Default OFF so the shared license-token path is unchanged for
# every other waypoint until this is deliberately enabled.
SERVICE_PRINCIPAL_AUTH_ENABLED = (
    os.getenv("SERVICE_PRINCIPAL_AUTH_ENABLED", "false").lower() == "true"
)
# Optional allowlist of caller client-ids (azp) permitted on the service-principal
# path. Comma-separated. EMPTY ⇒ allow ANY verified secure-apps SA token
# (backward-compatible with the original #22 behavior); set it to lock the path
# down to named first-party service accounts (e.g. "marketing-service-account").
# A verified SA whose client-id is not on a non-empty allowlist is refused and
# falls through to the standard missing_bearer_token 401. Parsed once at import;
# tests monkeypatch the frozenset directly.
SERVICE_PRINCIPAL_ALLOWED_CLIENTS: frozenset[str] = frozenset(
    c.strip() for c in os.getenv("SERVICE_PRINCIPAL_ALLOWED_CLIENTS", "").split(",") if c.strip()
)


# --- Service-principal identity resolution (azp -> licences tenant) ---------
# The service-principal fast path (below) emits metering with tenant_id="" —
# ATTRIBUTABLE to a caller but not BILLABLE to a tenant, because there was no
# licences-side identity to resolve it to. This map is that identity: for each
# known SP `azp`, the CANDIDATE tenant `name` to look up, by exact local
# match, in the name -> id map built from licences `GET /internal/tenants`
# (see `_resolve_service_principal_tenant`).
#
# Why the LIST endpoint and not per-name `GET /tenants/{name}`: licences
# deliberately refuses service accounts on `GET /tenants*` with 403 (its own
# test suite pins that billing cannot reach it). `GET /internal/tenants` is
# the sanctioned internal route (guard `require_internal_service`: any of
# service-internal | onboarding-internal | license-admin), this service's own
# SA token already carries `service-internal`, and billing reads the same
# route the same way (`billing_service/src/services/tenant_names.py`). So this
# needs NO licences change.
#
# The candidate names follow the registry's OWN convention for a service's
# own tenant: `svc-<service>` with kind=service (45 such rows live 2026-09-16:
# svc-kag, svc-karma, svc-fr-backend, ...; `provision_service_token.py` mints
# them with `name=<subject slug>`). Measured LIVE against the licences
# registry 2026-09-16 (`select id,name,kind from tenants where kind='service'`
# plus a regex sweep for marketing|sales|growth|relay|console|platform over
# ALL kinds): NONE of the five has a `svc-*` row today, so all five resolve to
# result="unresolved" until an operator provisions the row. Operator TODO,
# one per SA (the map then resolves with no code change):
#   marketing-service-account -> `svc-marketing`        NO ROW
#   sales-service-account     -> `svc-sales`            NO ROW
#   growth-service-account    -> `svc-growth`           NO ROW
#   control-relay-account     -> `svc-control-relay`    NO ROW
#   platform-console-backend  -> `svc-platform-console` NO ROW
#
# ⛔ Rows that LOOK like a match and were deliberately NOT bound, because the
# registry does not state the link and binding would be a billing decision
# taken here instead of by the operator:
#   - `marketing` / `sales` (kind=internal, cost_centre=merqi): they hold the
#     `marketing`/`sales` APP licences (subjects `marketing-service` /
#     `sales-service`, realm nonagentic) — the business tenants the apps are
#     licensed under, not the services' own tenants. Attributing SA traffic
#     there would land service overhead in merqi's cost centre; sales' own
#     control-service licence token is even minted for `tenant-dev`, so "the
#     sales service's tenant" has two candidate answers, which is exactly
#     why it must be an operator's choice.
#   - `svc-marketing-t-c46c9378` (kind=service): provisioned 2026-09-13 by
#     `provision_service_token.py` as an OPAQUE on-behalf-of token subject
#     (realm `service`), a different credential from the Keycloak azp.
#   - `platform-core` (kind=internal): differently named; not assumed.
# A candidate absent from the licences list resolves to result="unresolved",
# tenant_id="" — unchanged from today. Nothing here
# creates a tenant; it only asks a question licences is authoritative to
# answer. Env-overridable ("client=name,client=name,...") so an operator who
# decides one of the rows above IS the right identity can bind it without a
# code change — that decision is theirs, not this file's.
def _parse_sp_tenant_name_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        client_id, _, name = pair.partition("=")
        client_id, name = client_id.strip(), name.strip()
        if client_id and name:
            out[client_id] = name
    return out


SERVICE_PRINCIPAL_TENANT_NAME_MAP: dict[str, str] = _parse_sp_tenant_name_map(
    os.getenv(
        "SERVICE_PRINCIPAL_TENANT_NAME_MAP",
        "marketing-service-account=svc-marketing,"
        "sales-service-account=svc-sales,"
        "growth-service-account=svc-growth,"
        "control-relay-account=svc-control-relay,"
        "platform-console-backend=svc-platform-console",
    )
)
# Bounds the `client_id` label on metric_service_principal_identity to exactly
# the callers this map knows how to attempt — the same cardinality-clamp
# reasoning as `_sp_caller_label` / SERVICE_PRINCIPAL_METRIC_CALLERS. A
# verified azp outside this set is not attempted (no candidate name to ask
# licences for) and is counted under the `other` bucket.
SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS: frozenset[str] = frozenset(
    SERVICE_PRINCIPAL_TENANT_NAME_MAP
)

# --- Trusted tenant-proxy clients (JWT-mode X-Tenant-Id passthrough) --------
# `git grep -i trusted origin/main -- src/` (checked before adding this) turns
# up no equivalent setting in this service — the licenses service is the only
# place a tenant-proxy allowlist already exists
# (`licenses/auth/config.py:104`, `_DEFAULT_TENANT_PROXY_CLIENTS`). This is
# NOT a new policy: it names the same set of service accounts that service
# already trusts to retarget `X-Tenant-Id` (fr-backend forwards the end
# user's tenant; w2v-backend likewise). Same env var name, same default,
# deliberately, so the two allowlists cannot drift apart.
#
# Verified `azp` (client_id) of a JWT-mode caller's own bearer token, checked
# against this set, is what gates whether that caller's client-supplied
# `X-Tenant-Id` survives the header rewrite at the ext_authz response-header
# build (see the `x-tenant-id` injection below). This is a WIDENING: today
# every caller's `X-Tenant-Id` is unconditionally overwritten with its own
# tenant, so no caller — trusted or not — can ever assert another tenant
# through this service. A caller NOT on this list keeps that exact,
# unchanged behavior; only a listed, verified `azp` may have its asserted
# tenant preserved, and only when it is a well-formed UUID. Downstream
# (e.g. `licenses/auth/middleware.py:404`) still enforces its own tenant
# scope on top of this — this setting only decides whether the header
# reaches it unrewritten.
#
# NARROWED 2026-09-30 (license-authz#64): the default is `w2v-backend` ONLY.
# Measured over the feature's whole life (Loki, 09-16 -> 09-30): 66 retargets,
# 66/66 `w2v-backend`, 0 `fr-backend`. fr reaches licenses in-cluster
# (its Service DNS name), never through the gateway, so
# this service is never consulted for it and the privilege was pure exposure.
# ⚠️ This deliberately DIVERGES from `licenses/auth/config.py`'s
# `_DEFAULT_TENANT_PROXY_CLIENTS`, which still lists fr-backend and is what
# governs fr's in-cluster path. That is a separate repo's decision.
_DEFAULT_TRUSTED_TENANT_PROXY_CLIENTS = "w2v-backend"
TRUSTED_TENANT_PROXY_CLIENTS: frozenset[str] = frozenset(
    c.strip()
    for c in os.getenv("TRUSTED_TENANT_PROXY_CLIENTS", _DEFAULT_TRUSTED_TENANT_PROXY_CLIENTS).split(
        ","
    )
    if c.strip()
)

# --- secure-apps S2S tenant proxy (realm-per-app P7) -------------------------
# Realm `portal` is becoming STAFF-ONLY: service-to-service tokens move to realm
# `secure-apps` (gitops#4913). The last portal minter reaching this service is
# tts-backend on mini1/mini2, which presents a `w2v-backend` client-credentials
# JWT on the licences host and relies on the tenant-proxy retarget above. Its
# secure-apps twin could not take JWT mode: `secure-apps` is (rightly) refused in
# the user-JWT allowlist KEYCLOAK_TRUSTED_ISSUERS because that realm also holds
# human accounts, so a secure-apps bearer fell through to opaque introspection
# and was refused.
#
# This admits a secure-apps bearer into JWT mode ONLY when ALL of these hold,
# and nothing else about the realm is trusted:
#   1. `iss` == KEYCLOAK_SECURE_APPS_ISSUER, verified against the secure-apps
#      JWKS (`app.state.caller_jwks`) with `issuer=` pinned to that value;
#   2. the VERIFIED `azp` is in SECURE_APPS_TENANT_PROXY_CLIENTS (default
#      `w2v-backend` only) — a human's secure-apps login carries a browser
#      client's azp and can never match;
#   3. the VERIFIED `realm_access.roles` holds SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE
#      (default `license-reader`, which the secure-apps w2v-backend SA holds per
#      keycloak/realm-patch-job-p7-s2s-secure-apps.yaml). Empty = no role check.
# The unverified `iss`/`azp` only SELECT this arm; 2 and 3 are re-checked on the
# verified claims, and a miss is a 401 (`invalid_token`), never a fallback.
# Portal user JWTs for staff hosts (kanban / n8n / grafana) are untouched: they
# never match `iss` == secure-apps. Empty SECURE_APPS_TENANT_PROXY_CLIENTS
# disables the arm entirely (byte-identical to before).
_DEFAULT_SECURE_APPS_TENANT_PROXY_CLIENTS = "w2v-backend"
SECURE_APPS_TENANT_PROXY_CLIENTS: frozenset[str] = frozenset(
    c.strip()
    for c in os.getenv(
        "SECURE_APPS_TENANT_PROXY_CLIENTS", _DEFAULT_SECURE_APPS_TENANT_PROXY_CLIENTS
    ).split(",")
    if c.strip()
)
SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE: str = os.getenv(
    "SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE", "license-reader"
).strip()


def _secure_apps_issuer() -> str:
    return (_settings.keycloak_secure_apps_issuer or "").rstrip("/")


def _is_secure_apps_tenant_proxy_bearer(bearer: str) -> bool:
    """SELECT the secure-apps tenant-proxy arm from UNVERIFIED claims.

    True only for a JWT whose `iss` is exactly the configured secure-apps issuer
    and whose `azp` is allowlisted. Selection only — `_handle_jwt_mode`
    re-verifies both (plus the role) on signature-checked claims.
    """
    sa_issuer = _secure_apps_issuer()
    if not sa_issuer or not SECURE_APPS_TENANT_PROXY_CLIENTS:
        return False
    if unverified_issuer(bearer) != sa_issuer:
        return False
    return unverified_azp(bearer) in SECURE_APPS_TENANT_PROXY_CLIENTS


def _secure_apps_proxy_refusal(raw: dict) -> str | None:
    """Why a VERIFIED secure-apps token may NOT act as tenant proxy, or None."""
    azp = raw.get("azp")
    if not isinstance(azp, str) or azp not in SECURE_APPS_TENANT_PROXY_CLIENTS:
        return "azp_not_allowlisted"
    if SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE:
        realm_access = raw.get("realm_access")
        roles = realm_access.get("roles") if isinstance(realm_access, dict) else None
        if not isinstance(roles, list) or SECURE_APPS_TENANT_PROXY_REQUIRED_ROLE not in roles:
            return "required_role_missing"
    return None


# The ONLY target hosts on which a trusted proxy may retarget the tenant
# (license-authz#64). Before this, a listed `azp` could retarget on EVERY gated
# internal host (secrets, kag, workers, procedures, ...) although the only
# need ever observed is the licences host itself (66/66 measured). A retarget
# attempt on any other host is REFUSED with a 403 rather than silently
# downgraded to the caller's own tenant: a trusted proxy that asks for another
# tenant somewhere it may not have one is a bug or a stolen secret, and a
# silent downgrade would hand it an answer about the WRONG tenant.
#
# The target host is read from `x-forwarded-host`, which the ingress gateway's
# Lua filter sets from the original `:authority` (port-stripped) in the
# explicitly built httpCall header table (ext-authz-enforce.yaml, the
# `authz_req_headers` table) — on that call `:authority` is license-authz's
# own Service name, so it cannot identify the target. Falls back to
# `:authority`/`Host` for a direct in-mesh call that carries no
# `x-forwarded-host`. Entries are compared lowercased, port-stripped.
#
# ⚠️ DEPLOYMENT CONFIGURATION, NOT SOURCE (2026-11-14). This used to default
# to the licences host. The set is EMPTY unless `TENANT_PROXY_RETARGET_HOSTS`
# is set, so a deployment that does not set it simply never retargets — the
# safe direction for a privilege this narrow. The production value lives in
# the deployment manifest, not here.
TENANT_PROXY_RETARGET_HOSTS: frozenset[str] = frozenset(
    h.strip().lower().rstrip(".")
    for h in os.getenv("TENANT_PROXY_RETARGET_HOSTS", "").split(",")
    if h.strip()
)


def _retarget_target_host(request: Request) -> str:
    """The host a tenant-proxy retarget would apply to, normalised.

    `x-forwarded-host` first (the gateway's own record of the original
    `:authority`), then `:authority`/`Host`. Lowercased, port and trailing dot
    stripped. Returns "" when nothing usable is present — which is never in
    `TENANT_PROXY_RETARGET_HOSTS`, so an unidentifiable target is refused.
    """
    raw = (
        request.headers.get("x-forwarded-host")
        or request.headers.get(":authority")
        or request.headers.get("host")
        or ""
    )
    # x-forwarded-host may be a comma list after several proxies; the FIRST
    # entry is the original client-facing host.
    return raw.split(",")[0].strip().split(":")[0].strip().lower().rstrip(".")


# --- P1: licence self-read exemption ----------------------------------------
# DIRECTION: DEFAULT **ON**. Reading your OWN licence is introspection of the
# credential you already hold, not consumption of the `licenses` app.
#
# THE OUTAGE THIS CLOSES (prod, 2026-09-14): a bare-metal w2v host reads its own
# licence at `GET https://<licences-host>/licenses/active`. That path matches
# the REGISTERED prefix `/licenses/`, which demands app_key `licenses`; the
# caller holds a `w2v` licence, so `_check_app_route` denied 403
# `app_route_mismatch`, the caller's entitlement gate went INDETERMINATE, and it
# 503'd every request. In-cluster pods bypass the gateway entirely, so only bare
# metal broke and the defect looked host-specific. The gate had to be DISARMED in
# prod to restore service; this exemption is what lets it be re-armed.
#
# ⭐ WHY ON IS THE SAFE DEFAULT — the blast radius is bounded in one direction.
# The exemption can ONLY convert a 403 into the pre-existing downstream checks
# (budget, tenant, metering, and the licenses service's own authorization of the
# read). It cannot create a NEW allow of anything mutating: it fires only for
# GET/HEAD, only on the two paths in APP_ROUTE_SELF_READ, and only for a caller
# that already presented a VALID, ACTIVE, INTROSPECTED licence. A caller with no
# licence is 401'd long before this code runs, and `/licenses/active` returns
# only the calling tenant's own licence.
#
# ⛔ WHAT STAYS BOUND, and every one of these is pinned by a test: any non-GET
# method on the same two paths, `/licenses/issue`, `/licenses/revoke`, anything
# under `/licenses/admin/`, and every other `/licenses/` path. Also deliberately
# still bound: a token with NO app_key at all — that deny
# (`app_route_missing_key`) is a separate hardening ("one unkeyed token must not
# authorize all 28 registered prefixes") and the outage caller is keyed (`w2v`),
# so nothing requires re-opening it here.
#
# ⚠️ AMENDED 2026-09-16 — THIS FLAG NOW GOVERNS TWO ARMS, NOT ONE, and the
# paragraph above describes only the first. The sentences "only for a caller
# that already presented a VALID, ACTIVE, INTROSPECTED licence" and "a caller
# with no licence is 401'd long before this code runs" are true of the OPAQUE
# path and NOT of the JWT path, which has no introspected licence to speak of.
#
# The second arm is in `_handle_jwt_mode`'s `license_required` exit. mini1/mini2
# `tts_backend` moved from an opaque licence token to a `w2v-backend`
# client-credentials JWT, which sends no `x-app-key`; the key is then DERIVED
# from the route (`/licenses/` -> `licenses`), so the gate demanded the LICENSES
# app for a self-read and answered 402 -> entitlement `unknown` -> narration
# 503. That arm reuses `_is_self_read_route` and this flag, and is bounded
# additionally to the DERIVED-key case, but it DOES allow a caller holding no
# licence at all — bounded by the verified `claims.tenant_id`, with an EMPTY
# licence (no license_id/subject/plan), so budget skips on `no_license_id` and
# nothing is chargeable. Read that arm's own comment before changing this flag.
#
# Setting LICENSE_SELF_READ_EXEMPT_ENABLED=false restores the pre-fix behaviour
# byte-for-byte on BOTH arms (and re-opens both outages). The outcome is VISIBLE
# either way: the opaque arm counts `metric_app_route{result=
# "allowed_self_read_exempt"}` and logs `license_self_read_exempt`, the JWT arm
# counts `metric_jwt_mode{result="allowed_self_read_exempt"}` and logs
# `jwt_mode_self_read_exempt`; when the flag is off the same requests land on
# `denied_mismatch` / `license_required` exactly as they do today.
LICENSE_SELF_READ_EXEMPT_ENABLED = (
    os.getenv("LICENSE_SELF_READ_EXEMPT_ENABLED", "true").lower() == "true"
)

# --- P2: host -> app binding (OBSERVE-FIRST, DEFAULT OFF) --------------------
# THE GAP: marketing and sales are HOST-routed
# (`marketing.nonagentic.ai`, `marketing.merqi.nl`, `sales.merqi.nl`) and BOTH
# serve under the SAME `/api/...` prefix. A path-prefix registry physically
# cannot tell them apart, so neither is registered at all — and an unregistered
# path allows ANY key (`_check_app_route` returns True). Consequence today: any
# licensed token of any app passes BOTH waypoints.
#
# ⛔ THIS SHIPS INERT. `off` is the default and `off` does nothing at all — no
# deny, no allow-with-metric, not even a counter increment — so this PR cannot
# change a single live verdict. Arming (`observe`, then `enforce`) is a later,
# separately-measured operational step and deliberately does not appear here.
#
# BLAST RADIUS ON THE 29 LIVE PATH PREFIXES IS EXACTLY ZERO, structurally: the
# host registry is consulted ONLY when `_registered_app_key(path)` returned None.
# A request whose path matched a registered prefix never reaches this code, so
# every existing enforcement verdict is untouched by construction.
#
# ⚠️ THE app_key VALUES BELOW ARE THE UNVERIFIED HALF. The HOSTS are verified
# against gitops (`clusters/strato/services-istio/{marketing,sales}/
# virtualservice.yaml`, read 2026-09-15) — note there is NO `sales.nonagentic.ai`,
# sales publishes on merqi.nl only. What is NOT verified is which app_key the
# licences these callers present actually carry. That is precisely what `observe`
# is for: run it, read `mismatch_observed`, and correct the mapping via
# APP_HOST_REGISTRY_OVERRIDE before anyone types `enforce`.
#
# ⚠️ AND THE PARAGRAPH ABOVE WAS WRONG ABOUT THE VERIFIED HALF. "Verified
# against the VirtualService" only proves the host serves the app publicly — it
# says nothing about whether that authority ever reaches ext_authz. It does
# not. See the next comment block, immediately above the registry seed.
# ⛔ THE ONE FACT THAT MAKES OR BREAKS THIS TABLE: A PUBLIC HOSTNAME NEVER
# ARRIVES HERE. The gateway Lua (EnvoyFilter `ext-authz-enforce-mode`, ns
# `istio-system`) calls ext_authz ONLY for hosts listed in `AUTH_REQUIRED_HOSTS`,
# and that list is ~30 internal names — the marketing API,
# the sales API and the control API. The PUBLIC hosts
# (`marketing.nonagentic.ai`, `marketing.merqi.nl`, `sales.merqi.nl`) are in
# NONE of the gateway's gating tables — not `AUTH_REQUIRED_HOSTS`, not
# `BROWSER_AUTH_HOSTS`, not `STRICT_BROWSER_AUTH_HOSTS`, not
# `LICENSE_GATED_BROWSER_HOSTS`. Live traffic to them goes `via_upstream` with
# NO authz call at all.
#
# Measured 2026-09-15: seeded with the public hosts ONLY, this registry was
# INERT IN BOTH DIRECTIONS — the three keys could never arrive, and the
# authorities that DO arrive were all `unregistered`. Arming `observe` against
# it could only ever have produced `no_host`/`unregistered`, i.e. a measurement
# window that cannot measure anything.
#
# ⇒ RULE FOR THE NEXT PERSON: key this registry on the authority license-authz
# ACTUALLY RECEIVES. Before adding a host, confirm it appears in a gateway
# gating table; a host that does not is a decorative entry, not a binding.
#
# The public names are KEPT on purpose: they cost nothing (they simply never
# arrive today) and become correct automatically the day one of those hosts is
# added to a gating table.
#
# The control API host is deliberately NOT here: control_service's app_key is
# unconfirmed, and guessing it would be the same class of error as the one this
# table is fixing.
#
# ⚠️ WHAT THIS STILL DOES NOT PROVE. Making the keys arrivable is necessary, not
# sufficient: it is still UNPROVEN live that Envoy forwards the caller's
# authority into the check body (gitops #3919 added `host` to
# `includeRequestHeadersInCheck`; the counter has never incremented). The honest
# claim is that an `observe` window becomes MEANINGFUL rather than vacuously
# `no_host`/`unregistered` — not that the binding works.
# ⚠️⚠️ 2026-09-15 ADVERSARIAL AUDIT: THE TWO app_key VALUES BELOW ARE
# PROVISIONAL AND UNVERIFIED — DO NOT READ THEM AS ESTABLISHED FACT.
#
# THE REASON TO DOUBT THEM: the caller on BOTH the marketing API host and
# the sales API host is the SAME non-mesh growth app (the gateway injects
# `growth-service-account` on both hops — `ext-authz-enforce.yaml:1306/1331`).
# growth presents ONE per-tenant licence token — keyed by tenant id only
# (`GROWTH_TENANT_LICENSE_TOKENS`, `growth/src/clients/tenant-token.ts`) — on
# the marketing, sales AND control hops alike
# (`marketing-service-client.ts`, `sales-service-client.ts`,
# `control-service-client.ts` all call `this.tenantToken(tenantId)`). A single
# token carries a single `app_key`. It cannot legitimately BE both "marketing"
# and "sales" at once, so at least one of these two entries is wrong about
# what actually arrives. `marketing/src/core/license_gate.py:38` and
# `sales/src/core/license_gate.py:36` only confirm what each app queries ITS
# OWN entitlement as — not what key is in the token growth actually sends.
#
# ⇒ A `mismatch_observed` on EITHER of these two hosts under `observe` is an
# EXPECTED, INFORMATIVE possible outcome, NOT a bug and NOT evidence the
# mechanism is broken — it is exactly the signal that identifies the real
# app_key growth's token carries for that hop.
#
# ⛔ DO NOT ARM `enforce` ON EITHER the marketing API host OR
# the sales API host until an `observe` window has actually settled which
# app_key each one's token carries. Arming now could manufacture the very
# outage this feature exists to prevent (a legitimate growth call denied as
# `mismatch_denied`).
#
# THE FIX, WHEN observe SETTLES IT: `APP_HOST_REGISTRY_OVERRIDE` — no image
# build required.
#
# ══════════════════════════════════════════════════════════════════════════════
# 📏 MEASURED 2026-09-15 — THE `observe` WINDOW RAN AND SETTLED NOTHING.
# ══════════════════════════════════════════════════════════════════════════════
# `APP_HOST_ROUTE_MODE=observe` was live on both replicas
# (`license-authz-6bbf4b687-{h24tc,wkpns}`, ns `istio-system`, started
# 2026-09-15T10:30:22Z) for a ~6h window. Counters read from each pod's
# `/metrics` (NOT from logs — see the trap below), both replicas summed:
#
#     license_authz_app_host_route_total
#       result="no_host"             72   ← 35 + 37, ALL ORGANIC
#       result="unregistered"         0
#       result="matched"              0
#       result="mismatch_observed"    6   ← 2 + 2 + 2, ALL SYNTHETIC PROBE
#       result="mismatch_denied"      0
#
#     license_authz_service_principal_total
#       caller="marketing-service-account", result="allowed"   367  (195 + 172)
#       caller="sales-service-account"                       ABSENT — never once.
#
# ⚠️ TRAP THAT COST A PRIOR SESSION ITS CONCLUSION: `matched`, `no_host` and
# `unregistered` emit NO LOG LINE — only `mismatch_observed`/`mismatch_denied`
# log. Reading `kubectl logs` alone yields no_host=0 and the false conclusion
# that nothing reaches `_check_app_host`. SCRAPE `/metrics` (container `authz`,
# port 9001), never grep the log, when measuring this block.
#
# WHAT THE NUMBERS ESTABLISH, and what they do not:
#
#  1. THE MECHANISM RUNS. 72 organic requests DID reach `_check_app_host`. This
#     block is not dead code and the dispatch guard is not the obstacle.
#
#  2. BUT NO ORGANIC REQUEST EVER CARRIED A USABLE AUTHORITY. All 72 landed on
#     `no_host`, i.e. `_normalize_authority` returned "" — absent header, a bare
#     dotless name, or a cluster-domain name. `unregistered` is 0, so not one
#     organic request arrived with ANY dotted public authority, let alone a
#     registry member. gitops #3919 added `host` to `includeRequestHeadersInCheck`;
#     for east-west traffic it still does not deliver a public authority here.
#
#  3. EVERY non-zero `mismatch_observed` IS STILL THE PROBE. All 6 carry
#     `path=/api/__authz_probe_nonexistent` and `app_key=w2v` — a w2v token, so
#     they say nothing about marketing's or sales's key. There were exactly 6
#     probe requests in the window and they produced exactly these 6 counts;
#     the 72 `no_host` are therefore disjoint from the probe and fully organic.
#
#  4. WHY ORGANIC MARKETING-API TRAFFIC CANNOT REACH HERE. It exits on
#     the service-principal branch — 367 `allowed_service_principal`, which
#     `return`s a 200 Response directly, hundreds of lines BEFORE the `# --- P2:
#     host -> app binding ---` dispatch. Those are inbound marketing API paths
#     (`/api/deliverability-signals` 140, `/api/provider-health-signals` 140,
#     `/api/eval-results` 54, `/api/action-intents/decide` 21, …), i.e. exactly
#     the hop this registry entry is about. They never see a host check.
#
#  5. THE AUDIT BLOCK'S PREMISE IS UNMEASURED — NOT REFUTED. It asserts the
#     gateway injects `growth-service-account` on both hops. MEASURED, the
#     caller on 367/367 service-principal allows is `marketing-service-account`
#     and `growth-service-account` appears zero times — BUT that counter only
#     fires on the TOKENLESS service-principal branch. A growth hop presenting
#     a licence bearer token (i.e. exactly the hop this registry entry exists
#     for) takes the token path and can NEVER increment it. So this measurement
#     cannot refute the premise; it only shows the premise is untested on the
#     one path that can be counted today. Do not read it as a refutation.
#
#  6. THE SALES API HOST HAS NO MEASURED TRAFFIC AT ALL. Zero requests, zero
#     `sales-service-account` allows. Its entry is unfalsifiable, not merely
#     unverified.
#
# ⛔ THEREFORE BOTH THE MARKETING AND SALES API HOSTS REMAIN
# UNVERIFIED, and NO mapping was guessed or changed on this evidence. The
# audit block above is right that at least one of the two is wrong; `observe`
# as currently wired CANNOT say which, and running it longer will not help —
# the limit is structural (branch order + missing authority), not sample size.
#
# ✅ WHAT WOULD ACTUALLY SETTLE IT — any ONE of these, in preference order:
#   (a) Ask the licences service which app is bound to the licence that
#       `GROWTH_TENANT_LICENSE_TOKENS` mints for a tenant. This needs no
#       traffic, no mode change, and no risk — cheapest, do it first.
#       ⚠️ Do NOT expect to decode an `app_key` claim out of the token: on the
#       opaque-token path `app_key` comes from the licences-service RECORD
#       (`app_key=app_data.get("key")`), not from the token body. The only
#       header-asserted key is JWT-mode's `x-app-key`, which the gateway does
#       not forward.
#   (b) Make the host check reachable for these callers, by evaluating the
#       host binding BEFORE the service-principal early return (or by counting
#       host/app on that branch too). That is a dispatch-order change to this
#       file and must be argued on its own merits — it is NOT free, because the
#       SP branch exists precisely to allow non-licensed mesh callers.
#   (c) Get Envoy to forward a usable `:authority` east-west, then confirm the
#       `unregistered` counter starts moving off 0. Until `unregistered` > 0,
#       the authority is not arriving and no host-keyed measurement is possible.
#
# ⛔ WHAT WOULD NOT SETTLE IT, measured harmful, do not retry:
#   - Adding `/api/strategy/` or `/api/sales/` to APP_ROUTE_REGISTRY to "make
#     the paths visible". It flips live `400 missing_app_key` into `402
#     license_required` DENY on production paths, AND permanently excludes those
#     paths from this host check (the block only runs when
#     `_registered_app_key(path) is None`). Strictly counter-productive.
#   - Arming `enforce`. See the ⛔ two paragraphs up; nothing below is verified.
#   - Another synthetic probe. A probe proves the code path executes, which is
#     already item 1 above; it can never reveal growth's real app_key.
# ══════════════════════════════════════════════════════════════════════════════
# ⚠️ DEPLOYMENT CONFIGURATION, NOT SOURCE (2026-11-14). This mapping used to
# be a seed here. It is EMPTY unless `APP_HOST_REGISTRY_OVERRIDE` is set, so it
# is populated by the deployment, not by the repository. The audit history that
# used to annotate the seed values lives with the deployment that owns them.
APP_HOST_REGISTRY: dict[str, str] = {}


def _parse_host_registry_override(raw: str) -> dict[str, str]:
    """Parse `host=app_key,host2=app_key2` into a host -> app_key mapping.

    Runtime-configurable on purpose: a host can move (a new domain, a tenant
    vanity host, a correction to an app_key guess) without rebuilding and
    re-rolling the image that fronts every waypoint in the mesh. Malformed
    entries are skipped rather than fatal — a typo in an env var must not take
    the authz waypoint down at import time.
    """
    parsed: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        host, _, app = entry.partition("=")
        host = host.strip().split(":")[0].lower()
        app = app.strip()
        if host and app:
            parsed[host] = app
    return parsed


APP_HOST_REGISTRY.update(_parse_host_registry_override(os.getenv("APP_HOST_REGISTRY_OVERRIDE", "")))


# --- GATEWAY-ASSERTED host -> app key binding (asserted key, UNREGISTERED path) -
# THE OUTAGE THIS CLOSES (prod, measured 2026-10-05): every browser navigation to
# https://n8n.nonagentic.ai/<ui path> (e.g. `/workflow/<id>`) by a fully licensed
# user returned 403 `app_route_unregistered_asserted_key`
# (`denied_asserted_key_unregistered` = 4 over 7d — the clicks themselves).
#
# WHY: for browser-tier hosts the istio-ingressgateway Lua filter
# (gitops `clusters/strato/policies/envoy-filters/ext-authz-enforce.yaml`) calls
# this service with an EXPLICITLY BUILT header table. It first STRIPS the
# client's own `x-app-key`, then sets `x-app-key` from a STATIC host map
# (`LICENSE_GATED_APP_KEY`) and `x-forwarded-host` from the original authority.
# The key on those calls is therefore chosen by the gateway, not the caller —
# but it reaches us on the JWT path as an ASSERTED key, and n8n/kanban UI paths
# (`/workflow/...`, `/rest/...`, `/assets/...`) are not in APP_ROUTE_REGISTRY,
# so the asserted-key-unregistered deny refused the gateway's own key.
#
# THE BINDING: on that arm ONLY, an asserted key is allowed when the normalised
# `x-forwarded-host` maps here to EXACTLY that key. The host supplies the binding
# the route registry cannot. A different key, an unknown host, or no
# `x-forwarded-host` keeps the pre-existing behaviour byte-for-byte.
#
# ⛔ THIS MUST MIRROR the gateway Lua `LICENSE_GATED_APP_KEY` map. A host added
# there and not here is refused on every unregistered path; a host added here and
# not there is dead weight (the gateway never sends its key).
#
# ⛔ NOT APP_HOST_REGISTRY / `_check_app_host`: that is a separate observe-mode
# feature keyed on `:authority`/`Host`, which on the gateway's httpCall is this
# service's own Service name (all 9507 checks in 7d were `no_host`).
#
# RESIDUAL RISK, stated plainly: `x-forwarded-host` is itself an UNSIGNED header.
# It can reach this arm only (a) from the gateway Lua, which builds the header
# table explicitly, or (b) from a direct pod -> license-authz call. The mesh
# waypoint provider does NOT forward `x-app-key` or `x-forwarded-host`
# (`includeRequestHeadersInCheck` is authorization, x-forwarded-for, user-agent,
# x-caller-authorization, host), so a waypoint check never lands here with
# either. A direct call (b) obtains a VERDICT, not upstream access. And even a
# forged pair only lets the caller select the n8n or kanban licence of its OWN
# verified tenant (the JWT fixes the tenant), on paths of those two apps — the
# same mis-charging class the deny exists for, bounded to two named apps instead
# of a free choice.
_GATEWAY_ASSERTED_HOST_KEYS_SEED: dict[str, str] = {
    "n8n.nonagentic.ai": "n8n",
    # kanban.nonagentic.ai REMOVED 2026-11-18: kanban/Planka was deleted with the
    # S14 sweep; the gitops gateway no longer asserts an x-app-key for that host.
}


def _build_gateway_asserted_host_keys(override_raw: str) -> dict[str, str]:
    """Seed merged with `GATEWAY_ASSERTED_HOST_KEYS_OVERRIDE`.

    Same grammar and same fail-soft parsing as APP_HOST_REGISTRY_OVERRIDE
    (`host=app_key,host2=app_key2`; malformed entries skipped, never fatal at
    import), plus a trailing-dot strip so keys match `_gateway_asserted_host`.
    """
    merged = dict(_GATEWAY_ASSERTED_HOST_KEYS_SEED)
    for host, app in _parse_host_registry_override(override_raw).items():
        host = host.rstrip(".")
        if host:
            merged[host] = app
    return merged


GATEWAY_ASSERTED_HOST_KEYS: dict[str, str] = _build_gateway_asserted_host_keys(
    os.getenv("GATEWAY_ASSERTED_HOST_KEYS_OVERRIDE", "")
)


def _gateway_asserted_host(request: Request) -> str:
    """The gateway's `x-forwarded-host`, normalised; "" when absent.

    Deliberately reads `x-forwarded-host` ONLY — no `:authority`/`Host`
    fallback: on the gateway's httpCall those name this service, and a fallback
    would let a direct in-mesh call bind by its own Host header. First entry of a
    comma list, port stripped, lowercased, trailing dot stripped.
    """
    raw = request.headers.get("x-forwarded-host") or ""
    return raw.split(",")[0].strip().split(":")[0].strip().lower().rstrip(".")


def _asserted_key_host_bound(request: Request) -> bool:
    """True iff the gateway host maps to EXACTLY the asserted `x-app-key`."""
    asserted = request.headers.get("x-app-key", "").strip()
    host = _gateway_asserted_host(request)
    if not asserted or not host:
        return False
    return GATEWAY_ASSERTED_HOST_KEYS.get(host) == asserted


# Destination host -> billing service_key, for hosts whose paths share the bare
# `/api` prefix (billing-service SERVICE_KEY_MAP cannot disambiguate by path and
# asks the emitter to name itself). Gateway calls carry `x-forwarded-host`;
# waypoint calls carry the in-mesh `host`.
#
# ⚠️ DEPLOYMENT CONFIGURATION, NOT SOURCE (2026-11-14). The entries used to be
# seeds here; they are now supplied by `METERING_HOST_SERVICE_KEYS_OVERRIDE`
# (`host=service_key,...`), so the repository carries no estate topology. An
# unmapped destination meters as `unmapped:<path prefix>` — visible in billing,
# never guessed. The gateway-asserted hosts are still merged in last, because
# that binding is public and belongs to the protocol rather than a deployment.
METERING_HOST_SERVICE_KEYS: dict[str, str] = {
    **_parse_host_registry_override(os.getenv("METERING_HOST_SERVICE_KEYS_OVERRIDE", "")),
    **_GATEWAY_ASSERTED_HOST_KEYS_SEED,
}


def _destination_host(request: Request) -> str:
    """The destination host: gateway `x-forwarded-host`, else in-mesh authority."""
    raw = (
        request.headers.get("x-forwarded-host")
        or request.headers.get(":authority")
        or request.headers.get("host")
        or ""
    )
    host = raw.split(",")[0].strip().split(":")[0].strip().lower().rstrip(".")
    for suffix in (".svc.cluster.local", ".svc"):
        host = host.removesuffix(suffix)
    return host


def _metering_service_key(request: Request) -> str:
    """The service CONSUMED, from the destination host; "" when unknown (never guessed)."""
    return METERING_HOST_SERVICE_KEYS.get(_destination_host(request), "")


# Hosts that accept ONLY a licence for ONE app, on EVERY path and regardless of
# APP_HOST_ROUTE_MODE (which stays `off` for every other host): host -> app_key.
#   The classic case is a GPU endpoint. Its calls are never metered here (see
#   METERING_UNMETERED_SERVICES), so without this binding ANY active licence of
#   ANY tenant would get unaccounted accelerator time. Only that app's own
#   licence may pass; the router in front of it meters per end tenant itself.
# Resolved by `_destination_host`, the SAME function metering uses, so a request
# can never be skipped as that app without having been bound as it. A wrong
# key -> 403 `app_host_mismatch`, a non-licence caller (service principal) ->
# the standard 401. Health paths are exempted by the gateway before ext_authz.
#
# ⚠️ DEPLOYMENT CONFIGURATION, NOT SOURCE (2026-11-14). EMPTY unless
# `STRICT_APP_HOSTS_OVERRIDE` (`host=app_key,...`) is set. Setting it is a
# security decision, not a tuning knob: an empty map does not deny, it simply
# imposes no app binding on any host.
STRICT_APP_HOSTS: dict[str, str] = _parse_host_registry_override(
    os.getenv("STRICT_APP_HOSTS_OVERRIDE", "")
)


def _strict_app_for(request: Request) -> tuple[str, str] | None:
    """(host, required app_key) when the destination is a strict host."""
    host = _destination_host(request)
    app_key = STRICT_APP_HOSTS.get(host)
    return (host, app_key) if app_key else None


# Machine polls that are authorised like any other request but are NOT API
# consumption, per consumed service: service_key -> (reason, method, path regex).
# Matched on the exact method and the WHOLE normalised path, so every other
# route on the same host stays metered.
#   lead_bridge_poll - marketing's lead-auto-qualify bridge reconciles deferred
#       handovers by reading `GET /api/v2/leads/intakes/{intake_id}` on sales
#       every tick (60s in prod; SalesHandoverClient.get_intake). A pure read
#       that marketing issues on its own schedule, not usage a tenant asked for.
#       `/intakes/{id}/duplicates`, the list route and every POST do not match.
METERING_UNMETERED_ROUTES: dict[str, tuple[tuple[str, str, re.Pattern[str]], ...]] = {
    "sales": (("lead_bridge_poll", "GET", re.compile(r"/api/v2/leads/intakes/[^/]+")),),
}


# Consumed services whose EVERY authorised request is unmetered here, because
# the caller already meters it per END tenant: service_key -> reason.
#   metered_by_inference_router - `vllm.example` (pgx vLLM). Its only caller is
#       inference-router, which emits its own cost events per end tenant. A
#       license-authz event for the same call would double-count the spend.
# Host-level, never path-level, and ONLY after the STRICT_APP_HOSTS binding
# allowed: the skip is keyed on the strictly bound app, never on the host alone.
METERING_UNMETERED_SERVICES: dict[str, str] = {
    "vllm": "metered_by_inference_router",
}


def _unmetered_route(
    request: Request, method: str, path: str, *, strict_bound_app: str
) -> tuple[str, str] | None:
    """(reason, service_key) when this allowed request is deliberately not metered.

    `strict_bound_app` is the app a STRICT_APP_HOSTS binding ALLOWED for this
    request ("" when none did); a whole-service skip requires it.
    """
    if service_reason := METERING_UNMETERED_SERVICES.get(strict_bound_app):
        return service_reason, strict_bound_app
    service_key = _metering_service_key(request)
    rules = METERING_UNMETERED_ROUTES.get(service_key)
    if not rules:
        return None
    normalized = _normalize_route_path(path)
    for reason, rule_method, pattern in rules:
        if method == rule_method and pattern.fullmatch(normalized):
            return reason, service_key
    return None


# off (DEFAULT) | observe | enforce.
#   off     - not consulted at all; zero behaviour change, zero metrics.
#   observe - a mismatch is COUNTED and ALLOWED. This is the measurement step.
#   enforce - a mismatch is DENIED 403 with the NEW, DISTINCT reason
#             `app_host_mismatch`. It must never reuse `app_route_mismatch`:
#             the two are different defects with different fixes and an operator
#             reading logs has to be able to tell them apart.
# An unrecognised value falls back to `off` — fail-INERT, because the failure mode
# of a typo'd mode string on an ext_authz waypoint must not be "deny everything".
def _parse_host_route_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    return mode if mode in ("off", "observe", "enforce") else "off"


APP_HOST_ROUTE_MODE = _parse_host_route_mode(os.getenv("APP_HOST_ROUTE_MODE", "off"))

# Direction: FAILS OPEN by default. When false the service-principal branch
# allows before any budget check — but the allow is COUNTED
# (metric_service_principal_budget{result="not_enforced"}). When true (the live
# state), the caller's resolved tenant is checked against billing's
# TENANT-scoped token budgets (`MeteringClient.check_budget` with
# license_id=None) and a deny becomes a 402.
#
# This path carries no licence, so licence-scoped budgets and request quotas
# never bind it. A caller
# whose `azp` does not resolve to a tenant has nothing to key on: it fails open
# and is counted as `budget_fail_open{reason="no_license_id"}`.
SERVICE_PRINCIPAL_BUDGET_ENFORCED = (
    os.getenv("SERVICE_PRINCIPAL_BUDGET_ENFORCED", "false").lower() == "true"
)

# Cardinality clamp for the caller label on metric_service_principal_budget.
# `azp` comes from a VERIFIED secure-apps token, so it is not attacker-chosen,
# but it is still a remote-controlled string on a per-request label: one renamed
# or per-pod client id turns a bounded series into an unbounded one. Any caller
# not listed here is bucketed as `other`, so the series count is fixed at
# (len(this) + 1) x len(results) no matter what arrives.
#
# The default is the MEASURED census (24h to 2026-09-13): the only two callers
# that have ever taken this path. It is data, and it rots — when `other` starts
# moving, re-run the census and extend this rather than reading `other` as one
# caller. Kept separate from SERVICE_PRINCIPAL_ALLOWED_CLIENTS on purpose: that
# one is an AUTHZ decision and is empty (= allow any) in production, so reusing
# it as the label allowlist would collapse every caller into `other` and destroy
# the attribution this counter exists for.
SERVICE_PRINCIPAL_METRIC_CALLERS: frozenset[str] = frozenset(
    c.strip()
    for c in os.getenv(
        "SERVICE_PRINCIPAL_METRIC_CALLERS",
        "marketing-service-account,platform-console-backend",
    ).split(",")
    if c.strip()
)

# Fail direction of every billing HTTP check (token budget and request quota).
METERING_FAIL_OPEN = os.getenv("LICENSE_AUTHZ_METERING_FAIL_OPEN", "true").lower() == "true"

# Prometheus metrics
metric_budget_exceeded = Counter(
    "license_authz_budget_exceeded_total",
    "Requests blocked for exceeding budget or quota.",
    ["tenant", "app", "kind"],
)
metric_metering_lookup = Counter(
    "license_authz_metering_lookup_total",
    "Billing HTTP check outcomes (http/error/contract_error).",
    ["result"],
)
metric_jwt_mode = Counter(
    "license_authz_jwt_mode_total",
    "JWT-mode check outcomes.",
    # allowed, allowed_self_read_exempt, invalid_token, missing_app_key,
    # license_required, anonymous
    ["result"],
)
# Per-tenant JWT-mode counter — only emitted on paths where (tenant, app) are
# both known (allowed, allowed_self_read_exempt, license_required). Failure
# modes like invalid_token, missing_app_key, anonymous,
# jwt_mode_not_initialized, licenses_unavailable are excluded to bound
# cardinality (no UUID-from-sub fallback explosion).
metric_jwt_mode_by_tenant = Counter(
    "license_authz_jwt_mode_by_tenant_total",
    "JWT-mode check outcomes labeled by tenant and app "
    "(allowed, allowed_self_read_exempt, license_required only).",
    ["result", "tenant", "app"],
)
metric_jwks_refresh = Counter(
    "license_authz_jwks_refresh_total",
    "JWKS fetch outcomes (eager boot, lazy on kid-miss).",
    ["result"],  # success | failure
)
metric_cache_evictions = Counter(
    "license_authz_cache_evictions_total",
    "In-memory cache LRU evictions.",
    ["cache"],  # active_license | metering | introspect_stale
)
metric_internal_token = Counter(
    "license_authz_internal_token_issued_total",
    "Internal-token (Lane F1) issue outcomes.",
    ["result"],  # issued | denied
)
metric_service_principal = Counter(
    "license_authz_service_principal_total",
    "Service-principal (SA-token-only) authz outcomes on the opt-in path.",
    ["result", "caller"],  # result: allowed | denied_not_allowlisted ; caller: azp
)
# === The service-principal fast path, made visible ==========================
# THE AUDIT FINDING THIS EXISTS FOR: over 24h all 1,487 ext_authz checks took the
# service-principal branch, which returns 200 BEFORE the budget check and emits
# no metering. `metric_service_principal{result="allowed"}` already counted the
# volume, but it says nothing about ENFORCEMENT — an operator reading it cannot
# tell "allowed after passing its quota" from "allowed because this path has no
# quota gate at all", and the latter is what 100% of it was.
#
# So this counter answers only that question, and it is dimensioned by caller
# because the two facts an operator needs are inseparable: how much traffic is
# deliberately unenforced, and WHOSE. The caller label is clamped through
# `_sp_caller_label` (see SERVICE_PRINCIPAL_METRIC_CALLERS) so the series count
# is bounded regardless of what `azp` values turn up.
#
# Results:
#   not_enforced  - SERVICE_PRINCIPAL_BUDGET_ENFORCED is off (the default and
#                   the live state). THE ONE THAT MATTERS: it is the whole
#                   production volume of the audit finding, and it is a
#                   deliberate config state, not a fault — which is exactly why
#                   it must not share a series with a fail-open reason.
#   allowed       - the flag is on and the tenant budget check allowed. ⚠️ Read
#                   this WITH `budget_fail_open{reason="no_license_id"}` (an
#                   unresolved tenant) and `{reason="metering_http_error"}`:
#                   those allows were not enforced passes.
#   denied        - the flag is on and the caller's tenant-scoped token budget
#                   is exhausted (or billing failed with METERING_FAIL_OPEN off)
#                   → 402.
metric_service_principal_budget = Counter(
    "license_authz_service_principal_budget_total",
    "Budget-enforcement outcome of the service-principal fast path, by caller.",
    ["result", "caller"],  # result: not_enforced | allowed | denied
)
# Born at zero for the full (result x clamped caller) cross product — 3 x
# (census + `other`) series, bounded by construction. Without this, "this caller
# has never been denied" and "this build cannot report a denial" are the same
# empty vector, and an operator reads absence as health.
for _sp_result in ("not_enforced", "allowed", "denied"):
    for _sp_caller in (*SERVICE_PRINCIPAL_METRIC_CALLERS, "other"):
        metric_service_principal_budget.labels(result=_sp_result, caller=_sp_caller)
del _sp_result, _sp_caller
# === Service-principal metering IDENTITY (this PR) ==========================
# THE FOLLOW-ON AUDIT FINDING: metric_service_principal_budget answers whether
# the SP path is *enforced*; it says nothing about whether the metering event
# it emits (`_emit_metering_event(..., status="allowed_service_principal")`)
# is ATTRIBUTABLE to a tenant. Before this it never was — every SP allow
# emitted `tenant_id=""`, unconditionally, landing in billing unattributable.
# This counter is the honest measurement of the fix: how much of that volume
# was resolvable to a real tenant, and how much still is not.
#
# `client_id` is clamped to SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS (the
# name-map keys) — same cardinality reasoning as `_sp_caller_label`.
#   resolved   - `caller_client_id` had a candidate name in
#                SERVICE_PRINCIPAL_TENANT_NAME_MAP AND that exact name is in
#                the licences `GET /internal/tenants` list. The metering
#                event's tenant_id is that row's id.
#   unresolved - no candidate name, the name is absent from a usable list,
#                licences gave no usable list (unreachable / non-200 /
#                non-list / empty), or LICENSES_INTERNAL_TENANTS_URL is
#                unset. The metering
#                event's tenant_id stays "" — UNCHANGED from before this PR,
#                never fabricated.
metric_service_principal_identity = Counter(
    "license_authz_service_principal_identity_total",
    "Whether a service-principal metering event's tenant_id resolved to a real "
    "licences tenant, by caller.",
    ["client_id", "result"],  # result: resolved | unresolved
)
# Born at zero for the full (known client x result) cross product, same reason
# as metric_service_principal_budget's pre-seed: an operator must be able to
# tell "this client has never resolved" from "this build cannot report a
# resolution" without querying two different kinds of absence.
for _sp_id_client in (*SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS, "other"):
    for _sp_id_result in ("resolved", "unresolved"):
        metric_service_principal_identity.labels(client_id=_sp_id_client, result=_sp_id_result)
del _sp_id_client, _sp_id_result
# === Budget gate never even consulted =======================================
# Separate from `metric_budget_fail_open` ON PURPOSE, and the split is the whole
# point: these two reasons are PERMANENT, DELIBERATE config states, so folding
# them into the fail-open family would pin any `budget_fail_open > 0` alert
# firing forever and force it to be silenced — taking the five real defect
# reasons down with it. Same reasoning `metric_metering_events_dropped` records
# for keeping `metering_unconfigured` off the two loss reasons.
#
#   enforcement_disabled - BUDGET_ENFORCEMENT_ENABLED is false, so neither the
#                          token-budget nor the quota check is called on the licence-token/JWT path. This is
#                          the DEFAULT, and before this counter it produced no
#                          telemetry at all: a deployment with the entire quota
#                          gate switched off was indistinguishable, on the wire,
#                          from one enforcing every request.
#   no_license_id        - enforcement is on but the active introspect result
#                          carries no licence id, so the guard at the call site
#                          skips the check. Distinct from the `no_license_id`
#                          fail-open reason, which is the service-principal path
#                          with no resolved tenant.
metric_budget_enforcement_skipped = Counter(
    "license_authz_budget_enforcement_skipped_total",
    "Requests where the budget/quota check was never invoked at all.",
    ["reason"],  # enforcement_disabled | no_license_id
)
for _skip_reason in ("enforcement_disabled", "no_license_id"):
    metric_budget_enforcement_skipped.labels(reason=_skip_reason)
del _skip_reason
# === Tenant identity: claim vs `sub` fallback ================================
# WHY ALL THREE OUTCOMES ARE COUNTED, NOT JUST THE FALLBACK. The fallback count
# alone is a numerator with no denominator — it cannot answer "is the claim
# rolling out?" or "is the fallback now removable?", which is the only reason to
# measure it. `tenant_claim` is that denominator, and `absent` is the third state
# (neither claim nor `sub` → "anonymous" → 401) which must not be folded into
# either.
#
# The `sub` fallback is the root of the firing `LicenseTenantIdIsKeycloakPrincipal`
# alert: it silently promotes a per-USER Keycloak principal into the per-TENANT
# billing identity, so two humans at one customer bill as two tenants. Neither
# realm emits a tenant claim today, so `sub` is currently 100% of it and removing
# the fallback would 401 every browser caller. This series is what tells the
# operator when that has changed: the fallback is removable when `sub` reaches
# zero under real traffic, and not before.
metric_tenant_claim = Counter(
    "license_authz_tenant_claim_total",
    "Source of the tenant identity extracted from a verified Keycloak JWT.",
    ["source"],  # tenant_claim | sub | absent
)
for _tenant_source in ("tenant_claim", "sub", "absent"):
    metric_tenant_claim.labels(source=_tenant_source)
del _tenant_source
# === Audience verification, measured BEFORE it is armed ======================
# `jwt_mode.JWT_VERIFY_AUDIENCE` stays OFF, because nothing in this repo records
# which `aud` real tokens carry and arming it blind 401s every JWT caller at
# once. ⛔ The one audience the repo DOES know — `INTERNAL_TOKEN_AUDIENCE`, live
# value `internal` — is the audience this service MINTS onto tokens it ISSUES,
# not one it verifies, so it is the wrong answer and not a near miss.
#
# This counter is what replaces the guess: it shadow-evaluates every token that
# was verified WITHOUT an audience check and records what arming would have done
# to it, so the flag can be armed on evidence instead of on memory. See
# `jwt_mode.audience_shadow_verdict` for the four verdicts and how to read them.
#
# ⚠️ It reports `unconfigured` until an operator sets JWT_EXPECTED_AUDIENCES.
# That is not a broken instrument, it is the instrument saying it has not been
# given a hypothesis to test yet — and it is why all four are born at zero: an
# absent series would make "no token would fail" and "nothing measured" the same
# empty vector, which is the exact confusion that lets a flag get armed blind.
metric_jwt_audience_shadow = Counter(
    "license_authz_jwt_audience_shadow_total",
    "What arming JWT_VERIFY_AUDIENCE would have done to each verified JWT.",
    ["verdict"],  # unconfigured | absent | match | mismatch
)
for _aud_verdict in ("unconfigured", "absent", "match", "mismatch"):
    metric_jwt_audience_shadow.labels(verdict=_aud_verdict)
del _aud_verdict
# IN-POD verification instrument, NOT an alertable signal: /metrics is served but
# the Service carries no prometheus.io/scrape annotation (deliberate — ztunnel
# resets the plaintext scrape), so nothing collects this. The operator channel
# for a Redis outage is the metering_redis_unavailable log line.
metric_redis_connect = Counter(
    "license_authz_redis_connect_total",
    "Metering Redis connect attempts made by the lazy accessor.",
    ["result"],  # success | failure | skipped_backoff
)
metric_app_route = Counter(
    "license_authz_app_route_total",
    "App-route enforcement outcomes on the license-token path.",
    # allowed                            - the route check passed (attested key,
    #                                      or an asserted key on a registered
    #                                      prefix).
    # denied_mismatch                    - a key that is not the one the
    #                                      registered prefix expects.
    # denied_missing_key                 - no key at all on a registered prefix.
    # denied_asserted_key_unregistered   - an ASSERTED key (x-app-key, JWT path)
    #                                      on an UNREGISTERED path, refused.
    # allowed_asserted_key_unregistered  - the same population, allowed because
    #                                      ASSERTED_APP_KEY_ALLOW_UNREGISTERED
    #                                      restored the permissive behaviour.
    #
    # ⚠️ The last two are ONE population counted under two labels, and the split
    # is the point: an operator who reverts to permissive still needs to see how
    # much traffic is riding on the revert. Folding the revert case into
    # `allowed` would make the census go dark at exactly the moment it matters.
    # allowed_self_read_exempt           - a GET/HEAD of the caller's OWN licence
    #                                      (`/licenses/active`, `/licenses/user/`)
    #                                      that the `/licenses/` prefix would
    #                                      otherwise have denied. Split out of
    #                                      `allowed` on purpose: an exemption
    #                                      that fires invisibly is indistinguish-
    #                                      able from an exemption that is dead,
    #                                      and this one exists to fix a live
    #                                      outage — an operator re-arming the
    #                                      gate needs to SEE it carrying traffic.
    # allowed_asserted_key_host_bound    - an ASSERTED key on an UNREGISTERED
    #                                      path, allowed because the gateway's
    #                                      `x-forwarded-host` maps to exactly
    #                                      that key (GATEWAY_ASSERTED_HOST_KEYS).
    #                                      Its own label so the n8n/kanban UI
    #                                      population stays visible and is never
    #                                      mistaken for a registered-prefix allow.
    ["result"],
)
# Born at zero, all seven. A labelled prometheus_client Counter exports NO series
# until its first `.labels(...).inc()`, so without this loop "no request has ever
# been refused for an asserted key on an unregistered route" and "this build
# cannot refuse one" are the same empty vector on the wire.
for _route_result in (
    "allowed",
    "denied_mismatch",
    "denied_missing_key",
    "denied_asserted_key_unregistered",
    "allowed_asserted_key_unregistered",
    "allowed_self_read_exempt",
    "allowed_asserted_key_host_bound",
):
    metric_app_route.labels(result=_route_result)
del _route_result

metric_app_host_route = Counter(
    "license_authz_app_host_route_total",
    "Host->app binding outcomes, consulted ONLY on paths no prefix registered.",
    # result:
    #   matched           - the token's app_key is the one this host demands.
    #   mismatch_observed - it is not, and APP_HOST_ROUTE_MODE=observe ALLOWED it.
    #   mismatch_denied   - it is not, and APP_HOST_ROUTE_MODE=enforce DENIED it.
    #   no_host           - no usable public authority: empty, a bare service
    #                       name, or a cluster-domain name. ALWAYS ALLOWS — an
    #                       east-west in-mesh hop does not carry the public host
    #                       and must never be denied for not carrying it.
    #   unregistered      - a real host that is not in APP_HOST_REGISTRY. Allows.
    #
    # ⛔ CARDINALITY IS BOUNDED DELIBERATELY. `Host` is attacker-controlled: an
    # arbitrary value would mint an unbounded number of Prometheus series from
    # unauthenticated input, i.e. a memory-exhaustion vector on every waypoint.
    # So `host` carries the REAL value only when it is a registry MEMBER, and the
    # literal "other" otherwise; `app` carries the app_key the REGISTRY expects
    # (also bounded), never the caller-influenceable asserted key.
    ["result", "host", "app"],
)
# Born at zero on the bounded-label combination, same reason as above: without
# this, "host binding has never mismatched" and "this build is compiled without
# host binding" are the same empty vector.
for _host_result in (
    "matched",
    "mismatch_observed",
    "mismatch_denied",
    "no_host",
    "unregistered",
):
    metric_app_host_route.labels(result=_host_result, host="other", app="other")
del _host_result
# STRICT_APP_HOSTS outcomes: always-enforced, so `matched` here is the proof
# inference-router's calls pass and `strict_mismatch_denied` the proof a
# wrong-app licence was refused. Bounded: host/app are the table's own values.
# The names are bound BEFORE the loop because STRICT_APP_HOSTS is deployment
# configuration and is legitimately empty (a deployment that binds no host to a
# single app_key). An empty dict must not leave them unbound for the `del`.
_strict_host = _strict_app = _host_result = "other"
for _strict_host, _strict_app in STRICT_APP_HOSTS.items():
    for _host_result in ("strict_matched", "strict_mismatch_denied"):
        metric_app_host_route.labels(result=_host_result, host=_strict_host, app=_strict_app)
del _strict_host, _strict_app, _host_result

# === Dropped metering events ================================================
# Name deliberately follows the estate-wide
# `<service>_metering_events_dropped_total{reason=...}` convention already used
# by karma_service (src/metrics.py) and billing_service
# (src/observability/metrics.py). The fleet rules MeteringEventsDroppedNoTenant
# / MeteringEventsDroppedUnconfigured select on
# `{__name__=~".*metering_events_dropped_total", reason="..."}` — a permissive
# name match, not a bare metric name — so this series is picked up with no
# second rule, provided the `reason` value is on their allowlist.
#
# ⚠️ HOW THIS SERIES REACHES PROMETHEUS. license-authz cannot be scraped (root
# namespace ambient, mTLS STRICT — see src/metrics_push.py's docstring for the
# two failed attempts). It is exported only through the pushgateway, which IS
# armed: PUSHGATEWAY_URL is set on the Deployment in gitops
# (clusters/strato/services/license-authz/deployment.yaml). Pushed series are
# scraped by `job_name: kubernetes-pods` with `honor_labels` unset fleet-wide,
# so they arrive as app="pushgateway", exported_job="license-authz",
# job="kubernetes-pods". A rule grouping `by (app)` will therefore print
# "pushgateway", not "license-authz" — the metric NAME is what identifies the
# emitter here.
#
# The reasons are split because they are different defect classes, not two
# severities of one (the same reasoning karma records for its own split):
#   redis_unavailable     — the lazy accessor has no client: the connect failed
#                           or we are inside its backoff window. Metering is
#                           supposed to be ON, so every billable event in that
#                           window is lost with no record in the stream, in
#                           metering_events, or in any dead letter. THE DEFECT.
#   redis_error           — a client existed and the XADD itself failed.
#   metering_unconfigured — METERING_ENABLED=false, the deliberate kill switch.
#                           Benign and permanently non-zero wherever it is set,
#                           so it must never share a series with the two above
#                           or the alert on them has to be silenced.
#
# Every reason is born at 0 at import. A labelled prometheus_client Counter
# exports NO series until its first .labels(...).inc(), so without this loop
# "nothing has been dropped" and "this build has no instrument at all" are the
# same empty vector on the wire.
metric_metering_events_dropped = Counter(
    "license_authz_metering_events_dropped_total",
    "Billable metering events that were never written to the metering stream.",
    ["reason"],  # redis_unavailable | redis_error | metering_unconfigured
)
for _drop_reason in ("redis_unavailable", "redis_error", "metering_unconfigured"):
    metric_metering_events_dropped.labels(reason=_drop_reason)
del _drop_reason

# === Skipped metering events ================================================
# NOT a drop: these requests were allowed and deliberately never metered, so
# they get their own series rather than a `reason` on the dropped counter (whose
# alerts treat every non-kill-switch reason as lost billable usage).
#   gateway_browser_ui — a browser session on an editor UI host (n8n, kanban):
#                        verified JWT + gateway-set app key bound to the host +
#                        a path no app registered. SPA assets and editor REST,
#                        not API consumption. See the host-bound arm in
#                        `ext_authz_check`, the only place this is decided.
#   lead_bridge_poll   — marketing's once-a-minute intake read on sales; see
#                        METERING_UNMETERED_ROUTES. `app` is the service_key.
#   metered_by_inference_router — every authorised call to vllm.example; see
#                        METERING_UNMETERED_SERVICES. `app` is the service_key.
# Born at 0 per app, same reason as the dropped counter above.
metric_metering_events_skipped = Counter(
    "license_authz_metering_events_skipped_total",
    "Allowed requests deliberately not written to the metering stream.",
    ["reason", "app"],
)
for _skip_app in sorted(set(GATEWAY_ASSERTED_HOST_KEYS.values())):
    metric_metering_events_skipped.labels(reason="gateway_browser_ui", app=_skip_app)
del _skip_app
for _skip_service, _skip_rules in METERING_UNMETERED_ROUTES.items():
    for _skip_rule in _skip_rules:
        metric_metering_events_skipped.labels(reason=_skip_rule[0], app=_skip_service)
for _skip_service, _skip_service_reason in METERING_UNMETERED_SERVICES.items():
    metric_metering_events_skipped.labels(reason=_skip_service_reason, app=_skip_service)

# === Actor attribution on metering events ===================================
# Every metering event carries `actor_id` — a UUID Keycloak `sub` — when this
# service can vouch for one:
#   JWT mode           — the verified token's `claims.sub` (the acting human).
#   licence-token mode — the introspected `subject_actor_sub` (licenses#251: the
#                        subject's Keycloak sub, else its stated service-account
#                        sub), falling back to `subject_keycloak_sub` when talking
#                        to a licenses build that predates that field.
# When there is none, or it is not a UUID, the field is OMITTED (never "", never
# a sentinel) and this counter names why:
#   no_identity            — the event was emitted before any identity resolved
#                            (missing/invalid token, service-principal path).
#   jwt_no_sub             — JWT mode, but the verified token had no `sub`.
#   licence_no_subject_sub — introspect returned neither `subject_actor_sub` nor
#                            `subject_keycloak_sub`.
#   not_uuid               — a value was present but is not a UUID.
# Born at 0 per reason, same reason as the dropped counter above.
metric_actor_missing = Counter(
    "license_authz_actor_missing_total",
    "Metering events emitted without an actor_id, by why the actor is missing.",
    ["reason"],
)
_ACTOR_MISSING_REASONS = ("no_identity", "jwt_no_sub", "licence_no_subject_sub", "not_uuid")
for _actor_reason in _ACTOR_MISSING_REASONS:
    metric_actor_missing.labels(reason=_actor_reason)
del _actor_reason

# Inject `x-actor-id` downstream in LICENCE-TOKEN mode (JWT mode always does).
# Injected only when the inbound request carried no `x-actor-id` of its own, so a
# service forwarding the human it acts for keeps that human.
#
# ⛔ DEFAULT OFF, and it must stay off until the mesh lets this service SEE the
# inbound header. `x-actor-id` is not in the extension provider's
# `includeRequestHeadersInCheck` (gitops istio-meshconfig-patch.yaml), so on the
# east-west waypoint path the check request never carries it: "absent" here is
# indistinguishable from "not forwarded to the check", and because
# `x-actor-id` IS in `headersToUpstreamOnAllow`, injecting would OVERWRITE a
# forwarded human actor (core_app, fr, worker_management forward one) with the
# licence holder. The north-south gateway Lua copies `x-actor-id` only in its
# JWT branch, so there this flag is inert until that branch is extended too.
ACTOR_INJECT_LICENSE_TOKEN_MODE = (
    os.getenv("ACTOR_INJECT_LICENSE_TOKEN_MODE", "false").lower() == "true"
)


def _resolve_actor_id(raw: str | None, missing_reason: str) -> tuple[str | None, str | None]:
    """Return ``(canonical_uuid, None)`` or ``(None, reason_it_is_missing)``.

    Pure: counting is the caller's job, so one event is counted exactly once.
    """
    if not raw:
        return None, missing_reason
    try:
        return str(UUID(raw)), None
    except ValueError, AttributeError, TypeError:
        return None, "not_uuid"


# === Budget/quota enforcement fail-open =====================================
# Token-budget and quota enforcement ALLOW the request whenever enforcement
# state cannot be read or is simply absent. That is the documented policy; this
# counter makes it MEASURABLE.
#
# ⛔ Whether license-authz should fail CLOSED when billing is unreachable is an
# OPERATOR decision: LICENSE_AUTHZ_METERING_FAIL_OPEN.
#
# A DENY increments nothing.
metric_budget_fail_open = Counter(
    "license_authz_budget_fail_open_total",
    "Requests allowed because budget/quota enforcement state was unreadable or absent.",
    # no_license_id       - service-principal caller with no resolved tenant
    # no_tenant_id        - licence token with no tenant: billing's tenant-keyed
    #                       budget and quota checks have nothing to look up
    # metering_http_error - a billing HTTP check (token budget or request
    #                       quota) failed on transport or response shape and
    #                       LICENSE_AUTHZ_METERING_FAIL_OPEN allowed the request
    #                       anyway. With METERING_FAIL_OPEN=false the same arm
    #                       DENIES and is counted nowhere here.
    ["reason"],
)
for _fail_open_reason in ("no_license_id", "metering_http_error", "no_tenant_id"):
    metric_budget_fail_open.labels(reason=_fail_open_reason)
del _fail_open_reason


def _bump_jwks_refresh(result: str) -> None:
    metric_jwks_refresh.labels(result=result).inc()


def _bump_cache_eviction(cache: str) -> None:
    metric_cache_evictions.labels(cache=cache).inc()


def _bump_tenant_claim(source: str) -> None:
    metric_tenant_claim.labels(source=source).inc()


def _bump_audience_shadow(verdict: str) -> None:
    metric_jwt_audience_shadow.labels(verdict=verdict).inc()


def _sp_caller_label(caller_client_id: str) -> str:
    """Clamp a service-principal `azp` to a bounded metric label.

    Returns the caller id when it is on the measured census
    (SERVICE_PRINCIPAL_METRIC_CALLERS), otherwise the single `other` bucket. The
    label is on a per-request counter, so an unclamped remote-controlled string
    is an unbounded-series hazard even though `azp` arrives inside a VERIFIED
    secure-apps token: one client rename, or a per-pod client id, and the series
    count grows without limit on the hot path.

    ⚠️ `other` is a bucket, never a caller. A nonzero `other` means the census is
    stale — re-measure and extend the setting rather than attributing it.
    """
    return caller_client_id if caller_client_id in SERVICE_PRINCIPAL_METRIC_CALLERS else "other"


def _sp_identity_caller_label(caller_client_id: str) -> str:
    """Clamp `azp` for metric_service_principal_identity — see `_sp_caller_label`."""
    return (
        caller_client_id
        if caller_client_id in SERVICE_PRINCIPAL_IDENTITY_METRIC_CALLERS
        else "other"
    )


# The WHOLE licences tenant registry as one `name -> id` map, fetched from
# `GET /internal/tenants` (one call returns every non-archived tenant, so
# there is nothing to key per azp — same single-entry shape as billing's
# `tenant_names._name_cache`). An azp resolves by EXACT local match of its
# SERVICE_PRINCIPAL_TENANT_NAME_MAP candidate against this map.
#
# Three timers, and why each has the value it has:
#   _SP_TENANT_CACHE_TTL (300s)     - a USABLE map is reused this long. Matches
#       the other licences-backed caches; a tenant provisioned by an operator
#       starts resolving within 5 minutes with no restart.
#   _SP_TENANT_NEGATIVE_BACKOFF (30s) - after a NON-DEFINITIVE fetch, no new
#       fetch is attempted this long. Without it a down licences is hit on
#       EVERY service-principal auth request (each paying the timeout below).
#       Deliberately far shorter than the TTL, and the failure is NOT stored
#       as a map: caching "no tenants" for 300s would pin every SP to
#       unresolved for minutes after licences recovers — a wrong ABSENCE that
#       outlives its cause. 30s bounds both the load on a sick licences
#       (<= 2 calls/min per replica) and the post-recovery lag.
#   _SP_TENANT_FETCH_TIMEOUT (2s)   - this fetch sits on the auth hot path of
#       an ALLOWED request; the shared client's INTROSPECT_TIMEOUT (10s) is
#       sized for introspection, not for a metering label.
#
# ⛔ NON-DEFINITIVE (never cached, arms the backoff): transport error /
# timeout, non-200, a body that is not a JSON list, an EMPTY list, or a list
# with no usable {id, name} row. An empty list is treated as "licences gave no
# answer", never as "there are no tenants" — the same strict reading billing's
# tenant_names applies — because the registry is never legitimately empty.
_SP_TENANT_CACHE_TTL = float(os.getenv("SERVICE_PRINCIPAL_TENANT_CACHE_TTL", "300"))
_SP_TENANT_NEGATIVE_BACKOFF = float(os.getenv("SERVICE_PRINCIPAL_TENANT_NEGATIVE_BACKOFF", "30"))
_SP_TENANT_FETCH_TIMEOUT = float(os.getenv("SERVICE_PRINCIPAL_TENANT_FETCH_TIMEOUT", "2.0"))


@dataclass
class _SpTenantMapCache:
    names_to_ids: dict[str, str] | None = None
    expires_at: float = 0.0
    backoff_until: float = 0.0

    def clear(self) -> None:
        self.names_to_ids = None
        self.expires_at = 0.0
        self.backoff_until = 0.0


_sp_tenant_map_cache = _SpTenantMapCache()
# Single-flight: concurrent SP requests on a cold/expired cache wait for ONE
# fetch instead of each issuing their own. The lock is NOT created here:
# asyncio.Lock() binds to whichever event loop first contends it, so a
# module-level instance would raise "bound to a different event loop" the
# next time a different loop (a different test, or a future multi-loop
# process) contends it. Created fresh per-app in lifespan instead — see
# `application.state.redis_connect_lock`, same reasoning, same place.


def _parse_internal_tenants(payload: object) -> dict[str, str] | None:
    """Build `name -> id` from a `GET /internal/tenants` body, or None if unusable.

    Rows that are not objects, or lack a non-empty string `id`/`name`, are
    skipped. A name carried by two ids (licences' `tenants.name` is UNIQUE, so
    this should not happen) is DROPPED rather than last-one-wins'd: picking one
    would be a billing decision taken on a guess.
    """
    if not isinstance(payload, list):
        return None
    out: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        tenant_id, name = row.get("id"), row.get("name")
        if not (isinstance(tenant_id, str) and tenant_id and isinstance(name, str) and name):
            continue
        if name in out and out[name] != tenant_id:
            ambiguous.add(name)
        out[name] = tenant_id
    for name in ambiguous:
        out.pop(name, None)
    return out or None


async def _get_licences_tenant_name_map(request: Request) -> dict[str, str] | None:
    """Return the cached licences `name -> id` map, fetching on expiry.

    None means "no usable answer right now" (see the non-definitive list
    above). Never raises: this runs inside an ALLOWED request, and a metering
    label must not be able to turn an allow into a 500.
    """
    cache = _sp_tenant_map_cache
    now = time.monotonic()
    if cache.names_to_ids is not None and now < cache.expires_at:
        return cache.names_to_ids
    if now < cache.backoff_until:
        return None

    lock: asyncio.Lock | None = getattr(request.app.state, "sp_tenant_map_lock", None)
    if lock is None:
        # State was never installed (lifespan did not run): no single-flight
        # is possible, so answer non-definitively rather than fetch unlocked.
        return None

    async with lock:
        now = time.monotonic()
        if cache.names_to_ids is not None and now < cache.expires_at:
            return cache.names_to_ids
        if now < cache.backoff_until:
            return None

        names_to_ids: dict[str, str] | None = None
        detail = ""
        try:
            sa: ServiceTokenProvider = request.app.state.service_auth
            auth_headers = await sa.auth_header(request.app.state.http_client)
            resp = await request.app.state.http_client.get(
                LICENSES_INTERNAL_TENANTS_URL,
                headers=auth_headers,
                timeout=_SP_TENANT_FETCH_TIMEOUT,
            )
            if resp.status_code == 200:
                names_to_ids = _parse_internal_tenants(resp.json())
                if names_to_ids is None:
                    detail = "unusable_body"
            else:
                detail = f"status_{resp.status_code}"
        except Exception as e:  # noqa: BLE001 - see docstring: must never raise
            detail = f"{type(e).__name__}: {e}"

        now = time.monotonic()
        if names_to_ids is None:
            cache.backoff_until = now + _SP_TENANT_NEGATIVE_BACKOFF
            logger.warning(
                json.dumps(
                    {
                        "event": "sp_tenant_map_unavailable",
                        "detail": detail,
                        "backoff_seconds": _SP_TENANT_NEGATIVE_BACKOFF,
                    }
                )
            )
            # A previously good map is NOT served past its TTL: the TTL is the
            # promise that an archived/renamed tenant stops resolving.
            cache.names_to_ids = None
            return None
        cache.names_to_ids = names_to_ids
        cache.expires_at = now + _SP_TENANT_CACHE_TTL
        cache.backoff_until = 0.0
        return names_to_ids


async def _resolve_service_principal_tenant(request: Request, caller_client_id: str) -> str:
    """Resolve a VERIFIED service-principal `azp` to a licences tenant id.

    Returns the tenant's id string when the azp's candidate name is present
    in licences' `GET /internal/tenants` list, else "" (no candidate name,
    LICENSES_INTERNAL_TENANTS_URL unset, name absent from a usable list, or no
    usable list right now). Callers must never treat "" as a fabricated
    identity — see the emit site.
    """
    candidate_name = SERVICE_PRINCIPAL_TENANT_NAME_MAP.get(caller_client_id)
    if not candidate_name or not LICENSES_INTERNAL_TENANTS_URL:
        # No known candidate, or resolution is unconfigured: nothing to ask,
        # and no HTTP call is made.
        return ""
    names_to_ids = await _get_licences_tenant_name_map(request)
    if names_to_ids is None:
        return ""
    return names_to_ids.get(candidate_name, "")


jwt_mode.set_metrics_hooks(
    on_jwks_refresh=_bump_jwks_refresh,
    on_cache_eviction=_bump_cache_eviction,
    on_tenant_claim=_bump_tenant_claim,
    on_audience_shadow=_bump_audience_shadow,
)
_metering_client_mod.set_eviction_hook(_bump_cache_eviction)


# --------------------------------------------------------------------------
# Metering/budget Redis — lazy accessor.
#
# This is the ONLY construction site (lifespan delegates to it). redis-py
# already recovers from mid-life connection drops by itself; what it cannot
# recover from is never having had a client, which is exactly what a failed
# connect at boot produced: app.state.redis stayed None for the process
# lifetime, so metering emit and budget enforcement were silently dead on that
# replica until someone restarted the pod.
#
# The accessor is called from the ext_authz hot path AND from _readiness_checks.
# The readiness call is the one that matters today: production traffic is
# currently 100% allowed_service_principal, and that branch returns before any
# metering emit, so an accessor wired only into the request path would never
# fire. /ready runs every 10s and is therefore the reconnect driver — the same
# argument readiness.py already makes for JWKS.
# --------------------------------------------------------------------------
def _init_redis_state(application: FastAPI) -> None:
    """Install the accessor's state on the app. Called once, from lifespan."""
    application.state.redis = None
    application.state.redis_next_attempt_at = 0.0  # time.monotonic() deadline
    application.state.redis_failures = 0
    application.state.redis_connect_lock = asyncio.Lock()


def _redis_backoff_seconds(failures: int) -> float:
    """Capped exponential backoff (pre-jitter) for the Nth consecutive failure."""
    # Clamp the exponent as well as the result: 2 ** failures would otherwise
    # build a multi-thousand-bit int on a long outage just to be thrown away.
    exponent = min(max(failures - 1, 0), 32)
    return min(REDIS_RECONNECT_MAX_INTERVAL, REDIS_RECONNECT_MIN_INTERVAL * (2**exponent))


def _redis_retry_delay(failures: int) -> float:
    """Backoff with +/-20% jitter, so the two replicas do not retry in lockstep."""
    return _redis_backoff_seconds(failures) * random.uniform(0.8, 1.2)


async def _get_redis(application: FastAPI) -> aioredis.Redis | None:
    """Return a live Redis client, reconnecting at most once per backoff window.

    Hot path (every mesh request): one attribute read when connected, one
    monotonic compare when disconnected inside the window. The window is
    checked BEFORE the lock so the ext_authz path never queues behind an
    in-flight connect while Redis is down.
    """
    if not METERING_ENABLED:
        # Kill switch: never attempt a connect, never arm a backoff.
        return None

    state = application.state
    client: aioredis.Redis | None = getattr(state, "redis", None)
    if client is not None:
        return client

    if time.monotonic() < getattr(state, "redis_next_attempt_at", 0.0):
        metric_redis_connect.labels(result="skipped_backoff").inc()
        return None

    lock: asyncio.Lock | None = getattr(state, "redis_connect_lock", None)
    if lock is None:
        # State was never installed (lifespan did not run) — do not connect.
        return None

    async with lock:
        # Re-check under the lock: a concurrent caller may have connected, or
        # armed the window, while we were waiting for it.
        client = getattr(state, "redis", None)
        if client is not None:
            return client
        now = time.monotonic()
        if now < state.redis_next_attempt_at:
            metric_redis_connect.labels(result="skipped_backoff").inc()
            return None

        attempt = state.redis_failures + 1
        retry_in = _redis_retry_delay(attempt)
        # Arm the NEXT attempt BEFORE awaiting the connect. A hung connect
        # otherwise leaves the window open, so every subsequent probe tick
        # piles in behind it.
        state.redis_next_attempt_at = now + retry_in

        new_client: aioredis.Redis | None = None
        try:
            new_client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
            )
            # socket_connect_timeout does not bound DNS resolution; wait_for
            # bounds the whole connect+PING against /ready's total budget.
            await asyncio.wait_for(new_client.ping(), timeout=REDIS_CONNECT_TIMEOUT)
        except asyncio.CancelledError:
            # /ready wraps the whole probe in asyncio.wait_for(..., 2.0s). If that
            # budget blows while we are inside the PING above, CancelledError is
            # a BaseException and does NOT match the arm below: without this one
            # the ConnectionPool from_url just built leaks (fd + socket) and the
            # attempt goes uncounted, so the backoff never escalates past window 1.
            if new_client is not None:
                with contextlib.suppress(Exception):
                    await new_client.aclose()
            state.redis_failures = attempt
            metric_redis_connect.labels(result="cancelled").inc()
            raise
        except (redis.exceptions.RedisError, OSError, asyncio.TimeoutError) as e:
            # asyncio.TimeoutError IS the builtin TimeoutError (3.11+), itself an
            # OSError subclass — listed for readability, not coverage.
            if new_client is not None:
                # from_url builds a ConnectionPool; a discarded client leaks one
                # per attempt once we retry forever.
                with contextlib.suppress(Exception):
                    await new_client.aclose()
            state.redis_failures = attempt
            metric_redis_connect.labels(result="failure").inc()
            logger.warning(
                json.dumps(
                    {
                        "event": "metering_redis_unavailable",
                        "attempt": attempt,
                        "retry_in_s": round(retry_in, 2),
                        "error": type(e).__name__,
                    }
                )
            )
            return None

        state.redis = new_client
        state.redis_failures = 0
        state.redis_next_attempt_at = 0.0
        metric_redis_connect.labels(result="success").inc()
        logger.info(json.dumps({"event": "metering_redis_connected", "reconnect": attempt > 1}))
        return new_client


# App-route enforcement: maps URL path prefixes to expected app_key values.
# A value is ONE key (the norm), or a TUPLE of accepted keys for a prefix that is
# mid-transition between apps. For a tuple the FIRST key is the route's primary:
# it is the key `_registered_app_key` reports, and the one JWT mode derives for a
# caller that asserted none. Every key in the tuple satisfies `_check_app_route`.
APP_ROUTE_REGISTRY: dict[str, str | tuple[str, ...]] = {
    "/workers/": "workers",
    "/kanban/": "kanban",
    "/kag/": "kag",
    "/secrets/": "secrets",
    "/tools/": "tools",
    "/artifacts/": "artifacts",
    "/procedures/": "procedures",
    "/requirements/": "requirements",
    "/mcp/": "mcp",
    "/coding/": "coding",
    "/skills/": "skill-builder",
    "/licenses/": "licenses",
    "/inference-router/": "inference-router",
    "/model-registry/": "model-registry",
    "/data-collector/": "data-collector",
    "/model-evaluator/": "model-evaluator",
    "/training-data-generator/": "training-data-generator",
    "/improvement-loop/": "improvement-loop",
    "/rtx/": "vllm",
    "/vllm/": "vllm",
    # pgx laya (Jev-compatible System-1 classifier) behind its Jev bridge; the
    # api.nonagentic.ai `/laya/` route forwards to pgx:9210 (gitops-production
    # scripts/host-configs/pgx/laya/). A licence for app `laya` only.
    "/laya/": "laya",
    # api.nonagentic.ai `/tts/`, `/asr/`, `/image/` -> modality-gateway (gitops
    # #4775: the public base_urls for tenant BYO GPU registration). Before
    # 2026-10-20 they were unregistered and ANY active licence of any app
    # passed. `/<p>/health` stays open: the gateway Lua exempts 2-segment
    # health paths before ext_authz is ever called.
    # TTS/ASR: owner decision 2026-10-22 — shared PLATFORM services (like
    # vllm/laya), decoupled from Studio/w2v, bound to the platform app `speech`
    # (supersedes the 2026-10-20 `w2v`-only binding for these two).
    # ⚠️ TRANSITION SET, owner decision 2026-10-22: `w2v` is still accepted so
    # Studio callers on api.nonagentic.ai do not break. `w2v` is PRIMARY on
    # purpose: a JWT caller without `x-app-key` derives the primary, and today
    # that is `w2v` (Studio). Licence-token callers carry their own key, so a
    # `speech` token passes regardless of order. When Studio migrates, the order
    # flips to ("speech", "w2v"), or `w2v` is dropped and these become "speech".
    "/tts/": ("w2v", "speech"),
    "/asr/": ("w2v", "speech"),
    # `/image/` stays w2v's GPU modality (owner decision 2026-10-20) — a
    # licence for app `w2v` only.
    "/image/": "w2v",
    # `/image-gen/` is the NATIVE image host (POST /v1/images/generations), not
    # the modality-gateway dialect above: it forwards straight to
    # image-gen.internal:9204, the MOSS-Image server on mini1, and conveyor is
    # the caller that owns it (`media.generate_image`). Owner decision
    # 2026-11-22. A separate prefix on purpose — `/image/` is the legacy path
    # that has reached nothing since mlx_inference#25, and repointing it would
    # silently change what every existing `w2v` caller gets.
    "/image-gen/": "conveyor",
    "/kag-graph/": "kag-graph",
    "/kag-orchestrator/": "kag-orchestrator",
    "/graph-adapter/": "graph-adapter",
    "/karma/": "karma",
    "/hirag/": "hirag",
    "/nlp/": "nlp-service",
    "/storage/": "storage",
    "/ingestion/": "ingestion",
    "/npm/": "npm",
}

# Paths on which a caller INTROSPECTS ITS OWN LICENCE. See
# LICENSE_SELF_READ_EXEMPT_ENABLED for the outage and the direction argument.
#
# MATCHING IS DELIBERATELY TIGHT, and the shape of each entry carries meaning:
#   - an entry WITHOUT a trailing slash matches EXACTLY (or with one trailing
#     slash). `/licenses/active` must not be allowed to exempt
#     `/licenses/activedirectory-sync` by a naive startswith.
#   - an entry WITH a trailing slash matches as a prefix, because the remainder
#     is a path PARAMETER — `/licenses/user/{sub}`.
# Matching happens AFTER `_normalize_route_path`, so `/licenses/user/../admin/x`
# folds to `/licenses/admin/x` and is NOT exempt.
APP_ROUTE_SELF_READ: tuple[str, ...] = ("/licenses/active", "/licenses/user/")


def _normalize_route_path(path: str) -> str:
    """Fold a raw Envoy ``:path`` into the form the route registry matches on.

    ``ext_authz_check`` reads Envoy's RAW ``:path``. Envoy's HCM does NOT merge
    slashes — ``merge_slashes`` defaults to false and gitops sets it nowhere
    (verified: no ``merge_slashes``/``normalize_path`` anywhere in
    gitops-production) — so ``//vllm/x`` arrives verbatim and used to miss every
    registered prefix, dropping the request into the unregistered-allow bucket
    with no app_key binding at all.

    No live end-to-end bypass was found for that today: api.nonagentic.ai routes
    on the SAME prefix strings (``uri.prefix: /vllm/`` in
    ``istio/tunnel-mtls/virtualservice.yaml``) with no catch-all, so a path that
    misses the registry also misses the route; and the host-routed
    `internal` VirtualServices reach Starlette apps that do not fold ``//``
    either. It is a latent gap that goes live the moment any upstream normalises
    (a non-Starlette backend, ``merge_slashes: true``, or a catch-all route), so
    close it here rather than depend on the coincidence.

    Inert for well-formed paths: a path with no empty/dot segments and no
    escaped slash normalises to itself, trailing slash included.
    """
    # The query string is not part of the route; `.` and `..` inside it are data.
    path = path.split("?", 1)[0].split("#", 1)[0]
    # Envoy leaves %2F alone (path_with_escaped_slashes_action defaults to
    # KEEP_UNCHANGED) but plenty of upstreams decode it, so treat it as the
    # separator it will become rather than let it hide a registered prefix.
    path = path.replace("%2F", "/").replace("%2f", "/")
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    normalized = "/" + "/".join(segments)
    if path.endswith("/") and len(normalized) > 1:
        normalized += "/"
    return normalized


def _registered_app_keys(path: str) -> tuple[str, ...] | None:
    """Every app key a REGISTERED prefix accepts (primary first), or None when
    the path is unregistered. A single-key entry yields a 1-tuple."""
    normalized = _normalize_route_path(path)
    for prefix, accepted in APP_ROUTE_REGISTRY.items():
        if normalized.startswith(prefix):
            return (accepted,) if isinstance(accepted, str) else accepted
    return None


def _registered_app_key(path: str) -> str | None:
    """The PRIMARY app key a REGISTERED prefix demands, or None when the path is
    unregistered. For a transition tuple this is its first key; whether a key is
    ACCEPTED is `_registered_app_keys`' question, not this one's.

    Split out of ``_check_app_route`` so "which key does this route require?" and
    "is this route in the registry at all?" stop being the same question. The
    asserted-key rule needs the SECOND one on its own: an asserted key on an
    unregistered path has no expected key to be compared against, and that
    absence — not a mismatch — is what makes it refusable.

    First match wins, in ``APP_ROUTE_REGISTRY`` insertion order, exactly as the
    inline loop did.
    """
    accepted = _registered_app_keys(path)
    return accepted[0] if accepted else None


def _asserted_key_route_allowed(path: str) -> bool:
    """True = an ASSERTED (caller-supplied) app key may authorize this path.

    ⚠️ THE VERDICT IS ABOUT THE ROUTE, NOT THE KEY, and that is deliberate: the
    key's VALUE is unsigned, so no property of it is evidence of anything. The
    only question that has an answer is whether the route supplies a binding for
    it — i.e. whether the path is registered. It therefore takes no ``app_key``
    argument at all, so no future reader can be tempted to "improve" it by
    inspecting one.

    Applies ONLY to asserted keys (the `x-app-key` header on the JWT path). An
    ATTESTED key — returned by `/license/introspect` as part of the credential —
    is not subject to this: see ASSERTED_APP_KEY_ALLOW_UNREGISTERED.
    """
    return _registered_app_key(path) is not None


def _is_self_read_route(method: str, path: str) -> bool:
    """True = this is a GET/HEAD of one of the caller's own-licence routes.

    METHOD IS LOAD-BEARING, not decoration: the whole safety argument for
    defaulting the exemption ON is that it can never open a MUTATING route. A
    POST/PUT/PATCH/DELETE to the very same path stays fully bound. HEAD rides
    along with GET because it is the same read with the body suppressed.

    The path is matched against `_normalize_route_path`'s output, never the raw
    Envoy `:path` — otherwise `//licenses/active` or `/licenses/x/../active`
    would be judged by a different string than the registry judged.
    """
    if (method or "").strip().upper() not in ("GET", "HEAD"):
        return False
    normalized = _normalize_route_path(path)
    for entry in APP_ROUTE_SELF_READ:
        if entry.endswith("/"):
            if normalized.startswith(entry):
                return True
        elif normalized == entry or normalized == entry + "/":
            return True
    return False


# Every entry in APP_ROUTE_SELF_READ lives under this prefix, and that is the
# whole point: the exemption exists for a read of the caller's OWN licence,
# which only the licences route can serve.
SELF_READ_RAW_PREFIX = "/licenses/"


def _self_read_raw_path_stays_in_licenses(path: str) -> bool:
    """True = the RAW `:path`, not merely its normalised form, is under `/licenses/`.

    ⚠️ WHY A SECOND CHECK AT ALL. `_is_self_read_route` judges
    `_normalize_route_path`'s output, and that is correct for the registry
    question — but it makes the exemption reachable from a raw path that a
    ROUTER would send somewhere else entirely. `/vllm/../licenses/active`
    normalises to `/licenses/active` and is exempt here, while
    `uri.prefix: /vllm/` in the api.nonagentic.ai VirtualService matches the RAW
    string and forwards it to vllm. An unlicensed JWT caller would then be
    ALLOWED by us on a request that never reaches the licences app.

    That gap is only closed by checking the raw string, because whether Istio
    normalises BEFORE ext_authz is UNVERIFIED: the `istio-system/istio`
    ConfigMap carries no `pathNormalization` setting, so it falls to a default
    we are not willing to assume. An authorization verdict must not depend on
    unverified mesh behaviour, and the fail-closed direction is cheap: no
    legitimate caller sends `..`.

    BENIGN NORMALISATIONS ONLY, exactly the ones the existing code already
    treats as the same read — duplicate slashes, trailing slash, `%2F`. Dot
    segments are deliberately NOT resolved: a `..` anywhere is refused outright
    rather than folded, so a path that traverses OUT of `/licenses/` cannot
    earn the exemption by normalising back in.

    Inert for every shape the audit confirmed legitimate: `/licenses/active`,
    `//licenses/active`, `/licenses/active/` and `/licenses%2Factive` all pass.
    """
    raw = path.split("?", 1)[0].split("#", 1)[0]
    raw = raw.replace("%2F", "/").replace("%2f", "/")
    segments = [segment for segment in raw.split("/") if segment not in ("", ".")]
    if ".." in segments:
        return False
    return ("/" + "/".join(segments)).startswith(SELF_READ_RAW_PREFIX)


def _self_read_exemption_applies(app_key: str | None, path: str, method: str) -> bool:
    """True = the self-read exemption is what FLIPS this verdict from deny to allow.

    Narrower than "this is a self-read": it is false whenever the request would
    have been allowed anyway. That precision is what makes the metric honest —
    `allowed_self_read_exempt` then counts requests the exemption actually
    RESCUED, not every read of `/licenses/active` by the licenses app itself.

    Five things must all hold, and each one is a bound worth naming:
      1. the flag is on (the revert switch);
      2. the path is REGISTERED — an unregistered path already allows, so there
         is nothing to rescue;
      3. the caller presented a NON-EMPTY key that is not the expected one. A
         keyless token stays DENIED (`app_route_missing_key`): that deny is a
         separate hardening and the outage caller is keyed (`w2v`);
      4. it is a GET/HEAD of a self-read route;
      5. the RAW `:path` — not merely its normalised form — is also under
         `/licenses/`. Bound 4 alone lets `/vllm/../licenses/active` in: it
         normalises to `/licenses/active` and reads as a self-read here, while
         Envoy's `uri.prefix: /vllm/` VirtualService match is on the RAW,
         unnormalised string and would forward the request to vllm — a
         `w2v`-keyed caller reaching a GPU backend it never held a licence
         for, laundered through this exemption. The JWT arm
         (`_handle_jwt_mode`) already applies this same
         `_self_read_raw_path_stays_in_licenses` bound; this is the identical
         rule on the opaque arm, not a second implementation of it.
    """
    if not LICENSE_SELF_READ_EXEMPT_ENABLED:
        return False
    accepted_keys = _registered_app_keys(path)
    if accepted_keys is None:
        return False
    key = (app_key or "").strip()
    if not key or key in accepted_keys:
        return False
    return _is_self_read_route(method, path) and _self_read_raw_path_stays_in_licenses(path)


def _normalize_authority(authority: str) -> str:
    """Envoy `:authority` / `Host` -> bare lowercase hostname, or "" when unusable.

    Returns "" — meaning NO MATCH, ALWAYS ALLOW — for the three shapes that are
    not a public host, and each is a real east-west traffic pattern that must
    never be denied for failing to carry something it structurally cannot carry:
      - empty (no authority header at all);
      - a bare service name with no dot (`marketing`, `localhost`);
      - anything ending the cluster domain (in-mesh pod-to-pod).
    Port is stripped and the result lowercased, so `marketing.merqi.nl:443` and
    `MARKETING.MERQI.NL` are the same host as `marketing.merqi.nl`.
    """
    host = (authority or "").strip().split(":")[0].strip().lower().rstrip(".")
    if not host:
        return ""
    if host.endswith(".svc.cluster.local") or host == "svc.cluster.local":
        return ""
    if "." not in host:
        return ""
    return host


def _check_app_host(app_key: str | None, path: str, authority: str) -> tuple[bool, str, str, str]:
    """Host->app binding. Returns ``(allow, metric_result, host_label, app_label)``.

    ⛔ THE CALLER MUST ONLY INVOKE THIS WHEN `_registered_app_key(path)` IS None.
    That guard is what pins the blast radius on the 29 live path prefixes at
    exactly zero: a request whose path matched a prefix never reaches here, so no
    existing verdict can change.

    `allow` is False in exactly ONE case — mode `enforce` AND a registry host AND
    a key that is not the one that host demands. Mode `off` never reaches this
    function at all; mode `observe` allows every outcome and only counts.

    Label values are bounded on purpose (see `metric_app_host_route`): the real
    host is emitted only for registry members, "other" otherwise.
    """
    host = _normalize_authority(authority)
    if not host:
        return (True, "no_host", "other", "other")
    expected_app = APP_HOST_REGISTRY.get(host)
    if expected_app is None:
        return (True, "unregistered", "other", "other")
    if (app_key or "").strip() == expected_app:
        return (True, "matched", host, expected_app)
    if APP_HOST_ROUTE_MODE == "enforce":
        return (False, "mismatch_denied", host, expected_app)
    return (True, "mismatch_observed", host, expected_app)


def _check_app_route(app_key: str | None, path: str, method: str) -> bool:
    """True = allow.

    A REGISTERED prefix requires an exact app_key match. A missing, empty or
    whitespace-only key therefore DENIES: one unkeyed token must not authorize
    all 28 registered prefixes, /vllm/ and /rtx/ (GPU) among them.
    An UNREGISTERED path still allows any (or no) key — THIS predicate's answer
    is unchanged there, and must stay unchanged: the ATTESTED (introspected) key
    population depends on it.

    ONE EXEMPTION, added for the 2026-09-14 self-read outage: a GET/HEAD of the
    caller's OWN licence (`APP_ROUTE_SELF_READ`) is introspection of a credential
    already held, not consumption of the `licenses` app, so it allows even when
    the key is another app's. Non-GET on the same paths, and every other
    `/licenses/` path, stay bound. See `_self_read_exemption_applies`.

    An ASSERTED key on an unregistered path is refused separately, at the call
    site, by ``_asserted_key_route_allowed``.

    The path is normalised first (see ``_normalize_route_path``) so a doubled
    slash or a dot segment cannot walk a request out of the registry.

    DELIBERATE SURVIVING MUTANT: removing the ``.strip()`` below leaves the
    whole suite green (executed, 390/390). That is not a coverage hole — no
    registry value contains whitespace, so ``.strip()`` can only turn a
    NON-match into a match. Dropping it therefore makes this predicate strictly
    STRICTER (`" workers "` on `/workers/` would flip allow -> deny), and the
    only test that could kill the mutant is one asserting that lenient
    direction, i.e. ratifying whitespace tolerance inside an authz check. The
    whitespace-only case is already pinned by
    ``test_whitespace_only_app_key_denied_on_registered_route``, which denies
    with or without the strip; the strip's real job is the caller's
    ``app_route_missing_key`` vs ``app_route_mismatch`` split, and that call
    site keeps its own ``.strip()``.
    """
    # ⚠️ THIS PREDICATE STILL CANNOT TELL AN ATTESTED KEY FROM AN ASSERTED ONE,
    # and it is not supposed to. It is handed a key and a path and answers only
    # "does this route demand a different key?". On the JWT path the key it is
    # handed originates in the caller-supplied, unsigned `x-app-key` header, so a
    # caller that sets `x-app-key` to match the prefix it is calling passes HERE
    # while still having chosen which licence's quota funds the call.
    #
    # ⭐ That residue is bounded, and the bound is the registry: a caller can only
    # pick the licence the ROUTE already names. The free choice — an asserted key
    # on a path the registry says nothing about — is refused at the call site by
    # `_asserted_key_route_allowed`, NOT here, because that rule needs to know
    # where the key came from and this function deliberately does not.
    key = (app_key or "").strip()
    accepted_keys = _registered_app_keys(path)
    if accepted_keys is None:
        return True
    if key and key in accepted_keys:
        return True
    # LAST resort, never a shortcut: every other deny above is unchanged, and
    # this arm can only fire for a GET/HEAD of the caller's own licence. See
    # `_self_read_exemption_applies`.
    #
    # ⛔ `method` IS REQUIRED — NO DEFAULT, deliberately. A default of "GET" on an
    # AUTHORIZATION predicate is pure trap surface: nothing depends on it (an
    # audit mutated it to "" and the whole suite stayed green), so its only
    # possible effect is to let a future call site that FORGETS the argument
    # silently have its POST judged as a GET. Required means that mistake is a
    # TypeError at import/callsite instead of a silent widening.
    return _self_read_exemption_applies(app_key, path, method)


async def _prefetch_extra_issuer_jwks(registry: IssuerJwks) -> None:
    """Best-effort boot fetch for each EXTRA allowlisted issuer.

    Not readiness-gating: a second realm's Keycloak blip must not evict pods
    that still serve the primary realm. A cache whose boot fetch failed fetches
    lazily on the first token from that issuer (PyJWKClient), and fails closed.
    """
    for iss, cache in registry.extra.items():
        try:
            await asyncio.to_thread(cache.fetch)
            logger.info(
                json.dumps(
                    {"event": "issuer_jwks_initialized", "issuer": iss, "url": cache._jwks_url}
                )
            )
        except (jwt.PyJWKClientError, json.JSONDecodeError, OSError) as e:
            logger.error(
                json.dumps(
                    {
                        "event": "issuer_jwks_init_failed",
                        "issuer": iss,
                        "url": cache._jwks_url,
                        "error": str(e),
                    }
                )
            )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Initialize and teardown shared clients."""
    application.state.http_client = httpx.AsyncClient(timeout=INTROSPECT_TIMEOUT)
    _init_redis_state(application)
    application.state.service_auth = ServiceTokenProvider()
    # Keep the SA token warm. The first licence-gated request after an idle
    # window otherwise pays a Keycloak mint plus cold outbound connections
    # (4 939.7 ms measured 2026-11-15; the ingress budget is now 10 s). The loop
    # mints a FRESH token each tick — a cache hit would return the same token
    # with the same expiry. Only a configured provider can mint, and an
    # unconfigured one must not race the tests or boot with a pointless tick.
    # The loop stops on shutdown and never raises; see
    # ServiceTokenProvider.keep_warm.
    application.state.service_auth_stop = asyncio.Event()
    application.state.service_auth_task = None
    if SA_TOKEN_REFRESH_SECONDS > 0 and application.state.service_auth.configured:
        application.state.service_auth_task = asyncio.create_task(
            application.state.service_auth.keep_warm(
                application.state.http_client,
                SA_TOKEN_REFRESH_SECONDS,
                application.state.service_auth_stop,
            ),
            name="service-auth-token-refresh",
        )
    # Same reasoning as redis_connect_lock: created fresh per-app so it binds
    # to THIS app's event loop, never a module-level singleton.
    application.state.sp_tenant_map_lock = asyncio.Lock()
    # Token budgets and request quotas are decided by billing over HTTP, so the
    # client exists whenever either budget gate is armed.
    application.state.metering_client = (
        MeteringClient()
        if (BUDGET_ENFORCEMENT_ENABLED or SERVICE_PRINCIPAL_BUDGET_ENFORCED)
        else None
    )
    # Caller-identity: secure-apps SA JWT verification. Opt-in via config — the
    # JWKS client is only created when the secure-apps issuer URL is configured.
    # When unset, ext_authz_check skips caller-identity entirely and injects
    # empty x-caller-* headers (M1–M7 rollout semantics).
    application.state.caller_jwks = None
    if _settings.keycloak_secure_apps_issuer:
        application.state.caller_jwks = JwksCache(
            jwks_url=_settings.keycloak_secure_apps_jwks_url,
        )
        try:
            # JwksCache.fetch() performs the real network fetch, latches the
            # outcome on the instance for the /ready probe to read, and bumps the
            # jwks_refresh metric itself.
            await asyncio.to_thread(application.state.caller_jwks.fetch)
            logger.info(
                json.dumps(
                    {
                        "event": "caller_jwks_initialized",
                        "url": application.state.caller_jwks._jwks_url,
                    }
                )
            )
        except (jwt.PyJWKClientError, json.JSONDecodeError, OSError) as e:
            logger.error(
                json.dumps(
                    {
                        "event": "caller_jwks_init_failed",
                        "url": application.state.caller_jwks._jwks_url,
                        "error": str(e),
                    }
                )
            )
    if JWT_MODE_ENABLED:
        application.state.jwks = JwksCache()
        # Extra allowlisted user-JWT issuers (KEYCLOAK_TRUSTED_ISSUERS). Empty
        # by default; the primary stays `state.jwks` (readiness gates on it).
        application.state.issuer_jwks = IssuerJwks(application.state.jwks)
        await _prefetch_extra_issuer_jwks(application.state.issuer_jwks)
        application.state.active_resolver = ActiveLicenseResolver()
        # Eager JWKS fetch — surfaces network/DNS misconfiguration at boot, not
        # on first request. Anonymous JWT used so we exercise the actual fetch
        # path; signature verification failure is fine, JWKS reachability is not.
        try:
            await asyncio.to_thread(application.state.jwks.fetch)
            logger.info(
                json.dumps({"event": "jwks_initialized", "url": application.state.jwks._jwks_url})
            )
        except (jwt.PyJWKClientError, json.JSONDecodeError, OSError) as e:
            # PyJWKClientConnectionError (subclass of PyJWKClientError) for network/HTTP,
            # JSONDecodeError for malformed JWKS payload, OSError as a backstop for
            # anything urllib raises before pyjwt wraps it. Boot must not crash on
            # JWKS unavailability — the pod stays up but reports NOT ready via
            # /ready (fetch() left ever_loaded_ok False, so this pod holds no key
            # material at all), and the readiness probe drives a rate-limited
            # re-fetch until Keycloak comes back.
            logger.error(
                json.dumps(
                    {
                        "event": "jwks_init_failed",
                        "url": application.state.jwks._jwks_url,
                        "error": str(e),
                    }
                )
            )
    else:
        application.state.jwks = None
        application.state.issuer_jwks = None
        application.state.active_resolver = None
    if METERING_ENABLED:
        # Best-effort boot connect. A failure is no longer terminal for the
        # process: it just arms the accessor's backoff, and the /ready probe
        # drives the retry. Metering stays best-effort — the service must boot
        # without Redis.
        await _get_redis(application)
    # Pushgateway export. Disabled unless PUSHGATEWAY_URL is set, so this is a
    # no-op on merge. start() never blocks, and the loop never raises; see
    # metrics_push.py for why it is deliberately NOT hung off the /ready tick.
    application.state.metrics_pusher = MetricsPusher(
        url=_settings.pushgateway_url,
        interval_seconds=_settings.pushgateway_interval_seconds,
        timeout_seconds=_settings.pushgateway_timeout_seconds,
    )
    application.state.metrics_pusher.start()
    yield
    application.state.service_auth_stop.set()
    if application.state.service_auth_task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await application.state.service_auth_task
    await application.state.metrics_pusher.stop()
    await application.state.http_client.aclose()
    if application.state.redis:
        await application.state.redis.aclose()


app = FastAPI(
    title="License AuthZ Service",
    description="HTTP ext_authz service for Envoy/Istio license token validation",
    version="1.0.0",
    lifespan=lifespan,
)


class IntrospectResult(BaseModel):
    """Result from license introspection"""

    active: bool
    tenant_id: str | None = None
    subject_id: str | None = None
    # Keycloak `sub` of the licence's subject, as /license/introspect returns it
    # (licenses#224: a canonical UUID or null). Absent on /licenses/active, so it
    # is None on every JWT-mode result. Used ONLY as the licence-token-mode actor.
    subject_keycloak_sub: str | None = None
    # WHO ACTS for the licence subject (licenses migration 047): `subject_keycloak_sub`
    # when that is a real sub, else the subject's own service-account sub. Absent on
    # an older licenses build, hence the fallback at the use site.
    subject_actor_sub: str | None = None
    license_id: str | None = None
    app_id: str | None = None
    app_key: str | None = None
    plan_id: str | None = None
    plan_code: str | None = None
    reason: str | None = None


def hash_token(token: str) -> str:
    """Hash token for audit logging (never log raw tokens)"""
    return f"sha256:{hashlib.sha256(token.encode()).hexdigest()[:16]}"


_SAFE_HEADER_RE = re.compile(r"^[\w\-\.]+$")
# Caller roles arrive as a comma-separated list (Keycloak `realm_access.roles`
# joined with ","). Single header rather than repeated headers because Istio's
# headersToUpstreamOnAllow semantics for repeated headers aren't worth proving
# for an opt-in feature.
_SAFE_LIST_HEADER_RE = re.compile(r"^[\w\-\.,]+$")
_LIST_HEADERS = frozenset({"x-caller-roles"})

# Written into an audit field in place of a value the header-safety regex just
# rejected. Both regexes above are anchored `^[\w\-\.,]+$`, which admits neither
# `<` nor `>`, so no value that PASSES can spell this sentinel — and a value
# that spells it FAILS and is elided to it anyway. A row carrying this string
# therefore always means "the real value was refused here", never "the caller
# happened to be called that".
_UNSAFE_VALUE_ELIDED = "<unsafe-value-elided>"


def _elide_unsafe(value: str | None, rejected: str) -> str | None:
    """Return `value`, or the sentinel when `value` IS the rejected header value.

    The identity headers this service injects are the SAME strings as the
    identity fields `log_access` records: `x-license-id` is `result.license_id`,
    `x-app-id` is `result.app_id`, `x-plan-id` is `result.plan_id`, and on the
    service-principal path `x-caller-client-id` is `caller_client_id`, which
    that row puts in `reason`. (`x-tenant-id` is `tenant_uuid`, which is
    `tenant_id`, but it is UUID-validated or "" before either loop can see it —
    guarded, never observed to fail.)
    So when the header that fails the safety regex is one of those, an audit row
    built from those variables records the rejected value verbatim. `json.dumps`
    escapes it, so this is NOT log injection — it is the other failure: a
    wrong-but-complete value in a field downstream readers treat as a real
    licence id, in a row whose whole point is that the value was refused.

    ⛔ Matched by VALUE, never by a hand-kept map of header name → audit field.
    A map protects exactly the fields whoever wrote it thought of and silently
    stops covering the next one added; equality covers every field that carries
    the rejected string, including fields that reach it under a different name
    and fields that do not exist yet.

    `rejected` is never empty — both call sites `continue` on a falsy value — so
    this cannot collapse a legitimately empty field onto the sentinel.
    """
    return _UNSAFE_VALUE_ELIDED if value == rejected else value


def _introspect_result_from_dict(data: dict) -> IntrospectResult:
    """Build an IntrospectResult from a /license/introspect or /licenses/active body."""
    app_data = data.get("app") or {}
    plan_data = data.get("plan") or {}
    return IntrospectResult(
        active=data.get("active", False),
        tenant_id=data.get("tenant_id"),
        subject_id=data.get("subject_id"),
        subject_keycloak_sub=data.get("subject_keycloak_sub"),
        subject_actor_sub=data.get("subject_actor_sub"),
        license_id=data.get("license_id"),
        app_id=app_data.get("id"),
        app_key=app_data.get("key"),
        plan_id=plan_data.get("id"),
        plan_code=plan_data.get("code"),
        reason=data.get("reason"),
    )


# Stale-while-error store. A TRANSIENT introspect failure (timeout / network
# error / non-200) is NOT the same as a genuinely invalid token, but the caller
# treats any active=False result as a 403 license_invalid. So a valid token
# whose introspect happens to time out under load gets spuriously denied (the
# 2026-07 e2e 401 flake, and the internal deliverability/provider-health-signal
# 401s). Record every successful ACTIVE introspection here (keyed by token hash)
# and, on a later transient failure, serve the last-known-good result instead of
# denying. Bounded staleness preserves the posture: a definitive 200 active=false
# is authoritative and never served stale, and a token never seen before has no
# entry so a cold-cache failure still fails closed. No positive cache is added,
# so revocation latency on the happy path is unchanged (every request still
# introspects). A definitive 200 active=false additionally DROPS the entry (see
# ``_stale_drop``), so the deny is durable rather than true for one request only.
_INTROSPECT_STALE_TTL = float(os.getenv("INTROSPECT_STALE_TTL", "900"))
# Bound the store — cheap insurance, NOT a response to observed growth. The
# store only admits tokens that introspected ACTIVE, so its key space is the
# licenses `token` table, not caller-supplied input: an attacker cannot grow it
# by presenting garbage. Measured against prod 2026-08-08: 195 tokens ever
# issued, 186 not revoked, 6 used in the last 15 minutes, 0 created in the last
# 24 hours — a natural ceiling of a few hundred KB. The cap exists so that a
# future change in token-issuance volume cannot turn "entries are only dropped
# lazily on a _stale_get of that SAME key" into unbounded retention, and so the
# footprint is observable (evictions are metered). At the default 1000 against
# a 15-minute working set of 6 it provably never fires, so it cannot evict an
# entry before its TTL and cannot reintroduce the spurious-401 flake the stale
# store exists to fix. Same OrderedDict + oldest-first eviction as the jwt_mode
# active-license and metering caches, reporting on the same eviction counter.
#
# A cap of 0 (or negative) DISABLES the store entirely — see _stale_put. That
# is the natural reading of setting a max-entries knob to zero, and it doubles
# as an operator kill-switch for stale-while-error that needs no code change.
# Disabling fails CLOSED (a transient introspect failure denies, exactly as it
# did before this store existed), so a fat-fingered 0 costs availability under
# licenses flap, never security.
_INTROSPECT_STALE_MAX_ENTRIES = int(os.getenv("INTROSPECT_STALE_MAX_ENTRIES", "1000"))
_introspect_stale: OrderedDict[str, tuple[float, IntrospectResult]] = OrderedDict()


def _stale_enabled() -> bool:
    """False when the cap is 0 or negative: the store is turned off entirely."""
    return _INTROSPECT_STALE_MAX_ENTRIES > 0


def _stale_get(key: str) -> IntrospectResult | None:
    if not _stale_enabled():
        # Kill-switch takes effect immediately, not after the TTL drains.
        return None
    entry = _introspect_stale.get(key)
    if entry is None:
        return None
    expiry, result = entry
    if time.monotonic() >= expiry:
        _introspect_stale.pop(key, None)
        return None
    # LRU touch. This is the ONLY thing keeping a token alive in the LRU during
    # an outage of its own introspect: _stale_put runs on success only, so a
    # token being continuously stale-served would otherwise age out and be
    # evicted precisely while it is in active use.
    _introspect_stale.move_to_end(key)
    return result


def _stale_put(key: str, result: IntrospectResult) -> None:
    if not _stale_enabled():
        return
    if key in _introspect_stale:
        _introspect_stale.move_to_end(key)
    elif len(_introspect_stale) >= _INTROSPECT_STALE_MAX_ENTRIES:
        # Reachable only when the cap is >= 1, so the store is non-empty here
        # and popitem cannot raise KeyError. A KeyError escaping this function
        # is NOT caught by introspect_license's except tuple, so it would 500
        # every SUCCESSFUL introspect and fail the gateway closed fleet-wide.
        _introspect_stale.popitem(last=False)
        _bump_cache_eviction("introspect_stale")
    _introspect_stale[key] = (time.monotonic() + _INTROSPECT_STALE_TTL, result)


def _stale_drop(key: str) -> None:
    """Forget the last-known-good for one token (precise, never a cache flush)."""
    _introspect_stale.pop(key, None)


async def introspect_license(client: httpx.AsyncClient, token: str) -> IntrospectResult:
    """Call the licenses service to validate a token.

    On a TRANSIENT failure (timeout / network error / non-200), fall back to the
    last-known-good result in the stale store rather than denying. A definitive
    200 response (active true OR false) is authoritative. See ``_stale_get``.
    """
    key = hash_token(token)
    try:
        response = await client.post(
            LICENSE_INTROSPECT_URL,
            json={"token": token},
        )

        if response.status_code == 200:
            result = _introspect_result_from_dict(response.json())
            if result.active:
                # Record last-known-good for stale-while-error fallback.
                _stale_put(key, result)
            else:
                # A revocation must not be undone by the next blip. Without this
                # drop the pre-revocation active=True is still in the store, so
                # the very next transient failure serves it and the revoked
                # credential works again for the rest of the stale TTL. Honoring
                # the deny only on this one request is not honoring it.
                _stale_drop(key)
            # Definitive answer (active or a genuine active=false rejection).
            return result
        transient = IntrospectResult(
            active=False,
            reason=f"introspect_failed:{response.status_code}",
        )
    except httpx.TimeoutException:
        transient = IntrospectResult(active=False, reason="introspect_timeout")
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
        # httpx network/protocol errors, malformed JSON body, pydantic validation
        # failures — all transient/undeterminable.
        logger.error(json.dumps({"error": "introspect_exception", "detail": str(e)}))
        transient = IntrospectResult(active=False, reason="introspect_error")

    # Transient failure: serve the last-known-good result if we have one, so a
    # briefly slow/unreachable licenses service does not 403 a valid token.
    # Cold cache -> return the transient deny (fail-closed).
    stale = _stale_get(key)
    if stale is not None:
        logger.warning(
            json.dumps(
                {
                    "event": "introspect_stale_served",
                    "reason": transient.reason,
                    "token_hash": key,
                }
            )
        )
        return stale
    return transient


def log_access(
    *,
    license_id: str | None,
    tenant_id: str | None,
    app_id: str | None,
    plan_id: str | None,
    method: str,
    path: str,
    status: str,
    token_hash: str,
    source_ip: str,
    user_agent: str,
    reason: str | None = None,
    retarget_tenant_id: str | None = None,
) -> None:
    """Log access for audit trail (structured JSON).

    `retarget_tenant_id` is the tenant a trusted proxy was allowed to act for
    (license-authz#64). `tenant_id` stays the caller's OWN tenant. The key is
    emitted only when set, so non-retarget rows are unchanged.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "license_id": license_id,
        "tenant_id": tenant_id,
        "app_id": app_id,
        "plan_id": plan_id,
        "method": method,
        "path": path,
        "status": status,
        "token_hash": token_hash,
        "source_ip": source_ip,
        "user_agent": user_agent,
    }
    if reason:
        log_entry["reason"] = reason
    if retarget_tenant_id:
        log_entry["retarget_tenant_id"] = retarget_tenant_id

    logger.info(json.dumps(log_entry))


async def _emit_metering_event(
    request: Request,
    *,
    tenant_id: str,
    license_id: str | None,
    subject_id: str | None = None,
    app_id: str | None,
    plan_id: str | None,
    plan_code: str | None = None,
    method: str,
    path: str,
    status: str,
    source_ip: str,
    caller_client_id: str = "",
    retarget_tenant_id: str | None = None,
    actor_id: str | None = None,
    actor_missing_reason: str = "no_identity",
) -> None:
    """Emit a metering event to Redis Streams (non-blocking, never raises).

    `actor_id` is the UUID Keycloak `sub` of who acted (see
    `metric_actor_missing`). Added to the stream ONLY when it is a UUID; otherwise
    omitted and counted once under `actor_missing_reason` (or `not_uuid`).

    `retarget_tenant_id` (license-authz#64): the tenant a trusted proxy acted
    for, alongside the caller's own `tenant_id`. The stream field is added ONLY
    when set, so a non-retarget event carries exactly the fields it did before
    (same consumer caveat as `caller_client_id` below).

    `subject_id` is the licenses Subject UUID (stable, opaque, FK'd by License).
    Carries through to metering_service so per-user attribution is possible
    inside a tenant ("My usage" view, Top-N users). Empty string when unknown
    (e.g. requests denied before introspect succeeds).

    `caller_client_id` is the VERIFIED secure-apps `azp` of a first-party service
    principal. It is a THIRD, independent identity — not a tenant, not a subject,
    not a licence holder — and it is carried in its own field precisely so it can
    never be mistaken for one of those. Empty on every licence-token/JWT request.
    It exists because the service-principal path has no tenant, no licence and no
    subject, and `azp` is the only true identity that path holds: without a field
    of its own, attributing those requests at all would require writing the
    caller's name into a field that means something else.

    ⚠️ NEW FIELD ON AN EXISTING STREAM. Redis stream entries are field maps, so a
    consumer that reads named fields ignores an extra one; a consumer that
    validates with `extra="forbid"` would not. ⛔ This repo CANNOT verify which:
    the vendored contract (tests/contract/billing_openapi.json) covers billing's
    HTTP budget/quota surface and contains no `billing:events` stream schema at
    all. Verify against billing-service's consumer before relying on the field
    downstream; the failure, if it comes, is a consumer-side reject, not a
    rejected request here (this emit is best-effort and never raises).
    """
    resolved_actor_id, actor_missing = _resolve_actor_id(actor_id, actor_missing_reason)
    if actor_missing is not None:
        metric_actor_missing.labels(reason=actor_missing).inc()
    redis_client = await _get_redis(request.app)
    if redis_client is None:
        # A billable request just completed and there is nowhere to record it.
        # `_get_redis` returns None for two materially different reasons and
        # they must not share a series — see metric_metering_events_dropped.
        if not METERING_ENABLED:
            # Deliberate kill switch. Counted (usage IS going unmetered) but not
            # logged per request: it is a config state, not an incident, and it
            # would emit one WARNING for every request for the life of the
            # deployment.
            metric_metering_events_dropped.labels(reason="metering_unconfigured").inc()
            return
        metric_metering_events_dropped.labels(reason="redis_unavailable").inc()
        # WARNING, not debug, and carrying the event's identifying fields: this
        # line is the ONLY surviving record that this billable event existed, so
        # it has to be reconstructable from the log. `_get_redis` already logs
        # `metering_redis_unavailable`, but that fires once per CONNECT ATTEMPT
        # (bounded by the backoff), never once per lost event — it can tell you
        # Redis was down and can never tell you what that cost.
        logger.warning(
            json.dumps(
                {
                    "event": "metering_event_dropped",
                    "reason": "redis_unavailable",
                    "tenant_id": tenant_id,
                    "license_id": license_id or "",
                    "subject_id": subject_id or "",
                    "actor_id": resolved_actor_id or "",
                    # ⛔ Must be logged, not omitted "because it is usually
                    # empty": on the service-principal path it is the ONLY
                    # identity the event has, so dropping it from the line that
                    # is the event's last surviving record leaves a lost billable
                    # request with no attribution anywhere at all.
                    "caller_client_id": caller_client_id,
                    "app_id": app_id or "",
                    "plan_id": plan_id or "",
                    "method": method,
                    "path": path,
                    "status": status,
                    **({"retarget_tenant_id": retarget_tenant_id} if retarget_tenant_id else {}),
                }
            )
        )
        return
    # Annotated with redis-py's own xadd field types: dict is invariant, so an
    # inferred dict[str, str] does not match xadd's Dict[FieldT, EncodableT].
    event_fields: dict[FieldT, EncodableT] = {
        "tenant_id": tenant_id,
        "license_id": license_id or "",
        "subject_id": subject_id or "",
        "app_id": app_id or "",
        "plan_id": plan_id or "",
        "plan_code": plan_code or "",
        "caller_client_id": caller_client_id,
        "method": method,
        "path": path,
        "status": status,
        "source_ip": source_ip,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if retarget_tenant_id:
        event_fields["retarget_tenant_id"] = retarget_tenant_id
    # Omitted, not "", when unresolved: billing parses a non-empty actor_id as a
    # UUID and counts a failure as malformed — absence must stay absence.
    if resolved_actor_id:
        event_fields["actor_id"] = resolved_actor_id
    # Added ONLY when the destination host is known — same consumer caveat as
    # the two optional fields above; billing derives the key from `path` otherwise.
    service_key = _metering_service_key(request)
    if service_key:
        event_fields["service_key"] = service_key
    try:
        await redis_client.xadd(
            METERING_STREAM,
            event_fields,
            maxlen=100000,
            approximate=True,
        )
    # ⚠️ `except A, B:` without parentheses is VALID here — PEP 758, Python 3.14
    # (this repo is requires-python >=3.14). It is NOT a Python 2 except-clause
    # and must not be "fixed". Note the corollary: PEP 758 forbids the
    # unparenthesised form when an `as` clause is present, which is why the
    # exception is not bound below.
    except redis.exceptions.RedisError, OSError:
        # Metering is best-effort: redis errors and socket failures must not
        # disrupt the request flow. Auth decision has already been made.
        # Best-effort is not the same as unrecorded, though: the event is gone,
        # so it is counted on the same instrument as the null-client drop.
        metric_metering_events_dropped.labels(reason="redis_error").inc()
        logger.warning(
            json.dumps(
                {
                    "event": "metering_emit_failed",
                    "reason": "redis_error",
                    "tenant_id": tenant_id,
                    "license_id": license_id or "",
                    "caller_client_id": caller_client_id,
                    "app_id": app_id or "",
                    "method": method,
                    "path": path,
                    "status": status,
                    **({"retarget_tenant_id": retarget_tenant_id} if retarget_tenant_id else {}),
                }
            )
        )


async def _metering_http_verdict(
    request: Request,
    fetch: Callable[[MeteringClient, dict[str, str]], Awaitable[BudgetDecision]],
    *,
    license_id: str | None,
    tenant_id: str,
) -> tuple[bool, str | None]:
    """Run one billing HTTP check under METERING_FAIL_OPEN; return (allowed, reason).

    Used for the token-budget and request-quota checks: billing's Postgres is
    the only counter for both, and license-authz reads neither from Redis.

    ⚠️ The broad ``except Exception`` is load-bearing: anything escaping here
    becomes a 500, and Envoy turns a 500 from ext_authz into a DENY for EVERY
    request. A response shape the client cannot parse, a rename on billing's
    side or a programming error must be governed by the operator's kill switch
    exactly like a transport failure. asyncio.CancelledError derives from
    BaseException, so client disconnects are still propagated.
    """
    metering_client: MeteringClient | None = request.app.state.metering_client
    try:
        if metering_client is None:
            raise RuntimeError(
                "metering client not configured (BUDGET_ENFORCEMENT_ENABLED and "
                "SERVICE_PRINCIPAL_BUDGET_ENFORCED both false at startup)"
            )
        sa: ServiceTokenProvider = request.app.state.service_auth
        auth_headers = await sa.auth_header(request.app.state.http_client)
        decision = await fetch(metering_client, auth_headers)
        return decision.allowed, decision.reason
    except httpx.HTTPError as e:
        metric_metering_lookup.labels(result="error").inc()
        logger.warning(
            json.dumps(
                {
                    "event": "metering_http_error",
                    "error": type(e).__name__,
                    "detail": str(e),
                    "tenant_id": tenant_id,
                    "fail_open": METERING_FAIL_OPEN,
                }
            )
        )
    except Exception:
        metric_metering_lookup.labels(result="contract_error").inc()
        logger.exception(
            "metering_http_contract_error",
            extra={
                "license_id": license_id,
                "tenant_id": tenant_id,
                "fail_open": METERING_FAIL_OPEN,
            },
        )
    if not METERING_FAIL_OPEN:
        return False, "metering_unavailable"
    # The billing check was the only enforcement for this decision and it
    # failed while the request is allowed anyway: count the ALLOW, not just the
    # transport/shape defect counted above.
    metric_budget_fail_open.labels(reason="metering_http_error").inc()
    return True, None


@app.get("/metrics", operation_id="get_metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", operation_id="health_check")
async def health():
    """Process-alive health check — deliberately dependency-free.

    Kept as a literal 200: it is an ext_authz-exempt path that existing callers
    treat as "the service answers", and the readiness signal now lives on
    /ready. Do not add dependency checks here.
    """
    return {"status": "healthy", "service": "license-authz"}


@app.get("/healthz", operation_id="healthz")
async def healthz():
    """Kubernetes-style health check (alias). Dependency-free, see /health."""
    return {"status": "healthy"}


async def _readiness_checks() -> list[readiness.CheckResult]:
    """Run every readiness check. See src/readiness.py for the gating rationale."""
    results = [readiness.check_lifespan(app.state)]
    jwks_result = await readiness.check_jwks(
        getattr(app.state, "jwks", None), name="jwks_portal", required=JWT_MODE_ENABLED
    )
    if jwks_result is not None:
        results.append(jwks_result)
    caller_result = await readiness.check_jwks(
        getattr(app.state, "caller_jwks", None),
        name="jwks_secure_apps",
        required=bool(_settings.keycloak_secure_apps_issuer),
    )
    if caller_result is not None:
        results.append(caller_result)
    key_result = readiness.check_internal_token_key(
        enabled=_settings.internal_token_issuer_enabled,
        jwks_document=internal_token.jwks_document,
    )
    if key_result is not None:
        results.append(key_result)
    # Resolve through the accessor, not app.state: production traffic is
    # currently 100% allowed_service_principal, and that branch returns before
    # any metering emit — so /ready is the ONLY thing touching Redis in prod
    # today, and therefore the only place a reconnect can be driven from.
    # check_redis is NON-gating unless READINESS_REQUIRE_REDIS is armed — which
    # is exactly the tension: this call is the reconnect driver, so gating on it
    # can evict the pod that was healing itself. See readiness.check_redis.
    results.append(
        await readiness.check_redis(await _get_redis(app), metering_enabled=METERING_ENABLED)
    )
    return results


@app.get("/ready", operation_id="readiness_check")
async def readiness_probe():
    """Kubernetes readiness probe — REAL dependency checks.

    503 when a GATING dependency is unusable on THIS pod (lifespan incomplete,
    JWKS never loaded, internal-token PEM unparseable); 200 with status
    "degraded" when a SHARED dependency is down — Redis, or a live JWKS/Keycloak
    outage on a pod that already holds key material. See src/readiness.py for
    why the latter must not flip the status code. The whole check is bounded by
    readiness.TOTAL_BUDGET, which sits below the probe's timeoutSeconds so a
    slow dependency yields a diagnosable 503 rather than a kubelet-side timeout
    with no body.
    """
    try:
        results = await asyncio.wait_for(_readiness_checks(), timeout=readiness.TOTAL_BUDGET)
    except asyncio.TimeoutError:
        return Response(
            content=json.dumps(
                {
                    "status": "not_ready",
                    "service": "license-authz",
                    "failed": ["timeout"],
                    "checks": {},
                    "detail": f"readiness checks exceeded {readiness.TOTAL_BUDGET}s",
                }
            ),
            status_code=503,
            media_type="application/json",
        )
    status_code, body = readiness.build_report(results)
    return Response(
        content=json.dumps(body),
        status_code=status_code,
        media_type="application/json",
    )


@app.get("/health/live", operation_id="liveness_check")
async def liveness():
    """Kubernetes liveness probe — MUST stay dependency-free.

    A liveness probe that checks dependencies causes a restart storm: when a
    shared dependency (Redis/Keycloak) blips, every replica fails liveness at
    once and kubelet restarts them all, turning a degraded dependency into a
    total outage. This answers exactly one question — is the process alive.
    """
    return {"status": "alive"}


# --------------------------------------------------------------------------
# Internal-token issuer (Lane F1). Both routes are registered before the
# /{path:path} ext_authz catch-all, so FastAPI serves them directly (bypassing
# the auth check) — exactly how /health is "public". /jwks is public; the
# private signing key is never served. /internal-token requires a verified
# Keycloak SA bearer (secure-apps realm) before minting a token.
# --------------------------------------------------------------------------
def _internal_token_error(reason: str, status_code: int) -> Response:
    metric_internal_token.labels(result="denied").inc()
    return Response(
        content=json.dumps({"error": reason}),
        status_code=status_code,
        media_type="application/json",
        headers={"x-error-reason": reason},
    )


@app.get("/jwks", operation_id="get_jwks")
async def jwks():
    """Public JWKS for the internal-token signing key (no auth; never private)."""
    try:
        return internal_token.jwks_document()
    except IssuerDisabled:
        return Response(
            content=json.dumps({"error": "issuer_disabled"}),
            status_code=404,
            media_type="application/json",
        )


@app.post("/internal-token", operation_id="issue_internal_token")
async def issue_internal_token(request: Request):
    """Mint a short-lived RS256 internal token for a verified Keycloak SA caller."""
    if not _settings.internal_token_issuer_enabled:
        return Response(
            content=json.dumps({"error": "issuer_disabled"}),
            status_code=404,
            media_type="application/json",
        )

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return _internal_token_error("missing_bearer", 401)
    caller_token = auth[7:].strip()

    caller_jwks = request.app.state.caller_jwks
    if caller_jwks is None:
        # secure-apps realm not configured → cannot verify the caller; fail closed.
        return _internal_token_error("caller_verify_unavailable", 401)
    try:
        claims = verify_jwt(
            caller_token,
            caller_jwks,
            issuer=_settings.keycloak_secure_apps_issuer,
        )
    except InvalidJWT:
        return _internal_token_error("invalid_caller_token", 401)

    # SA tokens identify the client via azp (authorized party); fall back to sub.
    sub = claims.raw.get("azp") or claims.sub
    if not sub:
        return _internal_token_error("caller_sub_missing", 401)
    tenant_id = claims.tenant_id if claims.tenant_id != "anonymous" else None

    token, ttl = internal_token.mint_token(sub=sub, tenant_id=tenant_id)
    metric_internal_token.labels(result="issued").inc()
    return {"access_token": token, "token_type": "Bearer", "expires_in": ttl}


def _jwt_error(reason: str, status_code: int) -> Response:
    """Uniform error response for JWT-mode failures."""
    metric_jwt_mode.labels(result=reason).inc()
    return Response(
        content=json.dumps({"error": reason}),
        status_code=status_code,
        media_type="application/json",
        headers={"x-error-reason": reason},
    )


@dataclass
class JwtModeOutcome:
    """Result of JWT-mode handling. Exactly one of result/error_response is set.

    `actor_sub` is the verified Keycloak JWT `sub` (the acting human), set only
    alongside `result`. It is a SEPARATE identity from `result.subject_id` (the
    licences Subject UUID / billable party) and must never be conflated with
    it — see X-Actor-Id at the response_headers call site.
    """

    result: IntrospectResult | None = None
    error_response: Response | None = None
    actor_sub: str | None = None
    # Provenance of the app key the licence was resolved by, carried OUT of the
    # helper because only the helper knows which branch ran.
    #
    # True  = ASSERTED: the caller's unsigned `x-app-key` header chose it.
    # False = DERIVED: the caller asserted nothing and the ROUTE REGISTRY named
    #         the key. The caller chose nothing, so it is NOT an assertion and
    #         must not be logged/counted as one — the asserted-key rule at the
    #         call site exists precisely to bound a caller's free choice, and a
    #         derived key made no choice to bound.
    #
    # Defaults False so every error exit (which resolves no licence at all)
    # reports no assertion rather than inheriting one.
    app_key_asserted: bool = False
    # Verified `azp` (client_id) of the JWT-mode bearer token itself — the SAME
    # credential `claims.tenant_id` came from, NOT the separate
    # X-Caller-Authorization identity (`caller_client_id` elsewhere in this
    # file). "" when the token carries no `azp` or on any error exit. This is
    # the only signal the trusted-tenant-proxy check at the response-header
    # build may use: it is read straight from a signature-verified claim, never
    # from a header the caller controls.
    azp: str = ""
    # Realm of the VERIFIED issuer of the JWT-mode bearer (realm-per-app P0:
    # the realm names the app). "" on every error exit. Emitted downstream as
    # `x-auth-realm`.
    issuer_realm: str = ""


async def _handle_jwt_mode(request: Request, token: str, path: str, method: str) -> JwtModeOutcome:
    """Verify the Keycloak JWT and resolve (tenant, app_key) → license.

    Returns JwtModeOutcome with either `result` (caller continues to budget/
    app-route/metering-emit path) or `error_response` (caller returns it).

    `path` is the request's raw Envoy `:path`, passed in rather than re-read
    from the request so there is exactly ONE expression in this file deciding
    what "the path" is (`ext_authz_check` reads it once). It is used only to
    derive an app key when the caller asserts none — see the derivation block,
    and to answer the self-read question at the `license_required` exit.

    `method` is threaded in the SAME way and for the same reason, and — like
    `_check_app_route`'s — it is REQUIRED, no default. It feeds an
    authorization predicate (`_is_self_read_route`), where a default of "GET"
    would let a future caller that forgets the argument have its POST judged as
    a GET.
    """
    jwks: JwksCache | None = request.app.state.jwks
    resolver: ActiveLicenseResolver | None = request.app.state.active_resolver
    if jwks is None or resolver is None:
        # Flag flipped at request time without a restart. Fail closed.
        return JwtModeOutcome(error_response=_jwt_error("jwt_mode_not_initialized", 503))

    # Realm-per-app P0: pick the allowlisted issuer's OWN key set and pin
    # `issuer=` to it. The unverified `iss` only SELECTS; `verify_jwt` then
    # proves it. A primary-issuer token takes exactly the pre-allowlist call.
    issuer = unverified_issuer(token) or jwt_mode.KEYCLOAK_ISSUER
    # secure-apps S2S tenant proxy (P7): verified against the secure-apps key
    # set with `issuer=` pinned to it; azp + role re-checked after verification.
    sa_issuer = _secure_apps_issuer()
    secure_apps_proxy = bool(sa_issuer) and issuer == sa_issuer
    if issuer == jwt_mode.KEYCLOAK_ISSUER:
        issuer_jwks: JwksCache | None = jwks
    elif secure_apps_proxy:
        issuer_jwks = (
            getattr(request.app.state, "caller_jwks", None)
            if SECURE_APPS_TENANT_PROXY_CLIENTS
            else None
        )
    else:
        registry: IssuerJwks | None = getattr(request.app.state, "issuer_jwks", None)
        issuer_jwks = registry.get(issuer) if registry is not None else None
    if issuer_jwks is None:
        # Not allowlisted (ext_authz_check only routes allowlisted issuers here,
        # so this is defence in depth) — fail closed as an unverifiable token.
        logger.warning(
            json.dumps(
                {
                    "event": "jwt_verify_failed",
                    "reason": "issuer_not_trusted",
                    "token_hash": hash_token(token),
                }
            )
        )
        return JwtModeOutcome(error_response=_jwt_error("invalid_token", 401))

    try:
        claims = verify_jwt(token, issuer_jwks, issuer=issuer)
    except InvalidJWT as e:
        logger.warning(
            json.dumps(
                {"event": "jwt_verify_failed", "reason": str(e), "token_hash": hash_token(token)}
            )
        )
        return JwtModeOutcome(error_response=_jwt_error("invalid_token", 401))

    if claims.tenant_id == "anonymous":
        return JwtModeOutcome(error_response=_jwt_error("anonymous", 401))

    # VERIFIED `azp` (authorized party) of this signature-checked token. Used
    # ONLY to test membership in `TRUSTED_TENANT_PROXY_CLIENTS` at the
    # response-header build — never forwarded, never trusted for anything a
    # header could forge. Not-a-string / absent -> "", same treatment as the
    # `caller_client_id` / service-principal `azp` reads elsewhere in this file.
    jwt_azp_claim = claims.raw.get("azp")
    jwt_azp = jwt_azp_claim if isinstance(jwt_azp_claim, str) else ""

    if secure_apps_proxy:
        refusal = _secure_apps_proxy_refusal(claims.raw)
        if refusal is not None:
            logger.warning(
                json.dumps(
                    {
                        "event": "secure_apps_tenant_proxy_refused",
                        "reason": refusal,
                        "azp": jwt_azp,
                        "token_hash": hash_token(token),
                    }
                )
            )
            return JwtModeOutcome(error_response=_jwt_error("invalid_token", 401))

    # ⚠️ `x-app-key` is an UNSIGNED, CALLER-SUPPLIED header, and it is the key this
    # resolver looks the licence up by — so the caller chooses WHICH of its
    # tenant's licences (and therefore whose quota and whose budget) the request
    # is charged against. Nothing HERE cross-checks it, and Envoy does not strip
    # it. The risk is mischarged/underenforced billing rather than cross-tenant
    # access, because `/licenses/active` is queried with the VERIFIED `tenant_id`
    # and so cannot return another tenant's licence.
    #
    # ⭐ PARTIALLY CLOSED, AND NOT HERE. The caller's choice is now bounded to the
    # licence the ROUTE names: `ext_authz_check` marks this path's key ASSERTED
    # and refuses it on any path the route registry does not cover
    # (`_asserted_key_route_allowed`; flag ASSERTED_APP_KEY_ALLOW_UNREGISTERED).
    # The enforcement lives at that call site because it is where `path`, the
    # app-route counter, the audit log and the metering emit already are.
    #
    # ⛔ WHAT IS STILL OPEN, so no later reader stops looking: on a REGISTERED
    # prefix the caller can still assert the key that prefix expects — which is
    # the correct licence for that route, so it is bounded, but it is bounded by
    # the registry rather than by the credential. Closing it completely needs an
    # app-naming claim in the token, and ⛔ NO CLAIM IN EITHER REALM NAMES AN APP
    # today (same measurement recorded at jwt_mode.TENANT_CLAIM). Do not invent a
    # claim to match against.
    app_key = request.headers.get("x-app-key", "").strip()
    app_key_asserted = bool(app_key)
    if not app_key:
        # --- DERIVE the key from the route, instead of 400 ------------------
        #
        # WHY THIS IS NOT A WEAKENING. The 400 above refused a request that had
        # made NO claim about which app it was. Where the ROUTE ITSELF names the
        # app — i.e. the path matches a registered prefix — there is nothing for
        # the caller to claim: the registry, not the caller, supplies the key,
        # and `_check_app_route` downstream would have demanded exactly this
        # value anyway. Deriving it therefore admits precisely the requests the
        # registry already binds, and grants no key the caller could not have
        # asserted for that same path.
        #
        # THREE PROPERTIES THAT MUST SURVIVE ANY EDIT HERE:
        #  1. Derivation runs ONLY when the caller asserted nothing. An asserted
        #     key is never overwritten by the registry's — it still flows to
        #     `_check_app_route`, which still 403s `app_route_mismatch` when it
        #     contradicts the route. A caller can never launder a wrong key into
        #     a right one by way of this branch.
        #  2. An UNREGISTERED path derives nothing — `_registered_app_key`
        #     returns None — and the original 400 still stands. Nothing here
        #     invents a default, a wildcard or an empty-string key.
        #  3. The derived key is NOT asserted (see JwtModeOutcome), so audit and
        #     metrics keep saying the caller chose nothing.
        #
        # The lookup is `_registered_app_key` — the SAME function the enforcing
        # predicate uses, deliberately not a second copy, so the key that is
        # derived and the key that is demanded can never drift apart.
        derived_key = _registered_app_key(path)
        if not derived_key:
            return JwtModeOutcome(error_response=_jwt_error("missing_app_key", 400))
        app_key = derived_key
        logger.info(
            json.dumps(
                {
                    "event": "jwt_mode_app_key_derived_from_route",
                    "app_key": app_key,
                    "path": path,
                    "tenant_id": claims.tenant_id,
                }
            )
        )

    sa: ServiceTokenProvider = request.app.state.service_auth
    auth_headers = await sa.auth_header(request.app.state.http_client)

    # Realm of the VERIFIED issuer (`verify_jwt` pinned it). Forwarded to
    # licenses ONLY for a non-primary realm, so a primary-issuer request is
    # byte-identical to before; licenses owns `realm_owner` and enforces it.
    verified_realm = jwt_mode.realm_of_issuer(issuer)
    resolve_kwargs: dict[str, str] = {}
    # `realm` names the END USER's realm (licenses migration 072). A secure-apps
    # S2S proxy is not an end user — resolve it exactly as its portal twin was.
    if issuer != jwt_mode.KEYCLOAK_ISSUER and verified_realm and not secure_apps_proxy:
        resolve_kwargs["realm"] = verified_realm

    try:
        lic_data = await resolver.resolve(
            request.app.state.http_client,
            tenant_id=claims.tenant_id,
            app_key=app_key,
            auth_headers=auth_headers,
            **resolve_kwargs,
        )
    except httpx.HTTPError:
        return JwtModeOutcome(error_response=_jwt_error("licenses_unavailable", 503))

    if lic_data is None:
        # --- SELF-READ EXEMPTION, the SAME one the opaque path already honours -
        #
        # THE DEFECT THIS CLOSES (measured in prod 2026-09-16T09:01:26Z):
        # `w2v-backend` on mini1/mini2 stopped presenting an opaque licence
        # token and started presenting a client-credentials JWT, which moved it
        # onto THIS path. It sends `X-Tenant-Id` and no `x-app-key`, so the
        # derivation block above resolves the key from the ROUTE:
        # `APP_ROUTE_REGISTRY["/licenses/"] == "licenses"`. The gate then asked
        # "does this tenant hold the LICENSES app?" for a caller that was only
        # reading its OWN licence — 402 `license_required` -> the caller's
        # entitlement state went `unknown` -> narration 503'd.
        #
        # The opaque path never asks that question on these two paths, and the
        # reason is in `_check_app_route`: a GET/HEAD of the caller's own
        # licence is INTROSPECTION OF A CREDENTIAL ALREADY HELD, not
        # consumption of the `licenses` app. This arm applies that identical
        # policy here — same tuple, same methods, same revert switch, via the
        # same `_is_self_read_route` the opaque predicate calls. No second
        # exemption list exists, and neither `APP_ROUTE_SELF_READ` nor the
        # flag's default is touched.
        #
        # ⚠️ WHY NOT `_self_read_exemption_applies` VERBATIM. It is the right
        # policy wrapped in two narrowings that are FALSE BY CONSTRUCTION here,
        # so calling it would have been inert wiring that shipped green: it
        # demands a NON-EMPTY key that DIFFERS from the route's expected key, and
        # on this arm the key is either absent (derived, hence == expected) or
        # asserted-and-already-checked. Those narrowings exist to keep the
        # `allowed_self_read_exempt` app-route metric honest on the opaque path
        # — they are not the policy. `_is_self_read_route` IS the policy.
        #
        # THREE BOUNDS, each load-bearing:
        #  1. `not app_key_asserted` — this rescues ONLY the derived-key case,
        #     i.e. exactly the request that made no claim and was then charged
        #     against an app the route named for it. A caller that ASSERTS
        #     `x-app-key` still has to hold the licence it asserted; if it holds
        #     one, it resolves and meets the opaque-path exemption downstream at
        #     `_check_app_route` (already pinned by
        #     `test_jwt_mode_w2v_key_on_licenses_prefix_still_403`).
        #  2. `_is_self_read_route` — GET/HEAD only, and only the two paths in
        #     `APP_ROUTE_SELF_READ`, matched on `_normalize_route_path`'s output
        #     so a doubled slash or dot segment cannot walk in. Every other
        #     `/licenses/` path, and every mutating method on these two, still
        #     takes the 402 below unchanged.
        #  3. `_self_read_raw_path_stays_in_licenses` — the RAW `:path` must ALSO
        #     be under `/licenses/`, not merely its normalised form. Bound 2
        #     alone lets `/vllm/../licenses/active` in, and a router matching
        #     `uri.prefix: /vllm/` on the RAW string would send that to vllm; we
        #     would be allowing an unlicensed caller on a request that never
        #     reaches the licences app. Whether Istio normalises before
        #     ext_authz is UNVERIFIED (no `pathNormalization` in the
        #     `istio-system/istio` ConfigMap), so this does not depend on it.
        #  4. the flag — `LICENSE_SELF_READ_EXEMPT_ENABLED=false` restores the
        #     402 byte-for-byte, the same revert switch the opaque path has.
        #
        # ⚠️ BOUND 3 IS DELIBERATELY NOT IN `_is_self_read_route`. That helper is
        # SHARED with `_self_read_exemption_applies`, i.e. with the OPAQUE arm
        # that is live in production; tightening it there would change opaque
        # behaviour as a side effect of a JWT fix. The same residual does exist
        # on the opaque arm (it predates this branch) — it is reported, not
        # silently altered here.
        #
        # The licence it allows with is EMPTY, not fabricated: no license_id,
        # no subject, no plan, no app_id. Budget enforcement therefore skips on
        # `no_license_id` (counted), metering emits no licence, and the upstream
        # receives empty `x-license-id`/`x-subject-id`. The only thing asserted
        # is the VERIFIED `claims.tenant_id`, which is what `/licenses/active`
        # is scoped by anyway — so this cannot read another tenant's state.
        if (
            not app_key_asserted
            and LICENSE_SELF_READ_EXEMPT_ENABLED
            and _is_self_read_route(method, path)
            and _self_read_raw_path_stays_in_licenses(path)
        ):
            # Counted under its OWN label, never folded into `allowed`: an
            # exemption nobody can prove is alive cannot justify itself, and
            # this one is the difference between narration working and 503.
            metric_jwt_mode.labels(result="allowed_self_read_exempt").inc()
            metric_jwt_mode_by_tenant.labels(
                result="allowed_self_read_exempt",
                tenant=claims.tenant_id,
                app=app_key,
            ).inc()
            logger.info(
                json.dumps(
                    {
                        "event": "jwt_mode_self_read_exempt",
                        "app_key": app_key,
                        "method": method,
                        "path": path,
                        "tenant_id": claims.tenant_id,
                    }
                )
            )
            return JwtModeOutcome(
                result=IntrospectResult(
                    active=True,
                    tenant_id=claims.tenant_id,
                    app_key=app_key,
                    reason="self_read_exempt",
                ),
                actor_sub=claims.sub,
                # Still False — the key was DERIVED. The exemption changes the
                # verdict, never the provenance.
                app_key_asserted=app_key_asserted,
                azp=jwt_azp,
                issuer_realm=verified_realm,
            )
        metric_jwt_mode.labels(result="license_required").inc()
        metric_jwt_mode_by_tenant.labels(
            result="license_required",
            tenant=claims.tenant_id,
            app=app_key,
        ).inc()
        return JwtModeOutcome(
            error_response=Response(
                content=json.dumps(
                    {
                        "error": "license_required",
                        "tenant_id": claims.tenant_id,
                        "app_key": app_key,
                    }
                ),
                status_code=402,
                media_type="application/json",
                headers={"x-error-reason": "license_required"},
            )
        )

    metric_jwt_mode.labels(result="allowed").inc()
    metric_jwt_mode_by_tenant.labels(
        result="allowed",
        tenant=claims.tenant_id,
        app=app_key,
    ).inc()
    return JwtModeOutcome(
        result=_introspect_result_from_dict(lic_data),
        # Use `sub`: the stable opaque Keycloak subject identifier. Do NOT use
        # `preferred_username` or `email` (PII; never log/emit email) — `sub`
        # is what claims.raw carries as the durable per-human key.
        actor_sub=claims.sub,
        app_key_asserted=app_key_asserted,
        azp=jwt_azp,
        issuer_realm=verified_realm,
    )


def _extract_caller_identity(
    request: Request,
) -> tuple[str, list[str]] | Response:
    """Verify X-Caller-Authorization (secure-apps SA JWT) and extract identity.

    Returns (caller_client_id, caller_roles) on success, or empty values when
    the header is absent (opt-in semantics during M1–M7 rollout).

    Returns a 401 Response when the header is present but the token is invalid
    (bad signature / wrong issuer / expired) — caller sent a token, fail loud.
    """
    caller_auth = request.headers.get("x-caller-authorization", "")
    if not caller_auth:
        return ("", [])
    if not caller_auth.lower().startswith("bearer "):
        return Response(
            content='{"error": "unauthorized", "reason": "caller_token_invalid"}',
            status_code=401,
            media_type="application/json",
            headers={"x-error-reason": "caller_token_invalid"},
        )
    caller_jwks = request.app.state.caller_jwks
    if caller_jwks is None:
        # Header set but service has no secure-apps JWKS configured — refuse
        # rather than silently accept; misconfigured boundary is a security
        # issue, not a passthrough.
        return Response(
            content='{"error": "unauthorized", "reason": "caller_identity_not_configured"}',
            status_code=401,
            media_type="application/json",
            headers={"x-error-reason": "caller_identity_not_configured"},
        )
    token = caller_auth[7:]
    try:
        claims = verify_jwt(
            token,
            caller_jwks,
            issuer=_settings.keycloak_secure_apps_issuer,
            audience=None,
        )
    except InvalidJWT:
        return Response(
            content='{"error": "unauthorized", "reason": "caller_token_invalid"}',
            status_code=401,
            media_type="application/json",
            headers={"x-error-reason": "caller_token_invalid"},
        )
    raw = claims.raw
    azp = raw.get("azp", "")
    caller_client_id = azp if isinstance(azp, str) else ""
    realm_access = raw.get("realm_access") or {}
    roles_raw = realm_access.get("roles") if isinstance(realm_access, dict) else None
    if isinstance(roles_raw, list):
        caller_roles = [r for r in roles_raw if isinstance(r, str) and r]
    else:
        caller_roles = []
    return (caller_client_id, caller_roles)


async def ext_authz_check(request: Request) -> Response:
    """
    Main ext_authz endpoint.

    Envoy sends the original request headers and expects:
    - HTTP 200 = allow (with headers to inject)
    - HTTP 403 = deny (with error details)

    Headers to inject are returned as regular headers.
    Envoy's ExtensionProvider config (headersToUpstreamOnAllow) determines
    which headers are actually forwarded to the upstream service.
    """
    start_time = time.time()

    # Extract request metadata
    method = request.headers.get(":method", request.method)
    path = request.headers.get(":path", str(request.url.path))
    # No longer discarded: marketing and sales are HOST-routed and share the same
    # `/api/...` prefix, so the authority is the ONLY thing that can tell their
    # two waypoints apart. Consumed by the host->app binding below, which is
    # INERT unless APP_HOST_ROUTE_MODE is set (default `off`).
    authority = request.headers.get(":authority", request.headers.get("host", ""))
    source_ip = request.headers.get(
        "x-forwarded-for", request.client.host if request.client else "unknown"
    )
    user_agent = request.headers.get("user-agent", "unknown")

    # Extract authorization header
    auth_header = request.headers.get("authorization", "")

    # Caller-identity (opt-in during rollout): verify X-Caller-Authorization
    # secure-apps SA JWT and extract (azp, realm_access.roles). Absent header
    # = empty injected values. Present-but-invalid = 401 fail-loud.
    caller_result = _extract_caller_identity(request)
    if isinstance(caller_result, Response):
        # `_extract_caller_identity` has three 401 exits — a non-bearer
        # `X-Caller-Authorization`, no secure-apps JWKS configured, and a token
        # that fails signature/issuer/expiry. NONE of them wrote a `log_access`
        # row, emitted a `logger.warning`, or touched a Counter, so a FORGED
        # service-account JWT presented at the authz boundary was answered with
        # a bare 401 and left no trace anywhere in this service. That is the
        # same audit blindness as the two unsafe-header denies this branch
        # fixes, at a strictly more interesting exit.
        #
        # Audited HERE rather than inside the helper, deliberately: this one
        # site covers all three exits and any fourth one added later, and it is
        # where `method` / `path` / `source_ip` / `user_agent` already exist.
        # A helper-side emit would need all four threaded in and would have to
        # be repeated per exit — three chances to forget, and the next author
        # gets a fourth.
        log_access(
            # Nothing has been introspected at this point in the request: the
            # caller token is a SEPARATE credential from the licence token and
            # failing it aborts before any licence lookup. `None`, not "" —
            # the rule the rest of this function follows.
            license_id=None,
            tenant_id=None,
            app_id=None,
            plan_id=None,
            method=method,
            path=path,
            status="denied_caller_identity",
            # `token_hash` is non-optional and names the LICENCE token, which
            # this exit never reaches. Sentinel rather than a hash of the
            # caller token: hashing a credential that just failed verification
            # would put a correlatable-looking value in a field whose whole
            # meaning is "the licence token was this one".
            token_hash="caller_token",
            source_ip=source_ip,
            user_agent=user_agent,
            # Read back off the response the helper built. Every one of its
            # exits sets `x-error-reason`, and each value is a string literal
            # in this file from a closed set of two
            # (`caller_token_invalid`, `caller_identity_not_configured`) — so
            # this carries no caller-supplied bytes. The default exists so a
            # future exit that forgets the header degrades to an unlabelled
            # row rather than to no row.
            reason=f"caller_identity={caller_result.headers.get('x-error-reason', 'unknown')}",
        )
        return caller_result
    caller_client_id, caller_roles = caller_result

    if not auth_header.lower().startswith("bearer "):
        # Service-principal path (opt-in): a *verified* secure-apps SA caller token
        # authorizes the hop as a trusted first-party service even without a primary
        # license token. `caller_client_id` is non-empty ONLY when
        # _extract_caller_identity verified the SA token (signature + secure-apps
        # issuer + exp) above; an absent header yields ("",[]) and a present-but-
        # invalid one already returned 401 — so a truthy value here is proof of a
        # legitimate caller. Identity-only: inject the caller overlay with empty
        # tenant/license headers; downstream services enforce their own tenant scope.
        if SERVICE_PRINCIPAL_AUTH_ENABLED and caller_client_id:
            if strict_app := _strict_app_for(request):
                # A strict host demands a LICENCE for one app; a service
                # principal carries none. Fall through to the standard 401.
                metric_service_principal.labels(
                    result="denied_strict_host", caller=caller_client_id
                ).inc()
                log_access(
                    license_id=None,
                    tenant_id=None,
                    app_id=None,
                    plan_id=None,
                    method=method,
                    path=path,
                    status="denied_service_principal",
                    token_hash="service_principal",
                    source_ip=source_ip,
                    user_agent=user_agent,
                    reason=f"strict_host={strict_app[0]}",
                )
            elif (
                SERVICE_PRINCIPAL_ALLOWED_CLIENTS
                and caller_client_id not in SERVICE_PRINCIPAL_ALLOWED_CLIENTS
            ):
                # Verified secure-apps SA token, but its client-id is NOT on the
                # configured allowlist — a legitimate SA that is not a sanctioned
                # service principal. Refuse the identity-only allow: count it as a
                # security signal and fall through to the standard 401 below.
                metric_service_principal.labels(
                    result="denied_not_allowlisted", caller=caller_client_id
                ).inc()
                log_access(
                    license_id=None,
                    tenant_id=None,
                    app_id=None,
                    plan_id=None,
                    method=method,
                    path=path,
                    status="denied_service_principal",
                    token_hash="service_principal",
                    source_ip=source_ip,
                    user_agent=user_agent,
                    reason=f"caller_not_allowlisted={caller_client_id}",
                )
                logger.warning(
                    json.dumps(
                        {
                            "event": "authz_service_principal_denied",
                            "caller_client_id": caller_client_id,
                            "reason": "not_allowlisted",
                        }
                    )
                )
            else:
                sp_headers = {
                    "x-tenant-id": "",
                    "x-tenant-name": "",
                    "x-license-id": "",
                    "x-subject-id": "",
                    "x-app-id": "",
                    "x-plan-id": "",
                    "x-plan-code": "",
                    "x-app-key": "",
                    "x-caller-client-id": caller_client_id,
                    "x-caller-roles": ",".join(caller_roles),
                }
                for hdr_name, hdr_val in sp_headers.items():
                    if not hdr_val:
                        continue
                    regex = _SAFE_LIST_HEADER_RE if hdr_name in _LIST_HEADERS else _SAFE_HEADER_RE
                    if not regex.match(hdr_val):
                        # This deny used to leave NO audit row. `log_access` is
                        # the only structured audit emitter in this service, so
                        # a header-injection attempt against the authz boundary
                        # — precisely the event this regex exists to catch —
                        # was invisible to every query keyed on `status`. It
                        # fails CLOSED, so this was never a bypass; it was
                        # audit blindness.
                        #
                        # ⛔ NEVER put `hdr_val` in this row. `hdr_name` is a
                        # literal key from the `sp_headers` dict above, so it is
                        # safe; the VALUE is not, and the only two entries that
                        # can ever reach here non-empty are `x-caller-client-id`
                        # (== `caller_client_id`) and `x-caller-roles` — every
                        # other entry is a literal "" and is skipped by the
                        # `continue` above. So the one way the rejected value
                        # can enter this row is through `caller_client_id` in
                        # `reason`, and `_elide_unsafe` closes it.
                        log_access(
                            # None, not "" — this path never introspects, so
                            # there is genuinely no licence, tenant, app or
                            # plan. Same rule as the allow emit further down:
                            # wrong-but-complete is strictly worse than missing.
                            license_id=None,
                            tenant_id=None,
                            app_id=None,
                            plan_id=None,
                            method=method,
                            path=path,
                            status="denied_service_principal_unsafe_header",
                            # `token_hash` is non-optional and there is no
                            # licence token on this path to hash. Every other
                            # log_access on the service-principal path already
                            # uses this sentinel; a fabricated hash would be a
                            # false identity, and "" would read as "hashed to
                            # nothing".
                            token_hash="service_principal",
                            source_ip=source_ip,
                            user_agent=user_agent,
                            # `caller_client_id` is the VERIFIED secure-apps
                            # `azp` and the only true identity this path holds,
                            # so it is recorded — EXCEPT when it is itself the
                            # value that failed the regex, in which case
                            # `unsafe_header=x-caller-client-id` already tells
                            # the reader which identity was refused and naming
                            # it as the caller would assert an identity this
                            # service just declined to propagate. The
                            # pre-existing WARNING below still carries the raw
                            # azp for the alerting stack; nothing is lost.
                            reason=(
                                f"caller={_elide_unsafe(caller_client_id, hdr_val)},"
                                f"unsafe_header={hdr_name}"
                            ),
                        )
                        # Kept, not folded into the audit row: this is WARNING
                        # (what the alerting stack watches) while log_access is
                        # INFO (the audit trail). Deleting it to avoid a
                        # duplicate would remove an existing signal.
                        logger.warning(
                            json.dumps(
                                {
                                    "event": "unsafe_header_value",
                                    "header": hdr_name,
                                    "caller_client_id": caller_client_id,
                                }
                            )
                        )
                        return Response(
                            content='{"error": "unauthorized", "reason": "unsafe_header_value"}',
                            status_code=401,
                            media_type="application/json",
                            headers={"x-error-reason": "unsafe_header_value"},
                        )
                # --- Budget enforcement on the service-principal fast path ----
                # This branch is the audit finding: 100% of 24h traffic returned
                # 200 from HERE, before any budget check and with no metering
                # event. Both halves of that are now addressed, but NOT
                # symmetrically — read SERVICE_PRINCIPAL_BUDGET_ENFORCED's
                # comment for the measurement that forced the asymmetry.
                sp_caller = _sp_caller_label(caller_client_id)
                # ⭐ IDENTITY RESOLUTION hoisted ABOVE the budget branch (it used
                # to sit below, right before the metering emit — see the
                # metering-emit comment further down for the audit finding this
                # resolution serves). It has to run first now: the budget branch
                # needs `sp_tenant_id` to key the tenant-scoped budget check
                # on anything at all. Nothing observable sits between the old
                # position and the new one except the budget branch itself, so
                # the one thing that changes is that a request an armed
                # `SERVICE_PRINCIPAL_BUDGET_ENFORCED` denies now also resolves
                # identity and counts `metric_service_principal_identity` before
                # returning 402 — previously it returned before reaching this
                # code at all. That is a genuinely new HTTP call to licences on
                # the deny path, not just a metric; see the report for why that
                # is accepted rather than deferred. Never a fabricated identity:
                # an unresolved azp still resolves to tenant_id="", byte-for-byte
                # what this path produced before resolution existed.
                sp_tenant_id = await _resolve_service_principal_tenant(request, caller_client_id)
                sp_identity_result = "resolved" if sp_tenant_id else "unresolved"
                metric_service_principal_identity.labels(
                    client_id=_sp_identity_caller_label(caller_client_id),
                    result=sp_identity_result,
                ).inc()
                if not SERVICE_PRINCIPAL_BUDGET_ENFORCED:
                    # DEFAULT, and the live production state. Behaviour is
                    # byte-for-byte what it was: allow, no budget read, no
                    # metering. The only thing that changed is that the allow is
                    # now COUNTED and ATTRIBUTED, so "1,487 allows" stops being
                    # indistinguishable from "1,487 enforced passes".
                    metric_service_principal_budget.labels(
                        result="not_enforced", caller=sp_caller
                    ).inc()
                else:
                    # Token budget only — this path has NO licence, so there is
                    # no request quota to read. Tenant-scoped billing rows
                    # (license_id null / nil UUID) bind it, decided over HTTP
                    # like the licence-token path. `sp_tenant_id` is never
                    # fabricated (see `_resolve_service_principal_tenant`): an
                    # unresolved caller has nothing to key on, fails open, and is
                    # counted on the same `budget_fail_open{reason="no_license_id"}`
                    # series this path has always produced for it.
                    sp_reason: str | None
                    if not sp_tenant_id:
                        metric_budget_fail_open.labels(reason="no_license_id").inc()
                        sp_allowed, sp_reason = True, None
                    else:
                        metric_metering_lookup.labels(result="http").inc()
                        sp_allowed, sp_reason = await _metering_http_verdict(
                            request,
                            lambda mc, auth: mc.check_budget(
                                request.app.state.http_client,
                                tenant_id=sp_tenant_id,
                                license_id=None,
                                auth_headers=auth,
                            ),
                            license_id=None,
                            tenant_id=sp_tenant_id,
                        )
                    if not sp_allowed:
                        metric_service_principal_budget.labels(
                            result="denied", caller=sp_caller
                        ).inc()
                        # Deliberately NOT counted on `metric_budget_exceeded`:
                        # that counter is labelled by tenant and app, and this
                        # path has neither, so every increment would have to put a
                        # non-tenant string into a per-tenant series. The
                        # caller-attributed counter above is the instrument for
                        # this path.
                        sp_kind = "quota" if sp_reason == "request_quota_exhausted" else "budget"
                        log_access(
                            license_id=None,
                            tenant_id=None,
                            app_id=None,
                            plan_id=None,
                            method=method,
                            path=path,
                            status="denied_service_principal_budget",
                            token_hash="service_principal",
                            source_ip=source_ip,
                            user_agent=user_agent,
                            reason=f"caller={caller_client_id},{sp_reason}",
                        )
                        return Response(
                            content=json.dumps(
                                {
                                    "error": "budget_exceeded",
                                    "kind": sp_kind,
                                    "reason": sp_reason,
                                }
                            ),
                            status_code=402,
                            media_type="application/json",
                            headers={"x-error-reason": sp_reason or "budget_exceeded"},
                        )
                    metric_service_principal_budget.labels(result="allowed", caller=sp_caller).inc()
                # --- Metering on the service-principal path -------------------
                # THE AUDIT FINDING: 1,474 requests from `marketing-service-account`
                # and 17 from `platform-console-backend` over 24h returned 200 from
                # HERE and produced NO metering event. That traffic was not
                # "unattributed" — it was ABSENT from the measurement entirely.
                #
                # ⚠️ THIS REVERSES A PREVIOUS ROUND'S EXPLICIT DECISION not to emit,
                # so the reason that decision fell is recorded rather than deleted.
                # It rested on two claims:
                #   1. "a FABRICATED tenant writes false identity into billing's
                #      attribution" — STILL TRUE, and still honoured. Nothing is
                #      fabricated below: tenant_id, license_id, subject_id, app_id,
                #      plan_id and plan_code are all genuinely absent on this path
                #      and are all passed empty. `caller_client_id` carries the one
                #      identity that IS true — the VERIFIED secure-apps `azp`.
                #   2. "an HONEST tenantless row is dropped downstream and fires
                #      MeteringEventsDroppedNoTenant, so it would have to be
                #      silenced" — REFUTED BY THIS FILE. `ext_authz_check` ALREADY
                #      emits tenantless rows: the denied_missing_token emit passes
                #      a literal `tenant_id=""` unconditionally, and the
                #      denied_invalid_token emit passes `result.tenant_id or ""`,
                #      which is empty for exactly the invalid tokens that reach
                #      it. A tenantless row is
                #      therefore not a new class of event that this arm would
                #      introduce; whatever the consumer does with one, it already
                #      does it, and that behaviour is not a reason to leave a
                #      billable ALLOW with no record anywhere.
                #
                # ⛔ WHAT THIS STILL DOES NOT BUY, so nobody reads it as finished:
                # the event is ATTRIBUTABLE (to a caller) but not BILLABLE (to a
                # tenant/licence) UNLESS `_resolve_service_principal_tenant`
                # resolved a real tenant. Making every SP request billable needs a
                # licence identity for service principals, which is broader than
                # tenant resolution and is not here. See SERVICE_PRINCIPAL_BUDGET_ENFORCED.
                #
                # ⭐ IDENTITY RESOLUTION: ask licences whether this VERIFIED azp
                # names a tenant it already knows about — see
                # SERVICE_PRINCIPAL_TENANT_NAME_MAP for which candidate name each
                # known azp maps to, and why all 5 are currently absent from
                # licences' list (no `svc-*` registry row yet — not invented).
                # `sp_tenant_id` and `sp_identity_result` were resolved ABOVE the
                # budget branch now (the budget check needs the tenant id too) —
                # not recomputed here, so this is still exactly one resolution
                # and one `metric_service_principal_identity` increment per
                # request, same labels as before the hoist.
                await _emit_metering_event(
                    request,
                    # `sp_tenant_id` is either a licences-confirmed tenant id or ""
                    # — never fabricated. license_id/subject_id/app_id/plan_id/
                    # plan_code stay None: this branch never introspects a
                    # licence, so there genuinely is none, resolved tenant or not.
                    # ⛔ Do not "improve" any of these by defaulting to the caller
                    # id, a nil UUID or a literal: a false identity is
                    # unrecoverable once billing aggregates it, which is exactly
                    # why the `tenant_id or subject_id` fallbacks elsewhere in
                    # this file were removed.
                    tenant_id=sp_tenant_id,
                    license_id=None,
                    subject_id=None,
                    app_id=None,
                    plan_id=None,
                    plan_code=None,
                    method=method,
                    path=path,
                    status="allowed_service_principal",
                    source_ip=source_ip,
                    caller_client_id=caller_client_id,
                )
                metric_service_principal.labels(result="allowed", caller=caller_client_id).inc()
                log_access(
                    license_id=None,
                    tenant_id=sp_tenant_id or None,
                    app_id=None,
                    plan_id=None,
                    method=method,
                    path=path,
                    status="allowed_service_principal",
                    token_hash="service_principal",
                    source_ip=source_ip,
                    user_agent=user_agent,
                    # UNCHANGED format on purpose: downstream log parsers may key
                    # on `caller=<azp>`. The identity result goes on the
                    # structured event below instead, and only when resolution
                    # is configured.
                    reason=f"caller={caller_client_id}",
                )
                sp_event: dict[str, object] = {
                    "event": "authz_service_principal",
                    "caller_client_id": caller_client_id,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                }
                if LICENSES_INTERNAL_TENANTS_URL:
                    sp_event["identity"] = sp_identity_result
                logger.info(json.dumps(sp_event))
                return Response(
                    content='{"status": "ok"}',
                    status_code=200,
                    media_type="application/json",
                    headers=sp_headers,
                )

        log_access(
            license_id=None,
            tenant_id=None,
            app_id=None,
            plan_id=None,
            method=method,
            path=path,
            status="denied_missing_token",
            token_hash="no_token",
            source_ip=source_ip,
            user_agent=user_agent,
            reason="missing_bearer_token",
        )
        await _emit_metering_event(
            request,
            tenant_id="",
            license_id=None,
            subject_id=None,
            app_id=None,
            plan_id=None,
            method=method,
            path=path,
            status="denied_missing_token",
            source_ip=source_ip,
        )
        return Response(
            content='{"error": "unauthorized", "reason": "missing_bearer_token"}',
            status_code=401,
            media_type="application/json",
            headers={
                "x-error-reason": "missing_bearer_token",
            },
        )

    token = auth_header[7:]  # Remove "Bearer " prefix
    token_hash = hash_token(token)

    # Verified JWT `sub` (acting human), JWT-mode only. Stays None for
    # license-token mode and service-principal mode — neither has a real
    # verified end-user identity to vouch for, and this must never be
    # defaulted or inherited from anything else (see X-Actor-Id below).
    actor_sub: str | None = None
    # Verified issuer realm of a JWT-mode bearer; "" on every other path.
    auth_realm = ""

    # Verified `azp` of the bearer token itself, JWT-mode only. "" for the
    # opaque licence-token arm: `IntrospectResult` / the licenses service's
    # introspection response carry no client-id field (RFC 7662's optional
    # `client_id` is not modeled here), so an opaque caller has no verified
    # identity this service could check against a proxy allowlist — it stays
    # "" and therefore never matches `TRUSTED_TENANT_PROXY_CLIENTS` below,
    # which keeps today's overwrite behavior for that arm exactly as it is.
    jwt_azp = ""

    # Provenance of `result.app_key`, and therefore of the licence selection.
    # False = ATTESTED: `/license/introspect` returned the key as part of the
    # credential's own contents, so the caller chose nothing. True = ASSERTED:
    # the caller's unsigned `x-app-key` header is what `/licenses/active` was
    # queried by, so the caller picked which licence funds the call.
    #
    # ⚠️ This is set from WHICH BRANCH RAN, not from the value of any field. The
    # resolved licence usually echoes the asserted key back, so `result.app_key`
    # looks identical in both modes and cannot be used to tell them apart after
    # the fact — the provenance has to be captured here or it is lost.
    app_key_asserted = False

    # JWT-mode: Keycloak JWT from oauth2-proxy browser hosts. Resolve license
    # by (tenant_id, app_key) instead of opaque token lookup.
    if JWT_MODE_ENABLED and (is_jwt_bearer(token) or _is_secure_apps_tenant_proxy_bearer(token)):
        outcome = await _handle_jwt_mode(request, token, path, method)
        if outcome.error_response is not None:
            # `_handle_jwt_mode` has six error exits (jwt_mode_not_initialized,
            # invalid_token, anonymous, missing_app_key, licenses_unavailable,
            # license_required) and none writes a `log_access` row. They are
            # less blind than the caller-identity exit above — `_jwt_error`
            # increments `metric_jwt_mode`, and the 402 increments it too — but
            # a Counter is not the audit trail: it carries no path, no
            # source_ip, no token_hash, and cannot be joined to a request.
            #
            # Audited at this ONE call site for the same reason as the caller
            # -identity exit: it covers all six exits plus any added later, and
            # `token_hash` / `method` / `path` are already in scope here while
            # inside the helper they are not.
            log_access(
                # JWT mode resolves the licence INSIDE the helper, and every
                # error exit is a failure to resolve one — so on this path
                # there is genuinely no licence, tenant, app or plan to name.
                # The helper does hold a verified `claims.tenant_id` at four of
                # the six exits, but it is not on `JwtModeOutcome` and putting
                # it there is a signature change for a field this row does not
                # need. `None`, not a guess.
                license_id=None,
                tenant_id=None,
                app_id=None,
                plan_id=None,
                method=method,
                path=path,
                status="denied_jwt_mode",
                # Real, unlike the two paths above: the bearer token exists and
                # was hashed at the top of this block.
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                # Closed set of six string literals, all written in this file
                # (`_jwt_error`'s argument, plus the 402's explicit header).
                # No caller-supplied bytes.
                reason=f"jwt_mode={outcome.error_response.headers.get('x-error-reason', 'unknown')}",
            )
            return outcome.error_response
        # outcome.result is guaranteed non-None when error_response is None
        assert outcome.result is not None
        result = outcome.result
        actor_sub = outcome.actor_sub
        jwt_azp = outcome.azp
        auth_realm = outcome.issuer_realm
        # NOT unconditionally True any more. The JWT path can now DERIVE the key
        # from the route registry when the caller asserted none, and a derived
        # key is not an assertion — the helper is the only place that knows
        # which branch ran, so the flag is carried out of it rather than
        # re-guessed here.
        app_key_asserted = outcome.app_key_asserted
        actor_raw: str | None = actor_sub
        actor_reason = "jwt_no_sub"
        licence_token_mode = False
    else:
        # SECURITY: Always introspect - no passthrough, fail-closed
        result = await introspect_license(request.app.state.http_client, token)
        # Licence-token mode: the actor is the licence subject's Keycloak `sub`.
        # Only meaningful once the token is active; the inactive emit below
        # does not take it (an inactive answer carries no subject anyway).
        # `subject_actor_sub` covers service subjects whose keycloak_sub is a label
        # (`svc-kag`); `subject_keycloak_sub` keeps an older licenses build working.
        actor_raw = result.subject_actor_sub or result.subject_keycloak_sub
        actor_reason = "licence_no_subject_sub"
        licence_token_mode = True

    if not result.active:
        log_access(
            license_id=result.license_id,
            tenant_id=result.tenant_id or result.subject_id,
            app_id=result.app_id,
            plan_id=result.plan_id,
            method=method,
            path=path,
            status="denied_invalid_token",
            token_hash=token_hash,
            source_ip=source_ip,
            user_agent=user_agent,
            reason=result.reason,
        )
        await _emit_metering_event(
            request,
            # Same fix as the valid-token path below: do NOT fall back to
            # subject_id when tenant_id is missing — that poisons
            # cost_events.tenant_id with user UUIDs. subject_id is still
            # passed through correctly via the dedicated kwarg below.
            tenant_id=result.tenant_id or "",
            license_id=result.license_id,
            subject_id=result.subject_id,
            app_id=result.app_id,
            plan_id=result.plan_id,
            plan_code=result.plan_code,
            method=method,
            path=path,
            status="denied_invalid_token",
            source_ip=source_ip,
        )
        return Response(
            content=json.dumps({"error": "license_invalid", "reason": result.reason}),
            status_code=403,
            media_type="application/json",
            headers={
                "x-error-reason": result.reason or "license_invalid",
            },
        )

    # Token is valid - extract tenant context (may be empty).
    # IMPORTANT: do NOT fall back to subject_id here — that poisons the
    # cost_events.tenant_id column with user UUIDs and breaks per-tenant
    # rollups (Phase F.1 of the metering finish-the-job plan). When Keycloak
    # introspection omits tenant_id, the correct downstream value is empty.
    tenant_id = result.tenant_id or ""

    # Budget enforcement: token budget, then request quota, both decided by
    # billing over HTTP — billing's Postgres is the single counter for each.
    #
    # ⚠️ BOTH WAYS OF SKIPPING THIS BLOCK USED TO BE INVISIBLE. The guard has two
    # arms and neither emitted anything: a deployment with
    # BUDGET_ENFORCEMENT_ENABLED=false (the DEFAULT) produced telemetry identical
    # to one enforcing every request, and an active licence with no licence id
    # slipped past with no record. Counted on a SEPARATE instrument from
    # metric_budget_fail_open — see metric_budget_enforcement_skipped for why
    # folding a permanent config state into the fail-open family would pin an
    # alert on it forever.
    if not BUDGET_ENFORCEMENT_ENABLED:
        metric_budget_enforcement_skipped.labels(reason="enforcement_disabled").inc()
    elif not result.license_id:
        metric_budget_enforcement_skipped.labels(reason="no_license_id").inc()
    if BUDGET_ENFORCEMENT_ENABLED and result.license_id:
        license_id = result.license_id
        plan_id = result.plan_id or None
        if not tenant_id:
            # billing's budget and quota checks are tenant-scoped, so a licence
            # whose introspection carried no tenant has nothing to ask for.
            metric_budget_fail_open.labels(reason="no_tenant_id").inc()
            budget_allowed, budget_reason = True, None
        else:
            metric_metering_lookup.labels(result="http").inc()
            budget_allowed, budget_reason = await _metering_http_verdict(
                request,
                lambda mc, auth: mc.check_budget(
                    request.app.state.http_client,
                    tenant_id=tenant_id,
                    license_id=license_id,
                    auth_headers=auth,
                ),
                license_id=license_id,
                tenant_id=tenant_id,
            )
            if budget_allowed:
                metric_metering_lookup.labels(result="http").inc()
                budget_allowed, budget_reason = await _metering_http_verdict(
                    request,
                    lambda mc, auth: mc.check_quota(
                        request.app.state.http_client,
                        tenant_id=tenant_id,
                        license_id=license_id,
                        plan_id=plan_id,
                        auth_headers=auth,
                    ),
                    license_id=license_id,
                    tenant_id=tenant_id,
                )

        if not budget_allowed:
            kind = "quota" if budget_reason == "request_quota_exhausted" else "budget"
            metric_budget_exceeded.labels(
                tenant=tenant_id or "unknown",
                app=result.app_key or "unknown",
                kind=kind,
            ).inc()
            log_status = (
                "denied_quota" if budget_reason == "request_quota_exhausted" else "denied_budget"
            )
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status=log_status,
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=budget_reason,
            )
            await _emit_metering_event(
                request,
                actor_id=actor_raw,
                actor_missing_reason=actor_reason,
                tenant_id=tenant_id,
                license_id=result.license_id,
                subject_id=result.subject_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                plan_code=result.plan_code,
                method=method,
                path=path,
                status=log_status,
                source_ip=source_ip,
            )
            return Response(
                content=json.dumps(
                    {"error": "budget_exceeded", "kind": kind, "reason": budget_reason}
                ),
                status_code=402,
                media_type="application/json",
                headers={"x-error-reason": budget_reason or "budget_exceeded"},
            )

    # Set True ONLY in the host-bound arm below, never re-derived from headers:
    # `x-forwarded-host` / `x-app-key` are unsigned, and a licence-token caller
    # must not be able to forge the pair to dodge metering.
    gateway_browser_ui = False
    # App-route enforcement — reject tokens used on wrong app's routes (403)
    _self_read_exempt = _self_read_exemption_applies(result.app_key, path, method)
    if not _check_app_route(result.app_key, path, method):
        # Distinguish the new class from the original one. Contract-safe:
        # nothing in gitops switches on this value, the status stays 403 and
        # `error` stays "forbidden" — only the reason string is finer-grained.
        route_reason = (
            "app_route_missing_key" if not (result.app_key or "").strip() else "app_route_mismatch"
        )
        metric_app_route.labels(
            result="denied_missing_key"
            if route_reason == "app_route_missing_key"
            else "denied_mismatch"
        ).inc()
        log_access(
            license_id=result.license_id,
            tenant_id=tenant_id,
            app_id=result.app_id,
            plan_id=result.plan_id,
            method=method,
            path=path,
            status="denied_app_mismatch",
            token_hash=token_hash,
            source_ip=source_ip,
            user_agent=user_agent,
            reason=route_reason,
        )
        await _emit_metering_event(
            request,
            actor_id=actor_raw,
            actor_missing_reason=actor_reason,
            tenant_id=tenant_id,
            license_id=result.license_id,
            subject_id=result.subject_id,
            app_id=result.app_id,
            plan_id=result.plan_id,
            plan_code=result.plan_code,
            method=method,
            path=path,
            status="denied_app_mismatch",
            source_ip=source_ip,
        )
        return Response(
            content=json.dumps({"error": "forbidden", "reason": route_reason}),
            status_code=403,
            media_type="application/json",
            headers={"x-error-reason": route_reason},
        )
    # --- ASSERTED app key on an UNREGISTERED route ---------------------------
    # Reached only when `_check_app_route` allowed, so the two arms are disjoint:
    # a registered prefix has already been matched-or-denied above, and this one
    # fires only where the registry said nothing at all.
    #
    # `app_key_asserted` is true on the JWT path ONLY, where the key came from the
    # caller's unsigned `x-app-key` header rather than from the credential. See
    # ASSERTED_APP_KEY_ALLOW_UNREGISTERED for why an asserted key has no binding
    # here and is therefore refused.
    elif app_key_asserted and not _asserted_key_route_allowed(path):
        # The gateway host binding is checked FIRST: when the gateway's
        # `x-forwarded-host` names exactly this key, the key was set by the
        # gateway's static host map (client copy stripped), so the route-less
        # deny does not apply. See GATEWAY_ASSERTED_HOST_KEYS. Anything else
        # falls through to the unchanged flag/deny logic below.
        if _asserted_key_host_bound(request):
            metric_app_route.labels(result="allowed_asserted_key_host_bound").inc()
            gateway_browser_ui = True
        # ⚠️ COUNTED ON BOTH SIDES OF THE FLAG, on purpose. If the revert only
        # counted in the deny arm, flipping to permissive would take this series
        # to zero and an operator would read "the defect stopped happening"
        # when what stopped was the measurement.
        elif ASSERTED_APP_KEY_ALLOW_UNREGISTERED:
            metric_app_route.labels(result="allowed_asserted_key_unregistered").inc()
            logger.warning(
                json.dumps(
                    {
                        "event": "asserted_app_key_unregistered_route_allowed",
                        "reason": "ASSERTED_APP_KEY_ALLOW_UNREGISTERED",
                        "app_key": result.app_key or "",
                        "path": path,
                    }
                )
            )
        else:
            metric_app_route.labels(result="denied_asserted_key_unregistered").inc()
            route_reason = "app_route_unregistered_asserted_key"
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status="denied_app_mismatch",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=route_reason,
            )
            await _emit_metering_event(
                request,
                actor_id=actor_raw,
                actor_missing_reason=actor_reason,
                tenant_id=tenant_id,
                license_id=result.license_id,
                subject_id=result.subject_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                plan_code=result.plan_code,
                method=method,
                path=path,
                status="denied_app_mismatch",
                source_ip=source_ip,
            )
            # Contract-safe, same as the sibling arm: status stays 403 and
            # `error` stays "forbidden"; only the reason string is new. Nothing
            # in gitops switches on the reason value.
            return Response(
                content=json.dumps({"error": "forbidden", "reason": route_reason}),
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": route_reason},
            )
    else:
        # Counted ONCE, under the label that says WHY it was allowed. An
        # exemption that folds into the generic `allowed` bucket is an exemption
        # nobody can prove is alive — and this one exists to justify re-arming a
        # gate that had to be disarmed in production.
        metric_app_route.labels(
            result="allowed_self_read_exempt" if _self_read_exempt else "allowed"
        ).inc()
        if _self_read_exempt:
            logger.info(
                json.dumps(
                    {
                        "event": "license_self_read_exempt",
                        "app_key": result.app_key or "",
                        "method": method,
                        "path": path,
                    }
                )
            )

    # --- STRICT host -> app binding (always enforced) ------------------------
    # Every path on a STRICT_APP_HOSTS host, registered or not, independent of
    # APP_HOST_ROUTE_MODE. Sets the ONLY value that can unlock a whole-service
    # metering skip below.
    strict_bound_app = ""
    if strict := _strict_app_for(request):
        strict_host, strict_expected = strict
        if (result.app_key or "").strip() == strict_expected:
            strict_bound_app = strict_expected
            metric_app_host_route.labels(
                result="strict_matched", host=strict_host, app=strict_expected
            ).inc()
        else:
            metric_app_host_route.labels(
                result="strict_mismatch_denied", host=strict_host, app=strict_expected
            ).inc()
            strict_reason = "app_host_mismatch"
            logger.warning(
                json.dumps(
                    {
                        "event": "app_host_strict_mismatch",
                        "host": strict_host,
                        "expected_app": strict_expected,
                        "app_key": result.app_key or "",
                        "path": path,
                    }
                )
            )
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status="denied_app_mismatch",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=strict_reason,
            )
            await _emit_metering_event(
                request,
                actor_id=actor_raw,
                actor_missing_reason=actor_reason,
                tenant_id=tenant_id,
                license_id=result.license_id,
                subject_id=result.subject_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                plan_code=result.plan_code,
                method=method,
                path=path,
                status="denied_app_mismatch",
                source_ip=source_ip,
            )
            return Response(
                content=json.dumps({"error": "forbidden", "reason": strict_reason}),
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": strict_reason},
            )

    # --- P2: host -> app binding ---------------------------------------------
    # Reached only when the PATH registry said nothing (`_registered_app_key` is
    # None) — that guard is the zero-blast-radius argument, see `_check_app_host`.
    # Default mode `off` short-circuits before any work, so this block is
    # completely inert as shipped: no verdict, no metric, no log.
    if APP_HOST_ROUTE_MODE != "off" and _registered_app_key(path) is None:
        host_allow, host_result, host_label, host_app = _check_app_host(
            result.app_key, path, authority
        )
        metric_app_host_route.labels(result=host_result, host=host_label, app=host_app).inc()
        if host_result in ("mismatch_observed", "mismatch_denied"):
            logger.warning(
                json.dumps(
                    {
                        "event": "app_host_route_mismatch",
                        "mode": APP_HOST_ROUTE_MODE,
                        "enforced": not host_allow,
                        "host": host_label,
                        "expected_app": host_app,
                        "app_key": result.app_key or "",
                        "path": path,
                    }
                )
            )
        if not host_allow:
            # DISTINCT reason from `app_route_mismatch` on purpose: a path-prefix
            # violation and a host-binding violation are different defects with
            # different fixes, and an operator reading logs must be able to tell
            # them apart. Contract-safe otherwise — status stays 403, `error`
            # stays "forbidden".
            host_reason = "app_host_mismatch"
            # Audited on the way out, per the #47 rule that no deny exit may
            # leave the boundary without a row.
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status="denied_app_mismatch",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=host_reason,
            )
            await _emit_metering_event(
                request,
                actor_id=actor_raw,
                actor_missing_reason=actor_reason,
                tenant_id=tenant_id,
                license_id=result.license_id,
                subject_id=result.subject_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                plan_code=result.plan_code,
                method=method,
                path=path,
                status="denied_app_mismatch",
                source_ip=source_ip,
            )
            return Response(
                content=json.dumps({"error": "forbidden", "reason": host_reason}),
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": host_reason},
            )

    # Tenant identity: the licenses service owns it. introspect returns the
    # canonical tenant UUID (any version) — inject it directly as x-tenant-id and
    # use it in all downstream computations. A non-UUID tenant_id here means
    # un-migrated/invalid data — fail closed loudly rather than forwarding a bad
    # identity downstream.
    if tenant_id:
        if _is_uuid(tenant_id):
            tenant_uuid = tenant_id
        else:
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status="denied_invalid_tenant",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason="invalid_tenant",
            )
            return Response(
                content=json.dumps({"error": "forbidden", "reason": "invalid_tenant"}),
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": "invalid_tenant"},
            )
    else:
        tenant_uuid = ""

    # --- Trusted tenant-proxy passthrough (fixes the mini1 licence-gate 402) --
    # Every caller's `X-Tenant-Id` was UNCONDITIONALLY overwritten with its own
    # tenant (`tenant_uuid` above), which meant a trusted proxy — a service
    # acting on behalf of a DIFFERENT tenant, e.g. w2v-backend reading a
    # bare-metal host's licence — could never assert that other tenant here,
    # and the licenses service's own trusted-proxy retarget
    # (`auth/middleware.py:404`, `is_trusted_tenant_proxy`) could never fire
    # because the header it inspects already equaled the claim by the time it
    # arrived. This is a DELIBERATE WIDENING, not a pure bugfix: a listed,
    # verified `azp` may now have license-authz forward a tenant it does not
    # itself hold. It is bounded three ways:
    #   1. Trust comes from `jwt_azp` — a signature-VERIFIED claim on the
    #      caller's own bearer token — never from the `X-Tenant-Id` header
    #      itself, which is attacker-controlled.
    #   2. `jwt_azp` must be a non-empty member of `TRUSTED_TENANT_PROXY_CLIENTS`.
    #      Every other caller (unlisted azp, no azp, or opaque-mode — see
    #      `jwt_azp`'s declaration) keeps TODAY'S behavior byte-for-byte: its
    #      `X-Tenant-Id` is overwritten with `tenant_uuid`, exactly as before
    #      this change. That is the anti-spoofing control.
    #   3. The client-supplied value must itself be a well-formed UUID
    #      (`_is_uuid`, same check `tenant_uuid` above already passed) or the
    #      override does not apply and `tenant_uuid` (the caller's own tenant)
    #      is forwarded instead — never an arbitrary string.
    # Downstream (licenses' `_enforce_own_tenant_scope`, etc.) still enforces
    # its own scope on whatever tenant this passes through; this only decides
    # whether the delegation the code already models gets a chance to run.
    #
    # HEADER SOURCE (added 2026-09-16): on the NORTH-SOUTH path this branch was
    # structurally unreachable. The istio ingressgateway strips every
    # client-supplied identity header (15 of them, `x-tenant-id` among them) and
    # calls ext_authz with an EXPLICITLY built header table that carries no
    # tenant header at all — so `x-tenant-id` NEVER arrives here from outside
    # the mesh, and the retarget above could only ever fire on direct
    # Service-to-Service calls. The gateway therefore stamps its own
    # `x-asserted-tenant-id` (stripping any client copy first) and we prefer it.
    # It is gateway-controlled, but it is NOT the security control: trust still
    # comes ONLY from `jwt_azp` (signature-verified) being on the allowlist, and
    # the value must still be a well-formed UUID. Both guards are unchanged.
    #
    # HOST BOUND (license-authz#64, 2026-09-30): the retarget is honoured ONLY
    # when the target host is in `TENANT_PROXY_RETARGET_HOSTS` (deployment
    # configuration; empty means nothing is retargetable). Anywhere else it is
    # REFUSED with a 403 — see that
    # constant for why a refusal and not a silent downgrade. Asserting one's
    # OWN tenant is not a retarget and is not refused (nothing is widened).
    #
    # AUDIT (same change): the chosen tenant is recorded as the ADDITIVE field
    # `retarget_tenant_id` on the allowed audit row and the metering event, next
    # to the unchanged `tenant_id` (the caller's own). Absent on every
    # non-retarget request, so those rows are byte-for-byte unchanged.
    retarget_tenant_id: str | None = None
    asserted_tenant_header = request.headers.get("x-asserted-tenant-id", "").strip()
    client_tenant_header = asserted_tenant_header or request.headers.get("x-tenant-id", "").strip()
    if (
        jwt_azp
        and jwt_azp in TRUSTED_TENANT_PROXY_CLIENTS
        and client_tenant_header
        and _is_uuid(client_tenant_header)
        and client_tenant_header.lower() != (tenant_uuid or "").lower()
    ):
        retarget_host = _retarget_target_host(request)
        if retarget_host not in TENANT_PROXY_RETARGET_HOSTS:
            reason = "host_not_allowed" if retarget_host else "host_unknown"
            logger.warning(
                json.dumps(
                    {
                        "event": "trusted_tenant_proxy_retarget_refused",
                        "azp": jwt_azp,
                        "host": retarget_host,
                        "reason": reason,
                    }
                )
            )
            log_access(
                license_id=result.license_id,
                tenant_id=tenant_id,
                app_id=result.app_id,
                plan_id=result.plan_id,
                method=method,
                path=path,
                status="denied_tenant_proxy_host",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=f"tenant_proxy_retarget_{reason}",
            )
            return Response(
                content='{"error": "forbidden", "reason": "tenant_proxy_retarget_host_not_allowed"}',
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": "tenant_proxy_retarget_host_not_allowed"},
            )
        own_tenant_uuid = tenant_uuid
        tenant_uuid = client_tenant_header
        retarget_tenant_id = client_tenant_header
        # This branch logged NOTHING on success, which is precisely why it took
        # a day to establish that it had never fired. One INFO line makes
        # "fired" distinguishable from "never reached" in a single kubectl logs.
        logger.info(
            json.dumps(
                {
                    "event": "trusted_tenant_proxy_retarget",
                    "azp": jwt_azp,
                    "source_header": (
                        "x-asserted-tenant-id" if asserted_tenant_header else "x-tenant-id"
                    ),
                    "tenant_id": tenant_uuid,
                    "own_tenant_id": own_tenant_uuid,
                    "host": retarget_host,
                }
            )
        )

    # CRITICAL: These header names MUST match headersToUpstreamOnAllow in IstioOperator
    # x-subject-id propagates the licenses Subject UUID to upstream services so
    # they can attribute work to the originating user (per-user metering grain).
    # x-tenant-name is vestigial: the canonical tenant identity is the UUID
    # x-tenant-id, so the human label is no longer resolved or propagated (always
    # empty). Kept in the header set only to preserve the headersToUpstreamOnAllow
    # shape; removing it cluster-wide is a separate follow-up.
    response_headers = {
        "x-tenant-id": tenant_uuid,
        "x-tenant-name": "",
        "x-license-id": result.license_id or "",
        "x-subject-id": result.subject_id or "",
        "x-app-id": result.app_id or "",
        "x-plan-id": result.plan_id or "",
        "x-plan-code": result.plan_code or "",
        "x-app-key": result.app_key or "",
        "x-caller-client-id": caller_client_id,
        "x-caller-roles": ",".join(caller_roles),
    }
    # x-actor-id carries the verified JWT `sub` — the acting HUMAN — as an
    # ADDITIONAL header, distinct from x-subject-id (the licence holder).
    # JWT-mode only, and only when the token actually has a `sub`. Absence
    # must stay absence: no default, no empty string, key omitted entirely
    # rather than set to "" like the vestigial headers above — a hop that
    # cannot vouch for an identity must not forge or inherit one.
    if auth_realm:
        # Realm of the verified user JWT (JWT mode only). Inert until the mesh
        # config lists `x-auth-realm` in headersToUpstreamOnAllow.
        response_headers["x-auth-realm"] = auth_realm
    if actor_sub:
        response_headers["x-actor-id"] = actor_sub
    elif licence_token_mode and ACTOR_INJECT_LICENSE_TOKEN_MODE:
        # Licence-token mode: vouch for the licence subject's Keycloak `sub`,
        # but ONLY when the request did not already name its actor — a service
        # forwarding the human it acts for keeps that human. UUID-only, same
        # rule as the metering field; anything else injects nothing. See the
        # flag's declaration for why it is off by default.
        inbound_actor = request.headers.get("x-actor-id", "").strip()
        licence_actor, _ = _resolve_actor_id(actor_raw, actor_reason)
        if not inbound_actor and licence_actor:
            response_headers["x-actor-id"] = licence_actor

    # Validate header values contain only safe characters. List-valued headers
    # (x-caller-roles) accept commas; everything else uses the strict regex.
    for hdr_name, hdr_val in response_headers.items():
        if not hdr_val:
            continue
        regex = _SAFE_LIST_HEADER_RE if hdr_name in _LIST_HEADERS else _SAFE_HEADER_RE
        if not regex.match(hdr_val):
            # This deny sits immediately above the `status="allowed"` audit row
            # and used to emit NEITHER: a request that passed licence
            # validation on the merits and then tripped this regex produced no
            # audit row at all. It fails CLOSED — not a bypass, audit
            # blindness. The identity fields below are exactly the ones the
            # allow row it aborts would have carried.
            #
            # ⛔ NEVER put `hdr_val` in this row — and note that passing the
            # identity through verbatim DOES put it there. `x-license-id` IS
            # `result.license_id`, `x-app-id` IS `result.app_id` and `x-plan-id`
            # IS `result.plan_id`, so when the header that fails the regex is
            # one of those THREE, the matching field below carries the rejected
            # value. `_elide_unsafe` replaces exactly the field that failed and
            # leaves the others true. `hdr_name` is a literal key from
            # `response_headers` and is safe.
            #
            # ⚠️ `tenant_id` is guarded too, but that guard is a NO-OP TODAY and
            # is written as one deliberately rather than dropped: `x-tenant-id`
            # is `tenant_uuid`, which the block above sets to either a
            # UUID-validated `tenant_id` (always matches `^[\w\-\.]+$`) or "",
            # which the `continue` skips — so this loop can never see it fail.
            # It is covered because the value-equality form costs nothing and
            # does not depend on that upstream validation staying where it is.
            # Do not read its presence as evidence that a tenant leak was
            # observed; none was.
            log_access(
                license_id=_elide_unsafe(result.license_id, hdr_val),
                tenant_id=_elide_unsafe(tenant_id, hdr_val),
                app_id=_elide_unsafe(result.app_id, hdr_val),
                plan_id=_elide_unsafe(result.plan_id, hdr_val),
                method=method,
                path=path,
                status="denied_unsafe_header",
                token_hash=token_hash,
                source_ip=source_ip,
                user_agent=user_agent,
                reason=f"unsafe_header={hdr_name}",
            )
            # Kept, not folded: WARNING is what the alerting stack watches;
            # log_access is INFO and is the audit trail. Two different readers.
            logger.warning(
                json.dumps(
                    {"event": "unsafe_header_value", "header": hdr_name, "token_hash": token_hash}
                )
            )
            return Response(
                content='{"error": "license_invalid", "reason": "unsafe_header_value"}',
                status_code=403,
                media_type="application/json",
                headers={"x-error-reason": "unsafe_header_value"},
            )

    # Log successful access
    latency_ms = (time.time() - start_time) * 1000
    log_access(
        license_id=result.license_id,
        tenant_id=tenant_id,
        app_id=result.app_id,
        plan_id=result.plan_id,
        method=method,
        path=path,
        status="allowed",
        token_hash=token_hash,
        source_ip=source_ip,
        user_agent=user_agent,
        retarget_tenant_id=retarget_tenant_id,
    )
    if gateway_browser_ui:
        # Editor UI traffic is not API consumption: audited above, not metered.
        metric_metering_events_skipped.labels(
            reason="gateway_browser_ui",
            app=GATEWAY_ASSERTED_HOST_KEYS[_gateway_asserted_host(request)],
        ).inc()
    elif unmetered := _unmetered_route(request, method, path, strict_bound_app=strict_bound_app):
        # A known machine poll: authorised and audited above, not metered.
        metric_metering_events_skipped.labels(reason=unmetered[0], app=unmetered[1]).inc()
    else:
        await _emit_metering_event(
            request,
            actor_id=actor_raw,
            actor_missing_reason=actor_reason,
            tenant_id=tenant_id,
            license_id=result.license_id,
            subject_id=result.subject_id,
            app_id=result.app_id,
            plan_id=result.plan_id,
            plan_code=result.plan_code,
            method=method,
            path=path,
            status="allowed",
            source_ip=source_ip,
            retarget_tenant_id=retarget_tenant_id,
        )

    logger.info(json.dumps({"event": "authz_success", "latency_ms": round(latency_ms, 2)}))

    return Response(
        content='{"status": "ok"}',
        status_code=200,
        media_type="application/json",
        headers=response_headers,
    )


# Catch-all ext_authz route. Registered as separate routes per HTTP method so
# each gets a unique operation_id (OpenAPI requires unique operationIds across
# the whole document). FastAPI's @app.api_route with a methods list creates a
# single APIRoute and assigns the same operation_id to every method, which
# violates the spec — registering per-method avoids that.
for _method in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
    app.add_api_route(
        "/{path:path}",
        ext_authz_check,
        methods=[_method],
        operation_id=f"ext_authz_check_{_method.lower()}",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "9001"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=75)  # nosec B104 — container requires binding to all interfaces
