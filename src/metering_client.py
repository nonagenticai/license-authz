"""HTTP client for billing-service token-budget and request-quota checks.

Token budgets are decided ONLY here: billing's Postgres ``token_budgets`` row
(``consumed_tokens``) is the single budget counter, served by
``GET /api/budget/check``. license-authz no longer reads any Redis budget key.

Request quotas are decided ONLY here too, by ``GET /api/quota/check``;
license-authz reads no Redis quota key.

Both answers are cached in-process for ``METERING_HTTP_CACHE_TTL`` seconds to
soak bursty traffic.
"""

import asyncio
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from settings import get_settings

# Service URL comes from gitops Deployment env (no source-side fallback;
# see gitops-production/docs/config-externalization-policy.md).
METERING_SERVICE_URL = get_settings().metering_service_url
METERING_HTTP_CACHE_TTL = int(os.getenv("METERING_HTTP_CACHE_TTL", "5"))
MAX_CACHE_ENTRIES = int(os.getenv("LICENSE_AUTHZ_MAX_CACHE_ENTRIES", "1000"))


# Field names this client reads off billing-service's PUBLISHED schemas.
# tests/contract/test_billing_metering_contract.py asserts each set is a subset
# of the corresponding schema's declared properties in billing's vendored
# openapi.json. That is the only thing standing between this module and another
# round of parsing a server that does not exist: the previous implementation
# read "used"/"budget"/"quota"/"limit"/"period_end", none of which billing has
# ever declared, so every quota check silently returned "not exceeded".
BILLING_BUDGET_AGGREGATE_FIELDS = ("budget_count", "budgets", "next_expiry")
# `over_budget` / `remaining_tokens` are deliberately NOT read: the verdict is
# re-derived per SCOPE (licence rows vs tenant rows) from the raw counters, and
# listing a field the client does not read would make the contract guard's
# subset assertion a claim about nothing.
BILLING_BUDGET_ITEM_FIELDS = ("license_id", "budget_tokens", "used_tokens")
BILLING_QUOTA_FIELDS = ("over_quota", "quota_monthly", "used_this_month")

# billing declares these `required` and non-nullable on BudgetAggregateStatus.
# `budget_count` cross-checks `budgets`: a body claiming active budgets while
# omitting the list is not a budget answer and must not be read as "no rows".
# `budgets` itself is NOT here: it carries `default: []` and is legally
# omittable when there is nothing to list.
REQUIRED_BUDGET_AGGREGATE_FIELDS = ("budget_count",)
# Every TokenBudgetStatus field the verdict reads; billing declares all three
# required (license_id nullable). A row missing one is refused, never guessed.
REQUIRED_BUDGET_ITEM_FIELDS = BILLING_BUDGET_ITEM_FIELDS

# billing stores a TENANT-scoped budget with license_id NULL or the nil UUID.
NIL_LICENSE_ID = "00000000-0000-0000-0000-000000000000"


class MeteringContractError(ValueError):
    """billing answered 200 with a body license-authz cannot interpret.

    Distinct from httpx.HTTPError (transport/status). Raised instead of letting
    an AttributeError/TypeError escape, so the ext_authz call site can classify
    it and apply METERING_FAIL_OPEN rather than 500 — a 500 from ext_authz is a
    platform-wide DENY at the Envoy layer.
    """


def _unwrap_envelope(raw: Any) -> Any:
    """Peek into T3.2 {data: T, meta} envelope; fall back to bare T for older servers.

    Tightened heuristic: requires BOTH `data` AND `meta` keys present, and `data`
    being non-null. Otherwise returns ``raw`` as-is. This avoids two false positives:
    (1) a legitimate bare response that happens to carry a top-level ``data`` field
    (e.g. ``{"used": 5, "data": "...something..."}``); and (2) a malformed envelope
    where ``data`` is null, which previously turned into ``None`` and broke the
    downstream ``or {}``/``or []`` fallbacks inconsistently.
    """
    if isinstance(raw, dict) and "data" in raw and "meta" in raw and raw["data"] is not None:
        return raw["data"]
    return raw


# Eviction hook wired by main.py. Default is a no-op so this module imports
# standalone in tests.
def _noop_eviction(_cache: str) -> None:
    pass


_on_cache_eviction: Callable[[str], None] = _noop_eviction


def set_eviction_hook(hook: Callable[[str], None]) -> None:
    global _on_cache_eviction
    _on_cache_eviction = hook


@dataclass
class BudgetDecision:
    allowed: bool
    kind: str | None  # "budget" | "quota" | None when allowed
    reason: str | None
    used: int | None = None
    limit: int | None = None
    reset_at: str | None = None


_CacheKey = tuple[str, str, str | None]


class MeteringClient:
    """Thin HTTP client over billing-service /api/budget/check + /api/quota/check."""

    def __init__(
        self,
        base_url: str = METERING_SERVICE_URL,
        ttl: int = METERING_HTTP_CACHE_TTL,
        max_entries: int = MAX_CACHE_ENTRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ttl = ttl
        self._max_entries = max_entries
        self._cache: OrderedDict[_CacheKey, tuple[float, BudgetDecision]] = OrderedDict()

    def _cached(self, key: _CacheKey) -> BudgetDecision | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def _put(self, key: _CacheKey, value: BudgetDecision) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
            _on_cache_eviction("metering")
        self._cache[key] = (time.monotonic() + self._ttl, value)

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """GET with one retry on 5xx or a transport error (random jitter). 4xx is not retried.

        The transport retry covers the keep-alive race: billing's uvicorn closes an
        idle connection after 5s, httpx keeps pooled connections for 5s, so a reused
        connection can be closed under the request ("Server disconnected without
        sending a response"). Both checks are idempotent GETs.
        """
        attempt = 0
        while True:
            try:
                resp = await client.get(url, params=params, headers=headers)
            except httpx.TransportError:
                if attempt >= 1:
                    raise
                attempt += 1
                await asyncio.sleep(random.uniform(0.05, 0.2))  # nosec B311 — jitter, not crypto
                continue
            if resp.is_server_error and attempt < 1:
                attempt += 1
                await asyncio.sleep(random.uniform(0.05, 0.2))  # nosec B311 — jitter, not crypto
                continue
            return resp

    async def check_budget(
        self,
        client: httpx.AsyncClient,
        *,
        tenant_id: str,
        license_id: str | None,
        auth_headers: dict[str, str],
    ) -> BudgetDecision:
        """Token-budget verdict for (tenant, licence), decided from billing's rows.

        billing's ``/api/budget/check`` is tenant-keyed (it declares only
        ``tenant_id``), so the SCOPING happens here — see
        :func:`_scoped_budget_decision`. ``license_id=None`` (the
        service-principal path) checks the tenant scope only.

        Raises httpx.HTTPError on transport/status failure and
        MeteringContractError on a 200 body this client cannot interpret; the
        caller applies METERING_FAIL_OPEN to both.
        """
        key: _CacheKey = ("budget", tenant_id, license_id)
        cached = self._cached(key)
        if cached is not None:
            return cached

        resp = await self._request_with_retry(
            client,
            f"{self._base_url}/api/budget/check",
            {"tenant_id": tenant_id},
            {"X-Tenant-Id": tenant_id, **auth_headers},
        )
        resp.raise_for_status()
        decision = _scoped_budget_decision(_unwrap_envelope(resp.json()), license_id)
        self._put(key, decision)
        return decision

    async def check_quota(
        self,
        client: httpx.AsyncClient,
        *,
        tenant_id: str,
        license_id: str,
        plan_id: str | None,
        auth_headers: dict[str, str],
    ) -> BudgetDecision:
        """Request-quota verdict for a licence, per billing's ``/api/quota/check``.

        ``plan_id`` is sent only when known; billing resolves the limit from the
        licence's durable quota row. Same error contract as :meth:`check_budget`.
        """
        key: _CacheKey = ("quota", license_id, plan_id)
        cached = self._cached(key)
        if cached is not None:
            return cached

        resp = await self._request_with_retry(
            client,
            f"{self._base_url}/api/quota/check",
            {"license_id": license_id, **({"plan_id": plan_id} if plan_id else {})},
            {"X-Tenant-Id": tenant_id, **auth_headers},
        )
        resp.raise_for_status()
        q = _unwrap_envelope(resp.json())
        if _quota_exceeded(q):
            decision = BudgetDecision(
                allowed=False,
                kind="quota",
                reason="request_quota_exhausted",
                used=q.get("used_this_month"),
                limit=q.get("quota_monthly"),
                reset_at=None,  # billing's QuotaStatus declares no reset field
            )
        else:
            decision = BudgetDecision(allowed=True, kind=None, reason=None)
        self._put(key, decision)
        return decision


def _as_int(value: Any, field: str) -> int:
    """Coerce a declared-integer field, or fail loudly.

    Deliberately NOT a silent `return False` on garbage: swallowing a
    non-integer cap is how the previous implementation turned every unparsable
    response into "not exceeded" (i.e. allow). A contract error is surfaced and
    governed by METERING_FAIL_OPEN at the call site instead.

    bool is rejected explicitly (True/False ARE ints in Python):
    ``{"budget_tokens": false}`` would coerce to a block-all cap of 0 and
    ``{"used_tokens": false}`` to zero usage — an ALLOW that nothing raised, so METERING_FAIL_OPEN could not
    govern it. billing declares these fields ``integer``, so a bool is a
    contract violation and must be answered as one.
    """
    if isinstance(value, bool):
        raise MeteringContractError(
            f"billing field {field!r} is a bool ({value!r}); billing declares it an integer. "
            f"Refusing to read it as {int(value)}."
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MeteringContractError(
            f"billing field {field!r} is not an integer: {value!r}"
        ) from exc


def _budget_rows(aggregate: Any) -> list[dict[str, Any]]:
    """The TokenBudgetStatus rows of a BudgetAggregateStatus, or a contract error."""
    if not isinstance(aggregate, dict):
        raise MeteringContractError(
            f"/api/budget/check: expected a BudgetAggregateStatus object, "
            f"got {type(aggregate).__name__}"
        )
    for field in REQUIRED_BUDGET_AGGREGATE_FIELDS:
        if aggregate.get(field) is None:
            raise MeteringContractError(
                f"/api/budget/check: billing declares {field!r} required and non-nullable, "
                f"but it is missing or null. Refusing to infer a budget verdict; "
                f"METERING_FAIL_OPEN decides what happens next."
            )
    budget_count = _as_int(aggregate["budget_count"], "budget_count")
    budgets = aggregate.get("budgets")
    if budgets is None:
        budgets = []
    if not isinstance(budgets, list):
        raise MeteringContractError(
            f"/api/budget/check: 'budgets' is a {type(budgets).__name__}, not a list"
        )
    if budget_count > 0 and not budgets:
        # Reading this as "no rows" would turn every exhausted budget into an
        # ALLOW the moment billing stopped listing them.
        raise MeteringContractError(
            f"/api/budget/check: budget_count={budget_count} but no budgets listed"
        )
    for row in budgets:
        if not isinstance(row, dict):
            raise MeteringContractError(
                f"/api/budget/check: budget row is a {type(row).__name__}, not an object"
            )
        missing = [f for f in REQUIRED_BUDGET_ITEM_FIELDS if f not in row]
        if missing:
            raise MeteringContractError(
                f"/api/budget/check: budget row is missing required field(s) {missing}"
            )
        lic = row["license_id"]
        if lic is not None and not isinstance(lic, str):
            raise MeteringContractError(
                f"/api/budget/check: budget row license_id is a {type(lic).__name__}"
            )
    return budgets


def _scope_exhausted(rows: list[dict[str, Any]]) -> tuple[bool, int, int]:
    """(exhausted, used, limit) summed over one scope's rows.

    Exhausted when Σused_tokens >= Σbudget_tokens; a non-positive Σbudget_tokens
    is block-all (a cap of 0 is the operator's explicit "deny everything", the
    same reading the old Redis arm gave ``budget_tokens: 0``).
    """
    used = sum(_as_int(r["used_tokens"], "used_tokens") for r in rows)
    limit = sum(_as_int(r["budget_tokens"], "budget_tokens") for r in rows)
    return (limit <= 0 or used >= limit), used, limit


def _scoped_budget_decision(aggregate: Any, license_id: str | None) -> BudgetDecision:
    """Decide a token budget per SCOPE from billing's tenant-wide row list.

    billing returns every ACTIVE, unexpired budget row for the tenant with
    ``used_tokens`` already rolled to the current period. Pooling them all
    (``total_remaining``) would be WRONG here: a licence capped at 10M sitting
    at 10.6M inside a tenant pool of 1.45B must stay DENIED — exactly what the
    per-licence Redis key enforced before billing's Postgres counter became the
    single source of truth. So:

    * licence scope (only when ``license_id`` is given): the rows whose
      ``license_id`` equals it. Deny when that scope is exhausted.
    * tenant scope: rows whose ``license_id`` is null or the nil UUID. Deny when
      exhausted — for every licence of the tenant and for service principals.
    * rows of OTHER licences are ignored; a scope with no rows does not deny.

    Every period counts: a weekly row can only ADD a deny.
    """
    rows = _budget_rows(aggregate)
    licence_rows = [r for r in rows if r["license_id"] == license_id] if license_id else []
    tenant_rows = [r for r in rows if r["license_id"] in (None, NIL_LICENSE_ID)]
    reset_at = aggregate.get("next_expiry")
    for scope in (licence_rows, tenant_rows):
        if not scope:
            continue
        exhausted, used, limit = _scope_exhausted(scope)
        if exhausted:
            return BudgetDecision(
                allowed=False,
                kind="budget",
                reason="token_budget_exhausted",
                used=used,
                limit=limit,
                reset_at=reset_at,
            )
    return BudgetDecision(allowed=True, kind=None, reason=None)


def _quota_exceeded(quota: Any) -> bool:
    """Is the licence over its request quota, per billing's own verdict?

    billing computes ``over_quota = remaining is not None and remaining <= 0``
    (billing_service/src/services/quota.py) against a Redis counter that can be
    ahead of the ``used_this_month`` rollup. It OWNS this number, so its boolean
    wins over anything re-derived here. Note this also means the boundary is
    ``remaining <= 0`` — at-limit usage is over quota, and ``quota_monthly: 0``
    reports ``over_quota: true`` even at zero usage (BLOCK ALL), while
    ``quota_monthly: null`` is UNLIMITED and never over.

    The local derivation below is only reached if billing omits ``over_quota``
    entirely, and reproduces the same semantics.
    """
    if not isinstance(quota, dict):
        raise MeteringContractError(
            f"/api/quota/check: expected a QuotaStatus object, got {type(quota).__name__}"
        )

    over_quota = quota.get("over_quota")
    if over_quota is not None:
        return bool(over_quota)

    cap = quota.get("quota_monthly")
    if cap is None:
        return False  # unlimited
    cap_int = _as_int(cap, "quota_monthly")
    if cap_int <= 0:
        return True  # 0 → block all; negative → invalid → fail closed
    used = quota.get("used_this_month")
    if used is None:
        return False
    return _as_int(used, "used_this_month") >= cap_int
