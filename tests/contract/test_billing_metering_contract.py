"""Contract guard: MeteringClient must parse exactly what billing-service DECLARES.

Why this file exists
--------------------
``tests/test_metering_client.py`` was green for the entire life of the metering
HTTP fallback while the client could not parse a single real billing response.
Every payload in it was hand-written against a *fictional* server
(``[{"used": ..., "budget": ...}]``). Billing has never served that shape. A
hand-written fixture can agree with a hand-written client forever; only the
server's own published schema can break that loop.

So every payload in this module is SYNTHESISED FROM THE VENDORED SPEC
(:data:`SPEC`) rather than typed out by hand, and :func:`_synth` REFUSES any
field name the spec does not declare.

⚠️ SCOPE — this is a SNAPSHOT guard, not a drift guard
------------------------------------------------------
Nothing here re-fetches billing's spec. The guard proves that the client agrees
with the ``billing_openapi.json`` COMMITTED NEXT TO IT, and nothing more. If
billing renames ``used_tokens`` tomorrow, this file stays green until a human
re-vendors the spec with the command below — the rename is caught at the moment
of re-vendoring, not at the moment it ships. Do not read these tests as live
protection against billing drift; the protection they actually give is that a
change to THIS client can no longer invent a field, and that re-vendoring is a
one-command operation whose fallout is immediately visible.

Closing that gap properly needs a scheduled job that re-fetches the spec and
opens a PR on diff. That is a CI/gitops change, deliberately out of scope here.

Provenance of ``billing_openapi.json``
--------------------------------------
Fetched 2026-08-10 from the DEPLOYED billing-service, in-cluster, from inside
the license-authz pod::

    kubectl -n istio-system exec license-authz-7f7459b885-cwlpz -- python -c \\
      'import httpx; print(httpx.get(
         "http://billing-service.billing.example:8157/openapi.json").text)'

That live copy was compared against billing's committed ``docs/openapi.json``:
both declare 61 paths and are BYTE-IDENTICAL for every schema this guard reads
(``Envelope[BudgetAggregateStatus]``, ``BudgetAggregateStatus``,
``TokenBudgetStatus``, ``Envelope[QuotaStatus]``, ``QuotaStatus``, ``Meta``).
The live copy is the one vendored here, because the deployed contract — not the
repo's — is what license-authz actually talks to.

Refreshed 2026-10-31 from billing-service ``docs/openapi.json`` on branch
``feat/retire-redis-quota-keys`` (``/api/quota/check`` ``plan_id`` became
optional and ``QuotaStatus.plan_id`` nullable). That billing change deploys
right after this one.

This guard MUST NOT be skippable
--------------------------------
No ``pytest.skip``, no ``importorskip``, no ``xfail``, no ``skipif``, no module
``pytestmark``, no custom marker CI could ``-m``-deselect. ``test_guard_cannot_
skip_itself_green`` asserts that against this file's own source text, so the
property is enforced rather than merely promised. The spec is loaded at module
scope: a missing or corrupt ``billing_openapi.json`` is a collection ERROR, not
a silent pass.
"""

import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock

import httpx

import metering_client
from metering_client import MeteringClient

SPEC_PATH = pathlib.Path(__file__).parent / "billing_openapi.json"
SPEC: dict[str, Any] = json.loads(SPEC_PATH.read_text())
SCHEMAS: dict[str, Any] = SPEC["components"]["schemas"]

BUDGET_CHECK_PATH = "/api/budget/check"
QUOTA_CHECK_PATH = "/api/quota/check"


# --------------------------------------------------------------------------
# Spec-driven payload synthesis
# --------------------------------------------------------------------------


def _resolve(node: dict[str, Any]) -> dict[str, Any]:
    """Follow a single ``$ref`` into ``components.schemas``."""
    while "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        node = SCHEMAS[name]
    return node


def _default_for(prop: dict[str, Any]) -> Any:
    """A type-appropriate value for a declared property, from the spec alone."""
    if "anyOf" in prop:
        # Prefer the first non-null branch so required fields get a real value.
        for branch in prop["anyOf"]:
            if branch.get("type") != "null":
                return _default_for(branch)
        return None
    return {
        "string": "x",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(prop.get("type"), None)


def _synth(schema_name: str, **overrides: Any) -> dict[str, Any]:
    """Build an instance of ``schema_name`` from the vendored spec.

    Every ``required`` property is populated with a spec-derived default.
    ``overrides`` may only name properties billing DECLARES — inventing a field
    here would recreate the exact failure mode this module exists to prevent
    (a fixture that agrees with a broken client because both made the same
    field up).
    """
    schema = _resolve(SCHEMAS[schema_name])
    props: dict[str, Any] = schema.get("properties", {})
    unknown = set(overrides) - set(props)
    assert not unknown, (
        f"{schema_name} does not declare {sorted(unknown)}; billing declares "
        f"{sorted(props)}. Do not invent fields in a contract fixture."
    )
    instance = {name: _default_for(props[name]) for name in schema.get("required", [])}
    instance.update(overrides)
    return instance


def _envelope(schema_name: str, **overrides: Any) -> dict[str, Any]:
    """A T3.2 ``{data, meta}`` envelope around ``schema_name``, per the spec."""
    return {"data": _synth(schema_name, **overrides), "meta": _synth("Meta")}


def _budget_envelope(*budgets: dict[str, Any], total_remaining: int) -> dict[str, Any]:
    return _envelope(
        "BudgetAggregateStatus",
        tenant_id="t",
        total_remaining=total_remaining,
        budget_count=len(budgets),
        budgets=list(budgets),
    )


def _token_budget(**overrides: Any) -> dict[str, Any]:
    return _synth("TokenBudgetStatus", **overrides)


def _quota_envelope(**overrides: Any) -> dict[str, Any]:
    return _envelope("QuotaStatus", **overrides)


def _resp(body: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("GET", "http://test"))


def _client(*responses: httpx.Response) -> AsyncMock:
    c = AsyncMock(spec=httpx.AsyncClient)
    c.get.side_effect = list(responses)
    return c


# --------------------------------------------------------------------------
# What billing DECLARES (shape-level assertions on the spec itself)
# --------------------------------------------------------------------------


def test_budget_check_declares_an_object_whose_list_lives_under_budgets():
    """``/api/budget/check`` 200 is an OBJECT, not an array.

    The original client did ``for item in _unwrap_envelope(...)``, which iterates
    a dict's STRING KEYS. This assertion pins why that can never work.
    """
    ref = SPEC["paths"][BUDGET_CHECK_PATH]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    envelope = _resolve(ref)
    assert envelope["title"] == "Envelope[BudgetAggregateStatus]"
    data = _resolve(envelope["properties"]["data"])
    assert data["type"] == "object"
    assert data["properties"]["budgets"]["type"] == "array"
    assert _resolve(data["properties"]["budgets"]["items"])["title"] == "TokenBudgetStatus"


def test_quota_check_declares_an_object_with_an_authoritative_over_quota():
    ref = SPEC["paths"][QUOTA_CHECK_PATH]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    envelope = _resolve(ref)
    assert envelope["title"] == "Envelope[QuotaStatus]"
    data = _resolve(envelope["properties"]["data"])
    assert data["properties"]["over_quota"]["type"] == "boolean"
    assert "over_quota" in data["required"]


def test_budget_check_is_tenant_keyed_so_scoping_is_client_side():
    """``/api/budget/check`` declares only ``tenant_id``.

    FastAPI silently drops undeclared query params, so the client sends only
    ``tenant_id`` and selects licence-scoped vs tenant-scoped rows itself from
    each row's ``license_id`` (which billing declares required).
    """
    declared = {p["name"] for p in SPEC["paths"][BUDGET_CHECK_PATH]["get"].get("parameters", [])}
    assert declared == {"tenant_id"}
    assert "license_id" not in declared
    row = _resolve(SCHEMAS["TokenBudgetStatus"])
    assert "license_id" in row["required"]


# --------------------------------------------------------------------------
# What the CLIENT reads must be a subset of what billing declares
# --------------------------------------------------------------------------


def _declared(schema_name: str) -> set[str]:
    return set(_resolve(SCHEMAS[schema_name]).get("properties", {}))


def test_client_declares_the_billing_fields_it_reads():
    """The client must publish the field names it reads, so they can be checked."""
    for const in (
        "BILLING_BUDGET_AGGREGATE_FIELDS",
        "BILLING_BUDGET_ITEM_FIELDS",
        "BILLING_QUOTA_FIELDS",
    ):
        names = getattr(metering_client, const, None)
        assert names is not None, (
            f"metering_client must declare {const} — the set of billing field names it "
            "reads — so this guard can prove they exist in billing's published schema."
        )
        assert names, f"{const} must not be empty"


def test_client_budget_field_reads_are_all_declared_by_billing():
    agg = set(metering_client.BILLING_BUDGET_AGGREGATE_FIELDS)
    item = set(metering_client.BILLING_BUDGET_ITEM_FIELDS)
    assert agg <= _declared("BudgetAggregateStatus"), (
        f"client reads {sorted(agg - _declared('BudgetAggregateStatus'))} which "
        f"BudgetAggregateStatus does not declare"
    )
    assert item <= _declared("TokenBudgetStatus"), (
        f"client reads {sorted(item - _declared('TokenBudgetStatus'))} which "
        f"TokenBudgetStatus does not declare"
    )


def test_client_quota_field_reads_are_all_declared_by_billing():
    quota = set(metering_client.BILLING_QUOTA_FIELDS)
    assert quota <= _declared("QuotaStatus"), (
        f"client reads {sorted(quota - _declared('QuotaStatus'))} which "
        f"QuotaStatus does not declare"
    )


def test_the_fields_the_client_insists_on_are_the_ones_billing_declares_required():
    """The client refuses a body missing these. The spec must agree they are required.

    Two failure modes this closes:
    * insisting on a field billing may legally omit → needless contract errors;
    * NOT insisting on a field the verdict reads → a `.get()` fallback silently
      answers "not exceeded" for a body that is not a budget answer at all.

    ``budgets`` must NOT be insisted on: it carries ``default: []`` and is absent
    from ``required``, so billing may legally omit it (the client cross-checks it
    against ``budget_count`` instead).
    """
    agg_required = set(_resolve(SCHEMAS["BudgetAggregateStatus"])["required"])
    agg_insisted = set(metering_client.REQUIRED_BUDGET_AGGREGATE_FIELDS)
    assert agg_insisted <= agg_required
    assert "budget_count" in agg_insisted
    assert "budgets" not in agg_insisted
    assert "budgets" not in agg_required

    item_required = set(_resolve(SCHEMAS["TokenBudgetStatus"])["required"])
    item_insisted = set(metering_client.REQUIRED_BUDGET_ITEM_FIELDS)
    assert item_insisted <= item_required
    # Every row field the verdict reads must be insisted on.
    assert set(metering_client.BILLING_BUDGET_ITEM_FIELDS) <= item_insisted

    props = _resolve(SCHEMAS["TokenBudgetStatus"])["properties"]
    for field in ("budget_tokens", "used_tokens"):
        assert props[field]["type"] == "integer"


def test_a_body_missing_a_required_field_is_refused_not_allowed():
    """Spec-driven: drop each insisted field in turn; each must RAISE, not allow."""
    for field in metering_client.REQUIRED_BUDGET_AGGREGATE_FIELDS:
        body = _synth(
            "BudgetAggregateStatus",
            tenant_id="t",
            total_remaining=0,
            budget_count=1,
            budgets=[_token_budget(license_id="L", budget_tokens=1, used_tokens=1)],
        )
        del body[field]
        try:
            verdict = metering_client._scoped_budget_decision(body, "L")
        except metering_client.MeteringContractError:
            continue
        raise AssertionError(f"dropping {field!r} returned {verdict!r} instead of raising")
    for field in metering_client.REQUIRED_BUDGET_ITEM_FIELDS:
        row = _token_budget(license_id="L", budget_tokens=1, used_tokens=1)
        del row[field]
        body = _synth("BudgetAggregateStatus", tenant_id="t", budget_count=1, budgets=[row])
        try:
            verdict = metering_client._scoped_budget_decision(body, "L")
        except metering_client.MeteringContractError:
            continue
        raise AssertionError(f"dropping row {field!r} returned {verdict!r} instead of raising")


def test_the_fictional_fields_the_old_client_read_are_declared_nowhere():
    """Discriminating power: the guard must be able to tell right from wrong.

    ``used`` / ``budget`` / ``quota`` / ``period_end`` were what the client read
    for years. None of them exist anywhere in billing's budget or quota schemas.
    """
    everywhere = (
        _declared("BudgetAggregateStatus")
        | _declared("TokenBudgetStatus")
        | _declared("QuotaStatus")
    )
    for fictional in ("used", "budget", "quota", "limit", "period_end"):
        assert fictional not in everywhere


# --------------------------------------------------------------------------
# The real client against spec-synthesised payloads — BOTH directions
# --------------------------------------------------------------------------


async def test_under_budget_allows():
    c = _client(
        _resp(
            _budget_envelope(
                _token_budget(
                    license_id="L", budget_tokens=100_000, used_tokens=10, remaining_tokens=99_990
                ),
                total_remaining=99_990,
            )
        ),
    )
    d = await MeteringClient(base_url="http://meter").check_budget(
        c, tenant_id="t", license_id="L", auth_headers={}
    )
    assert d.allowed is True
    assert d.kind is None


async def test_licence_over_its_cap_denies_even_with_tenant_pool_remaining():
    """DENY direction, from real field values: the per-licence cap binds even
    though billing's pooled ``total_remaining`` is huge."""
    c = _client(
        _resp(
            _budget_envelope(
                _token_budget(
                    license_id="L",
                    budget_tokens=10_000_000,
                    used_tokens=10_649_903,
                    remaining_tokens=0,
                    over_budget=True,
                ),
                _token_budget(
                    license_id=None,
                    budget_tokens=1_450_000_000,
                    used_tokens=0,
                    remaining_tokens=1_450_000_000,
                ),
                total_remaining=1_450_000_000,
            )
        ),
    )
    d = await MeteringClient(base_url="http://meter").check_budget(
        c, tenant_id="t", license_id="L", auth_headers={}
    )
    assert d.allowed is False
    assert d.kind == "budget"
    assert d.reason == "token_budget_exhausted"
    assert d.used == 10_649_903
    assert d.limit == 10_000_000


async def test_no_active_budget_allows():
    """``budget_count == 0`` is billing's 'no_active_budget', not 'over_budget'."""
    c = _client(_resp(_budget_envelope(total_remaining=0)))
    d = await MeteringClient(base_url="http://meter").check_budget(
        c, tenant_id="t", license_id="L", auth_headers={}
    )
    assert d.allowed is True


async def test_billing_over_quota_true_denies():
    """DENY direction, quota. Billing's authoritative verdict is honoured."""
    c = _client(
        _resp(
            _quota_envelope(quota_monthly=500, used_this_month=500, remaining=0, over_quota=True)
        ),
    )
    d = await MeteringClient(base_url="http://meter").check_quota(
        c, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
    )
    assert d.allowed is False
    assert d.kind == "quota"
    assert d.reason == "request_quota_exhausted"
    assert d.used == 500
    assert d.limit == 500


async def test_over_quota_is_authoritative_over_a_locally_rederived_verdict():
    """If billing says over_quota, we deny even when used < quota_monthly.

    Billing computes ``over_quota = remaining is not None and remaining <= 0``
    from a Redis counter that can be ahead of ``used_this_month``. Re-deriving
    ``used > quota_monthly`` locally would silently overrule the service that
    owns the number.
    """
    c = _client(
        _resp(
            _quota_envelope(quota_monthly=1000, used_this_month=12, remaining=0, over_quota=True)
        ),
    )
    d = await MeteringClient(base_url="http://meter").check_quota(
        c, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
    )
    assert d.allowed is False
    assert d.kind == "quota"


async def test_unlimited_quota_allows():
    """``quota_monthly: null`` is billing's UNLIMITED branch (over_quota False)."""
    c = _client(
        _resp(
            _quota_envelope(
                quota_monthly=None, used_this_month=99, remaining=None, over_quota=False
            )
        ),
    )
    d = await MeteringClient(base_url="http://meter").check_quota(
        c, tenant_id="t", license_id="L", plan_id="P", auth_headers={}
    )
    assert d.allowed is True


async def test_the_exact_live_payload_from_the_deployed_tenant_parses():
    """Byte-for-byte the body deployed billing returned for tenant ``n8n-dev``.

    Captured 2026-08-10 in-cluster (200 OK). This is the payload that made the
    old client raise ``AttributeError: 'str' object has no attribute 'get'``.
    """
    live = {
        "data": {
            "tenant_id": "n8n-dev",
            "total_remaining": 200000,
            "budget_count": 2,
            "next_expiry": None,
            "budgets": [
                {
                    "budget_id": "019d3076-2a1f-739b-93e0-e18604846447",
                    "tenant_id": "n8n-dev",
                    "license_id": "069c6c62-f856-7383-9c71-5c9d54200a01",
                    "period": "monthly",
                    "budget_tokens": 100000,
                    "used_tokens": 0,
                    "remaining_tokens": 100000,
                    "usage_pct": 0.0,
                    "over_budget": False,
                    "priority": 100,
                    "starts_at": "2026-04-26T10:00:06.454876Z",
                    "expires_at": None,
                    "source": "manual",
                },
                {
                    "budget_id": "019d30c3-731a-7476-a702-2dfe997c05b4",
                    "tenant_id": "n8n-dev",
                    "license_id": "069c6d9f-88c0-78dd-9c95-b9855df6bd51",
                    "period": "monthly",
                    "budget_tokens": 100000,
                    "used_tokens": 0,
                    "remaining_tokens": 100000,
                    "usage_pct": 0.0,
                    "over_budget": False,
                    "priority": 100,
                    "starts_at": "2026-04-26T10:00:06.454876Z",
                    "expires_at": None,
                    "source": "manual",
                },
            ],
        },
        "meta": {"request_id": "019fea27-b5bd-7181-bc8b-d644f0000000"},
    }
    live_quota = {
        "data": {
            "license_id": "069c6c62-f856-7383-9c71-5c9d54200a01",
            "plan_id": "00000000-0000-7000-8000-000000000001",
            "quota_monthly": None,
            "used_this_month": 0,
            "remaining": None,
            "over_quota": False,
        },
        "meta": {
            "request_id": "019fea28-1e62-7353-bbea-e3c3c88b1bfb",
            "next_cursor": None,
            "total": None,
        },
    }
    c = _client(_resp(live), _resp(live_quota))
    m = MeteringClient(base_url="http://meter")
    d = await m.check_budget(
        c,
        tenant_id="n8n-dev",
        license_id="069c6c62-f856-7383-9c71-5c9d54200a01",
        auth_headers={},
    )
    assert d.allowed is True
    assert d.kind is None
    q = await m.check_quota(
        c,
        tenant_id="n8n-dev",
        license_id="069c6c62-f856-7383-9c71-5c9d54200a01",
        plan_id="P",
        auth_headers={},
    )
    assert q.allowed is True


# --------------------------------------------------------------------------
# The guard's own integrity
# --------------------------------------------------------------------------


def test_guard_cannot_skip_itself_green():
    """No escape hatch may exist in this file. Enforced, not promised.

    Parsed as an AST rather than grepped as text, so the module's own prose
    (which necessarily names the hatches it forbids) cannot trip it and, more
    importantly, cannot HIDE one either: a real ``pytest.skip(...)`` buried in a
    string would be invisible to the AST and a mention in a comment would be a
    false alarm. Only executable constructs count.
    """
    import ast

    tree = ast.parse(pathlib.Path(__file__).read_text())
    banned_calls = {"skip", "importorskip", "xfail", "exit"}
    banned_markers = {"skip", "skipif", "xfail"}
    offences: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in banned_calls:
                offences.append(f"call to .{func.attr}() at line {node.lineno}")
            elif isinstance(func, ast.Name) and func.id in banned_calls:
                offences.append(f"call to {func.id}() at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in banned_markers:
            # pytest.mark.skip / pytest.mark.skipif / pytest.mark.xfail
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "mark":
                offences.append(f"marker .mark.{node.attr} at line {node.lineno}")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    offences.append(f"module-level pytestmark at line {node.lineno}")

    assert not offences, f"contract guard must contain no skip/xfail escape hatch: {offences}"


def test_vendored_spec_is_the_billing_service_and_covers_both_endpoints():
    assert SPEC["info"]["title"] == "Billing Service"
    assert BUDGET_CHECK_PATH in SPEC["paths"]
    assert QUOTA_CHECK_PATH in SPEC["paths"]
    for name in (
        "Envelope_BudgetAggregateStatus_",
        "BudgetAggregateStatus",
        "TokenBudgetStatus",
        "Envelope_QuotaStatus_",
        "QuotaStatus",
        "Meta",
    ):
        assert name in SCHEMAS
