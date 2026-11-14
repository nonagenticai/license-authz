"""The metering event names the service CONSUMED (`service_key`).

billing-service maps a path to a service by prefix (SERVICE_KEY_MAP) and files
anything it cannot place under `unmapped:<first path segment>`. Several services
serve the bare `/api` prefix, so the path alone cannot tell them apart; the
emitter knows the destination host and says so.
"""

from unittest.mock import AsyncMock

import pytest
import redis.asyncio
import respx
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

import main
from main import LICENSE_INTROSPECT_URL as INTROSPECT_URL
from main import app, lifespan

TENANT_UUID = "0192a3b4-c5d6-7e8f-9012-345678901234"


CONTROL_SERVICE = "control-service"


def _active_response() -> dict:
    return {
        "active": True,
        "tenant_id": TENANT_UUID,
        "subject_id": "sub_123",
        "license_id": "lic_abc",
        "app": {"id": "app_workers", "key": "workers"},
        "plan": {"id": "premium", "code": "PRM"},
    }


async def _emitted_fields(base_url: str = "http://test", **headers: str) -> dict:
    """One allowed licence-token request on bare `/api`; returns the XADD fields."""
    mock_redis = AsyncMock(spec=redis.asyncio.Redis)
    mock_redis.xadd = AsyncMock()
    respx.post(INTROSPECT_URL).respond(200, json=_active_response())
    async with lifespan(app):
        app.state.redis = mock_redis
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url=base_url) as ac:
                response = await ac.get(
                    "/api/resource",
                    headers={"Authorization": "Bearer valid_token", **headers},
                )
        finally:
            app.state.redis = None
    assert response.status_code == 200
    mock_redis.xadd.assert_awaited_once()
    return mock_redis.xadd.await_args[0][1]


class TestMeteringServiceKey:
    @respx.mock
    async def test_waypoint_host_names_control_service(self):
        """⛔ FAILS WHEN REVERTED: origin/main emitted no `service_key` at all."""
        fields = await _emitted_fields("http://control-service.control-service:8010")
        assert fields["service_key"] == CONTROL_SERVICE
        assert fields["status"] == "allowed"

    @respx.mock
    async def test_gateway_inference_router_host_names_inference_router(self):
        """⛔ FAILS WHEN REVERTED: `/v1/...` on this host was billed `unmapped:v1`."""
        fields = await _emitted_fields(**{"x-forwarded-host": "inference-router.example"})
        assert fields["service_key"] == "inference-router"

    @respx.mock
    @pytest.mark.parametrize(
        "host",
        [
            "control-service.control-service.svc:8010",
            "control-service.control-service",
            "Control-Service.control-service.",
        ],
    )
    async def test_waypoint_host_forms_normalise(self, host):
        fields = await _emitted_fields(Host=host)
        assert fields["service_key"] == CONTROL_SERVICE

    @respx.mock
    @pytest.mark.parametrize(
        ("host", "service_key"),
        [
            ("sales.sales:8004", "sales"),
            ("sales.sales", "sales"),
            ("marketing.marketing:8003", "marketing"),
        ],
    )
    async def test_in_mesh_sales_and_marketing_hosts_are_named(self, host, service_key):
        """⛔ FAILS WHEN REVERTED: without the in-mesh entries these events carry
        no `service_key` and billing files them as `unmapped:api`."""
        fields = await _emitted_fields(Host=host)
        assert fields["service_key"] == service_key

    @respx.mock
    @pytest.mark.parametrize(
        ("forwarded_host", "service_key"),
        [
            ("control-api.example", "control-service"),
            ("marketing-api.example", "marketing"),
            ("sales-api.example", "sales"),
            ("Sales-API.example:443, proxy.example", "sales"),
        ],
    )
    async def test_gateway_forwarded_host_wins_over_host(self, forwarded_host, service_key):
        """On the gateway's httpCall `Host` names license-authz's caller-side
        target, not the consumed service — `x-forwarded-host` is the destination."""
        fields = await _emitted_fields(
            "http://control-service.control-service:8010",
            **{"x-forwarded-host": forwarded_host},
        )
        assert fields["service_key"] == service_key

    @respx.mock
    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"x-forwarded-host": "unknown.example"},
            {"Host": "control-service.other-namespace"},
        ],
    )
    async def test_unknown_host_adds_no_field(self, headers):
        """Never guessed: absent, so billing falls back to its own path mapping
        and a non-matching event carries exactly the fields it did before."""
        fields = await _emitted_fields(**headers)
        assert "service_key" not in fields


POLL_PATH = "/api/v2/leads/intakes/0192a3b4-0000-7000-8000-000000000001"


SKIPPED_METRIC = "license_authz_metering_events_skipped_total"


def _skipped(reason: str, app_key: str) -> float:
    return REGISTRY.get_sample_value(SKIPPED_METRIC, {"reason": reason, "app": app_key}) or 0.0


async def _request(method: str, path: str, **headers: str) -> tuple[int, AsyncMock]:
    """One licence-token request; returns (status, the XADD mock)."""
    mock_redis = AsyncMock(spec=redis.asyncio.Redis)
    mock_redis.xadd = AsyncMock()
    respx.post(INTROSPECT_URL).respond(200, json=_active_response())
    async with lifespan(app):
        app.state.redis = mock_redis
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.request(
                    method, path, headers={"Authorization": "Bearer valid_token", **headers}
                )
        finally:
            app.state.redis = None
    return response.status_code, mock_redis.xadd


class TestLeadBridgePollIsNotMetered:
    """Marketing reads `GET /api/v2/leads/intakes/{id}` on sales once a minute.
    It is authorised exactly as before and never written to the billing stream."""

    def test_skip_series_is_born_at_zero(self):
        labels = {"reason": "lead_bridge_poll", "app": "sales"}
        assert REGISTRY.get_sample_value(SKIPPED_METRIC, labels) is not None

    @respx.mock
    @pytest.mark.parametrize(
        "headers",
        [
            {"Host": "sales.sales:8004"},
            {"Host": "sales.sales"},
            {"x-forwarded-host": "sales-api.example"},
        ],
    )
    @pytest.mark.parametrize("path", [POLL_PATH, POLL_PATH + "?x=1"])
    async def test_poll_is_authorised_and_emits_no_event(self, headers, path):
        """⛔ FAILS WHEN REVERTED: without the `sales` rule the poll is XADDed
        (and billed per request) like any other sales call."""
        before = _skipped("lead_bridge_poll", "sales")
        ui_before = _skipped("gateway_browser_ui", "n8n")
        status, xadd = await _request("GET", path, **headers)
        assert status == 200
        xadd.assert_not_awaited()
        assert _skipped("lead_bridge_poll", "sales") == before + 1
        assert _skipped("gateway_browser_ui", "n8n") == ui_before

    @respx.mock
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/api/v2/leads/intakes"),
            ("GET", POLL_PATH + "/duplicates"),
            ("GET", POLL_PATH + "/"),
            ("GET", POLL_PATH + "/../../handover"),
            ("GET", POLL_PATH + "%2Fduplicates"),
            ("GET", "/api/v2/leads/handover"),
            ("GET", "/api/opportunities"),
            ("POST", POLL_PATH),
            ("POST", POLL_PATH + "/resume"),
            ("POST", "/api/v2/leads/handover"),
            ("HEAD", POLL_PATH),
        ],
    )
    async def test_every_other_sales_request_is_metered_as_sales(self, method, path):
        """NEGATIVE CONTROL: the rule is one method on one whole path."""
        before = _skipped("lead_bridge_poll", "sales")
        status, xadd = await _request(method, path, Host="sales.sales:8004")
        assert status == 200
        xadd.assert_awaited_once()
        fields = xadd.await_args[0][1]
        assert (fields["service_key"], fields["status"]) == ("sales", "allowed")
        assert _skipped("lead_bridge_poll", "sales") == before

    @respx.mock
    @pytest.mark.parametrize(
        ("headers", "service_key"),
        [
            ({"Host": "marketing.marketing:8003"}, "marketing"),
            ({"x-forwarded-host": "control-api.example"}, "control-service"),
            ({"Host": "sales.other-namespace"}, None),
            ({}, None),
            # The gateway's forwarded host is the destination; `Host` does not win.
            ({"x-forwarded-host": "marketing-api.example", "Host": "sales.sales"}, "marketing"),
        ],
    )
    async def test_same_path_on_a_non_sales_host_is_metered(self, headers, service_key):
        """NEGATIVE CONTROL: the rule belongs to `sales`, not to the path."""
        before = _skipped("lead_bridge_poll", "sales")
        status, xadd = await _request("GET", POLL_PATH, **headers)
        assert status == 200
        xadd.assert_awaited_once()
        assert xadd.await_args[0][1].get("service_key") == service_key
        assert _skipped("lead_bridge_poll", "sales") == before

    @respx.mock
    async def test_poll_without_a_valid_licence_is_still_denied(self):
        """Authorisation is untouched: the skip sits after the allow decision."""
        respx.post(INTROSPECT_URL).respond(200, json={"active": False})
        before = _skipped("lead_bridge_poll", "sales")
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.get(
                        POLL_PATH,
                        headers={"Authorization": "Bearer dead_token", "Host": "sales.sales"},
                    )
            finally:
                app.state.redis = None
        assert response.status_code in (401, 402, 403)
        assert _skipped("lead_bridge_poll", "sales") == before


VLLM_REASON = "metered_by_inference_router"
HOST_ROUTE_METRIC = "license_authz_app_host_route_total"


def _host_route(result: str) -> float:
    labels = {"result": result, "host": "vllm.example", "app": "vllm"}
    return REGISTRY.get_sample_value(HOST_ROUTE_METRIC, labels) or 0.0


def _licence(app_key: str) -> dict:
    return {**_active_response(), "app": {"id": f"app_{app_key}", "key": app_key}}


async def _licence_request(
    method: str, path: str, app_key: str, *, token: str | None = None, **headers: str
) -> tuple[int, AsyncMock]:
    """One request whose licence is for `app_key`; returns (status, the XADD mock).

    The bearer is unique per app so a cached introspection of one app's licence
    can never answer for another's.
    """
    mock_redis = AsyncMock(spec=redis.asyncio.Redis)
    mock_redis.xadd = AsyncMock()
    respx.post(INTROSPECT_URL).respond(200, json=_licence(app_key))
    auth = {} if token == "" else {"Authorization": f"Bearer {token or 'tok_' + app_key}"}
    async with lifespan(app):
        app.state.redis = mock_redis
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.request(method, path, headers={**auth, **headers})
        finally:
            app.state.redis = None
    return response.status_code, mock_redis.xadd


def _allowed_events(xadd: AsyncMock) -> list[dict]:
    return [c[0][1] for c in xadd.await_args_list if c[0][1].get("status") == "allowed"]


VLLM_CALLS = [
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"),
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions?stream=true"),
    ("GET", "/vllm/anything"),
]
VLLM_HOSTS = [
    {"x-forwarded-host": "vllm.example"},
    {"x-forwarded-host": "VLLM.example:443"},
    {"x-forwarded-host": "vllm.example."},
    {"Host": "vllm.example"},
]


class TestVllmIsStrictlyBoundAndNotMetered:
    """`vllm.example` accepts ONLY a `vllm` licence (inference-router's), on
    every path, whatever APP_HOST_ROUTE_MODE says; and a call it accepts is
    never written to the billing stream, because the router meters it per end
    tenant itself."""

    def test_series_are_born_at_zero(self):
        assert (
            REGISTRY.get_sample_value(SKIPPED_METRIC, {"reason": VLLM_REASON, "app": "vllm"})
            is not None
        )
        for result in ("strict_matched", "strict_mismatch_denied"):
            labels = {"result": result, "host": "vllm.example", "app": "vllm"}
            assert REGISTRY.get_sample_value(HOST_ROUTE_METRIC, labels) is not None

    def test_every_unmetered_service_is_strictly_bound(self):
        """A whole-service skip without a strict binding would be free GPU for
        any licence; the skip may only exist for a strictly bound app."""
        assert set(main.METERING_UNMETERED_SERVICES) <= set(main.STRICT_APP_HOSTS.values())

    @respx.mock
    @pytest.mark.parametrize("headers", VLLM_HOSTS)
    @pytest.mark.parametrize(("method", "path"), VLLM_CALLS)
    async def test_vllm_licence_is_allowed_and_emits_no_event(self, headers, method, path):
        """⛔ FAILS WHEN THE SKIP IS REVERTED: the call is XADDed and the router's
        own cost event double-counts it."""
        before = _skipped(VLLM_REASON, "vllm")
        matched = _host_route("strict_matched")
        poll_before = _skipped("lead_bridge_poll", "sales")
        status, xadd = await _licence_request(method, path, "vllm", **headers)
        assert status == 200
        xadd.assert_not_awaited()
        assert _skipped(VLLM_REASON, "vllm") == before + 1
        assert _host_route("strict_matched") == matched + 1
        assert _skipped("lead_bridge_poll", "sales") == poll_before

    @respx.mock
    @pytest.mark.parametrize("app_key", ["kanban", "workers", "inference-router", "", "vllm2"])
    @pytest.mark.parametrize("headers", VLLM_HOSTS)
    @pytest.mark.parametrize(("method", "path"), VLLM_CALLS)
    async def test_other_app_licence_is_denied_and_never_skipped(
        self, app_key, headers, method, path
    ):
        """⛔ FAILS WHEN THE STRICT CHECK IS REMOVED: any active licence of any
        tenant would get GPU time that nobody accounts for."""
        skipped = _skipped(VLLM_REASON, "vllm")
        denied = _host_route("strict_mismatch_denied")
        status, xadd = await _licence_request(method, path, app_key, **headers)
        assert status == 403
        assert _allowed_events(xadd) == []
        assert _skipped(VLLM_REASON, "vllm") == skipped
        if path.startswith("/v1/"):
            # `/vllm/` is a registered prefix, so it is refused one step earlier
            # (app_route_mismatch) before the strict binding is consulted.
            assert _host_route("strict_mismatch_denied") == denied + 1

    @respx.mock
    @pytest.mark.parametrize("mode", ["off", "observe"])
    async def test_strict_binding_ignores_the_global_host_mode(self, mode, monkeypatch):
        monkeypatch.setattr(main, "APP_HOST_ROUTE_MODE", mode)
        status, _ = await _licence_request(
            "POST", "/v1/chat/completions", "kanban", **{"x-forwarded-host": "vllm.example"}
        )
        assert status == 403

    @respx.mock
    async def test_no_token_is_401(self):
        status, xadd = await _licence_request(
            "POST",
            "/v1/chat/completions",
            "vllm",
            token="",
            **{"x-forwarded-host": "vllm.example"},
        )
        assert status == 401
        assert _allowed_events(xadd) == []

    @respx.mock
    async def test_dead_licence_is_denied(self):
        respx.post(INTROSPECT_URL).respond(200, json={"active": False})
        before = _skipped(VLLM_REASON, "vllm")
        mock_redis = AsyncMock(spec=redis.asyncio.Redis)
        mock_redis.xadd = AsyncMock()
        async with lifespan(app):
            app.state.redis = mock_redis
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    response = await ac.post(
                        "/v1/chat/completions",
                        headers={
                            "Authorization": "Bearer dead_vllm_token",
                            "x-forwarded-host": "vllm.example",
                        },
                    )
            finally:
                app.state.redis = None
        assert response.status_code in (401, 402, 403)
        assert _skipped(VLLM_REASON, "vllm") == before

    @respx.mock
    @pytest.mark.parametrize(
        ("headers", "service_key"),
        [
            ({"x-forwarded-host": "sales-api.example"}, "sales"),
            ({"x-forwarded-host": "marketing-api.example"}, "marketing"),
            ({"x-forwarded-host": "control-api.example"}, "control-service"),
            ({"x-forwarded-host": "vllm.example.evil"}, None),
            ({"x-forwarded-host": "notvllm.example"}, None),
            ({}, None),
            # The gateway's forwarded host is the destination; `Host` does not win.
            ({"x-forwarded-host": "sales-api.example", "Host": "vllm.example"}, "sales"),
        ],
    )
    @pytest.mark.parametrize("app_key", ["kanban", "vllm"])
    async def test_other_hosts_are_unaffected(self, headers, service_key, app_key):
        """NEGATIVE CONTROL: off the vllm host, any licence still passes an
        unregistered path exactly as on main, and is metered (even a `vllm` one)."""
        skipped = _skipped(VLLM_REASON, "vllm")
        status, xadd = await _licence_request("POST", "/v1/chat/completions", app_key, **headers)
        assert status == 200
        xadd.assert_awaited_once()
        assert xadd.await_args is not None
        fields = xadd.await_args[0][1]
        assert fields.get("service_key") == service_key
        assert fields["status"] == "allowed"
        assert _skipped(VLLM_REASON, "vllm") == skipped


def test_skip_requires_the_strict_binding_not_just_the_host():
    """The whole-service skip is keyed on the app the strict binding ALLOWED,
    never on the destination host alone: an unbound vllm.example request is
    not unmetered, even if some future path reached the emit without the deny."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"x-forwarded-host", b"vllm.example")],
    }
    request = Request(scope)
    route = main._unmetered_route
    assert route(request, "POST", "/v1/chat/completions", strict_bound_app="") is None
    assert route(request, "POST", "/v1/chat/completions", strict_bound_app="vllm") == (
        VLLM_REASON,
        "vllm",
    )
