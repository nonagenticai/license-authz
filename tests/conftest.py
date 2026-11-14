"""Test configuration for license-authz service.

Provides:
- Async HTTP client fixture (httpx.AsyncClient with ASGITransport)
- httpx client fixture for direct function tests

license-authz trusts the canonical tenant UUID from introspect directly, so there
is no name-resolution helper to stub here (removed with the tenant-identity refactor).
"""

# Set test env BEFORE any src import — Settings.model_post_init fail-fasts in
# production when required infra fields are empty. Tests import src modules
# that call get_settings() at import time; pin environment=test + provide
# placeholder URLs so respx can intercept against them.
import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "LICENSE_INTROSPECT_URL",
    "http://licenses.example:8544/license/introspect",
)
os.environ.setdefault(
    "LICENSES_ACTIVE_URL",
    "http://licenses.example:8544/licenses/active",
)
os.environ.setdefault(
    "METERING_SERVICE_URL",
    "http://metering.example:8156",
)
# ── host maps ────────────────────────────────────────────────────────────────
# Since 2026-11-14 these are DEPLOYMENT CONFIGURATION: the module ships no
# hostnames, so the suite supplies its own. Every name below is an RFC 2606
# `.example` placeholder — none of them is a real host, which is what lets this
# file live in a public repository.
#
# They are set with `setdefault`, so a developer can export a different map to
# reproduce a deployment's exact values locally.
os.environ.setdefault(
    "APP_HOST_REGISTRY_OVERRIDE",
    "marketing-api.example=marketing,"
    "sales-api.example=sales,"
    "marketing.nonagentic.example=marketing",
)
os.environ.setdefault(
    "METERING_HOST_SERVICE_KEYS_OVERRIDE",
    "control-service.control-service=control-service,"
    "control-api.example=control-service,"
    "marketing-api.example=marketing,"
    "sales-api.example=sales,"
    "marketing.marketing=marketing,"
    "sales.sales=sales,"
    "vllm.example=vllm,"
    "inference-router.example=inference-router",
)
os.environ.setdefault("STRICT_APP_HOSTS_OVERRIDE", "vllm.example=vllm")
os.environ.setdefault("TENANT_PROXY_RETARGET_HOSTS", "licenses.example")
os.environ.setdefault("KEYCLOAK_ISSUER", "https://auth.nonagentic.ai/realms/portal")
os.environ.setdefault(
    "KEYCLOAK_TOKEN_URL",
    "https://auth.nonagentic.ai/realms/portal/protocol/openid-connect/token",
)

from unittest.mock import patch

import httpx
import pytest
import redis.exceptions
from httpx import ASGITransport, AsyncClient

from main import app, lifespan


@pytest.fixture
async def client():
    """Async HTTP client for testing FastAPI endpoints.

    Uses httpx.AsyncClient with ASGITransport for proper async support.
    Wraps with lifespan to ensure app.state is initialized.
    """
    transport = ASGITransport(app=app)
    async with lifespan(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def http_client():
    """Bare httpx.AsyncClient for testing functions directly (e.g. introspect_license)."""
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture(autouse=True)
def _isolate_redis_accessor():
    """Reset the lazy Redis accessor's state around every test, and never dial.

    ``main.app`` is a module singleton, so the accessor's backoff window,
    failure count and connect lock all leak between tests.

    Parking the window at +inf is NOT enough to stop a live socket, and the
    docstring that used to claim it was is what this fixture is fixing:
    ``lifespan`` calls ``_init_redis_state`` (which zeroes the window) and then
    ``_get_redis`` with ``METERING_ENABLED`` defaulting true, so every test
    using ``client``/``healthy_client`` — and every ``async with lifespan(app)``
    written inline in a test — really dialled ``redis://localhost:6379/2``.
    ``tests/test_redis_accessor.py::test_lifespan_never_opens_a_real_connection``
    pins that at the event loop's connection layer.

    So the constructor itself is stubbed to refuse. That is byte-for-byte the
    outcome a real dial already produced in CI (nothing listens on 6379:
    ConnectionError -> ``state.redis is None``, ``redis_failures == 1``, window
    armed) — it just no longer touches the network. Tests that need a different
    connect outcome nest their own ``patch("redis.asyncio.from_url", ...)``,
    which wins over this one and is restored on exit.
    """
    import main

    main._init_redis_state(main.app)
    main.app.state.redis_next_attempt_at = float("inf")
    with patch(
        "redis.asyncio.from_url",
        side_effect=redis.exceptions.ConnectionError("no redis in the unit-test env"),
    ):
        yield
    main._init_redis_state(main.app)
    main.app.state.redis = None


@pytest.fixture(autouse=True)
def _reset_introspect_stale():
    """Clear the module-level stale-while-error store between tests.

    introspect_license records successful active results in a process-global
    dict; without a reset a prior test's success would leak a stale entry and
    flip cold-cache transient-failure assertions.
    """
    import main

    main._introspect_stale.clear()
    yield
    main._introspect_stale.clear()
