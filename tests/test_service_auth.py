"""Unit tests for ServiceTokenProvider (Keycloak SA token fetcher)."""

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest
import respx
from prometheus_client import REGISTRY

from service_auth import ServiceTokenProvider

TOKEN_URL = "https://auth.example.test/realms/portal/protocol/openid-connect/token"

SA_METRIC = "license_authz_sa_token_total"
# The label set is a PARTITION of get_token() calls: exactly one moves per call.
# Listed here so every test can assert the whole partition, not just its own
# label — a test that checks only the label it cares about cannot catch a mutant
# that ALSO increments a second one.
ALL_OUTCOMES = ("ok", "cached", "unconfigured", "mint_failed", "malformed_response")
# A recognisable canary. Every observability test that supplies a secret uses
# THIS string, and asserts it never reaches an emitted log line.
SECRET_CANARY = "sa-client-secret-CANARY-must-never-be-logged"
TOKEN_CANARY = "minted-jwt-CANARY-must-never-be-logged"
BODY_CANARY = "response-body-CANARY-must-never-be-logged"


def _sa_count(outcome: str) -> float:
    """Current value of license_authz_sa_token_total{outcome=...} (0.0 if unborn)."""
    return REGISTRY.get_sample_value(SA_METRIC, {"outcome": outcome}) or 0.0


def _snapshot() -> dict[str, float]:
    """Every outcome's current value, so deltas can be asserted exhaustively."""
    return {outcome: _sa_count(outcome) for outcome in ALL_OUTCOMES}


def _assert_deltas(before: dict[str, float], **expected: float) -> None:
    """Assert EVERY outcome moved by exactly the named delta; unnamed ones by 0."""
    unknown = set(expected) - set(ALL_OUTCOMES)
    assert not unknown, f"test names outcomes that do not exist: {sorted(unknown)}"
    for outcome in ALL_OUTCOMES:
        want = before[outcome] + expected.get(outcome, 0.0)
        got = _sa_count(outcome)
        assert got == want, f"{SA_METRIC}{{outcome={outcome!r}}}: expected {want}, got {got}"


def _autospec_client() -> AsyncMock:
    """A double that CANNOT be narrower than the real httpx.AsyncClient.

    ``create_autospec`` mirrors the real signatures recursively, so a call this
    provider makes with arguments the real client would reject fails here too —
    a plain ``AsyncMock`` would happily accept anything.
    """
    return create_autospec(httpx.AsyncClient, instance=True)


def _reject_json_constant(name: str):
    """json.loads hook: refuse the non-standard Infinity / -Infinity / NaN.

    Python's json ACCEPTS those literals by default; Loki's parser does not. The
    whole design rests on the log being the operator channel, so a line that a
    strict parser rejects is a broken line. Wiring this into `_events` makes
    every test that reads an event also a strict-JSON validity assertion.
    """
    raise AssertionError(f"emitted log line is not strict JSON: bare {name!r} literal")


def _events(caplog) -> list[tuple[str, int, dict]]:
    """(event, levelno, payload) for every record emitted by service_auth.

    Parsed STRICTLY — see ``_reject_json_constant``.
    """
    out: list[tuple[str, int, dict]] = []
    for rec in caplog.records:
        if rec.name != "service_auth":
            continue
        payload = json.loads(rec.getMessage(), parse_constant=_reject_json_constant)
        out.append((payload["event"], rec.levelno, payload))
    return out


def _emitted_text(caplog) -> str:
    """Every character service_auth emitted, for canary-absence assertions."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "service_auth")


@pytest.mark.asyncio
class TestServiceTokenProvider:
    async def test_not_configured_returns_none(self):
        """No client_secret → not configured → get_token() returns None."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="")
        assert provider.configured is False
        client = AsyncMock(spec=httpx.AsyncClient)
        out = await provider.get_token(client)
        assert out is None
        client.post.assert_not_called()

    async def test_cached_token_skips_http(self):
        """Token within expiry returned from cache without HTTP call."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        provider._token = "cached-token"
        provider._expires_at = time.monotonic() + 1000

        client = AsyncMock(spec=httpx.AsyncClient)
        out = await provider.get_token(client)
        assert out == "cached-token"
        client.post.assert_not_called()

    async def test_successful_fetch_caches_with_expiry(self):
        """Successful 200 caches token, sets expires_at = monotonic + expires_in - 30."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = httpx.Response(
            200,
            json={"access_token": "fresh-token", "expires_in": 300},
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = time.monotonic()
        out = await provider.get_token(client)
        after = time.monotonic()

        assert out == "fresh-token"
        assert provider._token == "fresh-token"
        # expires_at should be roughly monotonic + 300 - 30 = +270
        assert before + 270 - 1 <= provider._expires_at <= after + 270 + 1

        # Verify form body
        call = client.post.call_args
        assert call.args[0] == TOKEN_URL
        data = call.kwargs["data"]
        assert data["grant_type"] == "client_credentials"
        assert data["client_id"] == "cid"
        assert data["client_secret"] == "sec"

    async def test_default_expires_in_when_missing(self):
        """When response has no expires_in, defaults to 300 (so cache valid ~270s)."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = httpx.Response(
            200,
            json={"access_token": "tok"},
            request=httpx.Request("POST", TOKEN_URL),
        )
        before = time.monotonic()
        out = await provider.get_token(client)
        assert out == "tok"
        # Default expires_in=300 → expires_at ~ now + 270
        assert provider._expires_at >= before + 269

    async def test_non_200_returns_none(self):
        """Non-200 response → returns None, doesn't raise, doesn't cache."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = httpx.Response(
            401,
            json={"error": "invalid_client"},
            request=httpx.Request("POST", TOKEN_URL),
        )
        out = await provider.get_token(client)
        assert out is None
        assert provider._token is None

    async def test_http_error_returns_none(self):
        """httpx.HTTPError → returns None, doesn't raise."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.side_effect = httpx.ConnectError("refused")
        out = await provider.get_token(client)
        assert out is None
        assert provider._token is None

    async def test_auth_header_empty_when_no_token(self):
        """auth_header() returns {} when get_token() returns None."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="")
        client = AsyncMock(spec=httpx.AsyncClient)
        headers = await provider.auth_header(client)
        assert headers == {}

    async def test_auth_header_with_bearer_when_token_present(self):
        """auth_header() returns {Authorization: Bearer X} when token available."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        provider._token = "abc"
        provider._expires_at = time.monotonic() + 1000
        client = AsyncMock(spec=httpx.AsyncClient)
        headers = await provider.auth_header(client)
        assert headers == {"Authorization": "Bearer abc"}

    @respx.mock
    async def test_owns_client_path_creates_and_closes_client(self):
        """When called with client=None, provider creates and closes its own AsyncClient."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        respx.post(TOKEN_URL).respond(200, json={"access_token": "minted", "expires_in": 60})

        close_calls: list[int] = []
        real_async_client = httpx.AsyncClient

        class _SpyClient:
            def __init__(self, *args, **kwargs):
                self._inner = real_async_client(*args, **kwargs)

            async def post(self, *args, **kwargs):
                return await self._inner.post(*args, **kwargs)

            async def aclose(self):
                close_calls.append(1)
                await self._inner.aclose()

        with patch("service_auth.httpx.AsyncClient", _SpyClient):
            out = await provider.get_token(None)

        assert out == "minted"
        assert len(close_calls) == 1

    @respx.mock
    async def test_owns_client_path_closes_on_error(self):
        """Owns-client path closes its client even on HTTP error."""
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")
        respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectError("nope"))

        close_calls: list[int] = []
        real_async_client = httpx.AsyncClient

        class _SpyClient:
            def __init__(self, *args, **kwargs):
                self._inner = real_async_client(*args, **kwargs)

            async def post(self, *args, **kwargs):
                return await self._inner.post(*args, **kwargs)

            async def aclose(self):
                close_calls.append(1)
                await self._inner.aclose()

        with patch("service_auth.httpx.AsyncClient", _SpyClient):
            out = await provider.get_token(None)

        assert out is None
        assert len(close_calls) == 1

    async def test_env_var_defaults(self, monkeypatch):
        """Constructor reads env vars when args not provided."""
        monkeypatch.setenv("KEYCLOAK_TOKEN_URL", "https://x.test/token")
        monkeypatch.setenv("LICENSE_AUTHZ_SA_CLIENT_ID", "envclient")
        monkeypatch.setenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", "envsecret")
        provider = ServiceTokenProvider()
        assert provider._token_url == "https://x.test/token"
        assert provider._client_id == "envclient"
        assert provider._client_secret == "envsecret"
        assert provider.configured is True


@pytest.mark.asyncio
class TestServiceTokenObservability:
    """The three silent failure modes must be distinguishable.

    Every assertion here pins BOTH the log LEVEL and the metric LABEL, not just
    that some event fired: a mutant that leaves an event at the wrong level
    survives a name-only assertion (fleet incident 2026-08-11).
    """

    async def test_unconfigured_warns_once_but_counts_every_call(self, caplog):
        """unconfigured: WARNING exactly once (hot path), counter every call."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="")
        client = _autospec_client()

        before = _snapshot()

        assert await provider.get_token(client) is None
        assert await provider.get_token(client) is None
        assert await provider.get_token(client) is None

        # Behaviour unchanged, and Keycloak was never dialled.
        client.post.assert_not_called()

        # Label moved by exactly the number of calls — and ONLY that label.
        _assert_deltas(before, unconfigured=3)

        events = _events(caplog)
        assert [e[0] for e in events] == ["sa_token_unconfigured"], (
            f"expected exactly one latched unconfigured line, got {events}"
        )
        name, levelno, payload = events[0]
        assert levelno == logging.WARNING, f"unconfigured must be WARNING, got {levelno}"
        assert payload["client_secret_set"] is False
        assert payload["client_id"] == "cid"

    async def test_mint_failed_on_non_200_warns_every_time_with_the_status(self, caplog):
        """Keycloak refusal: WARNING on EVERY call (not latched), status included."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            401,
            json={"error": "invalid_client", "error_description": "Invalid client credentials"},
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = _snapshot()

        assert await provider.get_token(client) is None
        assert await provider.get_token(client) is None

        _assert_deltas(before, mint_failed=2)

        events = _events(caplog)
        assert [e[0] for e in events] == ["sa_token_mint_failed", "sa_token_mint_failed"], (
            f"non-200 must warn on every call, got {events}"
        )
        for _name, levelno, payload in events:
            assert levelno == logging.WARNING, f"mint_failed must be WARNING, got {levelno}"
            assert payload["reason"] == "http_status"
            assert payload["status"] == 401

        # Positive control: output is non-empty and the canary is absent from it.
        text = _emitted_text(caplog)
        assert "sa_token_mint_failed" in text
        assert SECRET_CANARY not in text

    async def test_mint_failed_on_transport_error_warns_with_the_exception_class(self, caplog):
        """Network failure: WARNING naming the exception CLASS, never str(exc)."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.side_effect = httpx.ConnectError("connection refused")

        before = _snapshot()

        assert await provider.get_token(client) is None

        _assert_deltas(before, mint_failed=1)

        events = _events(caplog)
        assert [e[0] for e in events] == ["sa_token_mint_failed"], f"got {events}"
        _name, levelno, payload = events[0]
        assert levelno == logging.WARNING, f"transport mint_failed must be WARNING, got {levelno}"
        assert payload["reason"] == "transport_error"
        assert payload["error_class"] == "ConnectError"

        text = _emitted_text(caplog)
        assert "sa_token_mint_failed" in text
        assert SECRET_CANARY not in text

    async def test_successful_mint_counts_ok_logs_debug_and_never_emits_the_token(self, caplog):
        """Success: `ok` label, DEBUG level, and neither secret nor token in the log."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            json={"access_token": TOKEN_CANARY, "expires_in": 300},
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = _snapshot()

        assert await provider.get_token(client) == TOKEN_CANARY

        _assert_deltas(before, ok=1)

        events = _events(caplog)
        assert [e[0] for e in events] == ["sa_token_minted"], f"got {events}"
        _name, levelno, payload = events[0]
        assert levelno == logging.DEBUG, f"success must be DEBUG, got {levelno}"
        assert payload["expires_in"] == 300

        # Positive control: the log really was captured, and holds no material.
        text = _emitted_text(caplog)
        assert "sa_token_minted" in text
        assert SECRET_CANARY not in text
        assert TOKEN_CANARY not in text

    async def test_cache_hit_counts_cached_and_stays_silent(self, caplog):
        """A cache hit is its own label, and emits nothing on the hot path."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        provider._token = TOKEN_CANARY
        provider._expires_at = time.monotonic() + 1000
        client = _autospec_client()

        before = _snapshot()

        assert await provider.get_token(client) == TOKEN_CANARY

        client.post.assert_not_called()
        _assert_deltas(before, cached=1)
        assert _events(caplog) == []

    async def test_auth_header_is_still_empty_on_every_failure_mode(self, caplog):
        """Observability only: the public contract of auth_header() is unchanged."""
        caplog.set_level(logging.DEBUG, logger="service_auth")

        unconfigured = ServiceTokenProvider(token_url=TOKEN_URL, client_id="c", client_secret="")
        assert await unconfigured.auth_header(_autospec_client()) == {}

        refused = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="c", client_secret=SECRET_CANARY
        )
        refused_client = _autospec_client()
        refused_client.post.return_value = httpx.Response(
            503, text="upstream down", request=httpx.Request("POST", TOKEN_URL)
        )
        assert await refused.auth_header(refused_client) == {}

        broken = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="c", client_secret=SECRET_CANARY
        )
        broken_client = _autospec_client()
        broken_client.post.side_effect = httpx.ReadTimeout("too slow")
        assert await broken.auth_header(broken_client) == {}

        malformed = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="c", client_secret=SECRET_CANARY
        )
        malformed_client = _autospec_client()
        malformed_client.post.return_value = httpx.Response(
            200, text=BODY_CANARY, request=httpx.Request("POST", TOKEN_URL)
        )
        assert await malformed.auth_header(malformed_client) == {}

        # All four surfaced, distinguishably — and none of them leaked.
        events = _events(caplog)
        assert [(e[0], e[1]) for e in events] == [
            ("sa_token_unconfigured", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
        ], f"every failure mode must surface at WARNING, got {events}"
        assert events[1][2]["reason"] == "http_status"
        assert events[1][2]["status"] == 503
        assert events[2][2]["reason"] == "transport_error"
        assert events[2][2]["error_class"] == "ReadTimeout"
        assert events[3][2]["reason"] == "malformed_response"
        text = _emitted_text(caplog)
        assert "sa_token_unconfigured" in text
        assert SECRET_CANARY not in text
        assert BODY_CANARY not in text


@pytest.mark.asyncio
class TestMalformed200DoesNotEscape:
    """A 200 that is not a token must resolve to None, not raise.

    THE CRASH THIS PINS. `resp.json()` raises json.JSONDecodeError (a
    ValueError) and `body["access_token"]` raises KeyError/TypeError. None of
    those is an httpx.HTTPError, so before 2026-08-11 they escaped get_token()
    -> escaped auth_header() -> escaped _handle_jwt_mode (unwrapped at
    main.py:1569) -> escaped the ext_authz handler -> 500, and Envoy converts a
    500 from ext_authz into a DENY for EVERY request on the platform.

    No pre-existing test fed a 200 whose body is not a token — every 200 in the
    original suite was well-formed — so the suite could not have caught it.
    """

    async def test_a_proxys_html_page_with_a_200_does_not_raise(self, caplog):
        """Not-JSON-at-all: the classic 'a proxy answered instead of Keycloak'."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            text=f"<html><body>502 Bad Gateway {BODY_CANARY}</body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = _snapshot()

        # Does not raise, and folds into the same public contract as every
        # other failure mode.
        assert await provider.get_token(client) is None
        assert await provider.auth_header(client) == {}
        # Nothing is cached from a body we could not parse.
        assert provider._token is None
        assert provider._expires_at == 0.0

        _assert_deltas(before, malformed_response=2)

        events = _events(caplog)
        assert [(e[0], e[1]) for e in events] == [
            ("sa_token_mint_failed", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
        ], f"malformed 200 must WARN on every call, got {events}"
        payload = events[0][2]
        assert payload["reason"] == "malformed_response"
        assert payload["error_class"] == "JSONDecodeError"
        # Shape hints only.
        assert payload["body_bytes"] == len(client.post.return_value.content)
        assert payload["content_type"] == "text/html"

        # Positive control, then the absence assertions it makes meaningful.
        text = _emitted_text(caplog)
        assert "malformed_response" in text
        assert BODY_CANARY not in text
        assert SECRET_CANARY not in text

    async def test_a_200_whose_json_lacks_access_token_does_not_raise(self, caplog):
        """Keycloak answered OIDC-shaped JSON, but not a token (wrong grant)."""
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            json={"token_type": "Bearer", "scope": BODY_CANARY, "expires_in": 300},
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = _snapshot()

        assert await provider.get_token(client) is None
        assert await provider.auth_header(client) == {}
        assert provider._token is None
        assert provider._expires_at == 0.0

        _assert_deltas(before, malformed_response=2)

        events = _events(caplog)
        assert [(e[0], e[1]) for e in events] == [
            ("sa_token_mint_failed", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
        ], f"got {events}"
        payload = events[0][2]
        assert payload["reason"] == "malformed_response"
        assert payload["error_class"] == "KeyError"

        text = _emitted_text(caplog)
        assert "malformed_response" in text
        assert BODY_CANARY not in text
        assert SECRET_CANARY not in text

    @pytest.mark.parametrize(
        ("label", "response_kwargs", "error_class"),
        [
            # `json=None` is "no json argument" to httpx, so the null body is
            # written literally — an empty 200 is a separate case below.
            ("json_null", {"content": b"null"}, "TypeError"),
            ("empty_body", {"content": b""}, "JSONDecodeError"),
            ("json_list", {"json": ["access_token", "nope"]}, "TypeError"),
            ("json_string", {"json": "just-a-string"}, "TypeError"),
            ("json_number", {"json": 42}, "TypeError"),
            ("token_not_a_string", {"json": {"access_token": 12345}}, "TypeError"),
            ("token_empty_string", {"json": {"access_token": ""}}, "TypeError"),
            (
                "expires_in_not_a_number",
                {"json": {"access_token": "t", "expires_in": "300"}},
                "TypeError",
            ),
            (
                # The OverflowError family. A legal JSON integer too large for
                # float — this is EXACTLY what escaped a (TypeError, ValueError)
                # catch in metering_client and shipped as finding N1 of PR #37.
                "expires_in_overflows_float",
                {"json": {"access_token": "t", "expires_in": 10**400}},
                "OverflowError",
            ),
            ("undecodable_bytes", {"content": b"\xff\xfe\x00not-utf8"}, "UnicodeDecodeError"),
            # The non-finite family. Python's json.loads accepts these
            # non-standard literals BY DEFAULT and returns real floats, so
            # nothing downstream raises for them — they are caught only because
            # expires_in is validated explicitly. Written as raw bytes so the
            # case does not depend on httpx's serialiser emitting them.
            (
                "expires_in_infinity",
                {"content": b'{"access_token":"tok","expires_in":Infinity}'},
                "TypeError",
            ),
            (
                "expires_in_negative_infinity",
                {"content": b'{"access_token":"tok","expires_in":-Infinity}'},
                "TypeError",
            ),
            (
                "expires_in_nan",
                {"content": b'{"access_token":"tok","expires_in":NaN}'},
                "TypeError",
            ),
            (
                # True IS an int in Python — the trap main.py::_budget_cap
                # guards and finding N2 of PR #37.
                "expires_in_bool_true",
                {"content": b'{"access_token":"tok","expires_in":true}'},
                "TypeError",
            ),
        ],
    )
    async def test_malformed_200_corpus_never_escapes(
        self, caplog, label, response_kwargs, error_class
    ):
        """A CORPUS, not a proof of universality — but every member is reachable.

        Each of these is a real 200 body that reaches the same parse. The point
        is that the catch tuple (ValueError, KeyError, TypeError, OverflowError)
        covers every family this parse can raise, including the two that a naive
        `except (TypeError, ValueError)` or `except json.JSONDecodeError` misses.
        """
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200, request=httpx.Request("POST", TOKEN_URL), **response_kwargs
        )

        before = _snapshot()

        assert await provider.get_token(client) is None, f"{label} raised or returned a token"
        assert provider._token is None, f"{label} cached something"
        assert provider._expires_at == 0.0, f"{label} armed an expiry from a body it rejected"

        _assert_deltas(before, malformed_response=1)

        events = _events(caplog)
        assert [(e[0], e[1]) for e in events] == [("sa_token_mint_failed", logging.WARNING)], (
            f"{label}: got {events}"
        )
        assert events[0][2]["reason"] == "malformed_response"
        assert events[0][2]["error_class"] == error_class, f"{label}: wrong exception family"
        assert SECRET_CANARY not in _emitted_text(caplog)

    async def test_an_infinite_expires_in_cannot_pin_the_token_for_the_pods_life(self, caplog):
        """The CONSEQUENCE, not just the label: no unexpiring cache entry.

        `{"expires_in": Infinity}` parses to inf with no exception anywhere, so
        `_expires_at = inf` would make `monotonic() < _expires_at` true forever.
        The token would be pinned for the pod's lifetime; once the real SA JWT
        expired, /licenses/active would 401 -> httpx.HTTPStatusError is not
        raised here but the resolver's failure surfaces as licenses_unavailable
        503 on every JWT-mode request until a restart. Nothing else in the file
        would have caught it — the arithmetic succeeds.
        """
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            content=b'{"access_token":"' + TOKEN_CANARY.encode() + b'","expires_in":Infinity}',
            request=httpx.Request("POST", TOKEN_URL),
        )

        before = _snapshot()

        assert await provider.get_token(client) is None
        assert provider._token is None
        assert provider._expires_at == 0.0

        # The second call must really re-attempt: if anything had been cached,
        # the cache branch would short-circuit and post would not be called
        # again. This is the assertion that distinguishes "rejected" from
        # "rejected but still poisoned the cache".
        assert await provider.get_token(client) is None
        assert client.post.call_count == 2

        _assert_deltas(before, malformed_response=2)
        assert [(e[0], e[1]) for e in _events(caplog)] == [
            ("sa_token_mint_failed", logging.WARNING),
            ("sa_token_mint_failed", logging.WARNING),
        ]
        assert TOKEN_CANARY not in _emitted_text(caplog)

    async def test_the_logged_content_type_is_capped_at_64_chars(self, caplog):
        """The content-type is proxy-controlled — the only untrusted field here.

        Without the cap an upstream can write an arbitrarily long string into
        every log line on this path.
        """
        caplog.set_level(logging.DEBUG, logger="service_auth")
        provider = ServiceTokenProvider(
            token_url=TOKEN_URL, client_id="cid", client_secret=SECRET_CANARY
        )
        long_content_type = "text/html; charset=" + "A" * 500
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            text="not json",
            headers={"content-type": long_content_type},
            request=httpx.Request("POST", TOKEN_URL),
        )

        assert await provider.get_token(client) is None

        payload = _events(caplog)[0][2]
        assert len(payload["content_type"]) == 64, "the [:64] cap is gone"
        assert payload["content_type"] == long_content_type[:64]
        # Nothing beyond the cap reached the output.
        assert "A" * 46 not in _emitted_text(caplog)


def test_the_strict_json_guard_would_actually_reject_a_bad_line():
    """Positive control for _events' strictness — a guard that cannot fire is not a guard.

    Every assertion about emitted JSON validity runs through `_events`, so if
    `parse_constant` were inert those assertions would be vacuous.
    """
    with pytest.raises(AssertionError, match="not strict JSON"):
        json.loads('{"expires_in": Infinity}', parse_constant=_reject_json_constant)
    # And the control: a well-formed line parses cleanly through the same path.
    assert json.loads('{"expires_in": 300}', parse_constant=_reject_json_constant) == {
        "expires_in": 300
    }


@pytest.mark.asyncio
class TestKeepWarm:
    """The background refresher: it renews, survives failures, and stops promptly."""

    @staticmethod
    def _provider() -> ServiceTokenProvider:
        return ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="sec")

    @staticmethod
    async def _wait_for_calls(client: AsyncMock, count: int, deadline_s: float = 2.0) -> None:
        """Wait until the loop has made ``count`` HTTP attempts (bounded)."""
        deadline = time.monotonic() + deadline_s
        while client.post.await_count < count and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert client.post.await_count >= count, "the refresher did not tick"

    async def test_renews_repeatedly_and_stops(self):
        """expires_in=1 makes the cache expire ~29 s in the past, so every tick mints."""
        provider = self._provider()
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            json={"access_token": "t", "expires_in": 1},
            request=httpx.Request("POST", TOKEN_URL),
        )
        stop = asyncio.Event()
        task = asyncio.create_task(provider.keep_warm(client, 1.0, stop))

        await self._wait_for_calls(client, 3, deadline_s=4.0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert not task.cancelled()

    async def test_a_failing_refresh_does_not_kill_the_loop(self):
        """A dead Keycloak must not end the background task; the next tick retries."""
        provider = self._provider()
        client = _autospec_client()
        client.post.side_effect = httpx.ConnectError("boom")
        stop = asyncio.Event()
        task = asyncio.create_task(provider.keep_warm(client, 1.0, stop))

        await self._wait_for_calls(client, 3, deadline_s=4.0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert not task.cancelled()

    async def test_stop_event_ends_a_long_interval_promptly(self):
        """Shutdown must not wait out the 200 s interval."""
        provider = self._provider()
        client = _autospec_client()
        client.post.return_value = httpx.Response(
            200,
            json={"access_token": "t", "expires_in": 300},
            request=httpx.Request("POST", TOKEN_URL),
        )
        stop = asyncio.Event()
        task = asyncio.create_task(provider.keep_warm(client, 60.0, stop))

        await self._wait_for_calls(client, 1)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_an_unconfigured_provider_never_reaches_http(self):
        provider = ServiceTokenProvider(token_url=TOKEN_URL, client_id="cid", client_secret="")
        client = _autospec_client()
        stop = asyncio.Event()
        task = asyncio.create_task(provider.keep_warm(client, 0.01, stop))

        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        client.post.assert_not_called()


async def test_lifespan_starts_and_stops_the_refresher_when_configured(monkeypatch):
    """The lifespan wiring: a configured provider gets a live task and shutdown ends it."""
    from main import app, lifespan

    monkeypatch.setenv("KEYCLOAK_TOKEN_URL", "http://127.0.0.1:9/token")
    monkeypatch.setenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", "sec")
    async with lifespan(app):
        task = app.state.service_auth_task
        assert task is not None, "a configured provider must be kept warm"
        assert not task.done()
    assert task.done(), "shutdown must stop the refresher"


async def test_lifespan_does_not_start_the_refresher_when_unconfigured(monkeypatch):
    """No SA credential: the tick would only pollute metrics and logs, so it must not run."""
    from main import app, lifespan

    monkeypatch.delenv("LICENSE_AUTHZ_SA_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("KEYCLOAK_TOKEN_URL", raising=False)
    async with lifespan(app):
        assert app.state.service_auth_task is None
