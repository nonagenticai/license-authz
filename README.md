# License AuthZ

[![CI](https://github.com/nonagenticai/license-authz/actions/workflows/ci-public.yml/badge.svg)](https://github.com/nonagenticai/license-authz/actions/workflows/ci-public.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/license-authz)](https://github.com/nonagenticai/license-authz/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

HTTP ext_authz service for Envoy/Istio that validates license tokens and injects trusted tenant
headers. It listens on port 9001.

| | |
|---|---|
| **Runtime** | Python 3.14 · FastAPI · httpx |
| **Depends on** | licences service (`/license/introspect`, `/licenses/active`, `/internal/tenants`); Keycloak (JWKS and service-account token endpoint); billing-service (budget and quota checks); Redis (metering events) |
| **Runs on** | Kubernetes, `istio-system` namespace; container port 9001 |

## What License AuthZ Does

When an external request arrives at the Istio IngressGateway carrying an `Authorization: Bearer <token>`
header, Envoy's ext_authz filter forwards the request headers to this service before the request reaches
the upstream. License AuthZ establishes the caller's tenant and licence through one of two paths:

- **Opaque licence tokens** are validated by calling the licences service's `/license/introspect`
  endpoint. This is the default path.
- **Keycloak JWTs** use JWT mode (`LICENSE_AUTHZ_JWT_MODE_ENABLED=true`): the token is verified against
  the configured issuer's JWKS and resolved to an active licence through `/licenses/active`.

There is no JWT passthrough and no trusted-network bypass: an end-user request is allowed only after one
of those checks succeeds. Separately, when `SERVICE_PRINCIPAL_AUTH_ENABLED=true`, a verified secure-apps
service-account token in `X-Caller-Authorization` can be admitted as an identity-only caller with empty
tenant/licence headers; that path is off by default.

If the credential resolves to an active licence, the service returns HTTP 200 with four headers injected
into the response: `x-tenant-id`, `x-license-id`, `x-app-id`, and `x-plan-id`. Envoy's
`headersToUpstreamOnAllow` configuration controls which of these headers are forwarded to the upstream
service. The upstream service then trusts these headers for tenant isolation, rate limiting, and feature
gating without performing its own token validation. If the token is missing or cannot be verified, the
service returns 401. If the licence is invalid, expired, or the licence check fails, the service returns
403.

The service follows a fail-closed security model. If the licences service is unreachable or returns a
non-200 status, the request is denied with a 403. Timeouts, connection errors, and unexpected exceptions
all result in denial. Raw tokens are never logged -- only a SHA256 hash prefix appears in the structured
JSON audit logs for traceability. Client-supplied tenant headers are stripped by an EnvoyFilter before
this service runs, preventing header injection attacks.

## Features

- **Fail-closed security model** -- opaque licence tokens must pass introspection and Keycloak JWTs must
  pass JWKS verification plus `/licenses/active`; there is no JWT passthrough or trusted-network bypass,
  so a compromised or forged token cannot bypass validation. The opt-in service-principal path is
  identity-only and grants no tenant/licence headers.
- **Tenant header injection** -- returns `x-tenant-id`, `x-license-id`, `x-app-id`, and `x-plan-id` on
  successful auth so upstream services can enforce tenant isolation without re-validating tokens
- **Envoy ext_authz contract** -- implements the HTTP ext_authz protocol (200 allow / 401+403 deny) so it
  integrates directly with Istio AuthorizationPolicy CUSTOM actions
- **Token audit logging** -- logs every access decision with SHA256 hash prefix of the token, source IP,
  user agent, method, and path in structured JSON for Loki/Grafana
- **Timeout resilience** -- configurable introspect timeout (default 10s) with graceful handling of
  timeouts and connection errors that always fail closed
- **Stale-while-error fallback** -- a recently introspected active token can be served from a short-lived
  store when a later introspect fails transiently (see [Stale-while-error introspection](#stale-while-error-introspection))
- **JWT mode** -- opt-in verification of Keycloak JWTs against the issuer's JWKS, resolving the licence
  through `/licenses/active` instead of `/license/introspect`
- **Subject ID fallback** -- when `tenant_id` is absent from introspection, falls back to `subject_id`
  for the `x-tenant-id` header so older license formats remain compatible
- **Health probes** -- `/health` + `/healthz` (dependency-free), `/health/live` (liveness), and `/ready`
  (readiness with REAL dependency checks; see [Probe semantics](#probe-semantics))

## Authentication

This service IS the authentication layer. It has no auth middleware of its own -- it runs in the
`istio-system` namespace and is called by Envoy, not by end users directly. It accepts either an opaque
licence token (introspection) or, when JWT mode is enabled, a Keycloak JWT (JWKS verification plus
`/licenses/active`).

Health endpoints (`/health`, `/healthz`, `/ready`, `/health/live`) are accessible without authentication.

## Quick start

```bash
git clone https://github.com/nonagenticai/license-authz.git
cd license-authz
uv sync --frozen
ENVIRONMENT=development PYTHONPATH=src \
  uv run uvicorn main:app --host 0.0.0.0 --port 9001 --timeout-keep-alive 75
```

`ENVIRONMENT=development` is required for a local run: the default `production` value makes
`src/settings.py` fail at boot unless the six service URLs listed under **Configuration** are set.

## Usage

```bash
# Liveness -- no dependencies required
curl -i http://localhost:9001/health

# An ext_authz check (needs a reachable licences service to allow)
curl -i -H 'Authorization: Bearer <license-token>' http://localhost:9001/api/anything
```

A successful check returns 200 with the injected headers; a rejected check returns 401 or 403 as
described in [API Endpoints](#api-endpoints).

## API Endpoints

This service implements the Envoy ext_authz HTTP protocol. All paths except `/health`, `/healthz`,
`/ready`, `/health/live`, `/metrics`, `/jwks` and `/internal-token` are handled by the catch-all
ext_authz check endpoint (those routes are registered before the `/{path:path}` catch-all, so FastAPI
serves them directly).

### ext_authz Check

| Method | Path | Description |
|--------|------|-------------|
| ANY | `/{path:path}` | Validate the bearer credential (introspection or JWT-mode licence resolution) |

**200 OK** (allow request) -- headers injected:
```
x-tenant-id: {subject_id}
x-license-id: {license_id}
x-app-id: {app_id}
x-plan-id: {plan_id}
```

**401 Unauthorized** (missing/invalid auth header):
```json
{"error": "unauthorized", "reason": "missing_bearer_token"}
```

**403 Forbidden** (invalid/expired license):
```json
{"error": "license_invalid", "reason": "license_expired"}
```

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check (dependency-free; startupProbe) |
| GET | `/healthz` | Alias of `/health` (dependency-free) |
| GET | `/ready` | Readiness probe -- real dependency checks, 503 when unservable |
| GET | `/health/live` | Liveness probe (dependency-free) |

### Probe semantics

`/health`, `/healthz` and `/health/live` are deliberately hardcoded 200s.

* **Liveness must stay dumb.** A liveness probe that checks dependencies causes
  a restart storm: when a shared dependency (Redis/Keycloak) blips, every
  replica fails liveness at once and the kubelet restarts them all, turning a
  degraded dependency into a total outage.
* **`/health` backs the startupProbe** and is an ext_authz-exempt path. Keeping
  it unconditional means a pod can always finish starting, even into a degraded
  dependency.

`/ready` is the only endpoint that reflects dependency state:

| Result | Meaning |
|--------|---------|
| `200 {"status":"ready"}` | every gating dependency usable |
| `200 {"status":"degraded", ...}` | a fail-open dependency (Redis) is down; the pod still serves |
| `503 {"status":"not_ready","failed":[...]}` | a gating dependency is unusable **on this pod** |

Gating vs. reported is decided by one rule: license-authz is an ext_authz auth
boundary with `statusOnError: "403"`, so if *every* replica goes NotReady the
mesh fails shut. Readiness therefore gates only on **per-pod, locally-latched
initialisation state** (JWKS loaded, internal-token key parses, lifespan
completed) -- cases where shedding this pod helps because a sibling is fine. It
never gates on shared remote dependencies (licenses service, Keycloak token
endpoint) or on fail-open paths (Redis metering events, billing checks), which are
reported in the body instead. Full rationale: `src/readiness.py`.

## Configuration

All configuration is via environment variables. Values without a module note come from the Pydantic
`BaseSettings` class in `src/settings.py`; values with a module note are read with `os.getenv()` at
import in that module.

| Variable | Default | Meaning |
|----------|---------|---------|
| `LICENSE_INTROSPECT_URL` | `""` | licences `/license/introspect` endpoint. Required in production: boot fails if unset. |
| `LICENSES_ACTIVE_URL` | `""` | licences `/licenses/active` endpoint used by JWT mode. Required in production. |
| `LICENSES_INTERNAL_TENANTS_URL` | `""` | licences `/internal/tenants`, used to attribute service-principal metering to a tenant. Empty disables resolution. |
| `METERING_SERVICE_URL` | `""` | billing-service base URL for token-budget and request-quota checks. Required in production. |
| `KEYCLOAK_ISSUER` | `""` | Keycloak realm issuer. Required in production. |
| `JWKS_URL` | `""` | Keycloak JWKS endpoint; defaults to `{KEYCLOAK_ISSUER}/protocol/openid-connect/certs` when the issuer is set. |
| `KEYCLOAK_TOKEN_URL` | `""` | Keycloak token endpoint for service-account `client_credentials`. Required in production. |
| `KEYCLOAK_SECURE_APPS_ISSUER` | `""` | secure-apps realm issuer for caller-identity/service-principal tokens. Required in production. |
| `INTERNAL_TOKEN_ISSUER_ENABLED` | `false` | Enable the `/jwks` and `/internal-token` issuer endpoints. Requires a signing key and kid. |
| `ENVIRONMENT` | `production` | Deployment environment; `production` enforces the required URLs at boot. Set `development` for a local run. |
| `LICENSE_AUTHZ_JWT_MODE_ENABLED` | `false` | Verify Keycloak JWTs via JWKS and resolve the licence through `/licenses/active`. |
| `SERVICE_PRINCIPAL_AUTH_ENABLED` | `false` | Accept a verified secure-apps service-account token in `X-Caller-Authorization` as an identity-only caller. |
| `SERVICE_PRINCIPAL_ALLOWED_CLIENTS` | `""` | Comma-separated `azp` allowlist for that path; empty allows any verified service-account token. |
| `INTROSPECT_TIMEOUT` | `10.0` | Timeout in seconds for introspect calls (`src/main.py`). |
| `INTROSPECT_STALE_TTL` | `900` | Seconds a last-known-good introspect result stays eligible for stale-while-error fallback (see below; `src/main.py`). |
| `INTROSPECT_STALE_MAX_ENTRIES` | `1000` | Max entries in the stale-while-error store; oldest-first LRU eviction. **`0` or negative disables the store entirely** (`src/main.py`). |
| `METERING_ENABLED` | `true` | Emit metering events to Redis (`src/main.py`). |
| `BUDGET_ENFORCEMENT_ENABLED` | `false` | Enforce per-licence and per-tenant token budgets and request quotas through billing-service (`src/main.py`). |
| `REDIS_URL` | `redis://localhost:6379/2` | Redis used for metering events (`src/main.py`). |
| `PUSHGATEWAY_URL` | `""` | Pushgateway for Prometheus metrics; empty disables the pusher (`src/settings.py`). |
| `PORT` | `9001` | Port to listen on (`src/main.py`). |

JWT-mode and readiness tuning (`JWKS_TTL_SECONDS`, `JWKS_HTTP_TIMEOUT`, `ACTIVE_LICENSE_CACHE_TTL`,
`READINESS_*`, `JWT_VERIFY_AUDIENCE` and others) is read with `os.getenv()` in `src/jwt_mode.py`,
`src/readiness.py` and `src/main.py`.

### Stale-while-error introspection

A *transient* introspect failure (timeout / network error / non-200) is not the
same as an invalid token, but callers treat any `active=false` as a 403. So
every successful **active** introspection is recorded (keyed by token hash) and
served as a fallback if a later introspect fails transiently, for up to
`INTROSPECT_STALE_TTL`. Posture is preserved:

- a definitive `200 active=false` is authoritative, is **never** served stale,
  and **drops** that token's stored entry — so a revocation cannot be undone by
  a subsequent blip;
- a token never seen before has no entry, so a cold-cache failure fails closed;
- no positive cache is added — every request still introspects, so revocation
  latency on the happy path is unchanged.

Setting `INTROSPECT_STALE_MAX_ENTRIES=0` turns the store off as an operator
kill-switch; transient failures then deny, as they did before the store
existed. Evictions are reported on `license_authz_cache_evictions_total{cache="introspect_stale"}`.

## Architecture

```
src/
├── __init__.py          # Package marker
├── internal_token.py    # Short-lived RS256 internal-token issuer and /jwks key material
├── jwt_mode.py          # Keycloak JWT verification and /licenses/active resolution
├── main.py              # FastAPI app, ext_authz handler, introspection, routes
├── metering_client.py   # billing-service token-budget and quota HTTP client
├── metrics_push.py      # Push Prometheus metrics to the cluster pushgateway
├── readiness.py         # Dependency probing behind /ready
├── service_auth.py      # Keycloak service-account token acquisition
└── settings.py          # Pydantic BaseSettings configuration
tests/
├── conftest.py          # Shared fixtures
├── test_*.py            # Unit and integration tests per module
└── contract/            # Billing contract guard and Schemathesis hooks
```

## Development

```bash
uv sync --frozen --all-extras
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

`ty` is available locally but is not part of the CI gate. There is no coverage threshold configured.

## CI/CD

`.github/workflows/ci-public.yml` runs on pull requests, pushes to `main`, and manual dispatches, on
GitHub-hosted runners:

1. **Secret scan** — betterleaks with `.betterleaks.toml`
2. **No private refs** — `scripts/ci/no-private-refs.sh`
3. **Checks** — `uv sync --frozen --all-extras`, `ruff check`, `ruff format --check`, `pytest -q`,
   `pip-audit` against the exported locked requirements
4. **Image** — when the repository variable `PUBLISH_IMAGE` is `true`, builds on every event and, on
   `main`, pushes `sha-<7>`, `main-<run>`, and `latest` tags to GitHub Container Registry

Type checking (`ty`) is deliberately not in the CI gate. The workflow performs no deployment.
Publishing an image to a registry, and rolling it out, is a decision for whoever operates the
deployment.

## Docker

```bash
docker build -t license-authz .
docker run --rm -p 9001:9001 -e ENVIRONMENT=development license-authz
docker run --rm -p 9001:9001 -e ENVIRONMENT=development \
  -e LICENSE_INTROSPECT_URL=http://localhost:8544/license/introspect license-authz
```

A production container needs the service URLs listed under **Configuration**; `ENVIRONMENT=development`
is the local-run escape from that requirement. The image sets `PYTHONPATH=/app/src` and runs
`main:app`.

## Istio Integration

### ExtensionProvider (in IstioOperator meshConfig)
```yaml
extensionProviders:
- name: license-authz
  envoyExtAuthzHttp:
    service: license-authz.example
    port: 9001
    headersToUpstreamOnAllow:
    - x-tenant-id
    - x-license-id
    - x-app-id
    - x-plan-id
```

### AuthorizationPolicy CUSTOM
```yaml
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: license-ext-authz
  namespace: istio-system
spec:
  selector:
    matchLabels:
      istio: ingressgateway
  action: CUSTOM
  provider:
    name: license-authz
  rules:
  - to:
    - operation:
        hosts: ["api.example.com"]
        notPaths: ["/health", "/healthz"]
```

## Dependencies

- **licences service** (port 8544) -- `/license/introspect`, `/licenses/active`, and `/internal/tenants`
- **Keycloak** -- JWKS for JWT-mode verification, and the token endpoint for service-account calls
- **billing-service** -- `/api/budget/check` and `/api/quota/check`
- **Redis** -- metering events on the `billing:events` stream
- **FastAPI / Uvicorn / httpx** -- web framework, ASGI server, async HTTP client
- **Pydantic / pydantic-settings** -- request models and configuration
- **PyJWT** -- JWT verification and internal-token signing
- **prometheus-client** -- metrics, including the pushgateway exporter

## Known Issues

- **JWT mode falls back to `sub` for the tenant** -- neither Keycloak realm emits a `tenant_id` claim
  today, so `src/jwt_mode.py` uses the per-user `sub` as the tenant. That splits a customer across
  tenants for licence and quota rollups. Removing the fallback requires the claim to land upstream
  first.
- **Type checking is not in the CI gate** -- `uv run ty check src` is documented above and passes
  locally, but `.github/workflows/ci-public.yml` deliberately does not run it.
- **Schemathesis hooks are not wired to a test module** -- `tests/contract/schemathesis_hooks.py` exists,
  but no Schemathesis test suite consumes it. The billing contract guard in `tests/contract/` does run
  in CI.

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for this repository's
rules on branches, pull requests, and DCO sign-off. The setup, test and lint commands are in
**Development** above.

## Licensing

This repository is part of an **optional** licensing system, and it is only the enforcement point:
an Envoy `ext_authz` service that an application calls only when that application is configured to
call it. No nonagenticai service requires a licence server to build, test or run. Each integrates
with the gate opt-in, and runs fully without one while the switch below is unset:

| repository | integration switch | default |
|---|---|---|
| `agent` | `LICENSE_GATE_2_ENABLED` | `false` |
| `care-bus-agents` | `LICENSE_GATE_2_ENABLED` | `false` |
| `strata_rag` | `STRATA_LICENSE_INTROSPECT_URL` | unset = off |
| `strata_ingest` | `STRATA_INGEST_LICENSE_INTROSPECT_URL` | unset = off |
| `conveyor` | `LICENCES_BASE_URL` | unset = off (only `licence.sync` jobs are affected) |
| `nonagenticai_mcp_server` (Python implementation, archived) | `LICENSE_INTROSPECT_URL` | unset = off (protected work disabled); the public Rust port carries no licence code yet |

When a service **is** configured, enforcement is deliberately fail-closed: if the licences service
is unreachable, this service denies the request (403) instead of allowing it. That is the intended
behaviour of a licence gate, not a defect. The Apache-2.0 licence covers the code: running this
software never requires a licence key.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
