# License AuthZ

HTTP ext_authz service for Envoy/Istio that validates license tokens and injects trusted tenant headers. Port 9001.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · httpx |
| **Runs on** | mf1 (K8s, istio-system namespace) |
| **Stateless** | No database |

## What License AuthZ Does

When an external request arrives at the Istio IngressGateway carrying an `Authorization: Bearer <token>` header, Envoy's ext_authz filter forwards the request headers to this service before the request reaches the upstream. License AuthZ extracts the bearer token, calls the licenses service's `/license/introspect` endpoint, and based on the introspection result either allows or denies the request. Every request with a token is introspected -- there is no JWT passthrough and no trusted-network bypass.

If introspection confirms the token is active, the service returns HTTP 200 with four headers injected into the response: `x-tenant-id`, `x-license-id`, `x-app-id`, and `x-plan-id`. Envoy's `headersToUpstreamOnAllow` configuration controls which of these headers are forwarded to the upstream service. The upstream service then trusts these headers for tenant isolation, rate limiting, and feature gating without performing its own token validation. If the token is missing, the service returns 401. If the token is invalid, expired, or introspection fails, the service returns 403.

The service follows a fail-closed security model. If the licenses service is unreachable or returns a non-200 status, the request is denied with a 403. Timeouts, connection errors, and unexpected exceptions all result in denial. Raw tokens are never logged -- only a SHA256 hash prefix appears in the structured JSON audit logs for traceability. Client-supplied tenant headers are stripped by an EnvoyFilter before this service runs, preventing header injection attacks.

## Features

- **Fail-closed security model** -- all tokens are introspected via the licenses service with no passthrough path, so a compromised or forged token cannot bypass validation
- **Tenant header injection** -- returns `x-tenant-id`, `x-license-id`, `x-app-id`, and `x-plan-id` on successful auth so upstream services can enforce tenant isolation without re-validating tokens
- **Envoy ext_authz contract** -- implements the HTTP ext_authz protocol (200 allow / 401+403 deny) so it integrates directly with Istio AuthorizationPolicy CUSTOM actions
- **Token audit logging** -- logs every access decision with SHA256 hash prefix of the token, source IP, user agent, method, and path in structured JSON for Loki/Grafana
- **Timeout resilience** -- configurable introspect timeout (default 2s) with graceful handling of timeouts and connection errors that always fail closed
- **Subject ID fallback** -- when `tenant_id` is absent from introspection, falls back to `subject_id` for the `x-tenant-id` header so older license formats remain compatible
- **Health probes** -- `/health` + `/healthz` (dependency-free), `/health/live` (liveness), and `/ready` (readiness with REAL dependency checks; see [Probe semantics](#probe-semantics))

## Authentication

This service IS the authentication layer. It has no auth middleware of its own -- it runs in the `istio-system` namespace and is called by Envoy, not by end users directly.

Health endpoints (`/health`, `/healthz`, `/ready`, `/health/live`) are accessible without authentication.

## Quick Start

```bash
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 9001 --timeout-keep-alive 75
uv run pytest tests/ -v
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

This service implements the Envoy ext_authz HTTP protocol. All paths except `/health`, `/healthz`, `/ready`, `/health/live`, `/metrics`, `/jwks` and `/internal-token` are handled by the catch-all ext_authz check endpoint (those routes are registered before the `/{path:path}` catch-all, so FastAPI serves them directly).

### ext_authz Check

| Method | Path | Description |
|--------|------|-------------|
| ANY | `/{path:path}` | Validate bearer token via license introspection |

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

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `LICENSE_INTROSPECT_URL` | `http://licenses.example:8544/license/introspect` | Licenses service introspect endpoint |
| `INTROSPECT_TIMEOUT` | `10.0` | Timeout in seconds for introspect calls |
| `INTROSPECT_STALE_TTL` | `900` | Seconds a last-known-good introspect result stays eligible for stale-while-error fallback (see below) |
| `INTROSPECT_STALE_MAX_ENTRIES` | `1000` | Max entries in the stale-while-error store; oldest-first LRU eviction. **`0` or negative disables the store entirely** |
| `PORT` | `9001` | Port to listen on |

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
├── __init__.py                    # Package marker
└── main.py                       # FastAPI app, ext_authz endpoint, introspect logic, health checks
tests/
├── conftest.py                   # Shared fixtures (AsyncClient)
├── test_main.py                  # Ext_authz endpoint tests, security hardening tests
├── test_introspect.py            # Introspect function tests, hash_token, log_access
└── contract/
    └── schemathesis_hooks.py     # Schemathesis contract test hooks (disabled in CI)
```

## Testing

```bash
uv run pytest tests/ -v                    # All tests
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
uv run ty check .                          # Type checking
```

Coverage threshold: 85% (enforced in pyproject.toml).

## CI/CD

`.github/workflows/ci-public.yml` runs on every push and pull request, on GitHub-hosted
runners:

1. **Lint, types, tests** — `ruff check`, `ruff format --check`, `ty`, `pytest`, `pip-audit`
2. **Secret scan** — betterleaks with `.betterleaks.toml`
3. **No private refs** — `scripts/ci/no-private-refs.sh`
4. **Image** — builds on `main` when the repository variable `PUBLISH_IMAGE` is `true`

The workflow performs no deployment. Publishing an image to a registry, and rolling it out, is
a decision for whoever operates the deployment; the steps below describe one way to do it.

## Docker

```bash
docker build -t license-authz .
docker run -p 9001:9001 license-authz
docker run -p 9001:9001 -e LOG_LEVEL=DEBUG -e LICENSE_INTROSPECT_URL=http://localhost:8544/license/introspect license-authz
```

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

- **licenses service** (port 8544) -- validates tokens via `/license/introspect` endpoint
- **FastAPI** -- web framework
- **Uvicorn** -- ASGI server
- **httpx** -- async HTTP client for license introspection calls
- **Pydantic** -- request/response validation

## Known Issues

- **Schemathesis contract tests disabled** -- the contract checks exist in `tests/contract/` but are not part of the CI gate.
- **No config.py** -- configuration is done via `os.getenv()` calls in `main.py` rather than a Pydantic settings class. This is fine for 3 variables but diverges from the standard pattern used by other services.

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
| `nonagenticai_mcp_server` | `LICENSE_INTROSPECT_URL` | unset = off (protected work disabled) |

When a service **is** configured, enforcement is deliberately fail-closed: if the licences service
is unreachable, this service denies the request (403) instead of allowing it. That is the intended
behaviour of a licence gate, not a defect. The Apache-2.0 licence covers the code: running this
software never requires a licence key.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
