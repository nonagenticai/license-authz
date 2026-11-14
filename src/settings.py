"""Application settings using Pydantic BaseSettings.

Per gitops-production/docs/config-externalization-policy.md: service URLs
ship with empty-string defaults; production deployments MUST set them via
env vars on the Kubernetes Deployment. `model_post_init` fails fast at
boot when required fields are missing in production.
"""

from functools import lru_cache
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the license-authz service.

    Env vars are read directly (no prefix) to match the existing module-level
    `os.getenv("LICENSE_INTROSPECT_URL", ...)` shape that surrounding services
    (gitops Deployment, Envoy ext_authz) already use.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # ------------------------------------------------------------------
    # Upstream service URLs — placeholders; real values come from the
    # license-authz Deployment env block in gitops-production.
    # ------------------------------------------------------------------
    license_introspect_url: str = Field(
        default="",
        description="licenses /license/introspect endpoint",
    )
    licenses_active_url: str = Field(
        default="",
        description="licenses /licenses/active endpoint (JWT-mode resolver)",
    )
    licenses_internal_tenants_url: str = Field(
        default="",
        description=(
            "FULL URL of licenses GET /internal/tenants (e.g. "
            "http://licenses.example:8544/internal/tenants), used to "
            "resolve a verified service-principal caller's azp to a tenant id for "
            "metering attribution via an exact name match on the returned list. Empty "
            "disables resolution: no HTTP call is made, every service-principal request "
            "is counted result=unresolved and emits tenant_id='' exactly as before this "
            "field existed — see SERVICE_PRINCIPAL_TENANT_NAME_MAP."
        ),
    )
    metering_service_url: str = Field(
        default="",
        description="billing-service base URL for token-budget checks and the quota HTTP fallback",
    )
    keycloak_issuer: str = Field(
        default="",
        description="Keycloak issuer URL (realm)",
    )
    jwks_url: str = Field(
        default="",
        description="Keycloak JWKS endpoint (defaults to {issuer}/protocol/openid-connect/certs)",
    )
    keycloak_token_url: str = Field(
        default="",
        description="Keycloak token endpoint for service-account client_credentials",
    )
    keycloak_trusted_issuers: str = Field(
        default="",
        description=(
            "Comma-separated allowlist of Keycloak issuer URLs whose END-USER JWTs are "
            "routed to JWT mode (realm-per-app P0). Empty = [KEYCLOAK_ISSUER], i.e. "
            "exactly the single-issuer behaviour. Never include the secure-apps issuer: "
            "it mints service accounts, not users (refused at boot)."
        ),
    )
    keycloak_secure_apps_issuer: str = Field(
        default="",
        description="Keycloak secure-apps realm issuer URL (caller-identity SA tokens)",
    )
    keycloak_secure_apps_jwks_url: str = Field(
        default="",
        description="secure-apps JWKS endpoint (defaults to {issuer}/protocol/openid-connect/certs)",
    )

    # ------------------------------------------------------------------
    # Internal-token issuer (Lane F1) — license-authz signs short-lived RS256
    # tokens for hostNetwork / non-SPIFFE callers and serves the public key at
    # /jwks. Gated off by default so the code ships before the key exists; when
    # the gate is on, model_post_init requires the signing key + kid.
    # ------------------------------------------------------------------
    internal_token_issuer_enabled: bool = Field(
        default=False,
        description="Enable the /jwks + /internal-token issuer endpoints",
    )
    internal_token_signing_key: str = Field(
        default="",
        description="RS256 private key (PEM) used to sign internal tokens",
    )
    internal_token_kid: str = Field(
        default="",
        description="Key id placed in the JWT header and JWKS (for rotation)",
    )
    internal_token_issuer: str = Field(
        default="license-authz",
        description="`iss` claim of issued internal tokens",
    )
    internal_token_audience: str = Field(
        default="internal",
        description="`aud` claim of issued internal tokens",
    )
    internal_token_ttl_seconds: int = Field(
        default=300,
        description="Lifetime of issued internal tokens in seconds",
    )

    # ------------------------------------------------------------------
    # Non-infra settings (timeouts, flags, identifiers) — keep defaults.
    # ------------------------------------------------------------------
    environment: str = Field(default="production", description="deploy environment")

    # ------------------------------------------------------------------
    # Pushgateway export. license-authz cannot be scraped (pod-level ambient
    # in the root namespace istio-system; mTLS STRICT refuses the plaintext
    # scrape at ztunnel before RBAC, and gitops #819/#833 both failed to relax
    # it). Measured 2026-08-24: 0 of Prometheus's 4224 series are
    # license_authz_*. Pushing is the only path that needs no mesh change.
    #
    # DISABLED BY DEFAULT: url="" means the pusher never starts, so merging
    # this changes nothing. Arming it is a separate gitops env change.
    # ------------------------------------------------------------------
    pushgateway_url: str = Field(
        default="",
        description="pushgateway host:port, e.g. pushgateway.example:9091. Empty disables the exporter.",
    )
    pushgateway_interval_seconds: float = Field(
        default=30.0,
        description="seconds between pushes",
    )
    pushgateway_timeout_seconds: float = Field(
        default=5.0,
        description="per-push timeout; the push runs in a worker thread and can never block the ext_authz event loop",
    )

    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "license_introspect_url",
        "licenses_active_url",
        "metering_service_url",
        "keycloak_issuer",
        "keycloak_token_url",
        "keycloak_secure_apps_issuer",
    )

    def model_post_init(self, __context: object) -> None:
        # Default jwks_url to {issuer}/protocol/openid-connect/certs when unset
        # but issuer is provided — preserves the legacy convenience without
        # baking a hardcoded fallback into the source.
        if not self.jwks_url and self.keycloak_issuer:
            object.__setattr__(
                self,
                "jwks_url",
                f"{self.keycloak_issuer}/protocol/openid-connect/certs",
            )
        if not self.keycloak_secure_apps_jwks_url and self.keycloak_secure_apps_issuer:
            object.__setattr__(
                self,
                "keycloak_secure_apps_jwks_url",
                f"{self.keycloak_secure_apps_issuer}/protocol/openid-connect/certs",
            )

        # Secure-apps is the S2S realm: its tokens are handled by the
        # caller-identity / service-principal paths. Admitting it to the USER
        # allowlist would silently re-route SA bearers into JWT mode, so an
        # operator who lists it gets a halted rollout, not a changed auth path.
        secure_apps = self.keycloak_secure_apps_issuer.strip().rstrip("/")
        if secure_apps and secure_apps in self.trusted_issuers():
            raise RuntimeError(
                "KEYCLOAK_TRUSTED_ISSUERS must not contain the secure-apps issuer "
                f"({secure_apps}): it mints service-account tokens, not user tokens."
            )

        # Internal-token issuer requires a signing key + kid only when enabled —
        # checked regardless of environment so a misconfigured opt-in fails fast
        # at boot rather than 500ing the first /internal-token request.
        if self.internal_token_issuer_enabled:
            issuer_missing = [
                n
                for n in ("internal_token_signing_key", "internal_token_kid")
                if not getattr(self, n)
            ]
            if issuer_missing:
                raise RuntimeError(
                    "license-authz internal-token issuer is enabled but missing required "
                    f"fields: {issuer_missing}. Set them via env vars on the Deployment."
                )

        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"license-authz settings missing required fields in production: {missing}. "
                "Set them via env vars on the Deployment (see gitops-production)."
            )

    def trusted_issuers(self) -> tuple[str, ...]:
        """User-JWT issuer allowlist: KEYCLOAK_TRUSTED_ISSUERS, else (KEYCLOAK_ISSUER,).

        Entries are stripped of whitespace and a trailing ``/`` (a token's ``iss``
        is compared EXACTLY against these, so a stray slash in the env would
        otherwise fail closed silently). Order kept, duplicates dropped.
        """
        raw = self.keycloak_trusted_issuers.split(",") if self.keycloak_trusted_issuers else []
        entries = [e.strip().rstrip("/") for e in raw]
        entries = [e for e in entries if e]
        if not entries:
            primary = self.keycloak_issuer.strip().rstrip("/")
            entries = [primary] if primary else []
        return tuple(dict.fromkeys(entries))

    def jwks_url_for_issuer(self, issuer: str) -> str:
        """JWKS URL for one allowlisted issuer.

        * the primary issuer keeps ``JWKS_URL`` exactly (today's behaviour);
        * another realm on the SAME Keycloak host reuses ``JWKS_URL`` with the
          realm segment swapped, so it is fetched in-cluster like the primary
          (``http://keycloak.keycloak.svc…/realms/<realm>/…``) rather than out
          through Cloudflare, which 403s urllib's User-Agent;
        * anything else falls back to ``{issuer}/protocol/openid-connect/certs``,
          the same derivation ``JWKS_URL`` itself defaults to.
        """
        primary = self.keycloak_issuer.strip().rstrip("/")
        if issuer == primary and self.jwks_url:
            return self.jwks_url
        if "/realms/" in issuer and "/realms/" in primary and self.jwks_url:
            base, _, realm = issuer.rpartition("/realms/")
            p_base, _, p_realm = primary.rpartition("/realms/")
            marker = f"/realms/{p_realm}/"
            if base == p_base and realm and p_realm and marker in self.jwks_url:
                return self.jwks_url.replace(marker, f"/realms/{realm}/", 1)
        return f"{issuer}/protocol/openid-connect/certs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
