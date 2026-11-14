"""Internal-token issuer (Lane F1): license-authz signs short-lived RS256 JWTs.

hostNetwork / non-SPIFFE callers (the alloy DaemonSet, future on-host agents)
have no mesh identity. This module lets license-authz mint a short-lived signed
token they can present, and serves the matching public key at /jwks so services
verify it locally with no hot-path network call.

Mirrors the verify-side conventions in `jwt_mode.py`: module-level settings,
fail-closed errors, no source-side URL fallbacks. The core functions here are
pure (take the PEM/kid/claims explicitly) so they unit-test without settings;
the thin `jwks_document()` / `mint_token()` wrappers bind them to live settings
and enforce the enable gate.
"""

import json
import time
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from jwt.algorithms import RSAAlgorithm

from settings import get_settings


class IssuerDisabled(Exception):
    """Raised when an issuer operation is attempted while the gate is off."""


# Loaded private keys cached by PEM contents — load_pem_private_key parses the
# key on every call otherwise, and the PEM only changes on a pod restart.
_key_cache: dict[str, RSAPrivateKey] = {}


def _load_private_key(pem: str) -> RSAPrivateKey:
    """Parse (and cache) an RS256 private key from PEM."""
    key = _key_cache.get(pem)
    if key is not None:
        return key
    loaded = load_pem_private_key(pem.encode(), password=None)
    if not isinstance(loaded, RSAPrivateKey):
        raise ValueError("internal-token signing key is not an RSA private key")
    _key_cache[pem] = loaded
    return loaded


def build_jwks(*, pem: str, kid: str) -> dict[str, Any]:
    """Public JWKS document for the given signing key. Never exposes private fields."""
    private_key = _load_private_key(pem)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


def issue_internal_token(
    *,
    pem: str,
    kid: str,
    issuer: str,
    audience: str,
    ttl_seconds: int,
    sub: str,
    tenant_id: str | None = None,
    now: int | None = None,
) -> tuple[str, int]:
    """Sign a short-lived RS256 internal token. Returns (token, ttl_seconds)."""
    private_key = _load_private_key(pem)
    issued_at = int(time.time()) if now is None else now
    claims: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": sub,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    if tenant_id:
        claims["tenant_id"] = tenant_id
    token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})
    return token, ttl_seconds


# --------------------------------------------------------------------------
# Settings-bound wrappers used by the routes — enforce the enable gate.
# --------------------------------------------------------------------------
def jwks_document() -> dict[str, Any]:
    settings = get_settings()
    if not settings.internal_token_issuer_enabled:
        raise IssuerDisabled("internal-token issuer is disabled")
    return build_jwks(pem=settings.internal_token_signing_key, kid=settings.internal_token_kid)


def mint_token(*, sub: str, tenant_id: str | None = None) -> tuple[str, int]:
    settings = get_settings()
    if not settings.internal_token_issuer_enabled:
        raise IssuerDisabled("internal-token issuer is disabled")
    return issue_internal_token(
        pem=settings.internal_token_signing_key,
        kid=settings.internal_token_kid,
        issuer=settings.internal_token_issuer,
        audience=settings.internal_token_audience,
        ttl_seconds=settings.internal_token_ttl_seconds,
        sub=sub,
        tenant_id=tenant_id,
    )
