"""Regression tests for OpenAPI schema stability.

Asserts every route declares an explicit operation_id and that all ids are
unique. Auto-generated FastAPI operation_ids (e.g. `metrics_metrics_get`) are
unstable across function renames and ugly in client SDKs — sibling FastAPI
services in Movies/ all set explicit ids; these tests prevent regression.
"""

from fastapi.routing import APIRoute

from main import app


def test_every_route_has_operation_id() -> None:
    missing = [r.path for r in app.routes if isinstance(r, APIRoute) and not r.operation_id]
    assert not missing, f"routes missing operation_id: {missing}"


def test_operation_ids_are_unique() -> None:
    # The ext_authz catch-all is registered once per HTTP method (see
    # add_api_route loop in main) so each method gets a unique
    # operation_id like ext_authz_check_get / _post / ... . OpenAPI 3.1
    # requires operationId to be unique across the whole document.
    ids = [r.operation_id for r in app.routes if isinstance(r, APIRoute)]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    assert not dupes, f"duplicate operation_ids: {dupes}"
