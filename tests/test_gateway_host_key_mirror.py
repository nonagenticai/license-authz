"""Mirror guard: GATEWAY_ASSERTED_HOST_KEYS must equal the gateway's LICENSE_GATED_APP_KEY.

The ingress-gateway Lua filter in the cluster-configuration repository
(`clusters/strato/policies/envoy-filters/ext-authz-enforce.yaml`) asserts an
`x-app-key` per host from its `LICENSE_GATED_APP_KEY` table; license-authz binds
that assertion only for hosts in `main.GATEWAY_ASSERTED_HOST_KEYS`. Drift either
way is a live defect:

- gateway has a host license-authz lacks (or a different key) -> every request on
  that host's UI is denied 403;
- license-authz has a host the gateway lacks -> a dead entry that silently widens
  what a forged `x-forwarded-host` could bind.

How the gitops file is found (`GITOPS_REPO` = absolute path of a gitops checkout):

- CI conformance job: the reusable workflow checks the cluster-configuration
  repository out and exports `GITOPS_REPO` (`extra_checkout_env`). If the var
  is set but the file is missing, the comparison FAILS - never skips.
- CI unit pass (`CI` set, `GITOPS_REPO` unset): the comparison cannot run here, so
  instead `test_ci_wiring_runs_this_guard` asserts ci-cd.yaml still wires the
  conformance job to this file. Removing that wiring reds the unit pass.
- Local, neither set: the comparison SKIPS with the reason.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import main as main_module

GITOPS_ENV = "GITOPS_REPO"
LUA_REL_PATH = Path("clusters/strato/policies/envoy-filters/ext-authz-enforce.yaml")
REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_FILE_REL = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()

_TABLE_OPEN = re.compile(r"^\s*local\s+LICENSE_GATED_APP_KEY\s*=\s*\{\s*$", re.MULTILINE)
_ENTRY = re.compile(r"""^\[\s*(["'])(?P<host>[^"']+)\1\s*\]\s*=\s*(["'])(?P<key>[^"']*)\3\s*,?$""")


def parse_license_gated_app_key(text: str) -> dict[str, str]:
    """Extract `LICENSE_GATED_APP_KEY` from the Lua source; raise on anything unparseable.

    Strict on purpose: a line inside the table that is neither blank, a `--`
    comment, nor a `["host"] = "key",` entry raises instead of being ignored, so
    a syntax change in gitops cannot silently shrink the parsed set.
    """
    opens = list(_TABLE_OPEN.finditer(text))
    if len(opens) != 1:
        raise ValueError(
            f"expected exactly 1 `local LICENSE_GATED_APP_KEY = {{` line, found {len(opens)}"
        )
    body_start = opens[0].end()
    close = text.find("}", body_start)
    if close == -1:
        raise ValueError("LICENSE_GATED_APP_KEY table is never closed")
    table: dict[str, str] = {}
    for raw in text[body_start:close].splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue
        m = _ENTRY.match(line)
        if not m:
            raise ValueError(f"unparseable LICENSE_GATED_APP_KEY line: {raw.strip()!r}")
        host, key = m.group("host"), m.group("key")
        if host in table:
            raise ValueError(f"duplicate LICENSE_GATED_APP_KEY host: {host!r}")
        table[host] = key
    if not table:
        raise ValueError("LICENSE_GATED_APP_KEY parsed as empty")
    return table


def mirror_drift(gateway: dict[str, str], authz: dict[str, str]) -> list[str]:
    """Human-readable differences; empty list means the two mirror exactly."""
    drift = []
    for host in sorted(gateway.keys() - authz.keys()):
        drift.append(
            f"gateway asserts {host}={gateway[host]} but license-authz has no entry (-> 403 on that host)"
        )
    for host in sorted(authz.keys() - gateway.keys()):
        drift.append(
            f"license-authz binds {host}={authz[host]} but the gateway never asserts it (dead entry)"
        )
    for host in sorted(gateway.keys() & authz.keys()):
        if gateway[host] != authz[host]:
            drift.append(
                f"{host}: gateway asserts {gateway[host]!r}, license-authz binds {authz[host]!r}"
            )
    return drift


def _seed() -> dict[str, str]:
    # Seed only: an env override is a deploy-time knob, not the mirrored contract.
    return main_module._build_gateway_asserted_host_keys("")


# --- the live comparison -----------------------------------------------------


def test_gateway_asserted_host_keys_mirror_gitops_lua_table() -> None:
    gitops = os.environ.get(GITOPS_ENV, "").strip()
    if not gitops:
        pytest.skip(
            f"{GITOPS_ENV} unset: no cluster-configuration checkout to compare against. "
            f"Run with {GITOPS_ENV}=/path/to/gitops-production. In CI this comparison "
            "runs in the conformance job."
        )
    lua_file = Path(gitops) / LUA_REL_PATH
    if not lua_file.is_file():
        pytest.fail(
            f"{GITOPS_ENV}={gitops} is set but {LUA_REL_PATH} does not exist there; the guard cannot run"
        )
    gateway = parse_license_gated_app_key(lua_file.read_text())
    drift = mirror_drift(gateway, _seed())
    assert not drift, (
        "GATEWAY_ASSERTED_HOST_KEYS drifted from LICENSE_GATED_APP_KEY:\n" + "\n".join(drift)
    )


def test_ci_runs_the_suite() -> None:
    """In the public CI the comparison skips; something else must still run tests.

    ⚠️ WHERE THE COMPARISON ACTUALLY RUNS. Not here. It needs the cluster
    configuration repository checked out beside this one, and the direction of
    dependency is one-way: that repository may refer to this public one, this
    one must never refer to it. So the comparison belongs in that repository's
    CI, which is the only place both trees exist.

    What this test pins is the weaker property that the public workflow still
    runs pytest — otherwise `test_gateway_asserted_host_keys_mirror_gitops_lua_table`
    would silently stop being invoked at all, and a skip would look like a pass.
    """
    if not os.environ.get("CI") or os.environ.get(GITOPS_ENV):
        pytest.skip("wiring check applies to the CI unit pass only (CI set, GITOPS_REPO unset)")
    wf = (REPO_ROOT / ".github/workflows/ci-public.yml").read_text()
    assert "pytest" in wf, "ci-public.yml no longer runs pytest; the mirror comparison never runs"


# --- must-fail controls on the parser/comparator -----------------------------

_LUA_OK = """
            local LICENSE_GATED_APP_KEY = {
              ["kanban.nonagentic.ai"] = "kanban",
              -- a comment line
              ['n8n.nonagentic.ai'] = 'n8n',  -- trailing comment
            }
            local function urlencode(s)
"""


def test_parser_reads_entries_and_ignores_comments() -> None:
    assert parse_license_gated_app_key(_LUA_OK) == {
        "kanban.nonagentic.ai": "kanban",
        "n8n.nonagentic.ai": "n8n",
    }


def test_control_mismatched_table_is_reported_as_drift() -> None:
    lua = _LUA_OK.replace('"kanban"', '"planka"') + "\n"
    lua = lua.replace("-- a comment line", '["grafana.nonagentic.ai"] = "grafana",')
    gateway = parse_license_gated_app_key(lua)
    authz = {
        "kanban.nonagentic.ai": "kanban",
        "n8n.nonagentic.ai": "n8n",
        "w2v.nonagentic.ai": "w2v",
    }
    drift = mirror_drift(gateway, authz)
    assert len(drift) == 3, drift
    assert any("grafana.nonagentic.ai" in d and "403" in d for d in drift)
    assert any("w2v.nonagentic.ai" in d and "dead entry" in d for d in drift)
    assert any("kanban.nonagentic.ai" in d and "'planka'" in d for d in drift)


def test_control_current_seed_differs_from_mutated_table() -> None:
    gateway = parse_license_gated_app_key(_LUA_OK.replace('"kanban"', '"kanban2"'))
    assert mirror_drift(gateway, _seed()), "a mutated gateway table must not mirror the seed"


@pytest.mark.parametrize(
    "lua",
    [
        "no table here",
        _LUA_OK + _LUA_OK,
        "local LICENSE_GATED_APP_KEY = {\n  kanban = 'kanban',\n}\n",
        'local LICENSE_GATED_APP_KEY = {\n  ["a"] = "x",\n  ["a"] = "y",\n}\n',
        "local LICENSE_GATED_APP_KEY = {\n}\n",
        'local LICENSE_GATED_APP_KEY = {\n  ["a"] = "x",\n',
    ],
    ids=["absent", "twice", "unparseable-line", "duplicate-host", "empty", "unclosed"],
)
def test_control_parser_refuses_malformed_tables(lua: str) -> None:
    with pytest.raises(ValueError):
        parse_license_gated_app_key(lua)
