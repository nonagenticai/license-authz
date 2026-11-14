"""Guard: uvicorn must hold idle keep-alive connections longer than envoy pools them.

uvicorn's default keep-alive is 5s; the istio-ingressgateway's upstream connection
pool keeps idle connections far longer (envoy default 1h). Envoy then reuses a
connection uvicorn already closed, and the gateway's licence-check Lua httpCall
returns `503 lua_response` ("upstream connect error or disconnect/reset before
headers. reset reason: connection termination") without license-authz ever seeing
the request. Seen live 2026-10-15 (02:19:59Z, 02:51:28Z).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_KEEP_ALIVE_S = 61  # must exceed a 60s envoy idleTimeout (fleet DestinationRule shape)


def _cmd_argv() -> list[str]:
    cmds = [
        line
        for line in (REPO / "Dockerfile").read_text().splitlines()
        if line.strip().startswith("CMD ")
    ]
    assert cmds, "Dockerfile has no CMD"
    # The LAST CMD is the effective one.
    return json.loads(cmds[-1].strip()[len("CMD ") :])


def test_dockerfile_cmd_runs_uvicorn() -> None:
    assert "uvicorn" in _cmd_argv()


def test_dockerfile_cmd_sets_keep_alive_above_envoy_idle() -> None:
    argv = _cmd_argv()
    assert "--timeout-keep-alive" in argv, f"uvicorn CMD lacks --timeout-keep-alive: {argv}"
    value = int(argv[argv.index("--timeout-keep-alive") + 1])
    assert value >= MIN_KEEP_ALIVE_S, f"--timeout-keep-alive {value} < {MIN_KEEP_ALIVE_S}"


def test_main_entrypoint_sets_keep_alive() -> None:
    src = (REPO / "src" / "main.py").read_text()
    match = re.search(r"uvicorn\.run\([^)]*timeout_keep_alive=(\d+)", src)
    assert match, "src/main.py uvicorn.run() lacks timeout_keep_alive"
    assert int(match.group(1)) >= MIN_KEEP_ALIVE_S
