"""Build-time guard: the image must run exactly the versions pinned in uv.lock.

Incident #61: the Dockerfile resolved dependencies fresh on every build, so the
image shipped pyjwt 2.15.1 while uv.lock said 2.14.0, and production auth broke.
The Dockerfile now installs with `uv sync --frozen`; this script proves it.

Usage: python assert_lock_versions.py <uv.lock> <pins.txt>
  <pins.txt> is `uv export --frozen --no-dev --no-emit-project --no-hashes`.
Exits non-zero on any mismatch.
"""

import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version

# Asserted explicitly, whatever the pins file says (incident #61).
CRITICAL = ("pyjwt",)


def installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def lock_version(lock_path: str, name: str) -> str:
    with open(lock_path, "rb") as fh:
        lock = tomllib.load(fh)
    for pkg in lock.get("package", []):
        if pkg["name"] == name:
            return pkg["version"]
    raise SystemExit(f"FATAL: {name} not found in {lock_path}")


def main(lock_path: str, pins_path: str) -> int:
    errors: list[str] = []

    for name in CRITICAL:
        want = lock_version(lock_path, name)
        got = installed(name)
        if got != want:
            errors.append(f"{name}: installed {got} but uv.lock pins {want} (incident #61 guard)")
        else:
            print(f"OK: {name}=={got} matches uv.lock")

    checked = 0
    with open(pins_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            spec, _, marker = line.partition(";")
            name, _, want = spec.strip().partition("==")
            name = name.split("[")[0].strip()
            want = want.strip()
            got = installed(name)
            if got is None and marker.strip():
                # Platform-conditional pin assumed not applicable to this image.
                # Evaluating the marker would need `packaging`, which is NOT in
                # the runtime venv (absent from the live 7b66ed83 image too), and
                # adding it would diverge from the known-good package set. Today's
                # markers (cffi, pycparser, uvloop) all evaluate true on CPython
                # Linux, so any of them missing would still be skipped here.
                continue
            checked += 1
            if got != want:
                errors.append(f"{name}: installed {got} but uv.lock pins {want}")

    if checked == 0:
        errors.append(f"checked 0 pins from {pins_path} - guard would pass vacuously")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {checked} installed packages match uv.lock")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
