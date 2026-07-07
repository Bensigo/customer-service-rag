"""Docs drift guard (issue #22): keep the README honest.

Two cheap assertions run in CI so documentation can never silently drift
from the code:

- Every ``make <target>`` the README tells a reader to run must exist as a
  real target in the Makefile.
- Every API route path the README documents (``/documents``, ``/chat``,
  ``/health``, …) must exist in the app's actual route table.

These are structural checks, not content checks: they hold the commands
and routes a reader will copy-paste to the ground truth in ``Makefile``
and ``create_app().routes``, so a renamed target or moved route breaks
the build instead of shipping a broken quickstart.
"""

import re
from pathlib import Path

from app.main import create_app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "README.md"
_MAKEFILE = _REPO_ROOT / "Makefile"

# A ``make foo`` invocation in a README code block or prose.
_MAKE_INVOCATION_RE = re.compile(r"\bmake\s+([a-z][a-z0-9-]*)\b")
# A ``.PHONY`` / target-definition line in the Makefile: "name:" at col 0.
_MAKE_TARGET_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:", re.MULTILINE)
# An API route the README documents inside backticks, e.g. `/chat`,
# `/documents/{doc_id}`, or `POST /documents` (an HTTP method may precede
# the path). Restricted to app routes: a leading slash and our path charset.
_ROUTE_RE = re.compile(r"`(?:[A-Z]+ )?(/[a-z][a-z0-9/{}_-]*)`")

# make sub-commands that are flags/args to python, not Makefile targets, plus
# anything a reader would never expect to be a target.
_NON_TARGET_TOKENS = {"sure", "it"}


def _makefile_targets() -> set[str]:
    return set(_MAKE_TARGET_RE.findall(_MAKEFILE.read_text(encoding="utf-8")))


def _app_route_paths() -> set[str]:
    """Every path the app serves, recursing into FastAPI's included routers.

    Modern FastAPI wraps ``include_router`` results in an ``_IncludedRouter``
    proxy whose real routes hang off ``original_router.routes``; walk both so
    ``/documents`` and ``/chat`` are found alongside ``/health`` and ``/ready``.
    """
    paths: set[str] = set()

    def walk(routes: object) -> None:
        for route in routes:  # type: ignore[attr-defined]
            path = getattr(route, "path", None)
            if isinstance(path, str):
                paths.add(path)
            sub = getattr(route, "routes", None)
            if sub:
                walk(sub)
            original = getattr(route, "original_router", None)
            if original is not None and getattr(original, "routes", None):
                walk(original.routes)

    walk(create_app().routes)
    return paths


def test_readme_make_targets_exist() -> None:
    """Every ``make <target>`` the README mentions is a real Makefile target."""
    readme = _README.read_text(encoding="utf-8")
    referenced = {
        token for token in _MAKE_INVOCATION_RE.findall(readme) if token not in _NON_TARGET_TOKENS
    }
    assert referenced, "expected the README to document at least one make target"
    targets = _makefile_targets()
    missing = referenced - targets
    assert not missing, (
        f"README references make targets absent from the Makefile: {sorted(missing)}"
    )


def test_readme_routes_exist() -> None:
    """Every API route path the README documents exists in the app."""
    readme = _README.read_text(encoding="utf-8")
    documented = {path for path in _ROUTE_RE.findall(readme)}
    # The core API surface the README's reference table must cover.
    expected_core = {"/documents", "/documents/{doc_id}", "/chat", "/health", "/ready"}
    assert expected_core <= documented, (
        f"README must document the core API routes; missing: {sorted(expected_core - documented)}"
    )
    served = _app_route_paths()
    missing = documented - served
    assert not missing, f"README documents routes the app does not serve: {sorted(missing)}"
