"""Pytest configuration: make the repo root importable.

The package is `agentrx` and `scripts` lives next to `run.py`; both are
importable from the repo root without `pip install -e .`. This conftest
ensures pytest finds them even when invoked from a subdirectory.

Also pins the tmp_path basetemp under the user's HOME so the suite is
runnable on hosts whose default system tmpdir is owned by a different user
(common on shared VMs).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _maybe_redirect_pytest_temproot() -> None:
    """Set PYTEST_DEBUG_TEMPROOT when system tmpdir is not owned by us.

    Pytest's TempPathFactory looks at PYTEST_DEBUG_TEMPROOT first, then falls
    back to ``tempfile.gettempdir()``. On hosts where ``pytest-of-$USER`` in
    the system tmpdir is owned by a different user (e.g. shared scratch
    mounts where /tmp is bind-mounted from a root-managed volume), pytest
    raises OSError on every tmp_path fixture setup. We probe the default
    location and, only if it is unhealthy, redirect to a repo-local
    ``.pytest_tmp`` directory that is guaranteed to be owned by the test
    runner (since the runner owns the repo checkout).

    This is a no-op on hosts where the default tmpdir is healthy or where
    the user has already set PYTEST_DEBUG_TEMPROOT.
    """
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return
    user = os.environ.get("USER", "user")
    system_tmp = Path(tempfile.gettempdir())
    pyt_of_user = system_tmp / f"pytest-of-{user}"
    if not pyt_of_user.exists() or pyt_of_user.owner() == user:
        return
    fallback = REPO_ROOT / ".pytest_tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(fallback)


_maybe_redirect_pytest_temproot()
