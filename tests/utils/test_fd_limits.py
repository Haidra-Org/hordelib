"""Tests for the open-file soft-limit raise that hardens cross-process tensor sharing.

The raise must only ever raise the ceiling (never lower it), must never raise an exception, and must be a
clean no-op on platforms without ``RLIMIT_NOFILE`` (Windows). These run on any platform.
"""

from __future__ import annotations

import sys

import pytest

from hordelib.utils.fd_limits import raise_open_file_soft_limit

_POSIX = sys.platform != "win32"


def _soft_limit() -> int | None:
    """The current soft ``RLIMIT_NOFILE``, or None on a platform without it."""
    try:
        import resource
    except ImportError:
        return None
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def test_raise_never_lowers_the_limit() -> None:
    """Raising returns a non-decreasing pair, or None when there is nothing to do."""
    before = _soft_limit()
    result = raise_open_file_soft_limit()
    after = _soft_limit()
    if result is None:
        assert before == after
    else:
        old_soft, new_soft = result
        assert new_soft >= old_soft
        assert after is None or after >= (before or 0)


def test_idempotent_second_call_is_a_noop() -> None:
    """Once the soft limit sits at the hard ceiling, a second raise finds nothing to do."""
    raise_open_file_soft_limit()
    assert raise_open_file_soft_limit() is None


@pytest.mark.skipif(_POSIX, reason="Windows has no RLIMIT_NOFILE; the raise is a documented no-op there")
def test_noop_on_windows() -> None:
    """On Windows there is no descriptor ceiling, so the raise reports nothing done."""
    assert raise_open_file_soft_limit() is None
