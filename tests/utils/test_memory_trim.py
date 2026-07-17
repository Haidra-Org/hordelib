"""Tests for the host memory trim that returns freed heap and cold mmap pages to the OS.

The trim must always return a ``bool`` and never raise, dispatch to the correct platform request, no-op
cleanly where no request exists, and swallow a refusing or absent OS symbol. The gating test proves
``clear_gc_and_torch_cache`` only trims when asked. None of these touch the GPU.
"""

from __future__ import annotations

from unittest import mock

from hordelib.utils import memory_trim
from hordelib.utils.memory_trim import trim_host_memory


def test_returns_bool_and_never_raises_on_this_platform() -> None:
    """The real call on whatever platform the suite runs on returns a bool and does not raise."""
    result = trim_host_memory()
    assert isinstance(result, bool)


def test_linux_dispatch_issues_malloc_trim(monkeypatch) -> None:
    """On Linux the trim loads libc and calls ``malloc_trim`` once, then reports success."""
    fake_libc = mock.MagicMock()
    fake_cdll = mock.MagicMock(return_value=fake_libc)
    monkeypatch.setattr(memory_trim.sys, "platform", "linux")
    monkeypatch.setattr(memory_trim.ctypes, "CDLL", fake_cdll)

    assert trim_host_memory() is True
    fake_cdll.assert_called_once_with("libc.so.6", use_last_error=True)
    fake_libc.malloc_trim.assert_called_once_with(0)


def test_windows_dispatch_empties_working_set(monkeypatch) -> None:
    """On Windows the trim empties the current process working set and reports the OS result."""
    fake_kernel32 = mock.MagicMock()
    fake_psapi = mock.MagicMock()
    fake_psapi.EmptyWorkingSet.return_value = 1
    fake_windll = mock.MagicMock(side_effect=lambda name, **_: {"kernel32": fake_kernel32, "psapi": fake_psapi}[name])
    monkeypatch.setattr(memory_trim.sys, "platform", "win32")
    monkeypatch.setattr(memory_trim.ctypes, "WinDLL", fake_windll, raising=False)

    assert trim_host_memory() is True
    fake_kernel32.GetCurrentProcess.assert_called_once_with()
    fake_psapi.EmptyWorkingSet.assert_called_once()


def test_windows_dispatch_reports_os_refusal(monkeypatch) -> None:
    """A zero return from ``EmptyWorkingSet`` is a refusal, surfaced as False without raising."""
    fake_kernel32 = mock.MagicMock()
    fake_psapi = mock.MagicMock()
    fake_psapi.EmptyWorkingSet.return_value = 0
    fake_windll = mock.MagicMock(side_effect=lambda name, **_: {"kernel32": fake_kernel32, "psapi": fake_psapi}[name])
    monkeypatch.setattr(memory_trim.sys, "platform", "win32")
    monkeypatch.setattr(memory_trim.ctypes, "WinDLL", fake_windll, raising=False)

    assert trim_host_memory() is False


def test_unsupported_platform_is_a_noop(monkeypatch) -> None:
    """A platform with no per-process release request (macOS) trims nothing and reports False."""
    monkeypatch.setattr(memory_trim.sys, "platform", "darwin")
    # No loader should be consulted on the no-op path.
    monkeypatch.setattr(memory_trim.ctypes, "CDLL", mock.MagicMock(side_effect=AssertionError("must not load")))

    assert trim_host_memory() is False


def test_missing_or_refusing_symbol_returns_false(monkeypatch) -> None:
    """When the loader itself raises (no libc, missing symbol), the trim swallows it and returns False."""
    monkeypatch.setattr(memory_trim.sys, "platform", "linux")
    monkeypatch.setattr(memory_trim.ctypes, "CDLL", mock.MagicMock(side_effect=OSError("no libc")))

    assert trim_host_memory() is False


def test_clear_gc_and_torch_cache_trims_only_when_asked(monkeypatch) -> None:
    """``clear_gc_and_torch_cache`` trims the host only under ``trim_host=True``; the default never trims."""
    from hordelib import comfy_horde

    # Isolate the gating decision from the device cache clear so the test does no GPU work.
    monkeypatch.setattr(comfy_horde, "clear_accelerator_cache", mock.MagicMock())
    spy = mock.MagicMock()
    monkeypatch.setattr(comfy_horde, "trim_host_memory", spy)

    comfy_horde.clear_gc_and_torch_cache()
    spy.assert_not_called()

    comfy_horde.clear_gc_and_torch_cache(trim_host=True)
    spy.assert_called_once_with()
