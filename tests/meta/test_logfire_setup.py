"""Guards for logfire initialization: idempotency and the external-host opt-out.

Embedders (the worker) configure logfire/loguru themselves in every process; hordelib
re-configuring on import used to clobber their handlers and spam "Attempting to
instrument while already instrumented" warnings.
"""

import logfire
import pytest
from loguru import logger

from hordelib.integrations import logfire_setup


def _fail_if_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("must not be called")


def test_second_initialization_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # hordelib's import already ran initialize_logfire once in this process.
    assert logfire_setup._logfire_initialized

    monkeypatch.setattr(logfire, "configure", _fail_if_called)
    monkeypatch.setattr(logger, "configure", _fail_if_called)

    logfire_setup.initialize_logfire()


def test_external_logfire_mode_skips_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logfire_setup, "_logfire_initialized", False)
    monkeypatch.setenv(logfire_setup.HORDELIB_EXTERNAL_LOGFIRE_ENV_VAR, "1")
    monkeypatch.setattr(logfire, "configure", _fail_if_called)
    monkeypatch.setattr(logfire, "instrument_pydantic", _fail_if_called)
    monkeypatch.setattr(logger, "configure", _fail_if_called)

    logfire_setup.initialize_logfire()

    # The guard must latch so repeated imports stay silent too.
    assert logfire_setup._logfire_initialized
