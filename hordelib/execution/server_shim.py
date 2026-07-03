"""The headless stand-in ComfyUI's executor uses in place of a real PromptServer.

ComfyUI's ``PromptExecutor`` takes a server object and, when running headless, touches
exactly this surface: it reads and writes ``client_id`` (assigned from ``extra_data`` each
run), writes ``last_node_id`` as nodes execute, reads ``sockets_metadata`` when preview
images are enabled, and delivers every execution event through ``send_sync``. That contract
is pinned against the vendored ComfyUI by ``tests/test_comfy_contract_drift.py`` (a strict
fake that raises on any attribute outside this surface).

:class:`HeadlessComfyServer` names that contract as a real class instead of duck-typing the
bridge object itself, and forwards events to a listener callable.

This module must remain importable before ``hordelib.initialise()``: it never imports ComfyUI.
"""

import typing
from collections.abc import Callable

EventListener = Callable[[str, dict, str | None], None]
"""Receives each executor event as ``(label, data, client_id)``."""


class HeadlessComfyServer:
    """Represents the server surface ComfyUI's executor requires when embedded without a web server.

    Instances are handed to ``PromptExecutor`` as its ``server``; ComfyUI mutates the
    attributes directly, so they are plain fields rather than properties.
    """

    def __init__(self, event_listener: EventListener) -> None:
        """Initialise the shim.

        Args:
            event_listener: Called with every event the executor delivers via ``send_sync``.
        """
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.sockets_metadata: dict[str, typing.Any] = {}
        self._event_listener = event_listener

    def send_sync(self, label: str, data: dict, sid: str | None = None) -> None:
        """Forward one executor event to the listener.

        Args:
            label: The event label (see
                :class:`hordelib.execution.comfy_events.ComfyEventLabel`).
            data: The raw event payload.
            sid: The client id the event addresses, when any.
        """
        self._event_listener(label, data, sid)
