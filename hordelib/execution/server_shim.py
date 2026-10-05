"""The headless stand-in ComfyUI's executor uses in place of a real PromptServer.

ComfyUI's ``PromptExecutor`` takes an ``ExecutionServer`` protocol object. It reads and writes
``client_id`` (assigned from ``extra_data`` each run), writes ``last_node_id`` as nodes execute,
reads ``sockets_metadata`` when preview images are enabled, and delivers every execution event
through ``send_sync``. The protocol also includes ``queue_updated`` for queue-owning callers;
there is no external queue in the embedded bridge, so that notification is a no-op here. This
contract is pinned against the vendored ComfyUI by ``tests/test_comfy_contract_drift.py``.

[`HeadlessComfyServer`][hordelib.execution.server_shim.HeadlessComfyServer] names that contract
as a real class instead of duck-typing the bridge object itself, and forwards events to a
listener callable.

This module must remain importable before ``hordelib.initialise()``: it never imports ComfyUI.
"""

from __future__ import annotations

import typing
from collections.abc import Callable

if typing.TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["HeadlessComfyServer"]

type _LifecycleEventPayload = dict[str, typing.Any]
type _PreviewImage = tuple[str, Image, int | None]
type _PreviewMetadata = dict[str, str | None]
type _BinaryPreviewEventPayload = _PreviewImage | tuple[_PreviewImage, _PreviewMetadata]
type _EventListener = Callable[[str, _LifecycleEventPayload, str | None], None]


class HeadlessComfyServer:
    """Represents the server surface ComfyUI's executor requires when embedded without a web server.

    Instances are handed to ``PromptExecutor`` as its ``server``; ComfyUI mutates the
    attributes directly, so they are plain fields rather than properties.
    """

    client_id: str | None
    last_node_id: str | None
    sockets_metadata: dict[str, dict[str, typing.Any]]
    _event_listener: _EventListener

    def __init__(self, event_listener: _EventListener) -> None:
        """Initialise the shim.

        Args:
            event_listener: Called with every event the executor delivers via ``send_sync``.
        """
        self.client_id = None
        self.last_node_id = None
        self.sockets_metadata = {}
        self._event_listener = event_listener

    @typing.overload
    def send_sync(self, event: str, data: _LifecycleEventPayload, sid: str | None = None) -> None: ...

    @typing.overload
    def send_sync(self, event: int, data: _BinaryPreviewEventPayload, sid: str | None = None) -> None: ...

    @typing.overload
    def send_sync[EventPayload](self, event: str | int, data: EventPayload, sid: str | None = None) -> None: ...

    def send_sync[EventPayload](self, event: str | int, data: EventPayload, sid: str | None = None) -> None:
        """Handle one event emitted by ComfyUI's embedded executor.

        Args:
            event: A lifecycle event label (see
                [`ComfyEventLabel`][hordelib.execution.comfy_events.ComfyEventLabel]) or
                ComfyUI binary event id.
            data: A lifecycle dictionary or ComfyUI preview-image tuple. This name preserves
                ComfyUI's external ``ExecutionServer`` signature.
            sid: The client id the event addresses, when any. This name preserves ComfyUI's
                external ``ExecutionServer`` signature.

        Raises:
            TypeError: A string lifecycle event contains a non-dictionary payload.
        """
        # ComfyUI's ExecutionServer protocol also carries PIL preview images to WebSocket
        # clients as integer events. This shim advertises no WebSocket feature flags and has
        # no binary consumer, so deliberately discard that transport-only side channel.
        if isinstance(event, int):
            return
        if not isinstance(data, dict):
            raise TypeError(f"Lifecycle event {event!r} must contain a dictionary payload")
        self._event_listener(event, data, sid)

    def queue_updated(self) -> None:
        """Accept ComfyUI queue notifications; the embedded executor has no external queue."""
