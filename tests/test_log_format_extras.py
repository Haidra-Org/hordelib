"""Regression tests for safe rendering of bound extras in the loguru format.

Logging a structured value (e.g. ``logger.debug("msg", node_types={...})``) used to splice the
value's ``repr`` straight into the format *template*. The plain file sink then raised
``KeyError`` on ``format_map`` (the dict keys looked like format fields) and the colorized
console sinks raised "Max string recursion exceeded" in the colorizer. ``_escape_for_format``
neutralises those metacharacters; see ``hordelib.utils.logger``.
"""

from loguru import logger

from hordelib.utils.logger import _color_format, _escape_for_format, _plain_format


def _log_markup_extra_through(fmt, *, colorize: bool) -> list:
    """Emit one record whose extra carries markup-like text through ``fmt``.

    External metadata (e.g. a model description fetched from a remote catalogue) can contain HTML.
    Loguru parses ``<...>`` in the format template on every sink, colorizing or not (the plain sink
    strips markup rather than ignoring it), so an unescaped tag raises and the record is dropped.
    """
    captured: list = []
    logger.remove()
    sink_id = logger.add(captured.append, format=fmt, colorize=colorize, level="DEBUG", catch=False)
    try:
        logger.debug(
            "lora.parse_version_data",
            version_data={"description": "<p>An HTML description.</p>"},
        )
    finally:
        logger.remove(sink_id)
    return captured


def _log_dict_extra_through(fmt, *, colorize: bool) -> list:
    """Emit one record carrying a dict extra through ``fmt`` and return what the sink received.

    ``catch=False`` makes a formatting failure propagate from ``logger.debug`` instead of being
    swallowed, so a regression surfaces as a raised ``KeyError``/``ValueError`` rather than a
    silently dropped message.
    """
    captured: list = []
    logger.remove()
    sink_id = logger.add(captured.append, format=fmt, colorize=colorize, level="DEBUG", catch=False)
    try:
        logger.debug(
            "Pipeline structure before validation",
            node_count=4,
            node_types={"image_loader": "HordeImageLoader", "output_image": "HordeImageOutput"},
        )
    finally:
        logger.remove(sink_id)
    return captured


def test_escape_for_format_neutralises_braces_and_markup():
    assert _escape_for_format("{'a': 'b'}") == "{{'a': 'b'}}"
    # Angle brackets are escaped for every sink: loguru parses <...> as markup even on a
    # non-colorizing sink (it strips markup rather than ignoring it), so an unescaped tag raises.
    assert _escape_for_format("<obj at 0x1>") == r"\<obj at 0x1>"


def test_plain_format_renders_html_extra_without_error():
    captured = _log_markup_extra_through(_plain_format, colorize=False)
    assert len(captured) == 1
    line = str(captured[0])
    assert "lora.parse_version_data" in line
    assert "description" in line


def test_color_format_renders_html_extra_without_error():
    captured = _log_markup_extra_through(_color_format, colorize=True)
    assert len(captured) == 1
    assert "lora.parse_version_data" in str(captured[0])


def test_plain_format_renders_dict_extra_without_error():
    captured = _log_dict_extra_through(_plain_format, colorize=False)
    assert len(captured) == 1
    line = str(captured[0])
    assert "Pipeline structure before validation" in line
    assert "node_types=" in line
    assert "image_loader" in line


def test_color_format_renders_dict_extra_without_error():
    captured = _log_dict_extra_through(_color_format, colorize=True)
    assert len(captured) == 1
    assert "Pipeline structure before validation" in str(captured[0])
