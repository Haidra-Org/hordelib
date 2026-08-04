import atexit
import contextlib
import sys
import threading
import time

from loguru import logger


def _escape_for_format(text: str, *, color: bool) -> str:
    """Neutralise characters loguru would otherwise parse inside a format template.

    Anything spliced directly into the returned format string (rather than referenced via a
    ``{...}`` field) is parsed by loguru: ``{``/``}`` as format fields and, when the sink
    colorizes, ``<...>`` as color markup. Escaping keeps repr'd extras literal, so a value
    like ``{'image_loader': ...}`` or ``<obj at 0x...>`` cannot raise ``KeyError`` on the
    plain sink or "Max string recursion exceeded" in the colorizer.
    """
    text = text.replace("{", "{{").replace("}", "}}")
    if color:
        text = text.replace("<", r"\<")
    return text


def _format_with_extras(record, *, color: bool) -> str:
    """Generate the log format string including any bound extras."""

    # Check if this log came from stdlib logging via InterceptHandler
    # If so, use the stdlib source location info for better accuracy
    extras = record["extra"]
    if "stdlib_pathname" in extras:
        # This is a stdlib logging message intercepted by our handler
        # Use the original source location from the LogRecord
        import os

        # Extract just the filename from the full path for readability. It is spliced
        # straight into the template below, so escape it like any other dynamic literal.
        pathname = extras.get("stdlib_pathname", "")
        filename = os.path.basename(pathname) if pathname else "unknown"
        filename = _escape_for_format(filename, color=color)

        if color:
            # Use {extra[key]} to safely access values without interpretation as color tags
            base = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                "<cyan>{extra[stdlib_loggername]}</cyan>:<cyan>" + filename + "</cyan>:"
                "<cyan>{extra[stdlib_funcname]}</cyan>:<cyan>{extra[stdlib_lineno]}</cyan> - "
                "<level>{message}</level>"
            )
        else:
            base = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[stdlib_loggername]}:"
                + filename
                + ":{extra[stdlib_funcname]}:{extra[stdlib_lineno]} - {message}"
            )
    else:
        # Normal loguru log - use loguru's own source tracking
        if color:
            base = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
        else:
            base = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

    # Avoid modifying the original extras when rendering the formatted string
    # Skip stdlib_ extras as we've already used them above
    extra_items = [
        f"{key}={value!r}"
        for key, value in sorted(record["extra"].items())
        if not key.startswith("_") and not key.startswith("stdlib_")
    ]
    extra_str = ""
    if extra_items:
        extra_repr = _escape_for_format(", ".join(extra_items), color=color)
        extra_str = f" | {extra_repr}"

    return base + extra_str + "\n{exception}"


def _color_format(record) -> str:
    return _format_with_extras(record, color=True)


def _plain_format(record) -> str:
    return _format_with_extras(record, color=False)


_log_throttle_lock = threading.Lock()
_log_throttle_last_emit_monotonic: dict[str, float] = {}


def throttled_log_level(
    key: str,
    interval_seconds: float,
    *,
    normal_level: str = "DEBUG",
    suppressed_level: str = "TRACE",
    now: float | None = None,
) -> str:
    """Pick the level a repeating log site should use so it emits at full level once per interval.

    Per-step and per-event sites fire far faster than a reader or a support bundle can absorb: at full
    level they crowd every other line out of the log and cost real time in the hot path. This returns
    ``normal_level`` for the first call on a ``key`` and for the first call once ``interval_seconds``
    has elapsed since that key last did so, and ``suppressed_level`` for every call in between. The
    site keeps a complete record at the quieter level while a normal-verbosity log keeps one line per
    interval.

    Callers pass the result to ``logger.log(...)`` rather than having this function emit, so the record
    still reports the real call site instead of this module.

    ``key`` separates independent sites. Give each message a stable key, and where one site's content
    varies meaningfully per call (an event type, a node name) fold that variant into the key, so each
    variant surfaces on its own schedule instead of whichever fires first masking the rest.

    ``now`` substitutes for the monotonic clock reading, letting a caller drive the schedule directly.
    Thread-safe: callers reached from several threads share one schedule per key.
    """
    reading = time.monotonic() if now is None else now
    with _log_throttle_lock:
        last_emit = _log_throttle_last_emit_monotonic.get(key)
        if last_emit is not None and (reading - last_emit) < interval_seconds:
            return suppressed_level
        _log_throttle_last_emit_monotonic[key] = reading
    return normal_level


def reset_log_throttle_state() -> None:
    """Forget every throttled site's last-emission time, so the next call on any key emits at full level.

    The schedule is process-global module state, so a caller that needs a known starting point (a test
    case, a reused process about to start unrelated work) clears it here.
    """
    with _log_throttle_lock:
        _log_throttle_last_emit_monotonic.clear()


class HordeLog:
    # By default we're at info level or higher
    verbosity: int = 20
    quiet: int = 0

    process_id: int | None = None

    CUSTOM_STATS_LEVELS = ["STATS"]

    # Our sink IDs
    sinks: list[int] = []  # default mutable because this is a class variable (class is a singleton)

    @classmethod
    def set_logger_verbosity(cls, count):
        if count == 2:
            cls.verbosity = 25
        else:
            cls.verbosity = 50 - (count * 10)

    @classmethod
    def is_stats_log(cls, record):
        if record["level"].name in HordeLog.CUSTOM_STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_not_stats_log(cls, record):
        if record["level"].name not in HordeLog.CUSTOM_STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_stderr_log(cls, record):
        if record["level"].name not in ["ERROR", "CRITICAL", "WARNING"]:
            return False
        return True

    @classmethod
    def is_trace_log(cls, record):
        if record["level"].name not in ["TRACE", "ERROR", "CRITICAL"]:
            return False
        return True

    @classmethod
    def is_stdout_log(cls, record):
        return not cls.is_stderr_log(record)

    @classmethod
    def test_logger(cls):
        logger.debug("Debug Message")
        logger.info("Info Message")
        logger.warning("Info Warning")
        logger.error("Error Message")
        logger.critical("Critical Message")

        # logger.log("STATS", "Stats Message")

        a = 0

        @logger.catch
        def main():
            a.item()  # This will raise an exception

        main()

        sys.exit()

    @classmethod
    def initialise(
        cls,
        setup_logging=True,
        process_id: int | None = None,
        verbosity_count: int = 1,
    ):
        cls.set_logger_verbosity(verbosity_count)
        if setup_logging:
            cls.process_id = process_id
            cls.set_sinks()
            atexit.register(cls.shutdown)

    @classmethod
    def set_sinks(cls) -> None:
        # Remove any existing sinks that we added
        for sink in cls.sinks:
            with contextlib.suppress(ValueError):
                # Suppress if someone else beat us to it
                logger.remove(sink)

        # Get the level corresponding to the verbosity
        # We want to log to stdout at that level

        levels_lookup: dict[int, str] = {
            5: "TRACE",
            10: "DEBUG",
            20: "INFO",
            25: "SUCCESS",
            30: "WARNING",
            40: "ERROR",
            50: "CRITICAL",
        }

        verbosity_level = "INFO"

        for level in levels_lookup:
            if cls.verbosity <= level:
                verbosity_level = levels_lookup[level]
                break

        # Use __stdout__/__stderr__ (the OS-level streams) for the main process
        # to avoid capturing pytest's temporary capture streams, which get closed
        # at the end of the test session and cause "I/O operation on closed file"
        # errors when background threads (e.g. OTel exporters) emit late log messages.
        stderr_sink = sys.__stderr__ if cls.process_id is None else sys.stderr
        stdout_sink = sys.__stdout__ if cls.process_id is None else sys.stdout

        config = {
            "handlers": [
                {
                    "sink": stderr_sink,
                    "colorize": True,
                    "filter": cls.is_stderr_log,
                    "level": verbosity_level,
                    "format": _color_format,
                },
                {
                    "sink": stdout_sink,
                    "colorize": True,
                    "filter": cls.is_stdout_log,
                    "level": verbosity_level,
                    "format": _color_format,
                },
                {
                    "sink": "logs/bridge.log" if cls.process_id is None else f"logs/bridge_{cls.process_id}.log",
                    "level": "DEBUG",
                    # Rotate on a 25MB size cap so a heavy run can't grow one file large enough to
                    # choke a tailing reader; zip rotated files and keep a bounded count so total
                    # disk use stays bounded regardless of how fast logs are written. (loguru 0.7.x
                    # takes a single rotation condition, not a list.)
                    "retention": 20,
                    "rotation": "25 MB",
                    "compression": "zip",
                    "format": _plain_format,
                    # Move disk writes off the hot (inference) thread; flushed by atexit shutdown.
                    "enqueue": True,
                },
                # {
                #     "sink": "logs/stats.log" if cls.process_id is None else f"logs/stats_{cls.process_id}.log",
                #     "level": "STATS",
                #     "filter": cls.is_stats_log,
                #     "retention": "7 days",
                #     "rotation": "1 days",
                # },
                {
                    "sink": "logs/trace.log" if cls.process_id is None else f"logs/trace_{cls.process_id}.log",
                    "level": "TRACE",
                    "filter": cls.is_trace_log,
                    # Trace is the most verbose sink; keep fewer rotated files than bridge.log.
                    "retention": 10,
                    "rotation": "25 MB",
                    "compression": "zip",
                    "backtrace": True,
                    "diagnose": True,
                    "format": _plain_format,
                    "enqueue": True,
                },
            ],
        }

        if cls.process_id is not None:
            # Remove the first 2 handlers, they're for the main process only
            config["handlers"] = config["handlers"][2:]

            # Redirect stdout/stderr to a file
            sys.stdout = open(f"logs/stdout_{cls.process_id}.log", "w")
            sys.stderr = open(f"logs/stderr_{cls.process_id}.log", "w")

        # logger.level("STATS", no=25, color="<yellow>", icon="📊")
        cls.sinks = logger.configure(**config)  # type: ignore

        if cls.process_id is not None:
            logger.debug("Logger finished setting up for process: process_id={}", cls.process_id)
        else:
            logger.debug("Setting up logger for main process")

    @classmethod
    def shutdown(cls) -> None:
        """Remove all loguru sinks that were added by this class.

        Called automatically via atexit to prevent "I/O operation on closed file"
        errors when background threads emit log messages during interpreter shutdown.
        """
        for sink in cls.sinks:
            with contextlib.suppress(ValueError):
                logger.remove(sink)
        cls.sinks.clear()
