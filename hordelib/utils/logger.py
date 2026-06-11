import atexit
import contextlib
import sys

from loguru import logger


def _format_with_extras(record, *, color: bool) -> str:
    """Generate the log format string including any bound extras."""

    # Check if this log came from stdlib logging via InterceptHandler
    # If so, use the stdlib source location info for better accuracy
    extras = record["extra"]
    if "stdlib_pathname" in extras:
        # This is a stdlib logging message intercepted by our handler
        # Use the original source location from the LogRecord
        import os

        name = extras.get("stdlib_loggername", "unknown")
        function = extras.get("stdlib_funcname", "unknown")
        line = extras.get("stdlib_lineno", 0)

        # Extract just the filename from the full path for readability
        pathname = extras.get("stdlib_pathname", "")
        filename = os.path.basename(pathname) if pathname else "unknown"

        if color:
            # Use {extra[key]} to safely access values without interpretation as color tags
            base = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                "<cyan>{extra[stdlib_loggername]}</cyan>:<cyan>" + filename + "</cyan>:"
                "<cyan>{extra[stdlib_funcname]}</cyan>:<cyan>{extra[stdlib_lineno]}</cyan> - "
                "<level>{message}</level>"
            )
        else:
            base = "{{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{extra[stdlib_loggername]}}:{filename}:{{extra[stdlib_funcname]}}:{{extra[stdlib_lineno]}} - {{message}}".format(
                filename=filename,
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
        extra_repr = ", ".join(extra_items)
        extra_str = f" | {extra_repr}"

    return base + extra_str + "\n{exception}"


def _color_format(record) -> str:
    return _format_with_extras(record, color=True)


def _plain_format(record) -> str:
    return _format_with_extras(record, color=False)


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
                    "retention": "2 days",
                    "rotation": "1 days",
                    "format": _plain_format,
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
                    "retention": "3 days",
                    "rotation": "1 days",
                    "backtrace": True,
                    "diagnose": True,
                    "format": _plain_format,
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
