import contextlib
import sys

from loguru import logger


class HordeLog:
    # By default we're at error level or higher
    verbosity: int = 20
    quiet: int = 0

    process_id: int | None = None

    CUSTOM_STATS_LEVELS = ["STATS"]

    # Our sink IDs
    sinks: list[int] = []  # default mutable because this is a class variable (class is a singleton)

    @classmethod
    def set_logger_verbosity(cls, count):
        cls.verbosity = 50 - (count * 10)

    @classmethod
    def is_stats_log(cls, record):
        if record["level"].name not in HordeLog.CUSTOM_STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_not_stats_log(cls, record):
        if record["level"].name in HordeLog.CUSTOM_STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_stderr_log(cls, record):
        if record["level"].name not in ["ERROR", "CRITICAL", "TRACE"]:
            return False
        return True

    @classmethod
    def is_trace_log(cls, record):
        if record["level"].name != "ERROR":
            return False
        return True

    @classmethod
    def test_logger(cls):
        logger.debug("Debug Message")
        logger.info("Info Message")
        logger.warning("Info Warning")
        logger.error("Error Message")
        logger.critical("Critical Message")

        logger.log("STATS", "Stats Message")

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
        quiet_count: int = 0,
    ):
        logger.debug(f"Initialising logger for process {process_id}")
        cls.set_logger_verbosity(verbosity_count)
        if setup_logging:
            cls.process_id = process_id
            cls.set_sinks()

    @classmethod
    def set_sinks(cls):
        # Remove any existing sinks that we added
        for sink in cls.sinks:
            with contextlib.suppress(ValueError):
                # Suppress if someone else beat us to it
                logger.remove(sink)

        cls.sinks = []

        # Get the level corresponding to the verbosity
        # We want to log to stdout at that level

        levels_lookup: list[str, int] = {
            5: "TRACE",
            10: "DEBUG",
            20: "INFO",
            25: "SUCCESS",
            30: "WARNING",
            40: "ERROR",
            50: "CRITICAL",
        }

        stdout_level = "INFO"

        for level in levels_lookup:
            if cls.verbosity <= level:
                stdout_level = levels_lookup[level]
                break

        print("logs/bridge.log" if cls.process_id is None else f"logs/bridge_{cls.process_id}.log")

        config = {
            "handlers": [
                {
                    "sink": sys.stderr,
                    "colorize": True,
                    "filter": cls.is_stderr_log,
                    "enqueue": True,
                },
                {
                    "sink": sys.stdout,
                    "colorize": True,
                    "enqueue": True,
                    "level": stdout_level,
                },
                {
                    "sink": "logs/bridge.log" if cls.process_id is None else f"logs/bridge_{cls.process_id}.log",
                    "level": "DEBUG",
                    "retention": "2 days",
                    "rotation": "1 days",
                    "enqueue": True,
                },
                {
                    "sink": "logs/stats.log" if cls.process_id is None else f"logs/stats_{cls.process_id}.log",
                    "level": "STATS",
                    "filter": cls.is_stats_log,
                    "retention": "7 days",
                    "rotation": "1 days",
                    "enqueue": True,
                },
                {
                    "sink": "logs/trace.log" if cls.process_id is None else f"logs/trace_{cls.process_id}.log",
                    "level": "ERROR",
                    "retention": "3 days",
                    "rotation": "1 days",
                    "backtrace": True,
                    "diagnose": True,
                    "enqueue": True,
                },
            ],
        }
        logger.debug(f"Setting up logger for process {cls.process_id}")

        logger.level("STATS", no=25, color="<yellow>", icon="📊")
        cls.sinks = logger.configure(**config)
