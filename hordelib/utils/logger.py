import contextlib
import sys

from loguru import logger


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

        config = {
            "handlers": [
                {
                    "sink": sys.stderr,
                    "colorize": True,
                    "filter": cls.is_stderr_log,
                    "level": verbosity_level,
                },
                {
                    "sink": sys.stdout,
                    "colorize": True,
                    "filter": cls.is_stdout_log,
                    "level": verbosity_level,
                },
                {
                    "sink": "logs/bridge.log" if cls.process_id is None else f"logs/bridge_{cls.process_id}.log",
                    "level": "DEBUG",
                    "retention": "2 days",
                    "rotation": "1 days",
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
            logger.debug(f"Logger finished setting up for process {cls.process_id}")
        else:
            logger.debug("Setting up logger for main process")
