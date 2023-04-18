import contextlib
import sys
from functools import partialmethod

from loguru import logger


class HordeLog:
    STDOUT_LEVELS = ["GENERATION", "PROMPT"]
    INIT_LEVELS = ["INIT", "INIT_OK", "INIT_WARN", "INIT_ERR"]
    MESSAGE_LEVELS = ["MESSAGE"]
    STATS_LEVELS = ["STATS"]

    # By default we're at error level or higher
    verbosity = 20
    quiet = 0

    # Our sink IDs
    sinks = []

    @classmethod
    def set_logger_verbosity(cls, count):
        # The count comes reversed. So count = 0 means minimum verbosity
        # While count 5 means maximum verbosity
        # So the more count we have, the lowe we drop the versbosity maximum
        cls.verbosity = 20 - (count * 10)

    @classmethod
    def quiesce_logger(cls, count):
        # The bigger the count, the more silent we want our logger
        cls.quiet = count * 10

    @classmethod
    def is_stdout_log(cls, record):
        if record["level"].name not in HordeLog.STDOUT_LEVELS:
            return False
        if record["level"].no < cls.verbosity + cls.quiet:
            return False
        return True

    @classmethod
    def is_init_log(cls, record):
        if record["level"].name not in HordeLog.INIT_LEVELS:
            return False
        if record["level"].no < cls.verbosity + cls.quiet:
            return False
        return True

    @classmethod
    def is_msg_log(cls, record):
        if record["level"].name not in HordeLog.MESSAGE_LEVELS:
            return False
        if record["level"].no < cls.verbosity + cls.quiet:
            return False
        return True

    @classmethod
    def is_stderr_log(cls, record):
        if (
            record["level"].name
            in HordeLog.STDOUT_LEVELS + HordeLog.INIT_LEVELS + HordeLog.MESSAGE_LEVELS + HordeLog.STATS_LEVELS
        ):
            return False
        if record["level"].no < cls.verbosity + cls.quiet:
            return False
        return True

    @classmethod
    def is_stats_log(cls, record):
        if record["level"].name not in HordeLog.STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_not_stats_log(cls, record):
        if record["level"].name in HordeLog.STATS_LEVELS:
            return False
        return True

    @classmethod
    def is_trace_log(cls, record):
        if record["level"].name not in ["TRACE", "ERROR"]:
            return False
        return True

    @classmethod
    def test_logger(cls):
        logger.generation(
            "This is a generation message\nIt is typically multiline\nThee Lines".encode(
                "unicode_escape",
            ).decode("utf-8"),
        )
        logger.prompt("This is a prompt message")
        logger.debug("Debug Message")
        logger.info("Info Message")
        logger.warning("Info Warning")
        logger.error("Error Message")
        logger.critical("Critical Message")
        logger.init("This is an init message", status="Starting")
        logger.init_ok("This is an init message", status="OK")
        logger.init_warn("This is an init message", status="Warning")
        logger.init_err("This is an init message", status="Error")
        logger.message("This is user message")
        sys.exit()

    @classmethod
    def initialise(cls, setup_logging=True):
        if setup_logging:
            cls.setup()
            cls.set_sinks()
        else:
            cls.setup_compatibility()

    @classmethod
    def setup_compatibility(cls):
        logger.__class__.generation = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.prompt = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.init = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.init_ok = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.init_warn = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.init_err = partialmethod(logger.__class__.log, "ERROR")
        logger.__class__.message = partialmethod(logger.__class__.log, "INFO")
        logger.__class__.stats = partialmethod(logger.__class__.log, "DEBUG")

    @classmethod
    def setup(cls):
        cls.logfmt = (
            "<level>{level: <10}</level> | <green>{time:YYYY-MM-DD HH:mm:ss.SSSSSS}</green> | "
            "<green>{name}</green>:<green>{function}</green>:<green>{line}</green> - <level>{message}</level>"
        )
        cls.genfmt = (
            "<level>{level: <10}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss.SSSSSS}</green> | <level>{message}</level>"
        )
        cls.initfmt = (
            "<magenta>INIT      </magenta> | <level>{extra[status]: <11}</level> | <magenta>{message}</magenta>"
        )
        cls.msgfmt = "<level>{level: <10}</level> | <level>{message}</level>"

        try:
            logger.level("GENERATION", no=24, color="<cyan>")
            logger.level("PROMPT", no=23, color="<yellow>")
            logger.level("INIT", no=31, color="<white>")
            logger.level("INIT_OK", no=31, color="<green>")
            logger.level("INIT_WARN", no=31, color="<yellow>")
            logger.level("INIT_ERR", no=31, color="<red>")
            # Messages contain important information without which this application might not be able to be used
            # As such, they have the highest priority
            logger.level("MESSAGE", no=61, color="<green>")
            # Stats are info that might not display well in the terminal
            logger.level("STATS", no=19, color="<blue>")
        except TypeError:
            pass

        logger.__class__.generation = partialmethod(logger.__class__.log, "GENERATION")
        logger.__class__.prompt = partialmethod(logger.__class__.log, "PROMPT")
        logger.__class__.init = partialmethod(logger.__class__.log, "INIT")
        logger.__class__.init_ok = partialmethod(logger.__class__.log, "INIT_OK")
        logger.__class__.init_warn = partialmethod(logger.__class__.log, "INIT_WARN")
        logger.__class__.init_err = partialmethod(logger.__class__.log, "INIT_ERR")
        logger.__class__.message = partialmethod(logger.__class__.log, "MESSAGE")
        logger.__class__.stats = partialmethod(logger.__class__.log, "STATS")

    @classmethod
    def set_sinks(cls):
        # Remove any existing sinks that we added
        for sink in cls.sinks:
            with contextlib.suppress(ValueError):
                # Suppress if someone else beat us to it
                logger.remove(sink)

        cls.sinks = []

        config = {
            "handlers": [
                {
                    "sink": sys.stderr,
                    "format": cls.logfmt,
                    "colorize": True,
                    "filter": cls.is_stderr_log,
                },
                {
                    "sink": sys.stdout,
                    "format": cls.genfmt,
                    "level": "PROMPT",
                    "colorize": True,
                    "filter": cls.is_stdout_log,
                },
                {
                    "sink": sys.stdout,
                    "format": cls.initfmt,
                    "level": "INIT",
                    "colorize": True,
                    "filter": cls.is_init_log,
                },
                {
                    "sink": sys.stdout,
                    "format": cls.msgfmt,
                    "level": "MESSAGE",
                    "colorize": True,
                    "filter": cls.is_msg_log,
                },
                {
                    "sink": "logs/bridge.log",
                    "format": cls.logfmt,
                    "level": "DEBUG",
                    "colorize": False,
                    "filter": cls.is_not_stats_log,
                    "retention": "2 days",
                    "rotation": "1 days",
                },
                {
                    "sink": "logs/stats.log",
                    "format": cls.logfmt,
                    "level": "STATS",
                    "colorize": False,
                    "filter": cls.is_stats_log,
                    "retention": "7 days",
                    "rotation": "1 days",
                },
                {
                    "sink": "logs/trace.log",
                    "format": cls.logfmt,
                    "level": "TRACE",
                    "colorize": False,
                    "filter": cls.is_trace_log,
                    "retention": "3 days",
                    "rotation": "1 days",
                    "backtrace": True,
                    "diagnose": True,
                },
            ],
        }

        cls.sinks = logger.configure(**config)
