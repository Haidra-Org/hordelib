import io
from collections import deque
from collections.abc import Callable
from enum import Enum
from time import perf_counter

import regex
from loguru import logger
from pydantic import BaseModel


class ComfyUIProgressUnit(Enum):
    """An enum to represent the different types of progress bars that ComfyUI can output.

    This is used to determine how to parse the progress bar and log it.
    """

    ITERATIONS_PER_SECOND = 1
    SECONDS_PER_ITERATION = 2
    UNKNOWN = 3


class ComfyUIProgress(BaseModel):
    """A dataclass to represent the progress of a ComfyUI job.

    This is used to determine how to parse the progress bar and log it.
    """

    percent: int
    current_step: int
    total_steps: int
    rate: float
    rate_unit: ComfyUIProgressUnit

    def __str__(self):
        return f"{self.percent}%: {self.current_step}/{self.total_steps} ({self.rate} {self.rate_unit})"


class OutputCollector(io.TextIOWrapper):
    """A wrapper around stdout/stderr that collects the output in a deque, and replays it on demand.

    Includes a time elapsed appending to each line, starting from when OutputCollector was instantiated.
    The captured output is typically from ComfyUI and originates from a `print()` or `tqdm.write()` call.
    """

    start_time: float
    slow_message_count: int = 0

    capture_deque: deque

    comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None
    """A callback function that is called when a progress bar is detected in the output. The callback function should \
    accept two arguments: a ComfyUIProgress object and a string. The ComfyUIProgress object contains the parsed \
    progress bar information, and the string contains the original message that was captured."""

    def __init__(self, *, comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None):
        logger.disable("tqdm")  # just.. no
        self.capture_deque = deque()
        self.comfyui_progress_callback = comfyui_progress_callback
        self.start_time = perf_counter()

        self.pattern_it_per_s = regex.compile(r"(\d+)%.*\s(\d+)/(\d+)\s.*\s([\?\d]+\.?\d*)it/s]")
        self.pattern_s_per_it = regex.compile(r"(\d+)%.*\s(\d+)/(\d+)\s.*\s([\?\d]+\.?\d*)s/it]")
        self.pattern_pipe = regex.compile(r"\|.*?\|")
        self.pattern_whitespace = regex.compile(r"\s{5,}")
        self.pattern_double_space = regex.compile(r"\s{2,}")

    def write(self, message: str):
        message = message.strip()
        if not message:
            return

        # If the has a percent followed by a pipe, it's a progress bar;
        # log at (approximately) 0%, 25%, 50%, 75%, 100%
        if "%|" in message:
            matches = None
            is_iterations_per_second = None
            if "DDIM Sampler:" in message:
                message = message[len("DDIM Sampler:") :]
                message = message.strip()
            if "it/s]" in message:
                matches = self.pattern_it_per_s.match(message)
                is_iterations_per_second = True
            elif "s/it]" in message:
                matches = self.pattern_s_per_it.match(message)
                is_iterations_per_second = False

            if not matches:
                logger.debug(f"Unknown progress bar format?: {message}")
                self.capture_deque.append(message)
                return

            # Remove everything in between '|' and '|'
            message = self.pattern_pipe.sub("", message)

            # Remove all cases of more than 5 whitespace characters in a row
            message = self.pattern_whitespace.sub(" ", message)

            # Add a timestamp to the log
            message = f"{message} ({perf_counter() - self.start_time:.2f} seconds in ComfyUI)"

            # found_percent_number = int(matches.group(1))
            found_current_step = int(matches.group(2))
            found_total_steps = int(matches.group(3))
            iteration_rate = matches.group(4)

            # Remove any double spaces
            message = self.pattern_double_space.sub(" ", message)
            from hordelib.comfy_horde import log_free_ram

            if (
                self.slow_message_count < 5
                and iteration_rate != "?"
                and (not is_iterations_per_second or float(iteration_rate) < 1.2)
            ):
                self.slow_message_count += 1
                if self.slow_message_count == 5:
                    logger.warning("Suppressing further slow job warnings. Please investigate.")

                    log_free_ram()
                else:
                    rate_unit = "iterations per second" if is_iterations_per_second else "*seconds per iterations*"
                    logger.warning(f"Job Slowdown: Job is running at {iteration_rate} {rate_unit}.")

            if found_current_step == 0:
                log_free_ram()
                logger.info("Job will show progress for the first three steps, and then every 10 steps.")

            # Log the first 3 steps, then every 10 steps, then the last step
            if (
                found_current_step in [1, 2, 3]
                or found_current_step % 10 == 0
                or found_current_step == found_total_steps
            ):
                logger.info(message)

            if self.comfyui_progress_callback:
                self.comfyui_progress_callback(
                    ComfyUIProgress(
                        percent=int(matches.group(1)),
                        current_step=found_current_step,
                        total_steps=found_total_steps,
                        rate=float(iteration_rate) if iteration_rate != "?" else -1.0,
                        rate_unit=(
                            ComfyUIProgressUnit.ITERATIONS_PER_SECOND
                            if is_iterations_per_second
                            else ComfyUIProgressUnit.SECONDS_PER_ITERATION
                        ),
                    ),
                    message,
                )

        self.capture_deque.append(message)

    def set_size(self, size):
        while len(self.capture_deque) > size:
            self.capture_deque.popleft()

    def flush(self):
        pass

    def isatty(self):
        # No, we are not a TTY
        return False

    def close(self):
        pass

    def replay(self):
        logger.debug("Replaying output. Seconds in parentheses is the elapsed time spent in ComfyUI. ")
        while len(self.capture_deque):
            logger.debug(self.capture_deque.popleft())
