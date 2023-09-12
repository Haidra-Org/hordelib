import io
from collections import deque
from time import perf_counter

import regex
from loguru import logger


class OutputCollector(io.TextIOWrapper):
    """A wrapper around stdout/stderr that collects the output in a deque, and replays it on demand.

    Includes a time elapsed appending to each line, starting from when OutputCollector was instantiated.
    The captured output is typically from ComfyUI and originates from a `print()` or `tqdm.write()` call.
    """

    start_time: float
    slow_message_count: int = 0

    def __init__(self):
        logger.disable("tqdm")  # just.. no
        self.deque = deque()
        self.start_time = perf_counter()

    def write(self, message: str):
        if message != "\n" and "MemoryEfficientCrossAttention." not in message:
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
                    matches = regex.match(r"(\d+)%.*\s(\d+)/(\d+)\s.*\s([\?\d]+\.?\d*)it/s]", message)
                    is_iterations_per_second = True
                elif "s/it]" in message:
                    matches = regex.match(r"(\d+)%.*\s(\d+)/(\d+)\s.*\s([\?\d]+\.?\d*)s/it]", message)
                    is_iterations_per_second = False

                if not matches:
                    logger.debug(f"Unknown progress bar format?: {message}")
                    self.deque.append(message)
                    return

                # Remove everything in between '|' and '|'
                message = regex.sub(r"\|.*?\|", "", message)

                # Remove all cases of more than 5 whitespace characters in a row
                message = regex.sub(r"\s{5,}", " ", message)

                # Add a timestamp to the log
                message = f"{message} ({perf_counter() - self.start_time:.2f} seconds in ComfyUI)"

                # found_percent_number = int(matches.group(1))
                found_current_step = int(matches.group(2))
                found_total_steps = int(matches.group(3))
                iteration_rate = matches.group(4)

                if (
                    self.slow_message_count < 5
                    and iteration_rate != "?"
                    and (not is_iterations_per_second or float(iteration_rate) < 1.2)
                ):
                    self.slow_message_count += 1
                    if self.slow_message_count == 5:
                        logger.warning("Suppressing further slow job warnings. Please investigate.")
                    else:
                        rate_unit = "iterations per second" if is_iterations_per_second else "*seconds per iterations*"
                        logger.warning(f"Job Slowdown: Job is running at {iteration_rate} {rate_unit}.")

                if found_current_step == 0:
                    logger.info("Job will show progress for the first three steps, and then every 10 steps.")

                # Log the first 2 steps, then every 10 steps, then the last step
                if (
                    found_current_step in [1, 2, 3]
                    or found_current_step % 10 == 0
                    or found_current_step == found_total_steps
                ):
                    logger.info(message)

            self.deque.append(message)

    def set_size(self, size):
        while len(self.deque) > size:
            self.deque.popleft()

    def flush(self):
        pass

    def isatty(self):
        # No, we are not a TTY
        return False

    def close(self):
        pass

    def replay(self):
        logger.debug("Replaying output. Seconds in parentheses is the elapsed time spent in ComfyUI. ")
        while len(self.deque):
            logger.debug(self.deque.popleft())
