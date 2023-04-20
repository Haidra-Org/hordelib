import io
from collections import deque

from loguru import logger


class OutputCollector(io.TextIOWrapper):
    def __init__(self):
        logger.disable("tqdm")  # just.. no
        self.deque = deque()

    def write(self, s):
        if s != "\n":
            self.deque.append(s.strip())

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
        while len(self.deque):
            logger.debug(self.deque.popleft())
