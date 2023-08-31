import os
import re
from collections import deque
from pathlib import Path

import psutil
from typing_extensions import Self

from hordelib import is_initialised
from hordelib.utils.switch import Switch


class UserSettings:
    """Container class for all user settings."""

    _instance: Self | None = None

    _ram_to_leave_free_mb: int
    """The amount of RAM to leave free, defaults to 50%, can be expressed as a number of MB or a percentage."""
    _vram_to_leave_free_mb: int
    """The amount of VRAM to leave free, defaults to 50% of the current machines VRAM, can be expressed as a number of
     MB or a percentage."""

    _model_directory: Path | None = None
    _basedir = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _is_percentage(value) -> float | bool:
        if isinstance(value, str):
            if re.match(r"^\d+(\.\d+)?%$", value):
                return float(value.strip("%"))
        return False

    @staticmethod
    def _get_total_vram_mb() -> int:
        # This might not be pretty, but it allows the import of hordelib on systems
        # which don't actually have a GPU (e.g. github ci containers)
        if not is_initialised():
            return 0

        from hordelib.utils.gpuinfo import GPUInfo

        try:
            gpu = GPUInfo()
            return gpu.get_total_vram_mb()
        except Exception:
            return 0

    @staticmethod
    def _get_total_ram_mb() -> int:
        virtual_memory = psutil.virtual_memory()
        return int(virtual_memory.total / (1024 * 1024))

    # Hordelib will try to leave at least this much VRAM free

    @classmethod
    def get_vram_to_leave_free_mb(cls) -> int:
        """Get the amount of VRAM being left free."""
        if not hasattr(cls, "_vram_to_leave_free_mb") or cls._vram_to_leave_free_mb is None:
            cls.set_vram_to_leave_free_mb("50%")
        return cls._vram_to_leave_free_mb

    @classmethod
    def set_vram_to_leave_free_mb(cls, value: str | int) -> None:
        """Set the amount of VRAM to leave free.

        Args:
            value (str | int): The amount of VRAM to leave free. Can be expressed as a number of MB or a percentage.
        """
        # Allow this to be expressed as a number (in MB) or a percentage
        if perc := cls._is_percentage(value):
            value = int((perc / 100) * cls._get_total_vram_mb())
        else:
            try:
                value = int(value)
            except ValueError:
                value = 0
        cls._vram_to_leave_free_mb = value

    # Hordelib will try to leave at least this much system RAM free
    @classmethod
    def get_ram_to_leave_free_mb(cls) -> int:
        """Get the amount of RAM being left free."""
        if not hasattr(cls, "_ram_to_leave_free_mb") or cls._ram_to_leave_free_mb is None:
            cls.set_ram_to_leave_free_mb("50%")
        return cls._ram_to_leave_free_mb

    @classmethod
    def set_ram_to_leave_free_mb(cls, value) -> None:
        """Set the amount of VRAM to leave free.

        Args:
            value (str | int): The amount of VRAM to leave free. Can be expressed as a number of MB or a percentage.
        """
        if perc := cls._is_percentage(value):
            value = int((perc / 100) * cls._get_total_ram_mb())
        else:
            try:
                value = int(value)
            except ValueError:
                value = 0
        cls._ram_to_leave_free_mb = value

    @classmethod
    def get_model_directory(cls) -> Path:
        """The directory where models are stored"""

        # We take this as a directory that may contain the worker
        # model directories, or they may be somewhere below this.
        basedir = os.environ.get("AIWORKER_CACHE_HOME", "models")

        # Maybe we already searched
        if cls._model_directory and basedir == cls._basedir:
            return cls._model_directory

        # Recursively walk the directory tree, breadth first or this will take forever
        queue = deque([basedir])
        while queue:
            dirpath = queue.popleft()
            try:
                dirnames = next(os.walk(dirpath))[1]
            except StopIteration:
                continue
            # Simple heuristic identification of our target directory
            if "compvis" in dirnames and "clip" in dirnames:
                basedir = dirpath
                break
            # Enqueue all subdirectories
            for dirname in dirnames:
                queue.append(os.path.join(dirpath, dirname))

        cls._model_directory = Path(basedir)
        cls._basedir = basedir
        return Path(basedir)

    # Disable the use of xformers
    disable_xformers = Switch()

    # Report idle time. If this is enabled a warning is issued if the time
    # between hordelib calls exceeds 1 second.
    enable_idle_time_warning = Switch()


_UserSettings = UserSettings()
