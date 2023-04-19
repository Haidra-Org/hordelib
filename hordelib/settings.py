from typing_extensions import Self

from hordelib.utils.switch import Switch


class UserSettings:
    """Container class for all worker settings."""

    _instance: Self | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    # Disable the use of xformers
    disable_xformers = Switch()

    # Disable the display of progress bars when downloading
    # FIXME We should enable these, but don't yet
    disable_download_progress = Switch()

    # Hordelib will try to leave at least this much VRAM free
    vram_to_leave_free_mb = 2048

    # Hordelib will try to leave at least this much system RAM free
    ram_to_leave_free_mb = 4 * 1024
