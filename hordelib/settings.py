from hordelib.utils.switch import Switch


class WorkerSettings:
    """Container class for all worker settings."""

    disable_xformers = Switch()
    disable_voodoo = Switch()
    enable_local_ray_temp = Switch()
    disable_progress = Switch()
    disable_download_progress = Switch()
    enable_ray_alternative = Switch()

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
