import hordelib

hordelib.initialise()
from hordelib.settings import UserSettings

UserSettings.set_vram_to_leave_free_mb("90%")
