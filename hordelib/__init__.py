from dotenv import load_dotenv

load_dotenv()

# Initialize logfire early to capture all subsequent operations
from hordelib.integrations.logfire_setup import initialize_logfire

initialize_logfire()

from hordelib.initialisation import initialise, is_initialised
from hordelib.settings import UserSettings

__all__ = [
    "initialise",
    "is_initialised",
    "UserSettings",
]
