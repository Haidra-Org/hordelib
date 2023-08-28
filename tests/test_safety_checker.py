# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.safety_checker import is_image_nsfw
from hordelib.shared_model_manager import SharedModelManager

# FIXME Should find a way to test for a positive NSFW result without something in the repo?


class TestHordeSaftyChecker:
    pass
