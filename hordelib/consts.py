# consts.py

# ComfyUI exact version we need
COMFYUI_VERSION = "c767e9426ae81bed4f52c7be0625f0efc4cbe16b"

# Default model managers to load
DEFAULT_MODEL_MANAGERS = {
    "blip": True,
    "clip": True,
    "codeformer": True,
    "compvis": True,
    "controlnet": True,
    "diffusers": True,
    "esrgan": True,
    "gfpgan": True,
    "safety_checker": True,
}

# Default location of model database
REMOTE_MODEL_DB = (
    "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/"
)
