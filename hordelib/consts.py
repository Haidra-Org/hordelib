# consts.py

# hordelib version
VERSION = "0.2.0"

# ComfyUI exact version we need
COMFYUI_VERSION = "ebd7f9bf80213a44a8e2cadc75875a4b980991e5"

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
