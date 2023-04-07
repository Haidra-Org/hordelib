# consts.py

# hordelib version
VERSION = "0.1.0"

# ComfyUI exact version we need
COMFYUI_VERSION = "44fea050649347ca4b4e7317a83d11c3b4b87f87"

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
