import pathlib

from PyInstaller.utils.hooks import collect_data_files  # type: ignore

hiddenimports = [
    "torch",
    "torchvision",
    "torchsde",
    "transformers",
    "safetensors",
    "scipy",
    "tqdm",
    "psutil",
    "opencv-python",
    "opencv-contrib-python",
    "timm",
    "addict",
    "fairscale",
    "scikit-image",
    "mediapipe",
    "unidecode",
    "fuzzywuzzy",
    "open_clip",
    "clip-interrogator",
]
warn_on_missing_hiddenimports = True

# module_collection_mode = "pyz+py"

project_root = pathlib.Path(__file__).parent.parent

datas = [
    (str(project_root / "_version.py"), "hordelib"),
]

excluded_folders = [
    "__pycache__",
    ".git",
]

# Add every file in the `hordelib/_comfyui` directory to datas, copying the directory structure
comfyui_dir = "hordelib._comfyui"
datas += collect_data_files(comfyui_dir, "hordelib/_comfyui", excludes=excluded_folders)

nodes_dir = "hordelib.nodes"
datas += collect_data_files(nodes_dir, "hordelib/nodes")


pipelines_dir = "hordelib.pipelines"
pipeline_designs_dir = "hordelib.pipeline_designs"

datas += collect_data_files(pipelines_dir, "hordelib/pipelines")
datas += collect_data_files(pipeline_designs_dir, "hordelib/pipeline_designs")
