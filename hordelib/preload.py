import requests
from loguru import logger

from hordelib.nodes.comfy_controlnet_preprocessors import canny, hed, leres, midas, mlsd, openpose, pidinet, uniformer

annotator_model_integrity: dict[str, str] = {
    "body_pose_model.pth": "25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746",
    "dpt_hybrid-midas-501f0c75.pt": "501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853",
    "hand_pose_model.pth": "b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600",
    "mlsd_large_512_fp32.pth": "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
    "network-bsds500.pth": "58a858782f5fa3e0ca3dc92e7a1a609add93987d77be3dfa54f8f8419d881a94",
    "res101.pth": "1d696b2ef3e8336b057d0c15bc82d2fecef821bfebe5ef9d7671a5ec5dde520b",
    "upernet_global_small.pth": "bebfa1264c10381e389d8065056baaadbdadee8ddc6e36770d1ec339dc84d970",
}


def download_all_controlnet_annotators() -> bool:
    """Will start the download of all the models needed for the controlnet annotators."""
    annotator_init_funcs = [
        canny.CannyDetector,
        hed.HEDdetector,
        midas.MidasDetector,
        mlsd.MLSDdetector,
        openpose.OpenposeDetector,
        uniformer.UniformerDetector,
        leres.download_model_if_not_existed,
        pidinet.download_if_not_existed,
    ]

    try:
        for annotator_init_func in annotator_init_funcs:
            annotator_init_func()
        return True
    except (OSError, requests.exceptions.RequestException) as e:
        logger.init_err(f"Failed to download annotator: {e}")
        return False
