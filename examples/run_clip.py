# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_clip
# You need all the deps in whatever environment you are running this.
import os

import hordelib


def main():
    hordelib.initialise()

    import json

    from loguru import logger
    from PIL import Image

    from hordelib.blip.caption import Caption
    from hordelib.clip.interrogate import Interrogator
    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    HordeLib()
    SharedModelManager.loadModelManagers(clip=True, blip=True)
    SharedModelManager.manager.load("ViT-L/14")
    SharedModelManager.manager.load("BLIP_Large")

    # Run CLIP
    model_info = SharedModelManager.manager.loaded_models["ViT-L/14"]
    interrogator = Interrogator(model_info)
    ranking_result = interrogator(
        image=Image.open("images/test_inpaint_original.png"),
        rank=True,
    )
    logger.warning(json.dumps(ranking_result, indent=4))

    # Run BLIP
    model = SharedModelManager.manager.loaded_models["BLIP_Large"]
    caption_class = Caption(model)
    caption = caption_class(
        image=Image.open("images/test_inpaint_original.png"),
        sample=True,
        num_beams=7,
        min_length=20,
        max_length=50,
        top_p=0.9,
        repetition_penalty=1.4,
    )
    logger.warning(json.dumps(caption, indent=4))


if __name__ == "__main__":
    main()
