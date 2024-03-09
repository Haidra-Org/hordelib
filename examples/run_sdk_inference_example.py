import hordelib

hordelib.initialise(setup_logging=False, logging_verbosity=5)

from uuid import UUID

from horde_sdk.ai_horde_api import KNOWN_SOURCE_PROCESSING
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
)
from horde_sdk.ai_horde_api.fields import JobID

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

horde = HordeLib()

SharedModelManager.load_model_managers()


def main():
    example_response = ImageGenerateJobPopResponse(
        ids=[JobID(root=UUID("00000000-0000-0000-0000-000000000000"))],
        source_processing=KNOWN_SOURCE_PROCESSING.txt2img,
        skipped=ImageGenerateJobPopSkippedStatus(),
        model="Deliberate",
        payload=ImageGenerateJobPopPayload(
            prompt="a cat in a hat",
            post_processing=["4x_AnimeSharp", "CodeFormers"],
            n_iter=1,
        ),
    )

    result = horde.basic_inference(example_response)
    print(result)


if __name__ == "__main__":
    main()
