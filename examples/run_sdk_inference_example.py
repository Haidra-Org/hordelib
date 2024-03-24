import hordelib

hordelib.initialise(setup_logging=False, logging_verbosity=5)

import asyncio
from uuid import UUID

from aiohttp import ClientSession
from horde_sdk.ai_horde_api import KNOWN_SOURCE_PROCESSING
from horde_sdk.ai_horde_api.apimodels import (
    ExtraSourceImageEntry,
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
)
from horde_sdk.ai_horde_api.fields import JobID

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

horde = HordeLib()

SharedModelManager.load_model_managers()


async def _download_images(payload: ImageGenerateJobPopResponse) -> None:
    async with ClientSession() as session:
        tasks = []
        tasks.append(payload.async_download_source_image(session))
        tasks.append(payload.async_download_source_mask(session))
        tasks.append(payload.async_download_extra_source_images(session))

        await asyncio.gather(*tasks)


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
        source_image="https://raw.githubusercontent.com/db0/Stable-Horde/main/img_stable/0.jpg",
        source_mask="https://raw.githubusercontent.com/db0/Stable-Horde/main/img_stable/1.jpg",
        extra_source_images=[
            ExtraSourceImageEntry(
                image="https://raw.githubusercontent.com/db0/Stable-Horde/main/img_stable/2.jpg",
            ),
            ExtraSourceImageEntry(
                image="https://raw.githubusercontent.com/db0/Stable-Horde/main/img_stable/3.jpg",
            ),
        ],
    )

    asyncio.run(_download_images(example_response))

    result = horde.basic_inference(example_response)
    print(result)


if __name__ == "__main__":
    main()
