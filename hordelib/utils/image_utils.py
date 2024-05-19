import base64

import numpy as np
import rembg  # type: ignore
from loguru import logger
from PIL import Image, ImageOps, PngImagePlugin, UnidentifiedImageError

IMAGE_CHUNK_SIZE = 64

DEFAULT_IMAGE_MIN_RESOLUTION = 512
DEFAULT_HIGHER_RES_MAX_RESOLUTION = 1024


class ImageUtils:
    @classmethod
    def resize_image_dimensions(
        cls,
        width: int,
        height: int,
        desired_dimension: int,
        use_min: bool,
    ) -> tuple[int, int]:
        """Resize the image dimensions to have one side equal to the desired resolution, keeping the aspect ratio.

        - If use_min is True, the side with the minimum length will be resized to the desired resolution.
          - For example, if the image is 1024x2048 and the desired resolution is 512, the image will be
            resized to 512x1024. (As desired for 512x trained models)
        - If use_min is False, the side with the maximum length will be resized to the desired resolution.
          - For example, if the image is 1024x2048 and the desired resolution is 1024, the image will be
            resized to 512x1024. (As desired for 1024x trained models)
        - If the image is smaller than the desired resolution, the image will not be resized.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            desired_dimension (int): The desired single side resolution.
            use_min (bool): Whether to use the minimum or maximum side.

        Returns:
            tuple[int, int]: The target first pass width and height of the image.
        """
        if desired_dimension is None or desired_dimension <= 0:
            raise ValueError("desired_resolution must be a positive integer.")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers.")

        if width < desired_dimension and height < desired_dimension:
            return width, height

        if use_min:
            ratio = min(
                height / desired_dimension,
                width / desired_dimension,
            )
        else:
            ratio = max(
                height / desired_dimension,
                width / desired_dimension,
            )

        new_width = int(width // (ratio * IMAGE_CHUNK_SIZE)) * IMAGE_CHUNK_SIZE
        new_height = int(height // (ratio * IMAGE_CHUNK_SIZE)) * IMAGE_CHUNK_SIZE

        return new_width, new_height

    @classmethod
    def get_first_pass_image_resolution_min(
        cls,
        width: int,
        height: int,
        min_dimension: int = DEFAULT_IMAGE_MIN_RESOLUTION,
    ):
        """Resize the image dimensions to have one side equal to the desired resolution, keeping the aspect ratio.

        - If the image is larger than the desired resolution, the side with the minimum length will be resized to the
          desired resolution.
        - If the image is smaller than the desired resolution, the image will not be resized.

        """
        if width > min_dimension and height > min_dimension:
            return cls.resize_image_dimensions(
                width,
                height,
                desired_dimension=min_dimension,
                use_min=True,
            )
        return width, height

    @classmethod
    def get_first_pass_image_resolution_max(
        cls,
        width: int,
        height: int,
        max_dimension: int = DEFAULT_HIGHER_RES_MAX_RESOLUTION,
    ):
        """Resize the image dimensions to have one side equal to the desired resolution, keeping the aspect ratio.

        - If the image is larger than the desired resolution, the side with the maximum length will be resized to the
          desired resolution.
        - If the image is smaller than the desired resolution, the image will not be resized.
        """

        if max(width, height) > max_dimension:
            return cls.resize_image_dimensions(
                width,
                height,
                desired_dimension=max_dimension,
                use_min=False,
            )
        return width, height

    @classmethod
    def add_image_alpha_channel(cls, source_image, alpha_image):
        # Convert images to RGBA mode
        source_image = source_image.convert("RGBA")
        # Convert alpha image to greyscale
        alpha_image = alpha_image.convert("L")
        # Resize alpha_image if its size is different from source_image
        if alpha_image.size != source_image.size:
            alpha_image = alpha_image.resize(source_image.size)
        # Create a new alpha channel from the second image
        alpha_image = ImageOps.invert(alpha_image)
        alpha_data = alpha_image.split()[0]
        source_image.putalpha(alpha_data)

        # Return the resulting image
        return source_image

    @classmethod
    def resize_sources_to_request(cls, payload):
        """Ensures the source_image and source_mask are at the size requested by the client"""
        source_image = payload.get("source_image")
        if not source_image:
            return
        try:
            newwidth = payload["width"]
            newheight = payload["height"]
            if payload.get("hires_fix") or payload.get("control_type"):
                newwidth, newheight = cls.get_first_pass_image_resolution_min(payload["width"], payload["height"])
            if source_image.size != (newwidth, newheight):
                payload["source_image"] = source_image.resize(
                    (newwidth, newheight),
                    Image.Resampling.LANCZOS,
                )
        except (UnidentifiedImageError, AttributeError):
            logger.warning("Source image could not be parsed. Falling back to text2img")
            del payload["source_image"]
            del payload["source_processing"]
            return

        source_mask = payload.get("source_mask")
        if not source_mask:
            return
        try:
            if source_mask.size != (payload["width"], payload["height"]):
                payload["source_mask"] = source_mask.resize(
                    (payload["width"], payload["height"]),
                )
        except (UnidentifiedImageError, AttributeError):
            logger.warning(
                "Source mask could not be parsed. Falling back to img2img with an all alpha mask.",
            )
            payload["source_mask"] = ImageUtils.create_alpha_image(payload["width"], payload["height"])

        if payload.get("source_mask"):
            payload["source_image"] = cls.add_image_alpha_channel(payload["source_image"], payload["source_mask"])

    @classmethod
    def shrink_image(cls, image, width, height, preserve_aspect=False):
        # Check if the provided image is an instance of the PIL.Image.Image class
        if not isinstance(image, Image.Image):
            logger.warning("Bad image passed to shrink_image")
            return None

        # If both width and height are not specified, return
        if width is None and height is None:
            logger.warning("Bad image size passed to shrink_image")
            return None

        # Only shrink
        if width >= image.width or height >= image.height:
            return image

        # Calculate new dimensions
        if preserve_aspect:
            aspect_ratio = float(image.width) / float(image.height)

            if width is not None:
                height = int(width / aspect_ratio)
            else:
                width = int(height * aspect_ratio)

        # Resize the image
        return image.resize((width, height), Image.LANCZOS)

    @classmethod
    def copy_image_metadata(cls, src_image, dest_image):
        metadata = src_image.info
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in metadata.items():
            if k not in ("dpi", "gamma", "transparency", "aspect"):
                pnginfo.add_text(k, v)
        dest_image.info["pnginfo"] = pnginfo
        return dest_image

    @classmethod
    def strip_background(cls, image: Image.Image):
        session = rembg.new_session("u2net")
        image = rembg.remove(
            image,
            session=session,
            only_mask=False,
            alpha_matting=10,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        del session
        return image

    @classmethod
    def create_alpha_image(cls, width: int = 512, height: int = 512):
        return Image.new("L", (width, height), 255)

    @classmethod
    def create_white_image(cls, width: int = 512, height: int = 512):
        return Image.new("RGB", (width, height), (255, 255, 255))

    @classmethod
    def create_black_image(cls, width: int = 512, height: int = 512):
        return Image.new("RGB", (width, height), (0, 0, 0))

    @classmethod
    def create_alpha_image_base64(cls, width: int = 512, height: int = 512):
        alpha_image = cls.create_alpha_image(width, height)
        alpha_image = alpha_image.tobytes()
        return base64.b64encode(alpha_image).decode("utf-8")

    @classmethod
    def has_alpha_channel(cls, image: Image.Image):
        return image.mode == "RGBA"

    @classmethod
    def create_noise_image(cls, width: int | None = 512, height: int | None = 512):
        if width is None:
            width = 512

        if height is None:
            height = 512

        return Image.fromarray((np.random.rand(height, width, 3) * 255).astype(np.uint8))
