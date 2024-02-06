# Some of the original implementation of the module was inspired from the following:
# - https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

DEFAULT_HISTOGRAM_BINS = 256
DEFAULT_HISTOGRAM_RANGE = (0, 512)


class CosineSimilarityResultCode(float, Enum):
    """Thresholds for image distance.\n
    In order of increasing strictness:\n
    - `SKIP`: Skip the image distance check.\n
    - `ORTHOGONAL`\n
    - `NOT_SIMILAR`\n
    - `SOMEWHAT_SIMILAR`\n
    - `PARTIALLY_SIMILAR`\n
    - `CONSIDERABLY_SIMILAR`\n
    - `EXTREMELY_SIMILAR`\n
    - `PERCEPTUALLY_IDENTICAL`\n
    """

    # Note: The values must be in ascending value order, as the first value is used as the default value.
    # (closer to 1 is more similar, closer to -1 becomes less similar/orthogonal)
    ORTHOGONAL = -1.0
    """Images are orthogonal, and in a certain sense, are the opposite of each other."""
    NOT_SIMILAR = 0
    """Images are probably very different, but may have some similarities or small regions which are the same."""
    SOMEWHAT_SIMILAR = 0.5
    """Images are somewhat similar, and may share some major features or regions, but it is unlikely they will be
    confused with each other."""
    PARTIALLY_SIMILAR = 0.87
    """Images are similar, but possibly not enough to be confused with each other at a glance."""
    CONSIDERABLY_SIMILAR = 0.92
    """Images are probably similar enough to be confused at a glance."""
    EXTREMELY_SIMILAR = 0.98
    """Images are extremely similar, and any differences are likely in the details."""
    PERCEPTUALLY_IDENTICAL = 0.99
    """Images are perceptually identical, but may not be byte-for-byte identical."""
    IDENTICAL = 1.0
    """Images are probably also byte-for-byte identical, but it is possible they may not be."""
    SKIP = 2**31
    """Skip the image distance check."""


@dataclass
class CosineSimilarityResult:
    @property
    def result_code(self) -> CosineSimilarityResultCode:
        """The result code for the cosine similarity."""

        return_code: CosineSimilarityResultCode = next(iter(CosineSimilarityResultCode))

        for code in CosineSimilarityResultCode:
            if self.cosine_similarity >= code:
                return_code = code
            else:
                return return_code
        return return_code

    cosine_similarity: float

    def __str__(self) -> str:
        return f"{self.cosine_similarity} ({self.result_code.name} " f"[-1.0, 1.0]: {self.result_code.value})"


class HistogramDistanceResultCode(float, Enum):
    """Thresholds for histogram distance.\n
    In order of increasing strictness:\n
    - `SKIP`: Skip the histogram distance check.\n
    - `VERY_DISSIMILAR_DISTRIBUTION`\n
    - `DISSIMILAR_DISTRIBUTION`\n
    - `SIMILAR_DISTRIBUTION`\n
    - `VERY_SIMILAR_DISTRIBUTION`\n
    - `EXTREMELY_SIMILAR_DISTRIBUTION`\n

    You should also consider the cosine similarity when evaluating the distance between two images.\n
    """

    # Note: The values must be in descending value order, as the first value is used as the default value.
    # (closer to 0 is more similar, higher values are a greater distance (more potential for being dissimilar))
    COMPLETELY_DISSIMILAR_DISTRIBUTION = 100000
    """The color distributions are completely different, and there is a very high potential for the two images to
    have completely different compositions."""
    VERY_DISSIMILAR_DISTRIBUTION = 70000
    """The color distributions are very similar, and there is a very high potential for them to be perceptually
    different."""
    DISSIMILAR_DISTRIBUTION = 50000
    """The color distributions are markedly dissimilar, and there is a high potential for them to be perceptually"""
    SIMILAR_DISTRIBUTION = 30000
    """The color distributions are close, and there is an increased potential for them to be perceptually different."""
    VERY_SIMILAR_DISTRIBUTION = 16000
    """The color distributions are very close, but there is potential for them to be perceptually different in
    certain situations."""
    EXTREMELY_SIMILAR_DISTRIBUTION = 10000
    """The color distributions are extremely close, and as such the composition of the images is likely to be very
    similar."""

    IDENTICAL = 0.0
    """The images are probably also byte-for-byte identical, but it is possible they may not be."""

    SKIP = 0 - 1e-7
    """Skip the histogram distance check."""


@dataclass
class HistogramDistanceResult:
    @property
    def result_code(self) -> HistogramDistanceResultCode:
        """The result code for the histogram distance."""
        return_code: HistogramDistanceResultCode = next(iter(HistogramDistanceResultCode))

        for code in HistogramDistanceResultCode:
            if self.histogram_distance <= code:
                return_code = code
            else:
                return return_code
        return return_code

    histogram_distance: float

    def __str__(self) -> str:
        return f"{self.histogram_distance} ({self.result_code.name} : {self.result_code.value})"


def is_cosine_similarity_fail(
    *,
    result_to_check: CosineSimilarityResult | CosineSimilarityResultCode,
    threshold: CosineSimilarityResultCode,
) -> bool:
    """Checks if the cosine similarity result is a failure.

    Args:
        result_to_check (CosineSimilarityResult): The result to check.
        threshold (CosineSimilarityResultCode): The result should be at least this similar.

    Returns:
        bool: True if the result is a failure, False otherwise.
    """
    if isinstance(result_to_check, CosineSimilarityResult):
        return result_to_check.cosine_similarity < threshold
    return result_to_check < threshold


def is_histogram_distance_fail(
    *,
    result_to_check: HistogramDistanceResult | HistogramDistanceResultCode,
    maximum: HistogramDistanceResultCode,
) -> bool:
    """Checks if the histogram distance result is a failure.

    Args:
        result_to_check (HistogramDistanceResult): The result to check.
        maximum_threshold (HistogramDistanceResultCode): The result should be at most this different.

    Returns:
        bool: True if the result is a failure, False otherwise.
    """
    if isinstance(result_to_check, HistogramDistanceResult):
        return result_to_check.histogram_distance > maximum
    return result_to_check > maximum


def parse_image(img_or_path: str | Path | PIL.Image.Image) -> PIL.Image.Image:
    """Parses an image from a path or a PIL.Image.Image object. If the passed object is already a PIL.Image.Image,
    it is returned as-is.

    Args:
        img_or_path (str | Path | PIL.Image.Image): The image to return back, or the path to read.

    Raises:
        ValueError: If the passed object is not a PIL.Image.Image or a valid path.

    Returns:
        PIL.Image.Image: The loaded image from the path, or the passed image if it was already a PIL.Image.Image.
    """
    if isinstance(img_or_path, Path) or isinstance(img_or_path, str):
        return Image.open(img_or_path)
    if isinstance(img_or_path, PIL.Image.Image):
        return img_or_path
    raise ValueError("Cannot parse provided input image. Comparing accepts only PIL.Image.Image objects or paths.")


def resize_to_thumbnail(img: PIL.Image.Image) -> PIL.Image.Image:
    """Resizes an image to a thumbnail size of 256x256.

    Args:
        img (PIL.Image.Image): The image to resize.

    Returns:
        PIL.Image.Image: The resized image.
    """
    img_width, img_height = img.size

    if img_width > 256 or img_height > 256:
        img_longest_side = max(img_width, img_height)
        img_resize_factor = 256 / img_longest_side
        img = img.resize((int(img_width * img_resize_factor), int(img_height * img_resize_factor)))

    return img


def cv2_image_similarity(
    img_or_path_1: str | Path | PIL.Image.Image,
    img_or_path_2: str | Path | PIL.Image.Image,
) -> CosineSimilarityResult:
    """Calculates the similarity between two images using the cv2 library. The images are converted to grayscale
    before calculating the similarity.

    Args:
        img_or_path_1 (str | Path | PIL.Image.Image): The first image to compare.
        img_or_path_2 (str | Path | PIL.Image.Image): The second image to compare.

    Raises:
        ValueError: If the passed objects are not valid images.

    Returns:
        float: The similarity between the two images, with 0 being completely different and 1 being identical.
    """
    img1 = parse_image(img_or_path_1)
    img2 = parse_image(img_or_path_2)
    if img1.size != img2.size:
        raise ValueError("Images must be of same size.")

    # Convert the images to grayscale
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)  # type: ignore # FIXME
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)  # type: ignore # FIXME

    # Calculate the similarity between the two images
    similarity = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]  # type: ignore # FIXME

    # Get the primitive (float) value of the similarity
    similarity = similarity.item()

    return CosineSimilarityResult(cosine_similarity=similarity)


def get_hist(img: PIL.Image.Image) -> np.ndarray:
    # Gets a numpy histogram of the image

    # Convert the image to RGB if it is a PNG
    if isinstance(img, PngImageFile):
        img = img.convert("RGB")

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Get the histogram of the image
    return np.histogram(img_array, bins=DEFAULT_HISTOGRAM_BINS, range=DEFAULT_HISTOGRAM_RANGE)[0]


def L2Norm(hist1: np.ndarray, hist2: np.ndarray) -> float:
    # Calculates the L2 norm between two histograms

    # Get the difference between the two histograms
    diff = hist1 - hist2

    # Square the difference
    diff = diff**2

    # Sum the difference
    diff = np.sum(diff)

    # Get the square root of the difference
    diff = np.sqrt(diff)

    return float(diff)


def get_image_histogram_distance(
    img_or_path_1: str | Path | PIL.Image.Image,
    img_or_path_2: str | Path | PIL.Image.Image,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    range: tuple[int, int] = DEFAULT_HISTOGRAM_RANGE,
) -> HistogramDistanceResult:
    img1 = parse_image(img_or_path_1)
    img2 = parse_image(img_or_path_2)
    if img1.size != img2.size:
        raise ValueError("Images must be of same size.")

    dist = L2Norm(
        get_hist(img1),
        get_hist(img2),
    )
    logger.debug(
        (f"The histogram (bins={bins}, range={range}) distance between images (dims: {img1.size}) is : {dist}"),
    )
    return HistogramDistanceResult(dist)


def get_image_thumbnail_histogram_distance(
    img_or_path_1: str | Path | PIL.Image.Image,
    img_or_path_2: str | Path | PIL.Image.Image,
):
    img1 = parse_image(img_or_path_1)
    img2 = parse_image(img_or_path_2)
    if img1.size != img2.size:
        raise ValueError("Images must be of same size.")

    img1 = resize_to_thumbnail(img1)
    img2 = resize_to_thumbnail(img2)

    return get_image_histogram_distance(img1, img2, bins=256, range=(0, 512))


def evaluate_image_distance(
    img_or_path_1: str | Path | PIL.Image.Image,
    img_or_path_2: str | Path | PIL.Image.Image,
) -> tuple[CosineSimilarityResult, HistogramDistanceResult]:
    cosine_similarity_result = cv2_image_similarity(img_or_path_1, img_or_path_2)
    histogram_distance_result = get_image_histogram_distance(img_or_path_1, img_or_path_2)
    logger.info(f"Cosine similarity: {cosine_similarity_result}")
    logger.info(f"Histogram distance: {histogram_distance_result}")
    return cosine_similarity_result, histogram_distance_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path_1", help="filepath for image 1 to compare")
    parser.add_argument("image_path_2", help="filepath for image 2 to compare")
    args = parser.parse_args()
    print(get_image_histogram_distance(args.image_path_1, args.image_path_1))
