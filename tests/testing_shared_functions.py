import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TypeAlias

import PIL.Image
import pytest
from loguru import logger

FilePathOrPILImage: TypeAlias = str | Path | PIL.Image.Image

from hordelib.utils.distance import (
    CosineSimilarityResult,
    CosineSimilarityResultCode,
    HistogramDistanceResult,
    HistogramDistanceResultCode,
    evaluate_image_distance,
    is_cosine_similarity_fail,
    is_histogram_distance_fail,
)


class ImageSimilarityDefaultsTypes(Enum):
    STANDARD_INFERENCE = auto()
    LORA = auto()


@dataclass
class ImageSimilarityConstraints:
    cosine_fail_floor: CosineSimilarityResultCode
    cosine_warn_floor: CosineSimilarityResultCode
    histogram_fail_threshold: HistogramDistanceResultCode
    histogram_warn_threshold: HistogramDistanceResultCode


INFERENCE_SIMILARITY_DEFAULTS = ImageSimilarityConstraints(
    cosine_fail_floor=CosineSimilarityResultCode.PARTIALLY_SIMILAR,
    cosine_warn_floor=CosineSimilarityResultCode.CONSIDERABLY_SIMILAR,
    histogram_fail_threshold=HistogramDistanceResultCode.SKIP,
    histogram_warn_threshold=HistogramDistanceResultCode.SKIP,
)

LORA_SIMILARITY_DEFAULTS = ImageSimilarityConstraints(
    cosine_fail_floor=CosineSimilarityResultCode.SKIP,
    cosine_warn_floor=CosineSimilarityResultCode.CONSIDERABLY_SIMILAR,
    histogram_fail_threshold=HistogramDistanceResultCode.SKIP,
    histogram_warn_threshold=HistogramDistanceResultCode.SKIP,
)


class ImageSimilarityResultCode(Enum):
    FAIL = 0
    SKIP = 1
    PASS = 2


@dataclass
class ImageSimilarityResult:
    result_code: ImageSimilarityResultCode
    cosine_similarity_result: CosineSimilarityResult | float
    histogram_distance_result: HistogramDistanceResult | float


def check_image_similarity_pytest(
    img1: FilePathOrPILImage,
    img2: FilePathOrPILImage,
    *,
    similarity_constraints: ImageSimilarityConstraints,
) -> ImageSimilarityResult:
    cosine_similarity_result, histogram_distance_result = evaluate_image_distance(img1, img2)

    cosine_fail = (
        is_cosine_similarity_fail(
            result_to_check=cosine_similarity_result,
            threshold=similarity_constraints.cosine_fail_floor,
        )
        if similarity_constraints.cosine_fail_floor != CosineSimilarityResultCode.SKIP
        else False
    )
    cosine_warn = (
        is_cosine_similarity_fail(
            result_to_check=cosine_similarity_result,
            threshold=similarity_constraints.cosine_warn_floor,
        )
        if similarity_constraints.cosine_warn_floor != CosineSimilarityResultCode.SKIP
        else False
    )

    histogram_fail = (
        is_histogram_distance_fail(
            result_to_check=histogram_distance_result,
            maximum=similarity_constraints.histogram_fail_threshold,
        )
        if similarity_constraints.histogram_fail_threshold != HistogramDistanceResultCode.SKIP
        else False
    )
    histogram_warn = (
        is_histogram_distance_fail(
            result_to_check=histogram_distance_result,
            maximum=similarity_constraints.histogram_warn_threshold,
        )
        if similarity_constraints.histogram_warn_threshold != HistogramDistanceResultCode.SKIP
        else False
    )

    if cosine_fail or histogram_fail:
        return ImageSimilarityResult(
            ImageSimilarityResultCode.FAIL,
            cosine_similarity_result,
            histogram_distance_result,
        )
    if cosine_warn or histogram_warn:
        return ImageSimilarityResult(
            ImageSimilarityResultCode.SKIP,
            cosine_similarity_result,
            histogram_distance_result,
        )

    return ImageSimilarityResult(
        ImageSimilarityResultCode.PASS,
        cosine_similarity_result,
        histogram_distance_result,
    )


def do_pytest_action(result: ImageSimilarityResult) -> bool:
    """Perform the appropriate pytest action (e.g. `pytest.fail(...)`) based on the image similarity result.

    Args:
        result (ImageSimilarityResult): _description_

    Returns:
        bool: _description_
    """
    if ImageSimilarityResult.result_code == ImageSimilarityResultCode.FAIL:
        pytest.fail(
            f"Image similarity check failed:\n"
            f"cosine_similarity_result={ImageSimilarityResult.cosine_similarity_result},\n"
            f"histogram_distance_result={ImageSimilarityResult.histogram_distance_result}",
        )
    elif ImageSimilarityResult.result_code == ImageSimilarityResultCode.SKIP:
        pytest.skip(
            f"Image similarity check skipped:\n"
            f"cosine_similarity_result={ImageSimilarityResult.cosine_similarity_result},\n"
            f"histogram_distance_result={ImageSimilarityResult.histogram_distance_result}",
        )

    return True


def check_single_inference_image_similarity(
    img1: FilePathOrPILImage,
    img2: FilePathOrPILImage,
    *,
    exception_on_fail: bool = False,
) -> bool:
    return check_list_inference_images_similarity(
        [(img1, img2)],
        exception_on_fail=exception_on_fail,
    )


def check_single_lora_image_similarity(
    img1: FilePathOrPILImage,
    img2: FilePathOrPILImage,
    *,
    exception_on_fail: bool = False,
) -> bool:
    return check_list_lora_images_similarity(
        [(img1, img2)],
        exception_on_fail=exception_on_fail,
    )


def check_list_lora_images_similarity(
    list_of_image_pairs: Iterable[tuple[FilePathOrPILImage, FilePathOrPILImage]],
    *,
    exception_on_fail: bool = False,
) -> bool:
    return check_list_images_similarity(
        list_of_image_pairs,
        similarity_constraints=LORA_SIMILARITY_DEFAULTS,
        exception_on_fail=exception_on_fail,
    )


def check_list_inference_images_similarity(
    list_of_image_pairs: Iterable[tuple[FilePathOrPILImage, FilePathOrPILImage]],
    *,
    exception_on_fail: bool = False,
) -> bool:
    return check_list_images_similarity(
        list_of_image_pairs,
        similarity_constraints=INFERENCE_SIMILARITY_DEFAULTS,
        exception_on_fail=exception_on_fail,
    )


def check_list_images_similarity(
    list_of_image_pairs: Iterable[tuple[FilePathOrPILImage, FilePathOrPILImage]],
    *,
    similarity_constraints: ImageSimilarityConstraints,
    exception_on_fail: bool = False,
) -> bool:
    all_results: list[tuple[tuple[FilePathOrPILImage, FilePathOrPILImage], ImageSimilarityResult]] = []
    for img1, img2 in list_of_image_pairs:
        image_similarity_result = check_image_similarity_pytest(
            img1,
            img2,
            similarity_constraints=similarity_constraints,
        )
        all_results.append(((img1, img2), image_similarity_result))

    if all(result[1].result_code == ImageSimilarityResultCode.PASS for result in all_results):
        return True

    all_failed_results = [result for result in all_results if result[1].result_code == ImageSimilarityResultCode.FAIL]
    all_skipped_results = [result for result in all_results if result[1].result_code == ImageSimilarityResultCode.SKIP]

    complete_error_message = ""

    if len(all_failed_results) > 0:
        for img_pair, result in all_failed_results:
            complete_error_message += (
                f"Image similarity check failed:\n"
                f"cosine_similarity_result={result.cosine_similarity_result},\n"
                f"histogram_distance_result={result.histogram_distance_result}\n"
            )
            complete_error_message += f"img1={img_pair[0]}\n"
            complete_error_message += f"img2={img_pair[1]}\n"
        complete_error_message += "all_failed_results:\n"
    if len(all_skipped_results) > 0:
        for img_pair, result in all_skipped_results:
            complete_error_message += (
                f"Image similarity check skipped:\n"
                f"cosine_similarity_result={result.cosine_similarity_result},\n"
                f"histogram_distance_result={result.histogram_distance_result}\n"
            )
            complete_error_message += f"img1={img_pair[0]}\n"
            complete_error_message += f"img2={img_pair[1]}\n"

    HORDELIB_SKIP_SIMILARITY_FAIL = os.getenv("HORDELIB_SKIP_SIMILARITY_FAIL", None)

    if len(all_failed_results) > 0:
        if exception_on_fail:
            raise AssertionError(complete_error_message)
        if not HORDELIB_SKIP_SIMILARITY_FAIL:
            pytest.fail(complete_error_message)
        else:
            logger.warning(complete_error_message)
            pytest.skip(complete_error_message)
    elif len(all_skipped_results) > 0:
        pytest.skip(complete_error_message)

    return True
