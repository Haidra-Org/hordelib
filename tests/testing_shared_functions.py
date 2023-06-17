from enum import Enum
from typing import Callable

import PIL.Image
import pytest

from hordelib.utils.distance import (
    CosineSimilarityResultCode,
    HistogramDistanceResultCode,
    evaluate_image_distance,
    is_cosine_similarity_fail,
    is_histogram_distance_fail,
)


def check_image_similarity_pytest(
    img1: PIL.Image.Image,
    img2: PIL.Image.Image,
    *,
    cosine_fail_floor: CosineSimilarityResultCode,
    cosine_warn_floor: CosineSimilarityResultCode | None = None,
    histogram_fail_threshold: HistogramDistanceResultCode | None = None,
    histogram_warn_threshold: HistogramDistanceResultCode | None = None,
) -> bool:
    cosine_similarity_result, histogram_distance_result = evaluate_image_distance(img1, img2)

    cosine_fail = is_cosine_similarity_fail(
        result_to_check=cosine_similarity_result,
        threshold=cosine_fail_floor,
    )
    cosine_warn = (
        is_cosine_similarity_fail(
            result_to_check=cosine_similarity_result,
            threshold=cosine_warn_floor,
        )
        if cosine_warn_floor is not None
        else False
    )

    histogram_fail = (
        is_histogram_distance_fail(
            result_to_check=histogram_distance_result,
            maximum=histogram_fail_threshold,
        )
        if histogram_fail_threshold is not None
        else False
    )
    histogram_warn = (
        is_histogram_distance_fail(
            result_to_check=histogram_distance_result,
            maximum=histogram_warn_threshold,
        )
        if histogram_warn_threshold is not None
        else False
    )

    if cosine_fail or histogram_fail:
        pytest.fail(
            f"Image similarity check failed: cosine_similarity_result={cosine_similarity_result}, "
            f"histogram_distance_result={histogram_distance_result}",
        )
    elif cosine_warn or histogram_warn:
        pytest.skip(
            f"Image similarity check failed: cosine_similarity_result={cosine_similarity_result}, "
            f"histogram_distance_result={histogram_distance_result}",
        )

    return True


def check_inference_image_similarity_pytest(
    img1: PIL.Image.Image,
    img2: PIL.Image.Image,
) -> bool:
    return check_image_similarity_pytest(
        img1,
        img2,
        cosine_fail_floor=CosineSimilarityResultCode.CONSIDERABLY_SIMILAR,
        cosine_warn_floor=CosineSimilarityResultCode.EXTREMELY_SIMILAR,
    )
