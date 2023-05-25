# Reference: https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/
from collections import Counter

import numpy as np
import PIL
from loguru import logger
from PIL import Image
from PIL.Image import Image as ImageType
from PIL.PngImagePlugin import PngImageFile


def get_hist(image):
    reference_image_arr = np.asarray(image)
    flat_array_1 = reference_image_arr.flatten()
    RH1 = Counter(flat_array_1)
    H1 = []
    for i in range(256):
        if i in RH1.keys():
            H1.append(RH1[i])
        else:
            H1.append(0)
    return H1


def L2Norm(H1, H2):
    distance = 0
    for i in range(len(H1)):
        distance += np.square(H1[i] - H2[i])
    return np.sqrt(distance)


def parse_image(img):
    if type(img) is str:
        return Image.open(img)
    if type(img) in [PngImageFile, ImageType]:
        return img
    raise Exception("Cannot parse provided input image. Comparing accepts only PIL or filenames path strings")


def get_image_distance(img1, img2):
    img1 = parse_image(img1)
    img2 = parse_image(img2)
    dist = L2Norm(
        get_hist(img1),
        get_hist(img2),
    )
    logger.info(f"The distance between images is : {dist}")
    return dist


def are_images_identical(img1, img2, identical_distance=10000):
    """Compares two images for distance
    If it is above a threshold, they are considered not-identical
    the distance is very sensitive to even things like compression artifacts
    For uncompressed images of the same format, distance 100 is prerry accurate.
    For differences that would be visible with the human eye, 10000 is a good number.
    Reminder that for comparing stable diffusion images, things like xformers versions
    can change the results almost imperceptibly,
    so make sure you provide enough of a distance buffer to take that into account.
    """
    img_dist = get_image_distance(img1, img2)
    # distance ~7000 is the distance caused
    # by compression artifacts and format changes
    return img_dist < identical_distance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("f1", help="filepath for image 1 to compare")
    parser.add_argument("f2", help="filepath for image 2 to compare")
    args = parser.parse_args()
    print(get_image_distance(args.f1, args.f2))
