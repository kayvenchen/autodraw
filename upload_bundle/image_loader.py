from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from models import ProcessingConfig


@dataclass(slots=True)
class LoadedImage:
    original_bgr: np.ndarray
    grayscale: np.ndarray
    binary: np.ndarray
    drawable_mask: np.ndarray


def load_and_process_image(image_path: str, config: ProcessingConfig) -> LoadedImage:
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    grayscale = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    grayscale = _resize_if_needed(grayscale, config)

    if config.blur_kernel > 1:
        kernel = _ensure_odd(config.blur_kernel)
        grayscale = cv2.GaussianBlur(grayscale, (kernel, kernel), 0)

    binary = cv2.threshold(grayscale, config.threshold, 255, cv2.THRESH_BINARY)[1]
    if config.invert:
        binary = cv2.bitwise_not(binary)

    drawable_mask = binary == 0
    if config.min_component_size > 1:
        drawable_mask = remove_small_objects(
            drawable_mask,
            min_size=config.min_component_size,
            connectivity=2,
        )

    preview_mask = skeletonize(drawable_mask) if config.skeletonize else drawable_mask
    processed_binary = np.where(preview_mask, 0, 255).astype(np.uint8)
    resized_bgr = _resize_if_needed(original_bgr, config)
    return LoadedImage(
        original_bgr=resized_bgr,
        grayscale=grayscale,
        binary=processed_binary,
        drawable_mask=drawable_mask.astype(bool),
    )


def _resize_if_needed(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.resize_width and not config.resize_height:
        return image

    height, width = image.shape[:2]
    target_width = config.resize_width
    target_height = config.resize_height

    if target_width and not target_height:
        scale = target_width / width
        target_height = max(1, int(round(height * scale)))
    elif target_height and not target_width:
        scale = target_height / height
        target_width = max(1, int(round(width * scale)))

    assert target_width is not None and target_height is not None
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
