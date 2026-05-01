from __future__ import annotations

import math

import numpy as np

from models import MappingConfig, Point, Stroke


def map_strokes_to_screen(
    strokes: list[Stroke],
    image_shape: tuple[int, int],
    config: MappingConfig,
) -> list[Stroke]:
    if not strokes:
        return []

    image_height, image_width = image_shape
    transformed = _rotate_strokes_if_needed(strokes, image_width, image_height, config.rotation_degrees)

    min_x, min_y, max_x, max_y = _stroke_bounds(transformed)
    source_width = max(max_x - min_x, 1.0)
    source_height = max(max_y - min_y, 1.0)

    if config.stretch_to_fit and not config.preserve_aspect_ratio:
        scale_x = config.width / source_width
        scale_y = config.height / source_height
    else:
        scale = min(config.width / source_width, config.height / source_height)
        scale_x = scale
        scale_y = scale

    scale_x *= config.scale_percent / 100.0
    scale_y *= config.scale_percent / 100.0

    used_width = source_width * scale_x
    used_height = source_height * scale_y
    anchor_x = config.top_left_x + (config.width - used_width) / 2.0
    anchor_y = config.top_left_y + (config.height - used_height) / 2.0
    anchor_x += config.offset_x
    anchor_y += config.offset_y

    mapped: list[Stroke] = []
    for stroke in transformed:
        mapped_stroke = [
            (
                anchor_x + (point[0] - min_x) * scale_x,
                anchor_y + (point[1] - min_y) * scale_y,
            )
            for point in stroke
        ]
        mapped.append(mapped_stroke)
    return mapped


def _rotate_strokes_if_needed(
    strokes: list[Stroke],
    image_width: int,
    image_height: int,
    rotation_degrees: float,
) -> list[Stroke]:
    if abs(rotation_degrees) < 1e-6:
        return [stroke[:] for stroke in strokes]

    center_x = image_width / 2.0
    center_y = image_height / 2.0
    theta = math.radians(rotation_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    rotated: list[Stroke] = []
    for stroke in strokes:
        rotated_stroke: Stroke = []
        for x, y in stroke:
            local_x = x - center_x
            local_y = y - center_y
            rotated_x = (local_x * cos_theta) - (local_y * sin_theta) + center_x
            rotated_y = (local_x * sin_theta) + (local_y * cos_theta) + center_y
            rotated_stroke.append((rotated_x, rotated_y))
        rotated.append(rotated_stroke)
    return rotated


def _stroke_bounds(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    xs = [point[0] for stroke in strokes for point in stroke]
    ys = [point[1] for stroke in strokes for point in stroke]
    return min(xs), min(ys), max(xs), max(ys)
