from __future__ import annotations

import math

from models import Point, Stroke


def reduce_point_spacing(stroke: Stroke, min_spacing: float) -> Stroke:
    if len(stroke) <= 2 or min_spacing <= 0:
        return stroke[:]

    reduced = [stroke[0]]
    for point in stroke[1:-1]:
        if _distance(reduced[-1], point) >= min_spacing:
            reduced.append(point)
    reduced.append(stroke[-1])
    return reduced


def rdp_simplify(stroke: Stroke, epsilon: float) -> Stroke:
    if len(stroke) <= 2 or epsilon <= 0:
        return stroke[:]

    index = -1
    max_distance = -1.0
    start = stroke[0]
    end = stroke[-1]

    for i in range(1, len(stroke) - 1):
        distance = _point_line_distance(stroke[i], start, end)
        if distance > max_distance:
            index = i
            max_distance = distance

    if max_distance <= epsilon:
        return [start, end]

    left = rdp_simplify(stroke[: index + 1], epsilon)
    right = rdp_simplify(stroke[index:], epsilon)
    return left[:-1] + right


def chunk_stroke(stroke: Stroke, max_points: int) -> list[Stroke]:
    if max_points <= 1 or len(stroke) <= max_points:
        return [stroke]

    chunks: list[Stroke] = []
    step = max_points - 1
    start = 0
    while start < len(stroke) - 1:
        end = min(len(stroke), start + max_points)
        chunk = stroke[start:end]
        if len(chunk) >= 2:
            chunks.append(chunk)
        start += step
    return chunks


def stroke_length(stroke: Stroke) -> float:
    return sum(_distance(stroke[i - 1], stroke[i]) for i in range(1, len(stroke)))


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    if start == end:
        return _distance(point, start)

    numerator = abs(
        (end[0] - start[0]) * (start[1] - point[1])
        - (start[0] - point[0]) * (end[1] - start[1])
    )
    denominator = math.hypot(end[0] - start[0], end[1] - start[1])
    return numerator / denominator
