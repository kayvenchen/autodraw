from __future__ import annotations

import math

from models import DrawConfig, MappingConfig, Point, Stroke
from simplifier import reduce_point_spacing, stroke_length


TAU = math.pi * 2.0


def refine_mapped_strokes(
    strokes: list[Stroke],
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
) -> list[Stroke]:
    return [
        _refine_stroke(stroke, mapping_config, draw_config, stroke_index)
        for stroke_index, stroke in enumerate(strokes)
        if len(stroke) >= 2
    ]


def measure_segment_detail(stroke: Stroke, segment_index: int) -> float:
    if segment_index < 0 or segment_index >= len(stroke) - 1:
        return 0.0

    start = stroke[segment_index]
    end = stroke[segment_index + 1]
    distance = _distance(start, end)
    if distance <= 1e-6:
        return 0.0

    turn_start = 0.0
    if segment_index > 0:
        turn_start = _turn_severity(stroke[segment_index - 1], start, end)

    turn_end = 0.0
    if segment_index + 2 < len(stroke):
        turn_end = _turn_severity(start, end, stroke[segment_index + 2])

    local_window = distance
    if segment_index > 0:
        local_window += _distance(stroke[segment_index - 1], start)
    if segment_index + 2 < len(stroke):
        local_window += _distance(end, stroke[segment_index + 2])

    turn_detail = max(turn_start, turn_end)
    short_detail = _clamp(1.0 - (distance / 12.0), 0.0, 1.0)
    dense_detail = _clamp(1.0 - (local_window / 24.0), 0.0, 1.0)
    return _clamp(max(turn_detail, short_detail * 0.55, dense_detail * 0.4), 0.0, 1.0)


def _refine_stroke(
    stroke: Stroke,
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
    stroke_index: int,
) -> Stroke:
    cleaned = _dedupe_adjacent_points(stroke)
    if len(cleaned) < 2:
        return cleaned

    resampled = _adaptive_resample(cleaned, draw_config)
    humanized = _humanize_stroke(resampled, draw_config, stroke_index)
    clamped = _clamp_stroke_to_region(humanized, mapping_config)
    finalized = reduce_point_spacing(clamped, min(0.45, draw_config.detail_path_spacing_pixels * 0.5))
    if len(finalized) < 2:
        return clamped
    return finalized


def _adaptive_resample(stroke: Stroke, draw_config: DrawConfig) -> Stroke:
    if len(stroke) < 2:
        return stroke[:]

    resampled = [stroke[0]]
    for segment_index, (start, end) in enumerate(zip(stroke, stroke[1:])):
        distance = _distance(start, end)
        if distance <= 1e-6:
            continue

        detail = measure_segment_detail(stroke, segment_index)
        spacing = _lerp(
            draw_config.straight_path_spacing_pixels,
            draw_config.detail_path_spacing_pixels,
            detail,
        )
        step_count = max(1, int(math.ceil(distance / max(0.1, spacing))))
        for step in range(1, step_count + 1):
            t = step / step_count
            resampled.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))

    return _dedupe_adjacent_points(resampled)


def _humanize_stroke(stroke: Stroke, draw_config: DrawConfig, stroke_index: int) -> Stroke:
    if not draw_config.humanize_paths or len(stroke) < 3:
        return stroke[:]

    total_length = stroke_length(stroke)
    if total_length <= 1e-6:
        return stroke[:]

    cumulative = [0.0]
    for start, end in zip(stroke, stroke[1:]):
        cumulative.append(cumulative[-1] + _distance(start, end))

    stroke_phase = (stroke_index * 1.61803398875 + total_length * 0.031) % TAU
    wavelength = max(8.0, draw_config.human_wobble_wavelength_pixels)
    length_factor = _clamp((total_length - 14.0) / 56.0, 0.0, 1.0)

    humanized: Stroke = [stroke[0]]
    for index in range(1, len(stroke) - 1):
        current = stroke[index]
        previous = stroke[index - 1]
        following = stroke[index + 1]
        progress = cumulative[index] / total_length
        endpoint_envelope = _smoothstep(_clamp(progress / 0.08, 0.0, 1.0)) * _smoothstep(
            _clamp((1.0 - progress) / 0.08, 0.0, 1.0)
        )
        detail = max(measure_segment_detail(stroke, index - 1), measure_segment_detail(stroke, index))
        straight_factor = (1.0 - detail) ** 1.25
        local_spacing = min(_distance(previous, current), _distance(current, following))
        amplitude_cap = min(
            draw_config.human_wobble_amplitude_pixels,
            max(0.0, local_spacing * (0.7 - (detail * 0.3))),
        )
        amplitude = amplitude_cap * endpoint_envelope * length_factor * straight_factor
        if amplitude <= 1e-3:
            humanized.append(current)
            continue

        phase = (cumulative[index] / wavelength) * TAU + stroke_phase
        wobble = (
            math.sin(phase) + (0.35 * math.sin(phase * 0.47 + stroke_phase * 1.37))
        ) / 1.35
        drift = amplitude * 0.12 * math.sin(phase * 1.11 + stroke_phase * 0.61)
        tangent = _point_tangent(stroke, index)
        normal = (-tangent[1], tangent[0])
        humanized.append(
            (
                current[0] + normal[0] * amplitude * wobble + tangent[0] * drift,
                current[1] + normal[1] * amplitude * wobble + tangent[1] * drift,
            )
        )

    humanized.append(stroke[-1])
    return _dedupe_adjacent_points(humanized)


def _clamp_stroke_to_region(stroke: Stroke, mapping_config: MappingConfig) -> Stroke:
    if mapping_config.width <= 0 or mapping_config.height <= 0:
        return stroke[:]

    min_x = mapping_config.top_left_x
    min_y = mapping_config.top_left_y
    max_x = mapping_config.top_left_x + mapping_config.width
    max_y = mapping_config.top_left_y + mapping_config.height
    return [
        (
            _clamp(point[0], min_x, max_x),
            _clamp(point[1], min_y, max_y),
        )
        for point in stroke
    ]


def _dedupe_adjacent_points(stroke: Stroke) -> Stroke:
    if not stroke:
        return []

    deduped = [stroke[0]]
    for point in stroke[1:]:
        if _distance(deduped[-1], point) > 1e-6:
            deduped.append(point)
    return deduped


def _point_tangent(stroke: Stroke, index: int) -> Point:
    if len(stroke) == 1:
        return (1.0, 0.0)

    previous = stroke[max(0, index - 1)]
    following = stroke[min(len(stroke) - 1, index + 1)]
    dx = following[0] - previous[0]
    dy = following[1] - previous[1]
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-6:
        return (1.0, 0.0)
    return (dx / magnitude, dy / magnitude)


def _turn_severity(previous: Point, current: Point, following: Point) -> float:
    incoming_x = current[0] - previous[0]
    incoming_y = current[1] - previous[1]
    outgoing_x = following[0] - current[0]
    outgoing_y = following[1] - current[1]
    incoming_magnitude = math.hypot(incoming_x, incoming_y)
    outgoing_magnitude = math.hypot(outgoing_x, outgoing_y)
    if incoming_magnitude <= 1e-6 or outgoing_magnitude <= 1e-6:
        return 0.0

    incoming_x /= incoming_magnitude
    incoming_y /= incoming_magnitude
    outgoing_x /= outgoing_magnitude
    outgoing_y /= outgoing_magnitude
    dot = _clamp((incoming_x * outgoing_x) + (incoming_y * outgoing_y), -1.0, 1.0)
    return _clamp((1.0 - dot) * 0.5, 0.0, 1.0)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - (2.0 * value))


def _lerp(start: float, end: float, amount: float) -> float:
    return start + ((end - start) * amount)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
