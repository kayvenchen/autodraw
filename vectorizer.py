from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from models import Point, Stroke, VectorizeConfig
from simplifier import chunk_stroke, rdp_simplify, reduce_point_spacing, stroke_length


NEIGHBORS_8 = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


@dataclass(slots=True)
class StrokeFeatures:
    length: float
    point_count: int
    mean_width: float
    max_width: float
    straightness: float
    curvature: float


@dataclass(slots=True)
class StrokeCandidate:
    stroke: Stroke
    role: str
    priority: int
    features: StrokeFeatures


@dataclass(slots=True)
class FillInterval:
    y: int
    start_x: int
    end_x: int
    used: bool = False


def vectorize_mask(
    mask: np.ndarray,
    config: VectorizeConfig,
    min_segment_length: int = 2,
    grayscale: np.ndarray | None = None,
) -> tuple[list[Stroke], int]:
    original_mask = mask.astype(bool)
    component_count, labels = _count_components(original_mask)
    distance_map = cv2.distanceTransform(original_mask.astype(np.uint8), cv2.DIST_L2, 3)
    candidates: list[StrokeCandidate] = []

    for component_label in range(1, component_count + 1):
        component_mask = labels == component_label
        path_mask = skeletonize(component_mask) if np.any(component_mask) else component_mask
        component_points = {tuple(point) for point in np.argwhere(path_mask)}
        if len(component_points) >= min_segment_length:
            raw_strokes = _extract_component_strokes(component_points)
            for raw_stroke in raw_strokes:
                features = _measure_stroke(raw_stroke, distance_map)
                role = _classify_stroke(features, config, min_segment_length)
                if role == "noise":
                    continue
                candidates.extend(_build_candidates(raw_stroke, features, role, config, min_segment_length, original_mask))

        candidates.extend(_generate_coverage_fill_candidates(component_mask, distance_map, config, grayscale))

    if config.enable_solid_region_fills:
        candidates.extend(
            _generate_solid_fill_candidates(
                original_mask,
                labels,
                component_count,
                distance_map,
                config,
                grayscale=grayscale,
            )
        )

    ordered = _order_candidates(candidates)
    return [candidate.stroke for candidate in ordered], component_count


def sort_strokes_by_proximity(strokes: list[Stroke]) -> list[Stroke]:
    ordered = _order_candidates(
        [
            StrokeCandidate(
                stroke=stroke,
                role="detail",
                priority=1,
                features=StrokeFeatures(
                    length=stroke_length(stroke),
                    point_count=len(stroke),
                    mean_width=1.0,
                    max_width=1.0,
                    straightness=1.0,
                    curvature=0.0,
                ),
            )
            for stroke in strokes
        ]
    )
    return [candidate.stroke for candidate in ordered]


def estimate_total_path_length(strokes: list[Stroke]) -> float:
    return sum(stroke_length(stroke) for stroke in strokes)


def _count_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels_count - 1, labels


def _extract_component_strokes(component_points: set[tuple[int, int]]) -> list[Stroke]:
    adjacency = {point: _neighbors(point, component_points) for point in component_points}
    traversed_edges: set[frozenset[tuple[int, int]]] = set()
    strokes: list[Stroke] = []
    nodes = [point for point, neighbors in adjacency.items() if len(neighbors) != 2]

    for node in nodes:
        for neighbor in adjacency[node]:
            edge_key = _edge(node, neighbor)
            if edge_key in traversed_edges:
                continue
            stroke = _walk_path(node, neighbor, adjacency, traversed_edges)
            if len(stroke) >= 2:
                strokes.append(_to_xy(stroke))

    for start in component_points:
        for neighbor in adjacency[start]:
            edge_key = _edge(start, neighbor)
            if edge_key in traversed_edges:
                continue
            stroke = _walk_cycle(start, neighbor, adjacency, traversed_edges)
            if len(stroke) >= 2:
                strokes.append(_to_xy(stroke))

    return strokes


def _walk_path(
    start: tuple[int, int],
    neighbor: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    traversed_edges: set[frozenset[tuple[int, int]]],
) -> list[tuple[int, int]]:
    path = [start, neighbor]
    traversed_edges.add(_edge(start, neighbor))
    previous = start
    current = neighbor

    while True:
        next_candidates = [candidate for candidate in adjacency[current] if candidate != previous]
        if len(adjacency[current]) != 2 or not next_candidates:
            break

        next_point = next_candidates[0]
        edge_key = _edge(current, next_point)
        if edge_key in traversed_edges:
            break
        path.append(next_point)
        traversed_edges.add(edge_key)
        previous, current = current, next_point

    return path


def _walk_cycle(
    start: tuple[int, int],
    neighbor: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    traversed_edges: set[frozenset[tuple[int, int]]],
) -> list[tuple[int, int]]:
    path = [start, neighbor]
    traversed_edges.add(_edge(start, neighbor))
    previous = start
    current = neighbor

    while True:
        next_candidates = [candidate for candidate in adjacency[current] if candidate != previous]
        if not next_candidates:
            break

        next_point = next_candidates[0]
        edge_key = _edge(current, next_point)
        if edge_key in traversed_edges:
            break
        path.append(next_point)
        traversed_edges.add(edge_key)
        previous, current = current, next_point
        if current == start:
            break

    return path


def _measure_stroke(stroke: Stroke, distance_map: np.ndarray) -> StrokeFeatures:
    length = stroke_length(stroke)
    point_count = len(stroke)
    width_samples = []
    for x, y in stroke:
        row = int(round(y))
        col = int(round(x))
        if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
            width_samples.append(float(distance_map[row, col]) * 2.0)
    if not width_samples:
        width_samples = [1.0]

    mean_width = float(np.mean(width_samples))
    max_width = float(np.max(width_samples))
    straightness = _compute_straightness(stroke, length)
    curvature = _compute_curvature(stroke)
    return StrokeFeatures(
        length=length,
        point_count=point_count,
        mean_width=mean_width,
        max_width=max_width,
        straightness=straightness,
        curvature=curvature,
    )


def _classify_stroke(features: StrokeFeatures, config: VectorizeConfig, min_segment_length: int) -> str:
    if features.point_count < max(2, min_segment_length):
        return "noise"
    if features.length < max(config.noise_length_threshold, 4.0):
        return "noise"
    if features.length < 8.0 and features.point_count <= 4 and features.mean_width < config.contour_width_threshold + 0.6:
        return "noise"

    is_contour = (
        features.length >= 54.0
        or (
            features.length >= 10.0
            and (
                features.mean_width >= config.contour_width_threshold
                or features.max_width >= config.contour_width_threshold + 1.4
            )
        )
        or (features.length >= 8.0 and features.curvature >= 0.20)
    )
    if is_contour:
        return "contour"

    is_hatch = (
        features.straightness >= config.hatch_straightness_threshold
        and features.mean_width <= config.hatch_width_threshold
        and features.length >= max(config.noise_length_threshold + 2.0, 8.0)
    )
    if is_hatch:
        return "hatch"

    if features.length < 5.0 and features.mean_width < 2.8:
        return "noise"

    return "detail"


def _build_candidates(
    raw_stroke: Stroke,
    features: StrokeFeatures,
    role: str,
    config: VectorizeConfig,
    min_segment_length: int,
    original_mask: np.ndarray,
) -> list[StrokeCandidate]:
    processed_strokes = _finalize_stroke(raw_stroke, config, min_segment_length, role)
    if not processed_strokes:
        return []

    priority = _role_priority(role)
    candidates = [
        StrokeCandidate(
            stroke=stroke,
            role=role,
            priority=priority,
            features=features,
        )
        for stroke in processed_strokes
    ]

    if role == "contour" and config.enable_fill_strokes:
        for base_stroke in processed_strokes:
            fill_strokes = _generate_parallel_fill_strokes(base_stroke, features, config, original_mask)
            candidates.extend(
                StrokeCandidate(
                    stroke=stroke,
                    role="fill",
                    priority=_role_priority("fill"),
                    features=features,
                )
                for stroke in fill_strokes
            )

    return candidates


def _generate_solid_fill_candidates(
    original_mask: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    distance_map: np.ndarray,
    config: VectorizeConfig,
    grayscale: np.ndarray | None = None,
) -> list[StrokeCandidate]:
    if grayscale is None:
        return []

    candidates: list[StrokeCandidate] = []
    image_area = max(1, original_mask.shape[0] * original_mask.shape[1])
    reference_island_area = max(
        max(20, config.solid_fill_min_core_area * 2),
        int(round(image_area * config.solid_fill_max_area_ratio)),
    )

    dark_mask = original_mask & (grayscale <= config.solid_fill_dark_pixel_value)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, open_kernel).astype(bool)
    if not np.any(dark_mask):
        return candidates

    seed_kernel_size = max(3, int(round(config.solid_fill_core_radius)) * 2 + 1)
    if seed_kernel_size % 2 == 0:
        seed_kernel_size += 1
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_kernel_size, seed_kernel_size))
    seed_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, seed_kernel).astype(bool)
    if not np.any(seed_mask):
        seed_mask = dark_mask

    island_count, island_labels = cv2.connectedComponents(seed_mask.astype(np.uint8), connectivity=8)
    minimum_area = max(30, config.solid_fill_min_core_area * 2)

    for island_label in range(1, island_count):
        island_mask = island_labels == island_label
        island_mask = cv2.dilate(island_mask.astype(np.uint8), seed_kernel).astype(bool) & dark_mask
        island_area = int(np.count_nonzero(island_mask))
        if island_area < minimum_area:
            continue

        ys, xs = np.where(island_mask)
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        bbox_density = island_area / max(bbox_width * bbox_height, 1)
        if bbox_density < config.solid_fill_min_bbox_density:
            continue
        if min(bbox_width, bbox_height) < 5:
            continue

        local_mask = island_mask[min_y : max_y + 1, min_x : max_x + 1]
        contour_ratio = _component_contour_ratio(local_mask)
        if contour_ratio < config.solid_fill_min_contour_ratio:
            continue
        local_distance = distance_map[min_y : max_y + 1, min_x : max_x + 1]
        local_pixels = grayscale[min_y : max_y + 1, min_x : max_x + 1][local_mask]
        if local_pixels.size == 0:
            continue
        dark_ratio = float(np.mean(local_pixels <= config.solid_fill_dark_pixel_value))
        if dark_ratio < config.solid_fill_min_dark_pixel_ratio:
            continue

        fill_mask = _build_fill_core_mask(local_mask, config, minimum_area)
        adaptive_spacing = max(1.0, config.solid_fill_spacing)
        if island_area > reference_island_area:
            adaptive_spacing = max(
                adaptive_spacing,
                min(4.0, config.solid_fill_spacing * math.sqrt(island_area / max(reference_island_area, 1))),
            )

        fill_strokes = _generate_dynamic_fill_strokes(
            fill_mask=fill_mask,
            origin_x=min_x,
            origin_y=min_y,
            spacing=adaptive_spacing,
            min_interval_length=max(2.0, config.solid_fill_min_interval_length),
            base_angle_degrees=_mask_dominant_angle(local_mask),
            bbox_density=bbox_density,
            island_area=island_area,
        )
        if not fill_strokes:
            continue

        local_dark_distance = local_distance[fill_mask]
        mean_width = max(1.0, float(np.mean(local_dark_distance)) * 2.0)
        max_width = max(1.0, float(np.max(local_dark_distance)) * 2.0)
        for stroke in fill_strokes:
            processed_strokes = _finalize_stroke(stroke, config, 2, "solid_fill")
            for processed_stroke in processed_strokes:
                features = StrokeFeatures(
                    length=stroke_length(processed_stroke),
                    point_count=len(processed_stroke),
                    mean_width=mean_width,
                    max_width=max_width,
                    straightness=_compute_straightness(processed_stroke, stroke_length(processed_stroke)),
                    curvature=_compute_curvature(processed_stroke),
                )
                candidates.append(
                    StrokeCandidate(
                        stroke=processed_stroke,
                        role="solid_fill",
                        priority=_role_priority("solid_fill"),
                        features=features,
                    )
                )

    return candidates


def _generate_coverage_fill_candidates(
    component_mask: np.ndarray,
    distance_map: np.ndarray,
    config: VectorizeConfig,
    grayscale: np.ndarray | None = None,
) -> list[StrokeCandidate]:
    if not config.enable_coverage_fills:
        return []

    fill_mask = component_mask
    if grayscale is not None:
        fill_mask = component_mask & (grayscale <= config.coverage_fill_dark_pixel_value)

    component_area = int(np.count_nonzero(fill_mask))
    if component_area < config.coverage_fill_min_area:
        return []

    local_distance_values = distance_map[fill_mask]
    if local_distance_values.size == 0:
        return []

    max_width = float(np.max(local_distance_values)) * 2.0
    if max_width < config.coverage_fill_min_width:
        return []

    ys, xs = np.where(fill_mask)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    local_mask = fill_mask[min_y : max_y + 1, min_x : max_x + 1]
    if min(local_mask.shape) <= 1:
        return []
    if _mask_bbox_density(local_mask) < config.coverage_fill_min_bbox_density:
        return []

    fill_strokes = _scanline_fill_strokes(
        fill_mask=local_mask,
        origin_x=min_x,
        origin_y=min_y,
        spacing=max(1.0, config.coverage_fill_spacing),
        min_interval_length=max(1.0, config.coverage_fill_min_interval_length),
    )
    candidates: list[StrokeCandidate] = []
    for fill_stroke in fill_strokes:
        processed_strokes = _finalize_stroke(fill_stroke, config, 2, "coverage_fill")
        for processed_stroke in processed_strokes:
            length = stroke_length(processed_stroke)
            features = StrokeFeatures(
                length=length,
                point_count=len(processed_stroke),
                mean_width=max(1.0, float(np.mean(local_distance_values)) * 2.0),
                max_width=max_width,
                straightness=_compute_straightness(processed_stroke, length),
                curvature=_compute_curvature(processed_stroke),
            )
            candidates.append(
                StrokeCandidate(
                    stroke=processed_stroke,
                    role="coverage_fill",
                    priority=_role_priority("coverage_fill"),
                    features=features,
                )
            )

    return candidates


def _finalize_stroke(
    stroke: Stroke,
    config: VectorizeConfig,
    min_segment_length: int,
    role: str,
) -> list[Stroke]:
    deduped = _dedupe_adjacent_points(stroke)
    if len(deduped) < max(2, min_segment_length):
        return []

    raw_length = stroke_length(deduped)
    bbox_width, bbox_height = _stroke_bbox_size(deduped)
    curvature = _compute_curvature(deduped)
    straightness = _compute_straightness(deduped, raw_length)
    spacing = config.min_point_spacing
    simplify_tolerance = config.simplify_tolerance

    if role == "hatch":
        spacing = max(spacing, 2.0)
        simplify_tolerance = max(simplify_tolerance, 0.8)
        if _compute_straightness(deduped, stroke_length(deduped)) >= 0.97:
            deduped = [deduped[0], deduped[-1]]
    elif role == "detail":
        spacing = max(1.0, min(spacing, 1.8))
        simplify_tolerance *= 0.75
    elif role == "contour":
        spacing = max(0.8, min(spacing, 1.4))
        simplify_tolerance *= 0.6
    elif role == "coverage_fill":
        spacing = max(0.5, min(spacing, max(0.65, config.coverage_fill_spacing * 0.75)))
        simplify_tolerance *= 0.1
    elif role == "solid_fill":
        spacing = max(0.8, min(spacing, max(1.2, config.solid_fill_spacing * 1.1)))
        simplify_tolerance *= 0.25

    preserve_compact_detail = role in {"detail", "contour"} and (
        (raw_length <= 95.0 and max(bbox_width, bbox_height) <= 34.0)
        or (raw_length <= 68.0 and curvature >= 0.14)
    )
    preserve_curved_geometry = role in {"detail", "contour"} and (
        curvature >= 0.08
        and straightness <= 0.985
        and raw_length <= 220.0
    )
    if preserve_compact_detail:
        spacing = min(spacing, 1.0 if role == "detail" else 1.15)
        simplify_tolerance *= 0.35
    if preserve_curved_geometry:
        spacing = min(spacing, 0.8 if role == "detail" else 0.95)
        simplify_tolerance *= 0.2 if role == "detail" else 0.28

    if spacing > 0:
        deduped = reduce_point_spacing(deduped, spacing)
    if simplify_tolerance > 0:
        deduped = rdp_simplify(deduped, simplify_tolerance)
    if len(deduped) < max(2, min_segment_length):
        return []
    if stroke_length(deduped) < config.minimum_stroke_length:
        return []

    return [chunk for chunk in chunk_stroke(deduped, config.max_points_per_stroke) if len(chunk) >= 2]


def _generate_parallel_fill_strokes(
    stroke: Stroke,
    features: StrokeFeatures,
    config: VectorizeConfig,
    original_mask: np.ndarray,
) -> list[Stroke]:
    if features.mean_width < config.contour_width_threshold + 0.9:
        return []
    if features.length < 20.0 or len(stroke) < 3:
        return []

    layers = min(2, int(round((features.mean_width - config.contour_width_threshold) / 1.4)))
    if layers <= 0:
        return []

    offsets: list[float] = []
    for layer_index in range(1, layers + 1):
        distance = layer_index * config.fill_spacing
        offsets.extend([-distance, distance])

    generated: list[Stroke] = []
    for offset in offsets:
        offset_stroke = _offset_stroke(stroke, offset, original_mask)
        if len(offset_stroke) >= 2 and stroke_length(offset_stroke) >= max(6.0, features.length * 0.35):
            generated.append(offset_stroke)
    return generated


def _scanline_fill_strokes(
    fill_mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    spacing: float,
    min_interval_length: float,
    angle_degrees: float = 0.0,
) -> list[Stroke]:
    strokes: list[Stroke] = []
    rotated_mask = fill_mask
    inverse_matrix = None
    if abs(angle_degrees) > 1.0:
        center = ((fill_mask.shape[1] - 1) / 2.0, (fill_mask.shape[0] - 1) / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
        rotated_mask = cv2.warpAffine(
            fill_mask.astype(np.uint8),
            rotation_matrix,
            (fill_mask.shape[1], fill_mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        inverse_matrix = cv2.invertAffineTransform(rotation_matrix)

    step = max(1, int(round(spacing)))
    interval_rows: list[list[FillInterval]] = []

    for local_y in range(0, rotated_mask.shape[0], step):
        row = rotated_mask[local_y]
        if not np.any(row):
            continue
        row_intervals = [
            FillInterval(y=local_y, start_x=start_x, end_x=end_x)
            for start_x, end_x in _row_intervals(row)
            if (end_x - start_x + 1) >= min_interval_length
        ]
        if row_intervals:
            interval_rows.append(row_intervals)

    if not interval_rows:
        return strokes

    center_x = (rotated_mask.shape[1] - 1) / 2.0
    sweep_rows = _build_fill_sweeps(interval_rows, center_x=center_x, max_row_gap=max(1, step * 2))

    for sweep in sweep_rows:
        if len(sweep) < 2:
            continue
        if inverse_matrix is None:
            stroke = [(float(origin_x + x), float(origin_y + y)) for x, y in sweep]
        else:
            stroke = [
                _transform_affine_point(inverse_matrix, float(x), float(y), origin_x, origin_y)
                for x, y in sweep
            ]
        stroke = _dedupe_adjacent_points(stroke)
        if len(stroke) >= 2:
            strokes.append(stroke)

    return strokes


def _generate_dynamic_fill_strokes(
    fill_mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    spacing: float,
    min_interval_length: float,
    base_angle_degrees: float,
    bbox_density: float,
    island_area: int,
) -> list[Stroke]:
    if not np.any(fill_mask):
        return []

    height, width = fill_mask.shape
    horizontal_bias = width >= height
    primary_slant = 16.0 if horizontal_bias else -16.0
    pass_specs = [(base_angle_degrees + primary_slant, spacing)]

    should_crosshatch = bbox_density >= 0.38 or island_area >= 420
    if should_crosshatch:
        pass_specs.append((base_angle_degrees - primary_slant * 0.85, spacing * 1.35))

    generated: list[Stroke] = []
    for angle, hatch_spacing in pass_specs:
        generated.extend(
            _scanline_fill_strokes(
                fill_mask=fill_mask,
                origin_x=origin_x,
                origin_y=origin_y,
                spacing=max(1.0, hatch_spacing),
                min_interval_length=min_interval_length,
                angle_degrees=angle,
            )
        )
    return generated


def _build_fill_core_mask(
    fill_mask: np.ndarray,
    config: VectorizeConfig,
    minimum_area: int,
) -> np.ndarray:
    radius = max(0, int(round(config.solid_fill_core_radius * 0.5)))
    if radius <= 0:
        return fill_mask
    kernel_size = radius * 2 + 1
    if min(fill_mask.shape) < kernel_size:
        return fill_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(fill_mask.astype(np.uint8), kernel).astype(bool)
    if int(np.count_nonzero(eroded)) < max(6, minimum_area // 4):
        return fill_mask
    return eroded


def _build_fill_sweeps(
    interval_rows: list[list[FillInterval]],
    center_x: float,
    max_row_gap: int,
) -> list[list[tuple[float, float]]]:
    sweeps: list[list[tuple[float, float]]] = []

    for row_index, intervals in enumerate(interval_rows):
        for interval in intervals:
            if interval.used:
                continue

            interval.used = True
            direction = _seed_fill_direction(interval, center_x)
            sweep = _interval_sweep_points(interval, direction)
            current = interval
            current_row_index = row_index
            current_exit_x = sweep[-1][0]

            while True:
                next_match = _find_next_fill_interval(
                    interval_rows,
                    current_row_index,
                    current,
                    current_exit_x,
                    max_row_gap,
                )
                if next_match is None:
                    break

                next_row_index, next_interval, bridge_x = next_match
                next_interval.used = True
                next_direction = _choose_fill_direction(next_interval, bridge_x, direction, center_x)
                bridge_point = (float(bridge_x), float(next_interval.y))
                if sweep[-1] != bridge_point:
                    sweep.append(bridge_point)

                next_exit_x = next_interval.end_x if next_direction > 0 else next_interval.start_x
                next_endpoint = (float(next_exit_x), float(next_interval.y))
                if sweep[-1] != next_endpoint:
                    sweep.append(next_endpoint)

                current = next_interval
                current_row_index = next_row_index
                current_exit_x = next_exit_x
                direction = next_direction

            sweeps.append(sweep)

    return sweeps


def _seed_fill_direction(interval: FillInterval, center_x: float) -> int:
    interval_center = (interval.start_x + interval.end_x) / 2.0
    return 1 if interval_center <= center_x else -1


def _interval_sweep_points(interval: FillInterval, direction: int) -> list[tuple[float, float]]:
    start_x = interval.start_x if direction > 0 else interval.end_x
    end_x = interval.end_x if direction > 0 else interval.start_x
    y = float(interval.y)
    return [(float(start_x), y), (float(end_x), y)]


def _find_next_fill_interval(
    interval_rows: list[list[FillInterval]],
    current_row_index: int,
    current_interval: FillInterval,
    current_exit_x: float,
    max_row_gap: int,
) -> tuple[int, FillInterval, int] | None:
    current_min = min(current_interval.start_x, current_interval.end_x)
    current_max = max(current_interval.start_x, current_interval.end_x)
    current_y = current_interval.y
    best_match: tuple[int, FillInterval, int] | None = None
    best_score: tuple[float, float, float] | None = None

    for next_row_index in range(current_row_index + 1, len(interval_rows)):
        next_intervals = interval_rows[next_row_index]
        if not next_intervals:
            continue

        row_gap = next_intervals[0].y - current_y
        if row_gap > max_row_gap:
            break

        for next_interval in next_intervals:
            if next_interval.used:
                continue

            overlap_start = max(current_min, next_interval.start_x)
            overlap_end = min(current_max, next_interval.end_x)
            if overlap_start > overlap_end:
                continue

            bridge_x = int(round(min(max(current_exit_x, overlap_start), overlap_end)))
            overlap_width = overlap_end - overlap_start + 1
            center_delta = abs(
                ((current_interval.start_x + current_interval.end_x) / 2.0)
                - ((next_interval.start_x + next_interval.end_x) / 2.0)
            )
            score = (
                float(overlap_width),
                -abs(current_exit_x - bridge_x),
                -center_delta,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_match = (next_row_index, next_interval, bridge_x)

        if best_match is not None:
            break

    return best_match


def _choose_fill_direction(
    interval: FillInterval,
    bridge_x: int,
    previous_direction: int,
    center_x: float,
) -> int:
    left_span = bridge_x - interval.start_x
    right_span = interval.end_x - bridge_x
    if abs(left_span - right_span) <= 1.0:
        interval_center = (interval.start_x + interval.end_x) / 2.0
        return previous_direction if abs(interval_center - center_x) <= 2.0 else _seed_fill_direction(interval, center_x)
    return -1 if left_span > right_span else 1


def _offset_stroke(stroke: Stroke, offset: float, original_mask: np.ndarray) -> Stroke:
    offset_points: Stroke = []
    for index, point in enumerate(stroke):
        normal = _point_normal(stroke, index)
        candidate = (
            point[0] + normal[0] * offset,
            point[1] + normal[1] * offset,
        )
        row = int(round(candidate[1]))
        col = int(round(candidate[0]))
        if 0 <= row < original_mask.shape[0] and 0 <= col < original_mask.shape[1] and original_mask[row, col]:
            offset_points.append(candidate)
        else:
            offset_points.append(point)
    return _dedupe_adjacent_points(offset_points)


def _point_normal(stroke: Stroke, index: int) -> Point:
    if len(stroke) == 1:
        return (0.0, 1.0)

    previous = stroke[max(0, index - 1)]
    current = stroke[index]
    following = stroke[min(len(stroke) - 1, index + 1)]
    tangent = (following[0] - previous[0], following[1] - previous[1])
    magnitude = math.hypot(tangent[0], tangent[1])
    if magnitude <= 1e-6:
        return (0.0, 1.0)
    tangent = (tangent[0] / magnitude, tangent[1] / magnitude)
    return (-tangent[1], tangent[0])


def _component_contour_ratio(local_mask: np.ndarray) -> float:
    contour_source = local_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour_area = max(cv2.contourArea(contour) for contour in contours)
    if contour_area <= 1e-6:
        return 0.0
    return float(np.count_nonzero(local_mask)) / contour_area


def _mask_bbox_density(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    width = int(xs.max()) - int(xs.min()) + 1
    height = int(ys.max()) - int(ys.min()) + 1
    return float(np.count_nonzero(mask)) / max(width * height, 1)


def _mask_dominant_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = coords - coords.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    dominant = eigenvectors[:, int(np.argmax(eigenvalues))]
    return math.degrees(math.atan2(float(dominant[1]), float(dominant[0])))


def _transform_affine_point(
    inverse_matrix: np.ndarray,
    x: float,
    y: float,
    origin_x: int,
    origin_y: int,
) -> Point:
    mapped_x = inverse_matrix[0, 0] * x + inverse_matrix[0, 1] * y + inverse_matrix[0, 2]
    mapped_y = inverse_matrix[1, 0] * x + inverse_matrix[1, 1] * y + inverse_matrix[1, 2]
    return (float(origin_x) + float(mapped_x), float(origin_y) + float(mapped_y))


def _row_intervals(row: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(row):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(row) - 1))
    return intervals


def _order_candidates(candidates: list[StrokeCandidate]) -> list[StrokeCandidate]:
    if not candidates:
        return []

    ordered: list[StrokeCandidate] = []
    current_end: Point | None = None

    for priority in sorted({candidate.priority for candidate in candidates}):
        bucket = [candidate for candidate in candidates if candidate.priority == priority]
        bucket.sort(key=lambda candidate: (-candidate.features.length, candidate.features.mean_width))
        if not bucket:
            continue

        if current_end is None:
            chosen = bucket.pop(0)
            ordered.append(chosen)
            current_end = chosen.stroke[-1]

        while bucket:
            best_index = 0
            best_distance = float("inf")
            reverse_best = False

            for index, candidate in enumerate(bucket):
                start_distance = _point_distance(current_end, candidate.stroke[0]) if current_end is not None else 0.0
                end_distance = _point_distance(current_end, candidate.stroke[-1]) if current_end is not None else 0.0
                if start_distance < best_distance:
                    best_index = index
                    best_distance = start_distance
                    reverse_best = False
                if end_distance < best_distance:
                    best_index = index
                    best_distance = end_distance
                    reverse_best = True

            next_candidate = bucket.pop(best_index)
            if reverse_best:
                next_candidate = StrokeCandidate(
                    stroke=list(reversed(next_candidate.stroke)),
                    role=next_candidate.role,
                    priority=next_candidate.priority,
                    features=next_candidate.features,
                )
            ordered.append(next_candidate)
            current_end = next_candidate.stroke[-1]

    return ordered


def _role_priority(role: str) -> int:
    if role in {"coverage_fill", "solid_fill"}:
        return 0
    if role == "contour":
        return 1
    if role == "detail":
        return 2
    if role == "fill":
        return 3
    return 4


def _neighbors(point: tuple[int, int], component_points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    neighbors: list[tuple[int, int]] = []
    for delta_col, delta_row in NEIGHBORS_8:
        candidate = (row + delta_row, col + delta_col)
        if candidate in component_points:
            neighbors.append(candidate)
    return neighbors


def _edge(a: tuple[int, int], b: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset((a, b))


def _to_xy(path: list[tuple[int, int]]) -> Stroke:
    return [(float(col), float(row)) for row, col in path]


def _compute_straightness(stroke: Stroke, length: float) -> float:
    if len(stroke) < 2 or length <= 1e-6:
        return 1.0
    end_to_end = _point_distance(stroke[0], stroke[-1])
    return end_to_end / max(length, 1e-6)


def _compute_curvature(stroke: Stroke) -> float:
    if len(stroke) < 3:
        return 0.0

    angle_sum = 0.0
    counted = 0
    for index in range(1, len(stroke) - 1):
        prev_vec = (stroke[index][0] - stroke[index - 1][0], stroke[index][1] - stroke[index - 1][1])
        next_vec = (stroke[index + 1][0] - stroke[index][0], stroke[index + 1][1] - stroke[index][1])
        prev_mag = math.hypot(prev_vec[0], prev_vec[1])
        next_mag = math.hypot(next_vec[0], next_vec[1])
        if prev_mag <= 1e-6 or next_mag <= 1e-6:
            continue
        dot = (prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1]) / (prev_mag * next_mag)
        dot = max(-1.0, min(1.0, dot))
        angle_sum += math.acos(dot)
        counted += 1
    if counted == 0:
        return 0.0
    return angle_sum / counted


def _point_distance(a: Point, b: Point) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _stroke_bbox_size(stroke: Stroke) -> tuple[float, float]:
    if not stroke:
        return (0.0, 0.0)
    xs = [point[0] for point in stroke]
    ys = [point[1] for point in stroke]
    return max(xs) - min(xs), max(ys) - min(ys)


def _dedupe_adjacent_points(stroke: Stroke) -> Stroke:
    if not stroke:
        return []

    deduped = [stroke[0]]
    for point in stroke[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    return deduped
