from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


Point = tuple[float, float]
Stroke = list[Point]


@dataclass(slots=True)
class ProcessingConfig:
    threshold: int = 180
    invert: bool = False
    resize_width: int | None = None
    resize_height: int | None = None
    blur_kernel: int = 0
    skeletonize: bool = False
    min_component_size: int = 8
    min_segment_length: int = 2


@dataclass(slots=True)
class VectorizeConfig:
    simplify_tolerance: float = 0.0
    min_point_spacing: float = 1.0
    max_points_per_stroke: int = 1000
    minimum_stroke_length: float = 6.0
    contour_width_threshold: float = 3.4
    hatch_straightness_threshold: float = 0.96
    hatch_width_threshold: float = 2.2
    noise_length_threshold: float = 6.0
    enable_fill_strokes: bool = True
    fill_spacing: float = 1.25
    enable_coverage_fills: bool = True
    coverage_fill_spacing: float = 1.0
    coverage_fill_min_area: int = 18
    coverage_fill_min_interval_length: float = 2.0
    coverage_fill_min_width: float = 2.6
    coverage_fill_min_bbox_density: float = 0.32
    coverage_fill_dark_pixel_value: int = 96
    enable_solid_region_fills: bool = True
    solid_fill_core_radius: float = 2.6
    solid_fill_spacing: float = 1.25
    solid_fill_min_core_area: int = 14
    solid_fill_min_bbox_density: float = 0.16
    solid_fill_min_contour_ratio: float = 0.18
    solid_fill_min_interval_length: float = 4.0
    solid_fill_max_area_ratio: float = 0.015
    solid_fill_dark_pixel_value: int = 64
    solid_fill_min_dark_pixel_ratio: float = 0.45


@dataclass(slots=True)
class MappingConfig:
    top_left_x: float
    top_left_y: float
    width: float
    height: float
    preserve_aspect_ratio: bool = True
    stretch_to_fit: bool = False
    scale_percent: float = 100.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation_degrees: float = 0.0


@dataclass(slots=True)
class DrawConfig:
    speed_pixels_per_second: float = 900.0
    pause_between_strokes: float = 0.15
    countdown_seconds: int = 3
    dry_run: bool = False
    fail_safe: bool = True
    log_every_n_strokes: int = 10
    max_drag_step_pixels: float = 2.5
    drag_event_interval_seconds: float = 0.003
    preview_brush_diameter_pixels: float = 6.0
    minimum_mapped_stroke_length: float = 1.0
    pen_down_settle_seconds: float = 0.012
    humanize_paths: bool = False
    detail_path_spacing_pixels: float = 0.7
    straight_path_spacing_pixels: float = 1.4
    human_wobble_amplitude_pixels: float = 0.85
    human_wobble_wavelength_pixels: float = 24.0
    reference_capture_rate_hz: float = 60.0
    detail_capture_step_pixels: float = 1.15
    straight_capture_step_pixels: float = 2.4
    minimum_segment_duration_seconds: float = 0.01


@dataclass(slots=True)
class PathBundle:
    image_shape: tuple[int, int]
    strokes: list[Stroke]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_shape": list(self.image_shape),
            "strokes": [[list(point) for point in stroke] for stroke in self.strokes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathBundle":
        return cls(
            image_shape=tuple(data["image_shape"]),
            strokes=[
                [(float(point[0]), float(point[1])) for point in stroke]
                for stroke in data["strokes"]
            ],
        )


@dataclass(slots=True)
class PipelineStats:
    component_count: int
    stroke_count: int
    total_points: int
    total_path_length: float
    estimated_draw_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
