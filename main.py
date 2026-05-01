from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_loader import LoadedImage, load_and_process_image
from mapper import map_strokes_to_screen
from models import DrawConfig, MappingConfig, PathBundle, PipelineStats, ProcessingConfig, VectorizeConfig
from mouse_drawer import MouseDrawer, estimate_draw_duration_seconds
from simplifier import stroke_length
from stroke_refiner import refine_mapped_strokes
from vectorizer import estimate_total_path_length, vectorize_mask


LOGGER = logging.getLogger("autodraw")


@dataclass(slots=True)
class PipelineArtifacts:
    loaded_image: LoadedImage
    bundle: PathBundle
    mapped_strokes: list[list[tuple[float, float]]]
    stats: PipelineStats
    component_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert line art into mouse-driven drawing strokes.")
    parser.add_argument("--image", help="Path to input image.")
    parser.add_argument("--import-json", help="Load vector paths from JSON instead of generating from an image.")
    parser.add_argument("--export-json", help="Save vector paths to JSON.")
    parser.add_argument("--export-preview", help="Save a preview image.")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI.")

    parser.add_argument("--top-left-x", type=float, default=0.0)
    parser.add_argument("--top-left-y", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=512.0)
    parser.add_argument("--height", type=float, default=512.0)
    parser.add_argument("--threshold", type=int, default=180)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--simplify", type=float, default=0.35)
    parser.add_argument("--spacing", type=float, default=0.9)
    parser.add_argument("--speed", type=float, default=900.0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--drag-step", type=float, default=1.5)
    parser.add_argument("--preview-brush-size", type=float, default=6.0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument("--resize-width", type=int)
    parser.add_argument("--resize-height", type=int)
    parser.add_argument("--skeletonize", action="store_true")
    parser.add_argument("--min-component-size", type=int, default=8)
    parser.add_argument("--min-segment-length", type=int, default=2)
    parser.add_argument("--max-points-per-stroke", type=int, default=1000)
    parser.add_argument("--min-stroke-length", type=float, default=6.0)
    parser.add_argument("--no-coverage-fill", action="store_true", help="Disable high-fidelity dark-region coverage fills.")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--scale-percent", type=float, default=100.0)
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument("--stretch-to-fit", action="store_true")
    parser.add_argument("--no-preserve-aspect", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Open a preview window.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui:
        from ui import launch_gui

        launch_gui()
        return

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    run_cli(args)


def run_cli(args: argparse.Namespace) -> None:
    if not args.image and not args.import_json:
        raise SystemExit("Either --image or --import-json is required unless using --gui.")

    processing_config = ProcessingConfig(
        threshold=args.threshold,
        invert=args.invert,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        blur_kernel=args.blur,
        skeletonize=args.skeletonize,
        min_component_size=args.min_component_size,
        min_segment_length=args.min_segment_length,
    )
    vectorize_config = VectorizeConfig(
        simplify_tolerance=args.simplify,
        min_point_spacing=args.spacing,
        max_points_per_stroke=args.max_points_per_stroke,
        minimum_stroke_length=args.min_stroke_length,
        enable_coverage_fills=not args.no_coverage_fill,
    )
    mapping_config = MappingConfig(
        top_left_x=args.top_left_x,
        top_left_y=args.top_left_y,
        width=args.width,
        height=args.height,
        preserve_aspect_ratio=not args.no_preserve_aspect,
        stretch_to_fit=args.stretch_to_fit,
        scale_percent=args.scale_percent,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        rotation_degrees=args.rotation,
    )
    draw_config = DrawConfig(
        speed_pixels_per_second=args.speed,
        pause_between_strokes=args.delay,
        countdown_seconds=args.countdown,
        dry_run=args.dry_run,
        max_drag_step_pixels=args.drag_step,
        preview_brush_diameter_pixels=args.preview_brush_size,
    )

    if args.import_json:
        bundle = _load_paths_json(args.import_json)
        artifacts = build_pipeline_from_bundle(
            bundle=bundle,
            mapping_config=mapping_config,
            draw_config=draw_config,
            component_count=len(bundle.strokes),
        )
    else:
        artifacts = process_image_pipeline(
            image_path=args.image,
            processing_config=processing_config,
            vectorize_config=vectorize_config,
            mapping_config=mapping_config,
            draw_config=draw_config,
        )
        bundle = artifacts.bundle

    if args.export_json:
        _save_paths_json(args.export_json, artifacts.bundle)

    _log_stats(artifacts.stats)

    preview_image = render_preview(
        artifacts.loaded_image.original_bgr,
        artifacts.loaded_image.binary,
        artifacts.bundle.strokes,
        artifacts.mapped_strokes,
        mapping_config,
        draw_config,
    )
    if args.export_preview:
        export_path = Path(args.export_preview)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(export_path), preview_image)
        LOGGER.info("Saved preview image to %s", export_path)
    if args.preview:
        cv2.imshow("AutoDraw Preview", preview_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    drawer = MouseDrawer(draw_config)
    result = drawer.draw(artifacts.mapped_strokes)
    if draw_config.dry_run:
        LOGGER.info("Dry run complete.")
    elif result.stopped:
        LOGGER.warning("Drawing stopped early.")
    else:
        LOGGER.info("Drawing complete.")


def process_image_pipeline(
    image_path: str,
    processing_config: ProcessingConfig,
    vectorize_config: VectorizeConfig,
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
) -> PipelineArtifacts:
    loaded = load_and_process_image(image_path, processing_config)
    strokes, component_count = vectorize_mask(
        loaded.drawable_mask.astype(np.uint8),
        vectorize_config,
        min_segment_length=processing_config.min_segment_length,
        grayscale=loaded.grayscale,
    )
    bundle = PathBundle(image_shape=loaded.binary.shape[:2], strokes=strokes)
    return build_pipeline_from_bundle(
        bundle=bundle,
        mapping_config=mapping_config,
        draw_config=draw_config,
        component_count=component_count,
        loaded_image=loaded,
    )


def build_pipeline_from_bundle(
    bundle: PathBundle,
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
    component_count: int,
    loaded_image: LoadedImage | None = None,
) -> PipelineArtifacts:
    if loaded_image is None:
        blank_binary = np.full(bundle.image_shape, 255, dtype=np.uint8)
        loaded_image = LoadedImage(
            original_bgr=cv2.cvtColor(blank_binary, cv2.COLOR_GRAY2BGR),
            grayscale=blank_binary.copy(),
            binary=blank_binary,
            drawable_mask=blank_binary == 0,
        )

    mapped_strokes = map_strokes_to_screen(bundle.strokes, bundle.image_shape, mapping_config)
    mapped_strokes = refine_mapped_strokes(mapped_strokes, mapping_config, draw_config)
    filtered_source_strokes, filtered_mapped_strokes = _filter_short_mapped_strokes(
        bundle.strokes,
        mapped_strokes,
        draw_config.minimum_mapped_stroke_length,
    )
    filtered_bundle = PathBundle(image_shape=bundle.image_shape, strokes=filtered_source_strokes)
    stats = _build_stats(filtered_bundle.strokes, filtered_mapped_strokes, component_count, draw_config)
    return PipelineArtifacts(
        loaded_image=loaded_image,
        bundle=filtered_bundle,
        mapped_strokes=filtered_mapped_strokes,
        stats=stats,
        component_count=component_count,
    )


def render_preview(
    original_bgr: np.ndarray,
    processed_binary: np.ndarray,
    source_strokes: list[list[tuple[float, float]]],
    mapped_strokes: list[list[tuple[float, float]]],
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
) -> np.ndarray:
    original_panel = render_original_panel(original_bgr, (400, 400))
    binary_panel = render_processed_panel(processed_binary, (400, 400))
    vector_panel = render_vector_panel(source_strokes, processed_binary.shape[:2], (400, 400))
    mapped_panel = render_mapped_panel(mapped_strokes, mapping_config, draw_config, (400, 400))

    panels = [
        _annotate_panel(original_panel, "Original"),
        _annotate_panel(binary_panel, "Processed"),
        _annotate_panel(vector_panel, "Stroke Plan"),
        _annotate_panel(mapped_panel, "Simulated Output"),
    ]
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    return np.vstack([top, bottom])


def _ensure_bgr_and_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def render_original_panel(image: np.ndarray, size: tuple[int, int] = (400, 400)) -> np.ndarray:
    return _fit_image_to_canvas(image, size)


def render_processed_panel(binary_image: np.ndarray, size: tuple[int, int] = (400, 400)) -> np.ndarray:
    return _fit_image_to_canvas(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), size)


def render_vector_panel(
    strokes: list[list[tuple[float, float]]],
    image_shape: tuple[int, int] | None = None,
    size: tuple[int, int] = (400, 400),
    color: tuple[int, int, int] = (0, 140, 255),
) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 248, dtype=np.uint8)
    bounds = None
    if image_shape is not None:
        image_height, image_width = image_shape
        bounds = (0.0, 0.0, max(1.0, float(image_width - 1)), max(1.0, float(image_height - 1)))
    _draw_strokes_scaled(strokes, canvas, color, bounds=bounds)
    return canvas


def render_mapped_panel(
    strokes: list[list[tuple[float, float]]],
    mapping_config: MappingConfig,
    draw_config: DrawConfig,
    size: tuple[int, int] = (400, 400),
    color: tuple[int, int, int] = (30, 30, 30),
    screen_size: tuple[int, int] | None = None,
    selection_rect: tuple[float, float, float, float] | None = None,
    show_strokes: bool = True,
) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 246, dtype=np.uint8)
    effective_rect = selection_rect
    if effective_rect is None and mapping_config.width > 0 and mapping_config.height > 0:
        effective_rect = (
            mapping_config.top_left_x,
            mapping_config.top_left_y,
            mapping_config.width,
            mapping_config.height,
        )

    if effective_rect is None or effective_rect[2] <= 0 or effective_rect[3] <= 0:
        cv2.putText(
            canvas,
            "Select a draw area to simulate output",
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (115, 115, 115),
            1,
            cv2.LINE_AA,
        )
        return canvas

    rect_x, rect_y, rect_width, rect_height = effective_rect
    offset_x, offset_y, scale = _fit_rect_to_canvas(rect_width, rect_height, canvas.shape[1], canvas.shape[0], padding=18)
    anchor_x = offset_x - rect_x * scale
    anchor_y = offset_y - rect_y * scale
    preview_x0 = int(round(offset_x))
    preview_y0 = int(round(offset_y))
    preview_x1 = int(round(offset_x + rect_width * scale))
    preview_y1 = int(round(offset_y + rect_height * scale))
    cv2.rectangle(canvas, (preview_x0, preview_y0), (preview_x1, preview_y1), (255, 255, 255), -1)
    cv2.rectangle(canvas, (preview_x0, preview_y0), (preview_x1, preview_y1), (0, 165, 255), 2)
    cv2.putText(
        canvas,
        "Draw area (zoomed)",
        (preview_x0 + 8, max(24, preview_y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 120, 220),
        1,
        cv2.LINE_AA,
    )

    if show_strokes and strokes:
        _draw_screen_space_strokes(strokes, canvas, color, scale, anchor_x, anchor_y, draw_config)
    return canvas


def _draw_strokes_scaled(
    strokes: list[list[tuple[float, float]]],
    canvas: np.ndarray,
    color: tuple[int, int, int],
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    if not strokes:
        return

    if bounds is None:
        xs = [point[0] for stroke in strokes for point in stroke]
        ys = [point[1] for stroke in strokes for point in stroke]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, min_y, max_x, max_y = bounds

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    offset_x, offset_y, scale = _fit_rect_to_canvas(width, height, canvas.shape[1], canvas.shape[0], padding=10)

    for stroke in strokes:
        points = np.array(
            [
                [int(round(offset_x + (point[0] - min_x) * scale)), int(round(offset_y + (point[1] - min_y) * scale))]
                for point in stroke
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)

def _draw_screen_space_strokes(
    strokes: list[list[tuple[float, float]]],
    canvas: np.ndarray,
    color: tuple[int, int, int],
    scale: float,
    offset_x: float,
    offset_y: float,
    draw_config: DrawConfig,
) -> None:
    if not strokes:
        return

    stamp_size = max(2, int(round(draw_config.preview_brush_diameter_pixels * scale)))
    step_pixels = max(0.5, draw_config.max_drag_step_pixels)

    for stroke in strokes:
        if not stroke:
            continue
        _stamp_preview_brush(canvas, offset_x + stroke[0][0] * scale, offset_y + stroke[0][1] * scale, stamp_size, color)
        for start, end in zip(stroke, stroke[1:]):
            distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            steps = max(1, int(np.ceil(distance / step_pixels)))
            for step in range(1, steps + 1):
                t = step / steps
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                _stamp_preview_brush(canvas, offset_x + x * scale, offset_y + y * scale, stamp_size, color)


def _stamp_preview_brush(
    canvas: np.ndarray,
    x: float,
    y: float,
    stamp_size: int,
    color: tuple[int, int, int],
) -> None:
    radius = max(1, stamp_size // 2)
    center_x = int(round(x))
    center_y = int(round(y))
    cv2.circle(canvas, (center_x, center_y), radius, color, -1, lineType=cv2.LINE_AA)


def _fit_rect_to_canvas(
    source_width: float,
    source_height: float,
    canvas_width: int,
    canvas_height: int,
    padding: int,
) -> tuple[float, float, float]:
    usable_width = max(1.0, canvas_width - padding * 2)
    usable_height = max(1.0, canvas_height - padding * 2)
    scale = min(usable_width / max(source_width, 1.0), usable_height / max(source_height, 1.0))
    offset_x = padding + (usable_width - source_width * scale) / 2.0
    offset_y = padding + (usable_height - source_height * scale) / 2.0
    return offset_x, offset_y, scale


def _fit_image_to_canvas(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 248, dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    offset_x, offset_y, scale = _fit_rect_to_canvas(image_width, image_height, size[0], size[1], padding=10)
    target_width = max(1, int(round(image_width * scale)))
    target_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    x0 = int(round(offset_x))
    y0 = int(round(offset_y))
    canvas[y0 : y0 + target_height, x0 : x0 + target_width] = resized
    cv2.rectangle(canvas, (x0, y0), (x0 + target_width, y0 + target_height), (220, 220, 220), 1)
    return canvas


def _resolve_screen_size(screen_size: tuple[int, int] | None, mapping_config: MappingConfig) -> tuple[int, int]:
    if screen_size is not None and screen_size[0] > 0 and screen_size[1] > 0:
        return screen_size

    right = mapping_config.top_left_x + max(mapping_config.width, 1.0)
    bottom = mapping_config.top_left_y + max(mapping_config.height, 1.0)
    return max(1920, int(round(right + 80))), max(1080, int(round(bottom + 80)))


def _annotate_panel(panel: np.ndarray, label: str) -> np.ndarray:
    annotated = panel.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (220, 220, 220), 1)
    cv2.putText(annotated, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    return annotated


def _build_stats(
    source_strokes: list[list[tuple[float, float]]],
    mapped_strokes: list[list[tuple[float, float]]],
    component_count: int,
    draw_config: DrawConfig,
) -> PipelineStats:
    total_points = sum(len(stroke) for stroke in source_strokes)
    total_path_length = estimate_total_path_length(mapped_strokes)
    estimated_draw_seconds = estimate_draw_duration_seconds(mapped_strokes, draw_config)
    return PipelineStats(
        component_count=component_count,
        stroke_count=len(source_strokes),
        total_points=total_points,
        total_path_length=total_path_length,
        estimated_draw_seconds=estimated_draw_seconds,
    )


def _filter_short_mapped_strokes(
    source_strokes: list[list[tuple[float, float]]],
    mapped_strokes: list[list[tuple[float, float]]],
    minimum_mapped_stroke_length: float,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    if minimum_mapped_stroke_length <= 0:
        return [stroke[:] for stroke in source_strokes], [stroke[:] for stroke in mapped_strokes]

    filtered_source_strokes: list[list[tuple[float, float]]] = []
    filtered_mapped_strokes: list[list[tuple[float, float]]] = []
    for source_stroke, mapped_stroke in zip(source_strokes, mapped_strokes):
        if stroke_length(mapped_stroke) < minimum_mapped_stroke_length:
            continue
        filtered_source_strokes.append(source_stroke)
        filtered_mapped_strokes.append(mapped_stroke)
    return filtered_source_strokes, filtered_mapped_strokes


def _log_stats(stats: PipelineStats) -> None:
    LOGGER.info("Components: %s", stats.component_count)
    LOGGER.info("Strokes: %s", stats.stroke_count)
    LOGGER.info("Total points: %s", stats.total_points)
    LOGGER.info("Mapped path length: %.1f px", stats.total_path_length)
    LOGGER.info("Estimated draw time: %.1f s", stats.estimated_draw_seconds)


def _save_paths_json(path: str, bundle: PathBundle) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Saved paths JSON to %s", output_path)


def _load_paths_json(path: str) -> PathBundle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bundle = PathBundle.from_dict(data)
    LOGGER.info("Loaded %s strokes from %s", len(bundle.strokes), path)
    return bundle


if __name__ == "__main__":
    main()
