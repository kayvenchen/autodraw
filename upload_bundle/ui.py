from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import hashlib
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional at runtime
    Image = None
    ImageTk = None

from main import (
    PipelineArtifacts,
    process_image_pipeline,
    render_mapped_panel,
    render_original_panel,
    render_vector_panel,
)
from models import DrawConfig, MappingConfig, ProcessingConfig, VectorizeConfig
from mouse_drawer import MouseDrawer, mouse_runtime_error
from screen_selector import ScreenRegionSelector, ScreenSelection


LOGGER = logging.getLogger(__name__)
STATE_FILE = Path.home() / ".autodraw_gui_state.json"
PREVIEW_CACHE_DIR = Path.home() / ".autodraw_preview_cache"
STATE_VERSION = 2

GUI_DEFAULTS = {
    "threshold": "180",
    "simplify": "0.35",
    "spacing": "0.9",
    "minimum_stroke_length": "6.0",
    "speed": "900.0",
    "delay": "0.05",
    "drag_step": "1.5",
    "preview_brush_size": "6.0",
    "countdown": "3",
}

LEGACY_DEFAULTS = {
    "simplify": "1.25",
    "spacing": "2.0",
    "speed": "1200.0",
    "drag_step": "2.5",
}


@dataclass(frozen=True)
class NumericFieldSpec:
    default: float
    minimum: float | None
    maximum: float | None
    increment: float
    decimals: int
    integer: bool = False


class AutoDrawGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AutoDraw Roblox Drawer")
        self.root.geometry("1460x980")
        self.root.minsize(1180, 820)
        self.preview_size = (430, 300)

        self.image_path = tk.StringVar()
        self.threshold = tk.StringVar(value=GUI_DEFAULTS["threshold"])
        self.invert = tk.BooleanVar(value=False)
        self.skeletonize = tk.BooleanVar(value=True)
        self.simplify = tk.StringVar(value=GUI_DEFAULTS["simplify"])
        self.spacing = tk.StringVar(value=GUI_DEFAULTS["spacing"])
        self.minimum_stroke_length = tk.StringVar(value=GUI_DEFAULTS["minimum_stroke_length"])
        self.speed = tk.StringVar(value=GUI_DEFAULTS["speed"])
        self.delay = tk.StringVar(value=GUI_DEFAULTS["delay"])
        self.drag_step = tk.StringVar(value=GUI_DEFAULTS["drag_step"])
        self.preview_brush_size = tk.StringVar(value=GUI_DEFAULTS["preview_brush_size"])
        self.countdown = tk.StringVar(value=GUI_DEFAULTS["countdown"])
        self.dry_run = tk.BooleanVar(value=False)

        self.selection_x = tk.IntVar(value=0)
        self.selection_y = tk.IntVar(value=0)
        self.selection_width = tk.IntVar(value=0)
        self.selection_height = tk.IntVar(value=0)

        if sys.platform == "darwin":
            hotkey_text = "Hotkeys: Ctrl+O open image | F6 select draw area | F5 draw | stop by moving mouse to top-left or Ctrl+C"
        else:
            hotkey_text = "Hotkeys: Ctrl+O open image | F6 select draw area | F5 draw | ESC emergency stop"
        self.hotkeys_var = tk.StringVar(value=hotkey_text)
        self.area_var = tk.StringVar(value="Draw area: not selected")
        self.stats_var = tk.StringVar(value="Load an image to build the drawing paths.")
        self.status_var = tk.StringVar(value="Ready.")

        self.raw_preview_label: ttk.Label | None = None
        self.vector_preview_label: ttk.Label | None = None
        self.target_preview_label: ttk.Label | None = None
        self.log_widget: ScrolledText | None = None

        self._raw_photo = None
        self._vector_photo = None
        self._target_photo = None
        self._pipeline: PipelineArtifacts | None = None
        self._selector: ScreenRegionSelector | None = None
        self._draw_thread: threading.Thread | None = None
        self._event_queue: Queue[tuple[str, object]] = Queue()
        self._state_save_after_id: str | None = None
        self._state_ready = False
        self._numeric_specs: dict[str, NumericFieldSpec] = {}
        self._control_widgets: list[tk.Widget] = []

        self._load_state_into_vars()

        self._build()
        self._bind_state_traces()
        self._bind_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_area_label()
        self._state_ready = True
        self.root.after(50, self._restore_preview_if_possible)
        self.root.after(100, self._poll_ui_events)

    def _build(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.columnconfigure(8, weight=1)

        self.open_button = ttk.Button(toolbar, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=(0, 8))
        self.select_area_button = ttk.Button(toolbar, text="Select Draw Area (F6)", command=self.select_draw_area)
        self.select_area_button.grid(row=0, column=1, padx=(0, 8))
        self.draw_button = ttk.Button(toolbar, text="Draw (F5)", command=self.start_draw)
        self.draw_button.grid(row=0, column=2, padx=(0, 8))
        self.refresh_button = ttk.Button(toolbar, text="Refresh", command=self.refresh_preview)
        self.refresh_button.grid(row=0, column=3, padx=(0, 8))
        dry_run_check = ttk.Checkbutton(toolbar, text="Dry Run", variable=self.dry_run)
        dry_run_check.grid(row=0, column=4, padx=(0, 12))
        ttk.Label(toolbar, textvariable=self.hotkeys_var).grid(row=0, column=8, sticky="e")
        self._control_widgets.extend(
            [
                self.open_button,
                self.select_area_button,
                self.draw_button,
                self.refresh_button,
                dry_run_check,
            ]
        )

        controls = ttk.LabelFrame(container, text="Vector Settings", padding=10)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column in range(6):
            controls.columnconfigure(column, weight=1)

        self._add_numeric_field(
            controls,
            "Threshold",
            self.threshold,
            NumericFieldSpec(default=180, minimum=1, maximum=255, increment=1, decimals=0, integer=True),
            0,
            0,
        )
        self._add_numeric_field(
            controls,
            "Simplify",
            self.simplify,
            NumericFieldSpec(default=0.35, minimum=0.0, maximum=20.0, increment=0.05, decimals=2),
            0,
            1,
        )
        self._add_numeric_field(
            controls,
            "Spacing",
            self.spacing,
            NumericFieldSpec(default=0.9, minimum=0.0, maximum=20.0, increment=0.05, decimals=2),
            0,
            2,
        )
        self._add_numeric_field(
            controls,
            "Min Length",
            self.minimum_stroke_length,
            NumericFieldSpec(default=6.0, minimum=0.0, maximum=100.0, increment=0.5, decimals=2),
            0,
            3,
        )
        self._add_numeric_field(
            controls,
            "Speed px/s",
            self.speed,
            NumericFieldSpec(default=900.0, minimum=1.0, maximum=10000.0, increment=50.0, decimals=1),
            0,
            4,
        )
        self._add_numeric_field(
            controls,
            "Stroke Delay",
            self.delay,
            NumericFieldSpec(default=0.05, minimum=0.0, maximum=2.0, increment=0.01, decimals=2),
            0,
            5,
        )
        self._add_numeric_field(
            controls,
            "Countdown",
            self.countdown,
            NumericFieldSpec(default=3, minimum=0, maximum=30, increment=1, decimals=0, integer=True),
            1,
            2,
        )
        self._add_numeric_field(
            controls,
            "Drag Step",
            self.drag_step,
            NumericFieldSpec(default=1.5, minimum=0.5, maximum=20.0, increment=0.25, decimals=2),
            1,
            3,
        )
        self._add_numeric_field(
            controls,
            "Brush Size",
            self.preview_brush_size,
            NumericFieldSpec(default=6.0, minimum=1.0, maximum=40.0, increment=0.5, decimals=2),
            1,
            4,
        )
        invert_check = ttk.Checkbutton(controls, text="Invert", variable=self.invert)
        invert_check.grid(row=1, column=0, sticky="w", pady=(8, 0))
        skeleton_check = ttk.Checkbutton(controls, text="Skeletonize", variable=self.skeletonize)
        skeleton_check.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.area_var).grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))
        self._control_widgets.extend([invert_check, skeleton_check])

        content = ttk.Frame(container)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=1)
        content.rowconfigure(0, weight=1)

        raw_frame = ttk.LabelFrame(content, text="Raw Image")
        raw_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.raw_preview_label = ttk.Label(raw_frame, anchor="center")
        self.raw_preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        vector_frame = ttk.LabelFrame(content, text="Vectorized Paths")
        vector_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.vector_preview_label = ttk.Label(vector_frame, anchor="center")
        self.vector_preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        target_frame = ttk.LabelFrame(content, text="Simulated Output")
        target_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.target_preview_label = ttk.Label(target_frame, anchor="center")
        self.target_preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        footer = ttk.Frame(container)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)

        ttk.Label(footer, textvariable=self.stats_var).grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 8))

        self.log_widget = ScrolledText(footer, height=8, wrap=tk.WORD, state="disabled")
        self.log_widget.grid(row=2, column=0, sticky="ew")

    def _add_numeric_field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        spec: NumericFieldSpec,
        row: int,
        column: int,
    ) -> None:
        variable_key = str(variable)
        self._numeric_specs[variable_key] = spec
        field = ttk.Frame(parent)
        field.grid(row=row, column=column, sticky="ew", padx=4)
        ttk.Label(field, text=label).pack(anchor="w")
        editor = ttk.Frame(field)
        editor.pack(anchor="w", pady=(2, 0))

        down_button = ttk.Button(
            editor,
            text="-",
            width=2,
            command=lambda current=variable: self._step_numeric_var(current, -1),
        )
        down_button.pack(side=tk.LEFT)

        validate_command = (
            self.root.register(self._validate_numeric_input),
            "%P",
            "1" if spec.integer else "0",
        )
        entry = ttk.Entry(
            editor,
            textvariable=variable,
            width=10,
            justify="right",
            validate="key",
            validatecommand=validate_command,
        )
        entry.pack(side=tk.LEFT, padx=4)
        entry.bind("<FocusOut>", lambda _event, current=variable: self._normalize_numeric_var(current))
        entry.bind("<Return>", lambda _event, current=variable: self._normalize_numeric_var(current))
        entry.bind("<KP_Enter>", lambda _event, current=variable: self._normalize_numeric_var(current))
        entry.bind("<Up>", lambda _event, current=variable: self._step_numeric_var(current, 1))
        entry.bind("<Down>", lambda _event, current=variable: self._step_numeric_var(current, -1))

        up_button = ttk.Button(
            editor,
            text="+",
            width=2,
            command=lambda current=variable: self._step_numeric_var(current, 1),
        )
        up_button.pack(side=tk.LEFT)

        self._control_widgets.extend([down_button, entry, up_button])

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<F5>", lambda _event: self.start_draw())
        self.root.bind_all("<F6>", lambda _event: self.select_draw_area())

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select line art image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        self.image_path.set(path)
        self._append_log(f"Loaded image: {path}")
        self.refresh_preview()

    def refresh_preview(self) -> None:
        if not self.image_path.get():
            messagebox.showerror("Missing image", "Select an image first.")
            return

        try:
            self._pipeline = process_image_pipeline(
                image_path=self.image_path.get(),
                processing_config=self._build_processing_config(),
                vectorize_config=self._build_vectorize_config(),
                mapping_config=self._build_mapping_config(),
                draw_config=self._build_draw_config(dry_run_override=True),
            )
            self._update_preview_labels(self._pipeline)
            self._update_stats(self._pipeline)
            self._update_status("Preview refreshed.")
        except Exception as exc:  # pragma: no cover - GUI path
            self._update_status("Preview failed.")
            messagebox.showerror("Preview failed", str(exc))

    def select_draw_area(self) -> None:
        if not self.image_path.get():
            messagebox.showerror("Missing image", "Select an image before choosing a draw area.")
            return
        if self._draw_thread is not None and self._draw_thread.is_alive():
            messagebox.showinfo("Busy", "Wait for the current draw operation to finish.")
            return
        if self._pipeline is None:
            self.refresh_preview()
            if self._pipeline is None:
                return

        aspect_ratio = self._current_image_aspect_ratio()
        self._append_log("Opening area selector.")
        self._update_status("Select the drawing area on screen.")
        self.root.iconify()
        self.root.after(180, lambda: self._show_selector(aspect_ratio))

    def _show_selector(self, aspect_ratio: float | None) -> None:
        self._selector = ScreenRegionSelector(self.root, aspect_ratio, self._on_area_selected)
        self._selector.show()

    def _on_area_selected(self, selection: ScreenSelection | None) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._selector = None

        if selection is None:
            self._append_log("Area selection cancelled.")
            self._update_status("Area selection cancelled.")
            return

        self.selection_x.set(selection.x)
        self.selection_y.set(selection.y)
        self.selection_width.set(selection.width)
        self.selection_height.set(selection.height)
        self._refresh_area_label()
        self._append_log(
            f"Selected draw area x={selection.x}, y={selection.y}, width={selection.width}, height={selection.height}"
        )
        self.refresh_preview()

    def start_draw(self) -> None:
        if self._draw_thread is not None and self._draw_thread.is_alive():
            messagebox.showinfo("Busy", "A draw operation is already running.")
            return
        if not self.image_path.get():
            messagebox.showerror("Missing image", "Select an image first.")
            return
        if self.selection_width.get() <= 0 or self.selection_height.get() <= 0:
            messagebox.showerror("Missing draw area", "Select the draw area first.")
            return

        try:
            self._pipeline = process_image_pipeline(
                image_path=self.image_path.get(),
                processing_config=self._build_processing_config(),
                vectorize_config=self._build_vectorize_config(),
                mapping_config=self._build_mapping_config(),
                draw_config=self._build_draw_config(),
            )
        except Exception as exc:
            messagebox.showerror("Processing failed", str(exc))
            return

        runtime_error = None if self.dry_run.get() else mouse_runtime_error(request_permissions=True)
        if runtime_error is not None:
            messagebox.showerror("Missing dependency", runtime_error)
            return

        self._set_busy_state(True)
        self._append_log("Starting draw request.")
        self._update_stats(self._pipeline)

        if not self.dry_run.get():
            self.root.iconify()
            self.root.after(220, self._launch_draw_worker)
        else:
            self._launch_draw_worker()

    def _launch_draw_worker(self) -> None:
        assert self._pipeline is not None
        draw_config = self._build_draw_config()
        strokes = self._pipeline.mapped_strokes

        def run_draw() -> None:
            try:
                drawer = MouseDrawer(draw_config)
                result = drawer.draw(strokes, progress_callback=lambda message: self._event_queue.put(("progress", message)))
                self._event_queue.put(("done", result))
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._event_queue.put(("error", str(exc)))

        self._draw_thread = threading.Thread(target=run_draw, daemon=True)
        self._draw_thread.start()
        self._update_status("Drawing started." if not draw_config.dry_run else "Dry run started.")

    def _poll_ui_events(self) -> None:
        try:
            while True:
                event_type, payload = self._event_queue.get_nowait()
                if event_type == "progress":
                    message = str(payload)
                    self._append_log(message)
                    self._update_status(message)
                elif event_type == "done":
                    self._handle_draw_complete(payload)
                elif event_type == "error":
                    self._handle_draw_error(str(payload))
        except Empty:
            pass
        finally:
            self.root.after(100, self._poll_ui_events)

    def _handle_draw_complete(self, result: object) -> None:
        self._draw_thread = None
        self._set_busy_state(False)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        stopped = bool(getattr(result, "stopped", False))
        strokes_completed = int(getattr(result, "strokes_completed", 0))
        if stopped:
            message = f"Drawing stopped early after {strokes_completed} strokes."
        elif self.dry_run.get():
            message = f"Dry run finished with {strokes_completed} strokes prepared."
        else:
            message = f"Drawing finished. Completed {strokes_completed} strokes."
        self._append_log(message)
        self._update_status(message)

    def _handle_draw_error(self, message: str) -> None:
        self._draw_thread = None
        self._set_busy_state(False)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._append_log(f"Draw failed: {message}")
        self._update_status("Draw failed.")
        messagebox.showerror("Draw failed", message)

    def _build_processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            threshold=self._safe_int(
                self.threshold,
                default=int(GUI_DEFAULTS["threshold"]),
                minimum=1,
                maximum=255,
            ),
            invert=bool(self.invert.get()),
            skeletonize=bool(self.skeletonize.get()),
            min_component_size=8,
            min_segment_length=2,
        )

    def _build_vectorize_config(self) -> VectorizeConfig:
        return VectorizeConfig(
            simplify_tolerance=self._safe_float(
                self.simplify,
                default=float(GUI_DEFAULTS["simplify"]),
                minimum=0.0,
                maximum=20.0,
            ),
            min_point_spacing=self._safe_float(
                self.spacing,
                default=float(GUI_DEFAULTS["spacing"]),
                minimum=0.0,
                maximum=20.0,
            ),
            max_points_per_stroke=1000,
            minimum_stroke_length=self._safe_float(
                self.minimum_stroke_length,
                default=float(GUI_DEFAULTS["minimum_stroke_length"]),
                minimum=0.0,
                maximum=100.0,
            ),
        )

    def _build_mapping_config(self) -> MappingConfig:
        width = int(self.selection_width.get())
        height = int(self.selection_height.get())
        if width <= 0 or height <= 0:
            default_width = 1000
            aspect_ratio = self._current_image_aspect_ratio()
            default_height = int(round(default_width / aspect_ratio)) if aspect_ratio and aspect_ratio > 0 else 1000
            width = default_width
            height = max(1, default_height)
        return MappingConfig(
            top_left_x=float(self.selection_x.get()),
            top_left_y=float(self.selection_y.get()),
            width=float(width),
            height=float(height),
            preserve_aspect_ratio=True,
            stretch_to_fit=False,
        )

    def _build_draw_config(self, dry_run_override: bool | None = None) -> DrawConfig:
        dry_run = bool(self.dry_run.get()) if dry_run_override is None else dry_run_override
        return DrawConfig(
            speed_pixels_per_second=self._safe_float(
                self.speed,
                default=float(GUI_DEFAULTS["speed"]),
                minimum=1.0,
                maximum=10000.0,
            ),
            pause_between_strokes=self._safe_float(
                self.delay,
                default=float(GUI_DEFAULTS["delay"]),
                minimum=0.0,
                maximum=2.0,
            ),
            countdown_seconds=self._safe_int(
                self.countdown,
                default=int(GUI_DEFAULTS["countdown"]),
                minimum=0,
                maximum=30,
            ),
            dry_run=dry_run,
            max_drag_step_pixels=self._safe_float(
                self.drag_step,
                default=float(GUI_DEFAULTS["drag_step"]),
                minimum=0.5,
                maximum=20.0,
            ),
            preview_brush_diameter_pixels=self._safe_float(
                self.preview_brush_size,
                default=float(GUI_DEFAULTS["preview_brush_size"]),
                minimum=1.0,
                maximum=40.0,
            ),
        )

    def _current_image_aspect_ratio(self) -> float | None:
        if self._pipeline is None:
            return None
        image_height, image_width = self._pipeline.bundle.image_shape
        if image_height <= 0:
            return None
        return image_width / image_height

    def _update_preview_labels(self, pipeline: PipelineArtifacts) -> None:
        panel_size = self.preview_size
        raw_panel = render_original_panel(pipeline.loaded_image.original_bgr, panel_size)
        vector_panel = self._get_cached_vector_panel(pipeline, panel_size)
        selection_rect = (0.0, 0.0, 0.0, 0.0)
        has_selection = self.selection_width.get() > 0 and self.selection_height.get() > 0
        mapped_strokes = pipeline.mapped_strokes if has_selection else []
        if has_selection:
            selection_rect = (
                float(self.selection_x.get()),
                float(self.selection_y.get()),
                float(self.selection_width.get()),
                float(self.selection_height.get()),
            )
        target_panel = render_mapped_panel(
            mapped_strokes,
            self._build_mapping_config(),
            self._build_draw_config(dry_run_override=True),
            panel_size,
            selection_rect=selection_rect,
            show_strokes=has_selection,
        )

        self._raw_photo = self._to_photo(raw_panel, panel_size)
        self._vector_photo = self._to_photo(vector_panel, panel_size)
        self._target_photo = self._to_photo(target_panel, panel_size)

        assert self.raw_preview_label is not None
        assert self.vector_preview_label is not None
        assert self.target_preview_label is not None
        self.raw_preview_label.configure(image=self._raw_photo)
        self.vector_preview_label.configure(image=self._vector_photo)
        self.target_preview_label.configure(image=self._target_photo)

    def _update_stats(self, pipeline: PipelineArtifacts) -> None:
        self.stats_var.set(
            f"Components: {pipeline.stats.component_count} | "
            f"Strokes: {pipeline.stats.stroke_count} | "
            f"Points: {pipeline.stats.total_points} | "
            f"Estimated draw time: {pipeline.stats.estimated_draw_seconds:.1f}s"
        )

    def _to_photo(self, image_bgr, max_size: tuple[int, int]):
        if Image is None or ImageTk is None:
            raise RuntimeError("Pillow is required for GUI previews.")
        image_rgb = image_bgr[:, :, ::-1]
        image = Image.fromarray(image_rgb)
        image.thumbnail(max_size)
        return ImageTk.PhotoImage(image)

    def _append_log(self, message: str) -> None:
        LOGGER.info(message)
        if self.log_widget is None:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, message + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def _set_busy_state(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for widget in self._control_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                continue

    def _refresh_area_label(self) -> None:
        width = self.selection_width.get()
        height = self.selection_height.get()
        if width > 0 and height > 0:
            self.area_var.set(
                f"Draw area: x={self.selection_x.get()}, y={self.selection_y.get()}, width={width}, height={height}"
            )
        else:
            self.area_var.set("Draw area: not selected")

    def _safe_float(
        self,
        variable: tk.StringVar,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        raw = variable.get().strip()
        if raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, tk.TclError):
                value = default

        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)

        variable.set(str(value))
        return value

    def _safe_int(
        self,
        variable: tk.StringVar,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        raw = variable.get().strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(float(raw))
            except (TypeError, ValueError, tk.TclError):
                value = default

        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)

        variable.set(str(value))
        return value

    def _validate_numeric_input(self, proposed: str, integer_only: str) -> bool:
        if proposed == "":
            return True
        if integer_only == "1":
            return proposed.isdigit()
        if proposed == ".":
            return True
        if proposed.count(".") > 1:
            return False
        return all(character.isdigit() or character == "." for character in proposed)

    def _normalize_numeric_var(self, variable: tk.StringVar) -> str:
        spec = self._numeric_specs.get(str(variable))
        if spec is None:
            return variable.get()

        raw = variable.get().strip()
        if raw in {"", "."}:
            value = float(spec.default)
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, tk.TclError):
                value = float(spec.default)

        if spec.minimum is not None:
            value = max(float(spec.minimum), value)
        if spec.maximum is not None:
            value = min(float(spec.maximum), value)

        if spec.integer:
            normalized = str(int(round(value)))
        else:
            normalized = f"{value:.{spec.decimals}f}".rstrip("0").rstrip(".")
            if spec.decimals > 0 and "." not in normalized:
                normalized += ".0"

        if variable.get() != normalized:
            variable.set(normalized)
        return normalized

    def _step_numeric_var(self, variable: tk.StringVar, direction: int) -> str:
        spec = self._numeric_specs.get(str(variable))
        if spec is None:
            return "break"

        normalized = self._normalize_numeric_var(variable)
        try:
            current_value = float(normalized)
        except ValueError:
            current_value = float(spec.default)

        stepped_value = current_value + (float(spec.increment) * direction)
        if spec.minimum is not None:
            stepped_value = max(float(spec.minimum), stepped_value)
        if spec.maximum is not None:
            stepped_value = min(float(spec.maximum), stepped_value)

        if spec.integer:
            variable.set(str(int(round(stepped_value))))
        else:
            stepped_text = f"{stepped_value:.{spec.decimals}f}".rstrip("0").rstrip(".")
            if spec.decimals > 0 and "." not in stepped_text:
                stepped_text += ".0"
            variable.set(stepped_text)
        return "break"

    def _bind_state_traces(self) -> None:
        tracked_variables: list[tk.Variable] = [
            self.image_path,
            self.threshold,
            self.invert,
            self.skeletonize,
            self.simplify,
            self.spacing,
            self.minimum_stroke_length,
            self.speed,
            self.delay,
            self.drag_step,
            self.preview_brush_size,
            self.countdown,
            self.dry_run,
            self.selection_x,
            self.selection_y,
            self.selection_width,
            self.selection_height,
        ]
        for variable in tracked_variables:
            variable.trace_add("write", self._on_state_variable_changed)

    def _on_state_variable_changed(self, *_args: object) -> None:
        self._refresh_area_label()
        if not self._state_ready:
            return
        self._schedule_state_save()

    def _schedule_state_save(self) -> None:
        if self._state_save_after_id is not None:
            self.root.after_cancel(self._state_save_after_id)
        self._state_save_after_id = self.root.after(200, self._save_state)

    def _snapshot_state(self) -> dict[str, object]:
        return {
            "state_version": STATE_VERSION,
            "image_path": self.image_path.get(),
            "threshold": self.threshold.get(),
            "invert": bool(self.invert.get()),
            "skeletonize": bool(self.skeletonize.get()),
            "simplify": self.simplify.get(),
            "spacing": self.spacing.get(),
            "minimum_stroke_length": self.minimum_stroke_length.get(),
            "speed": self.speed.get(),
            "delay": self.delay.get(),
            "drag_step": self.drag_step.get(),
            "preview_brush_size": self.preview_brush_size.get(),
            "countdown": self.countdown.get(),
            "dry_run": bool(self.dry_run.get()),
            "selection_x": int(self.selection_x.get()),
            "selection_y": int(self.selection_y.get()),
            "selection_width": int(self.selection_width.get()),
            "selection_height": int(self.selection_height.get()),
        }

    def _save_state(self) -> None:
        self._state_save_after_id = None
        try:
            STATE_FILE.write_text(json.dumps(self._snapshot_state(), indent=2), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to save GUI state: %s", exc)

    def _load_state_into_vars(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to load GUI state: %s", exc)
            return

        self.image_path.set(str(data.get("image_path", "")))
        self.threshold.set(str(data.get("threshold", GUI_DEFAULTS["threshold"])))
        self.invert.set(bool(data.get("invert", False)))
        self.skeletonize.set(bool(data.get("skeletonize", True)))
        self.simplify.set(str(data.get("simplify", GUI_DEFAULTS["simplify"])))
        self.spacing.set(str(data.get("spacing", GUI_DEFAULTS["spacing"])))
        self.minimum_stroke_length.set(str(data.get("minimum_stroke_length", GUI_DEFAULTS["minimum_stroke_length"])))
        self.speed.set(str(data.get("speed", GUI_DEFAULTS["speed"])))
        self.delay.set(str(data.get("delay", GUI_DEFAULTS["delay"])))
        self.drag_step.set(str(data.get("drag_step", GUI_DEFAULTS["drag_step"])))
        self.preview_brush_size.set(str(data.get("preview_brush_size", GUI_DEFAULTS["preview_brush_size"])))
        self.countdown.set(str(data.get("countdown", GUI_DEFAULTS["countdown"])))
        self.dry_run.set(bool(data.get("dry_run", False)))
        self.selection_x.set(int(data.get("selection_x", 0)))
        self.selection_y.set(int(data.get("selection_y", 0)))
        self.selection_width.set(int(data.get("selection_width", 0)))
        self.selection_height.set(int(data.get("selection_height", 0)))

        try:
            state_version = int(data.get("state_version", 0))
        except (TypeError, ValueError):
            state_version = 0
        if state_version < STATE_VERSION:
            self._migrate_legacy_defaults(data)

    def _migrate_legacy_defaults(self, data: dict[str, object]) -> None:
        if str(data.get("simplify", "")) == LEGACY_DEFAULTS["simplify"]:
            self.simplify.set(GUI_DEFAULTS["simplify"])
        if str(data.get("spacing", "")) == LEGACY_DEFAULTS["spacing"]:
            self.spacing.set(GUI_DEFAULTS["spacing"])
        if str(data.get("speed", "")) == LEGACY_DEFAULTS["speed"]:
            self.speed.set(GUI_DEFAULTS["speed"])
        if str(data.get("drag_step", "")) == LEGACY_DEFAULTS["drag_step"]:
            self.drag_step.set(GUI_DEFAULTS["drag_step"])

    def _restore_preview_if_possible(self) -> None:
        image_path = self.image_path.get().strip()
        if not image_path:
            return
        if not Path(image_path).exists():
            self._update_status("Previous image path was not found.")
            self.image_path.set("")
            self._save_state()
            return
        try:
            self.refresh_preview()
            self._append_log(f"Restored previous image: {image_path}")
        except Exception as exc:
            self._update_status("Failed to restore previous image.")
            LOGGER.warning("Failed to restore previous image: %s", exc)

    def _on_close(self) -> None:
        if self._state_save_after_id is not None:
            self.root.after_cancel(self._state_save_after_id)
            self._state_save_after_id = None
        self._save_state()
        self.root.destroy()

    def _get_cached_vector_panel(self, pipeline: PipelineArtifacts, panel_size: tuple[int, int]):
        PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fingerprint_payload = {
            "image_path": self.image_path.get(),
            "threshold": self.threshold.get(),
            "invert": bool(self.invert.get()),
            "skeletonize": bool(self.skeletonize.get()),
            "simplify": self.simplify.get(),
            "spacing": self.spacing.get(),
            "minimum_stroke_length": self.minimum_stroke_length.get(),
            "image_shape": list(pipeline.bundle.image_shape),
            "stroke_count": len(pipeline.bundle.strokes),
            "size": list(panel_size),
            "cache_version": 4,
        }
        cache_key = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = PREVIEW_CACHE_DIR / f"{cache_key}.png"
        if cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached is not None:
                return cached

        vector_panel = render_vector_panel(pipeline.bundle.strokes, pipeline.bundle.image_shape, panel_size)
        cv2.imwrite(str(cache_path), vector_panel)
        return vector_panel


def launch_gui() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = tk.Tk()
    AutoDrawGui(root)
    root.mainloop()
