from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from typing import Callable


@dataclass(slots=True)
class ScreenSelection:
    x: int
    y: int
    width: int
    height: int


class ScreenRegionSelector:
    def __init__(
        self,
        root: tk.Tk,
        aspect_ratio: float | None,
        on_complete: Callable[[ScreenSelection | None], None],
    ) -> None:
        self.root = root
        self.aspect_ratio = aspect_ratio
        self.on_complete = on_complete
        self.overlay = tk.Toplevel(root)
        self.overlay.withdraw()
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.18)
        self.overlay.configure(bg="black")

        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.canvas = tk.Canvas(
            self.overlay,
            width=self.screen_width,
            height=self.screen_height,
            highlightthickness=0,
            cursor="crosshair",
            bg="black",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._start_x = 0
        self._start_y = 0
        self._current_selection: ScreenSelection | None = None
        self._selection_rect_id: int | None = None
        self._shade_rect_ids: list[int] = []
        self._selection_fill_id: int | None = None
        self._hint_id: int | None = None

        self._bind_events()

    def show(self) -> None:
        self.overlay.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.overlay.deiconify()
        self.overlay.lift()
        self.overlay.focus_force()
        self._draw_hint("Drag over Roblox to choose the draw area. ESC cancels. The box stays locked to the image aspect ratio.")

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.overlay.bind("<Escape>", self._cancel)

    def _on_press(self, event: tk.Event) -> None:
        self._start_x = int(event.x)
        self._start_y = int(event.y)
        self._current_selection = None

    def _on_drag(self, event: tk.Event) -> None:
        selection = constrain_selection_to_aspect(
            start_x=self._start_x,
            start_y=self._start_y,
            end_x=int(event.x),
            end_y=int(event.y),
            aspect_ratio=self.aspect_ratio,
            max_width=self.screen_width,
            max_height=self.screen_height,
        )
        self._current_selection = selection
        self._render_selection(selection)

    def _on_release(self, event: tk.Event) -> None:
        if self._current_selection is None:
            self._cancel()
            return

        if self._current_selection.width < 8 or self._current_selection.height < 8:
            self._cancel()
            return

        selection = self._current_selection
        self._close()
        self.on_complete(selection)

    def _cancel(self, event: tk.Event | None = None) -> None:
        self._close()
        self.on_complete(None)

    def _close(self) -> None:
        if self.overlay.winfo_exists():
            self.overlay.destroy()

    def _render_selection(self, selection: ScreenSelection) -> None:
        if self._selection_rect_id is not None:
            self.canvas.delete(self._selection_rect_id)
        if self._selection_fill_id is not None:
            self.canvas.delete(self._selection_fill_id)
        for rect_id in self._shade_rect_ids:
            self.canvas.delete(rect_id)
        self._shade_rect_ids.clear()

        x1 = selection.x
        y1 = selection.y
        x2 = selection.x + selection.width
        y2 = selection.y + selection.height

        self._shade_rect_ids.extend(
            [
                self.canvas.create_rectangle(0, 0, self.screen_width, y1, fill="#101010", stipple="gray50", outline=""),
                self.canvas.create_rectangle(0, y2, self.screen_width, self.screen_height, fill="#101010", stipple="gray50", outline=""),
                self.canvas.create_rectangle(0, y1, x1, y2, fill="#101010", stipple="gray50", outline=""),
                self.canvas.create_rectangle(x2, y1, self.screen_width, y2, fill="#101010", stipple="gray50", outline=""),
            ]
        )
        self._selection_fill_id = self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", stipple="gray25", outline="")
        self._selection_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00d7ff", width=3)

        label = f"{selection.width} x {selection.height}"
        self._draw_hint(label, x=x1 + 12, y=max(24, y1 - 20), anchor="w")

    def _draw_hint(self, text: str, x: int = 24, y: int = 24, anchor: str = "nw") -> None:
        if self._hint_id is not None:
            self.canvas.delete(self._hint_id)
        self._hint_id = self.canvas.create_text(
            x,
            y,
            anchor=anchor,
            text=text,
            fill="white",
            font=("Segoe UI", 12, "bold"),
        )


def constrain_selection_to_aspect(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    aspect_ratio: float | None,
    max_width: int,
    max_height: int,
) -> ScreenSelection:
    raw_dx = end_x - start_x
    raw_dy = end_y - start_y
    width = abs(raw_dx)
    height = abs(raw_dy)

    if aspect_ratio is not None and aspect_ratio > 0:
        if width == 0 and height == 0:
            constrained_width = 0
            constrained_height = 0
        elif height == 0:
            constrained_width = width
            constrained_height = int(round(constrained_width / aspect_ratio))
        elif width == 0:
            constrained_height = height
            constrained_width = int(round(constrained_height * aspect_ratio))
        elif width / height > aspect_ratio:
            constrained_height = height
            constrained_width = int(round(constrained_height * aspect_ratio))
        else:
            constrained_width = width
            constrained_height = int(round(constrained_width / aspect_ratio))
    else:
        constrained_width = width
        constrained_height = height

    sign_x = 1 if raw_dx >= 0 else -1
    sign_y = 1 if raw_dy >= 0 else -1
    candidate_x2 = start_x + sign_x * constrained_width
    candidate_y2 = start_y + sign_y * constrained_height

    x1 = max(0, min(start_x, candidate_x2))
    y1 = max(0, min(start_y, candidate_y2))
    x2 = min(max_width, max(start_x, candidate_x2))
    y2 = min(max_height, max(start_y, candidate_y2))
    return ScreenSelection(x=int(x1), y=int(y1), width=int(max(0, x2 - x1)), height=int(max(0, y2 - y1)))
