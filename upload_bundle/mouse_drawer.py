from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable

from models import DrawConfig, Stroke
from stroke_refiner import measure_segment_detail

if sys.platform != "darwin":
    try:
        import pyautogui
    except ImportError:  # pragma: no cover - depends on local runtime
        pyautogui = None
else:  # pragma: no cover - skipped on non-macOS
    pyautogui = None

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover - depends on local runtime
    keyboard = None

try:
    import ApplicationServices
except ImportError:  # pragma: no cover - macOS only
    ApplicationServices = None

try:
    import Quartz
except ImportError:  # pragma: no cover - macOS only
    Quartz = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DrawResult:
    stopped: bool
    strokes_completed: int


class EmergencyStop(Exception):
    pass


MAC_FAILSAFE_CORNER_THRESHOLD = 8
MIN_INTERPOLATED_DRAG_STEPS = 2


def estimate_draw_duration_seconds(strokes: list[Stroke], config: DrawConfig) -> float:
    if not strokes:
        return 0.0

    total_duration = float(config.countdown_seconds)
    for index, stroke in enumerate(strokes):
        total_duration += _stroke_duration_seconds(stroke, config)
        if index < len(strokes) - 1:
            total_duration += config.pause_between_strokes
    return total_duration


def _stroke_duration_seconds(stroke: Stroke, config: DrawConfig) -> float:
    if len(stroke) < 2:
        return 0.0

    total_duration = max(0.0, config.pen_down_settle_seconds)
    for segment_index in range(len(stroke) - 1):
        total_duration += _segment_duration_for_config(stroke, segment_index, config)
    return total_duration


def _segment_duration_for_config(stroke: Stroke, segment_index: int, config: DrawConfig) -> float:
    start = stroke[segment_index]
    end = stroke[segment_index + 1]
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance <= 1e-6:
        return 0.0

    base_duration = 0.0
    if config.speed_pixels_per_second > 0:
        base_duration = distance / config.speed_pixels_per_second

    detail = measure_segment_detail(stroke, segment_index)
    capture_step = _lerp(config.straight_capture_step_pixels, config.detail_capture_step_pixels, detail)
    capture_speed = max(1e-6, capture_step * max(1.0, config.reference_capture_rate_hz))
    capture_duration = distance / capture_speed
    return max(base_duration, capture_duration, max(0.0, config.minimum_segment_duration_seconds))


def _ease_drag_progress(progress: float) -> float:
    return progress * progress * (3.0 - (2.0 * progress))


def _lerp(start: float, end: float, amount: float) -> float:
    return start + ((end - start) * amount)


class MouseDrawer:
    def __init__(self, config: DrawConfig) -> None:
        self.config = config
        self._stop_requested = False
        self._listener = None
        self._use_quartz_backend = sys.platform == "darwin"
        self._mouse_is_down = False

    def draw(
        self,
        strokes: list[Stroke],
        progress_callback: Callable[[str], None] | None = None,
    ) -> DrawResult:
        if self.config.dry_run:
            self._emit_progress(progress_callback, f"Dry run enabled; skipping mouse movement for {len(strokes)} strokes")
            return DrawResult(stopped=False, strokes_completed=len(strokes))

        runtime_error = mouse_runtime_error(request_permissions=False)
        if runtime_error is not None:
            raise RuntimeError(runtime_error)

        self._configure_backend()
        self._start_listener()
        completed = 0

        try:
            self._countdown(progress_callback)
            for index, stroke in enumerate(strokes, start=1):
                self._check_stop()
                self._draw_stroke(stroke)
                completed = index
                if index < len(strokes):
                    time.sleep(self.config.pause_between_strokes)
                if index % self.config.log_every_n_strokes == 0 or index == len(strokes):
                    self._emit_progress(progress_callback, f"Completed {index}/{len(strokes)} strokes")
            return DrawResult(stopped=False, strokes_completed=len(strokes))
        except EmergencyStop:
            self._emit_progress(progress_callback, "Emergency stop triggered")
            return DrawResult(stopped=True, strokes_completed=completed)
        finally:
            self._stop_listener()
            self._safe_mouse_up()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _draw_stroke(self, stroke: Stroke) -> None:
        if len(stroke) < 2:
            return

        start_x, start_y = stroke[0]
        self._move_to(start_x, start_y)
        self._check_stop()
        self._mouse_down(start_x, start_y)
        try:
            if self.config.pen_down_settle_seconds > 0:
                time.sleep(self.config.pen_down_settle_seconds)
            for segment_index, current in enumerate(stroke[1:]):
                self._check_stop()
                duration = _segment_duration_for_config(stroke, segment_index, self.config)
                self._move_to(current[0], current[1], duration=duration)
        finally:
            self._mouse_up()

    def _countdown(self, progress_callback: Callable[[str], None] | None) -> None:
        for remaining in range(self.config.countdown_seconds, 0, -1):
            self._check_stop()
            self._emit_progress(progress_callback, f"Starting in {remaining}...")
            time.sleep(1)

    def _check_stop(self) -> None:
        if self._use_quartz_backend and self._macos_corner_failsafe_triggered():
            self._stop_requested = True
        if self._stop_requested:
            raise EmergencyStop()

    def _start_listener(self) -> None:
        if self._use_quartz_backend:
            LOGGER.info("macOS emergency ESC listener is disabled for stability; use the terminal to stop the process if needed.")
            self._listener = None
            return

        def on_press(key: object) -> bool | None:
            if key == keyboard.Key.esc:
                self.request_stop()
                return False
            return None

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def _stop_listener(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _configure_backend(self) -> None:
        if self._use_quartz_backend:
            return
        assert pyautogui is not None
        pyautogui.FAILSAFE = self.config.fail_safe
        pyautogui.PAUSE = 0

    def _move_to(self, x: float, y: float, duration: float = 0.0) -> None:
        if self._use_quartz_backend:
            self._quartz_move_to(x, y, duration)
            return
        assert pyautogui is not None
        if self._mouse_is_down:
            self._pyautogui_drag_to(x, y, duration)
            return
        pyautogui.moveTo(x, y, duration=duration)

    def _mouse_down(self, x: float, y: float) -> None:
        if self._use_quartz_backend:
            self._quartz_post_mouse_event(Quartz.kCGEventLeftMouseDown, x, y)
            self._mouse_is_down = True
            return
        assert pyautogui is not None
        pyautogui.mouseDown()
        self._mouse_is_down = True

    def _mouse_up(self) -> None:
        if not self._mouse_is_down:
            return
        if self._use_quartz_backend:
            x, y = self._current_quartz_position()
            self._quartz_post_mouse_event(Quartz.kCGEventLeftMouseUp, x, y)
            self._mouse_is_down = False
            return
        assert pyautogui is not None
        pyautogui.mouseUp()
        self._mouse_is_down = False

    def _safe_mouse_up(self) -> None:
        try:
            self._mouse_up()
        except Exception:
            pass

    def _quartz_move_to(self, x: float, y: float, duration: float) -> None:
        if Quartz is None:
            raise RuntimeError("Quartz is unavailable for macOS mouse control.")

        target_x = float(x)
        target_y = float(y)
        start_x, start_y = self._current_quartz_position()
        distance = math.hypot(target_x - start_x, target_y - start_y)
        event_type = Quartz.kCGEventLeftMouseDragged if self._mouse_is_down else Quartz.kCGEventMouseMoved
        steps = self._movement_step_count(distance, duration)
        if steps <= 1:
            self._check_stop()
            self._quartz_post_mouse_event(event_type, target_x, target_y)
            return

        sleep_time = duration / steps
        for step in range(1, steps + 1):
            self._check_stop()
            t = _ease_drag_progress(step / steps) if self._mouse_is_down else step / steps
            current_x = start_x + (target_x - start_x) * t
            current_y = start_y + (target_y - start_y) * t
            self._quartz_post_mouse_event(event_type, current_x, current_y)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _pyautogui_drag_to(self, x: float, y: float, duration: float) -> None:
        assert pyautogui is not None
        start_position = pyautogui.position()
        start_x = float(start_position.x)
        start_y = float(start_position.y)
        target_x = float(x)
        target_y = float(y)
        distance = math.hypot(target_x - start_x, target_y - start_y)
        steps = self._movement_step_count(distance, duration)
        if steps <= 1:
            pyautogui.moveTo(target_x, target_y, duration=0)
            return

        sleep_time = duration / steps
        for step in range(1, steps + 1):
            self._check_stop()
            t = _ease_drag_progress(step / steps) if self._mouse_is_down else step / steps
            current_x = start_x + (target_x - start_x) * t
            current_y = start_y + (target_y - start_y) * t
            pyautogui.moveTo(current_x, current_y, duration=0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _movement_step_count(self, distance: float, duration: float) -> int:
        if distance <= 1e-6:
            return 1

        if not self._mouse_is_down:
            if duration <= 0:
                return 1
            return max(1, int(math.ceil(duration / 0.01)))

        max_step = max(0.5, self.config.max_drag_step_pixels)
        step_count = int(math.ceil(distance / max_step))
        if duration > 0:
            interval = max(0.0005, self.config.drag_event_interval_seconds)
            step_count = max(step_count, int(math.ceil(duration / interval)))
        return max(MIN_INTERPOLATED_DRAG_STEPS, step_count)

    def _current_quartz_position(self) -> tuple[float, float]:
        if Quartz is None:
            return (0.0, 0.0)
        event = Quartz.CGEventCreate(None)
        location = Quartz.CGEventGetLocation(event)
        return float(location.x), float(location.y)

    def _quartz_post_mouse_event(self, event_type: int, x: float, y: float) -> None:
        if Quartz is None:
            raise RuntimeError("Quartz is unavailable for macOS mouse control.")
        event = Quartz.CGEventCreateMouseEvent(
            None,
            event_type,
            Quartz.CGPointMake(float(x), float(y)),
            Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def _macos_corner_failsafe_triggered(self) -> bool:
        x, y = self._current_quartz_position()
        return x <= MAC_FAILSAFE_CORNER_THRESHOLD and y <= MAC_FAILSAFE_CORNER_THRESHOLD

    def _emit_progress(self, progress_callback: Callable[[str], None] | None, message: str) -> None:
        LOGGER.info(message)
        if progress_callback is not None:
            progress_callback(message)


def mouse_runtime_error(request_permissions: bool = False) -> str | None:
    if sys.platform != "darwin" and keyboard is None:
        return "pynput is not installed. Install it on the target Windows machine."
    if sys.platform != "darwin" and pyautogui is None:
        return "pyautogui is not installed. Install it on the target machine."
    if sys.platform == "darwin" and Quartz is None:
        return "Quartz is unavailable, so macOS mouse control cannot start."
    macos_error = _macos_mouse_permission_error(request_permissions=request_permissions)
    if macos_error is not None:
        return macos_error
    return None


def _macos_mouse_permission_error(request_permissions: bool = False) -> str | None:
    if sys.platform != "darwin":
        return None
    if ApplicationServices is None or Quartz is None:
        return "macOS permission checks are unavailable because Quartz/ApplicationServices could not be imported."

    if request_permissions:
        try:
            ApplicationServices.AXIsProcessTrustedWithOptions(
                {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
            )
        except Exception:
            pass
        try:
            Quartz.CGRequestListenEventAccess()
        except Exception:
            pass
        try:
            Quartz.CGRequestPostEventAccess()
        except Exception:
            pass

    missing: list[str] = []
    try:
        if not bool(ApplicationServices.AXIsProcessTrusted()):
            missing.append("Accessibility")
    except Exception:
        missing.append("Accessibility")

    try:
        if not bool(Quartz.CGPreflightPostEventAccess()):
            missing.append("Post Event Access")
    except Exception:
        missing.append("Post Event Access")

    if not missing:
        return None

    unique_missing = []
    for item in missing:
        if item not in unique_missing:
            unique_missing.append(item)

    return (
        "macOS permissions are missing for live drawing: "
        + ", ".join(unique_missing)
        + ". In System Settings > Privacy & Security, allow the host app and Python in Accessibility. "
        + "If needed, also add /Library/Frameworks/Python.framework/Versions/3.10/Resources/Python.app "
        + "and then restart the app."
    )


def _macos_can_listen_for_events() -> bool:
    if sys.platform != "darwin" or Quartz is None:
        return keyboard is not None
    if keyboard is None:
        return False
    try:
        return bool(Quartz.CGPreflightListenEventAccess())
    except Exception:
        return False
