# AutoDraw Roblox Line Art Tool

External desktop drawing tool that converts black-and-white line art into mouse-driven strokes for a Roblox drawing surface.

## Features

- Load PNG/JPG line art
- Threshold, invert, blur, resize, and skeletonize images
- Convert connected black pixels into ordered strokes
- Simplify and resample paths
- Select the draw region with a snipping-style drag box
- Lock the selected draw region to the current image aspect ratio
- Preview raw image, vectorized paths, and mapped output
- Save and load vector paths as JSON
- Dry-run or draw with mouse automation
- GUI hotkeys: `Ctrl+O` open image, `F6` select draw area, `F5` draw
- Emergency stop hotkey via `ESC`

## Modules

- `image_loader.py`
- `vectorizer.py`
- `simplifier.py`
- `mapper.py`
- `mouse_drawer.py`
- `main.py`
- `ui.py`

## Install

Recommended on Windows:

```bash
python -m pip install opencv-python numpy scikit-image pyautogui pynput Pillow
```

## GUI Usage

Recommended on Windows:

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Launch the GUI:

```bash
python main.py --gui
```

Or double-click `launch_autodraw.bat`.

If you want a one-click dependency install first, run `setup_windows.bat`.

## macOS Notes

Use the same Python app identity each time:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python /Users/kayvenchen/autodraw/main.py --gui
```

Or double-click `launch_autodraw_mac.command`.

To verify what macOS currently grants to this Python:

```bash
/Library/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python /Users/kayvenchen/autodraw/check_macos_permissions.py
```

3. Workflow:

- Click `Open Image`
- Adjust threshold / simplify / spacing if needed
- Click `Select Draw Area (F6)`
- Drag a selection box over the Roblox drawing area
- Click `Draw (F5)` to start
- Press `ESC` at any time to stop the mouse drawing

## CLI Usage

```bash
python main.py --image path/to/image.png --top-left-x 100 --top-left-y 100 --width 500 --height 500 --dry-run
```

Useful options:

- `--threshold 180`
- `--invert`
- `--simplify 1.5`
- `--spacing 2.0`
- `--speed 900`
- `--delay 0.15`
- `--countdown 3`
- `--skeletonize`
- `--min-component-size 16`
- `--max-points-per-stroke 800`
- `--export-json out/paths.json`
- `--import-json out/paths.json`
- `--export-preview out/preview.png`
- `--gui`

## Notes

- This tool only simulates normal mouse input.
- It does not inject into Roblox or modify game memory.
- Test with `--dry-run` first.
- On Windows, disable Roblox camera movement or shortcuts that may interfere with dragging.
- The snipping-style area picker currently targets the primary display.
