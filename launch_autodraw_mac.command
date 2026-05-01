#!/bin/bash
set -e
cd "$(dirname "$0")"
export VIRTUAL_ENV="$PWD/.venv"
export PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages:$PWD"
/Library/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python "$PWD/main.py" --gui
