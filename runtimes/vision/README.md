# Isolated Ultralytics / Torch runtime for the AI Learning Lab.
#
# Why separate?
#   Main Lab uses OpenCV 4.x. Ultralytics currently pulls OpenCV 5.x.
#   Keep this environment separate to avoid dependency conflicts.
#
# Setup (from this directory):
#   uv sync
#
# Examples:
#   uv run ai-lab-vision info
#   uv run ai-lab-vision detect-image path\to\image.jpg -o output\out.jpg -m yolo12n.pt
#   uv run ai-lab-vision detect-video path\to\video.mp4 -o output\out.mp4 --max-frames 100
#
# Local weights: models/*.pt (also mirrored under sessions/session6/models/)
#
# Course Session 6 scratchpad:
#   uv run --project runtimes/vision python sessions/session6/main.py
#   (from repo root; uncomment regions inside that file)
#
# Desktop UI: projects/detection-studio
# Web: only add a Streamlit adapter when explicitly requested; call this same CLI.
