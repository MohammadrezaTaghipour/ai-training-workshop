"""Classic computer vision (Haar cascades, etc.) — no deep-learning deps."""

from core.vision_classic.faces import (
    detect_eyes,
    detect_faces,
    draw_faces,
    draw_faces_and_eyes,
)

__all__ = [
    "detect_eyes",
    "detect_faces",
    "draw_faces",
    "draw_faces_and_eyes",
]
