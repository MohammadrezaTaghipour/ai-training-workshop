"""Haar-cascade face / eye detection (OpenCV). UI-free."""

from __future__ import annotations

import cv2
import numpy as np

from core.image.filters import FilterResult, ensure_bgr

Box = tuple[int, int, int, int]  # x, y, w, h


def _cascade(filename: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + filename
    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load cascade: {filename}")
    return classifier


def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> list[Box]:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    faces = _cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        gray, scale_factor, min_neighbors
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def detect_eyes(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> list[Box]:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    eyes = _cascade("haarcascade_eye.xml").detectMultiScale(
        gray, scale_factor, min_neighbors
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in eyes]


def draw_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> FilterResult:
    """Detect faces and draw boxes; returns annotated BGR image."""
    result = ensure_bgr(image)
    faces = detect_faces(result, scale_factor, min_neighbors)
    for x, y, w, h in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            result,
            "face",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    return result, f"Face detect ({len(faces)})"


def draw_faces_and_eyes(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> FilterResult:
    """Detect faces and eyes; draw green faces and blue eyes."""
    result = ensure_bgr(image)
    faces = detect_faces(result, scale_factor, min_neighbors)
    eyes = detect_eyes(result, scale_factor, min_neighbors)
    for x, y, w, h in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in eyes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            result,
            "eye",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return result, f"Faces {len(faces)}, eyes {len(eyes)}"
