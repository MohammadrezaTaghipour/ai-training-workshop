"""
Image processing operations covered in AI Workshop sessions.

Topics so far:
  - Resize
  - Mean / Gaussian / Median / Bilateral blur
  - Sharpen (convolution kernel)
  - Flip (horizontal, vertical, both)
  - Rotate (keep size / fit canvas)
  - Gaussian noise
  - Random filter pipeline
"""

from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np


FilterResult = tuple[np.ndarray, str]


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a BGR image copy suitable for OpenCV filters."""
    if image is None:
        raise ValueError("Image is empty.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def resize(image: np.ndarray, width: int, height: int) -> FilterResult:
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return result, f"Resize {width}x{height}"


def mean_blur(image: np.ndarray, kernel_size: int = 5) -> FilterResult:
    k = _odd(kernel_size)
    result = cv2.blur(image, (k, k))
    return result, f"Mean Blur {k}x{k}"


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> FilterResult:
    k = _odd(kernel_size)
    result = cv2.GaussianBlur(image, (k, k), 0)
    return result, f"Gaussian Blur {k}x{k}"


def median_blur(image: np.ndarray, kernel_size: int = 5) -> FilterResult:
    k = _odd(kernel_size)
    result = cv2.medianBlur(image, k)
    return result, f"Median Blur {k}x{k}"


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> FilterResult:
    result = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return result, f"Bilateral d={diameter}, σc={sigma_color:.0f}, σs={sigma_space:.0f}"


def sharpen(image: np.ndarray) -> FilterResult:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    result = cv2.filter2D(image, -1, kernel)
    return result, "Sharpen"


def flip(image: np.ndarray, mode: str = "horizontal") -> FilterResult:
    codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if mode not in codes:
        raise ValueError(f"Unknown flip mode: {mode}")
    result = cv2.flip(image, codes[mode])
    return result, f"Flip {mode.title()}"


def rotate(
    image: np.ndarray,
    angle: float = 45.0,
    fit: bool = True,
    scale: float = 1.0,
) -> FilterResult:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    if fit:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        result = cv2.warpAffine(image, matrix, (new_w, new_h))
        label = f"Rotate {angle:.0f}° (fit)"
    else:
        result = cv2.warpAffine(image, matrix, (w, h))
        label = f"Rotate {angle:.0f}°"

    return result, label


def add_gaussian_noise(image: np.ndarray, sigma: float = 15.0) -> FilterResult:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, f"Gaussian Noise σ={sigma:.0f}"


def apply_random_pipeline(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of workshop filters (session Part 8)."""
    result = image.copy()
    applied: list[str] = []

    steps: list[Callable[[np.ndarray], FilterResult | None]] = [
        lambda img: mean_blur(img, random.choice([3, 5, 7, 9])) if random.random() > 0.5 else None,
        lambda img: gaussian_blur(img, random.choice([3, 5, 7])) if random.random() > 0.5 else None,
        lambda img: sharpen(img) if random.random() > 0.7 else None,
        lambda img: median_blur(img, random.choice([3, 5])) if random.random() > 0.5 else None,
        lambda img: flip(img, random.choice(["horizontal", "vertical", "both"]))
        if random.random() > 0.5
        else None,
        lambda img: rotate(img, random.choice([45, 90, 180, 270]), fit=False)
        if random.random() > 0.3
        else None,
        lambda img: add_gaussian_noise(img, float(random.choice([5, 10, 15])))
        if random.random() > 0.8
        else None,
    ]

    for step in steps:
        outcome = step(result)
        if outcome is None:
            continue
        result, label = outcome
        applied.append(label)

    if not applied:
        result, label = gaussian_blur(result, 5)
        applied.append(label)

    return result, applied


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
