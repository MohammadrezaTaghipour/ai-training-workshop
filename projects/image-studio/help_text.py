"""
Help text for Image Studio tools and effects (workshop topics).
"""

from __future__ import annotations

HELP: dict[str, dict[str, str]] = {
    "pointer": {
        "title": "Pointer",
        "summary": "Default navigation tool.",
        "details": (
            "Use the Pointer to work with the canvas without applying an effect.\n\n"
            "It does not change pixels. Select another tool when you want to "
            "transform or filter the image."
        ),
    },
    "hand": {
        "title": "Hand (Pan)",
        "summary": "Move around a zoomed image.",
        "details": (
            "When the image is larger than the window, use the Hand tool mindset "
            "and the scrollbars to pan.\n\n"
            "Does not modify the image — only how you view it."
        ),
    },
    "resize": {
        "title": "Resize",
        "summary": "Change image width and height in pixels.",
        "details": (
            "Resamples the image to a new size using OpenCV interpolation "
            "(area-friendly downscaling).\n\n"
            "Use cases:\n"
            "• Prepare images for display or training\n"
            "• Reduce resolution to speed up processing\n"
            "• Match a required input size\n\n"
            "Note: shrinking loses detail; enlarging cannot invent true detail."
        ),
    },
    "flip_h": {
        "title": "Flip Horizontal",
        "summary": "Mirror the image left ↔ right (around the Y axis).",
        "details": (
            "Equivalent to OpenCV flip code 1.\n\n"
            "Common in data augmentation so a model sees mirrored versions of "
            "the same object without collecting new photos."
        ),
    },
    "flip_v": {
        "title": "Flip Vertical",
        "summary": "Mirror the image top ↔ bottom (around the X axis).",
        "details": (
            "Equivalent to OpenCV flip code 0.\n\n"
            "Useful for augmentation and for correcting upside-down captures."
        ),
    },
    "flip_both": {
        "title": "Flip Both Axes",
        "summary": "Mirror horizontally and vertically (180° reflection).",
        "details": (
            "Equivalent to OpenCV flip code -1.\n\n"
            "Same result as rotating 180° in many cases, but implemented as a flip."
        ),
    },
    "rotate": {
        "title": "Rotate",
        "summary": "Rotate the image around its center by an angle in degrees.",
        "details": (
            "Uses OpenCV getRotationMatrix2D + warpAffine.\n\n"
            "• Positive angles rotate counter-clockwise\n"
            "• “Fit canvas” expands the canvas so corners are not cropped\n"
            "• Without fit, the image size stays the same and corners may clip\n\n"
            "Workshop tip: 90° / 180° / 270° are common; other angles need "
            "fit-canvas for a full view."
        ),
    },
    "mean": {
        "title": "Mean Blur",
        "summary": "Average each pixel with its neighbors (box / mean filter).",
        "details": (
            "Also called a box blur. Every pixel becomes the average of a "
            "K×K neighborhood (Kernel option).\n\n"
            "What it does:\n"
            "• Softens the image and reduces fine detail\n"
            "• Larger kernels = stronger blur\n\n"
            "Workshop uses:\n"
            "• Create softer / “matte” looking images\n"
            "• Simple object / data augmentation\n\n"
            "OpenCV: cv2.blur"
        ),
    },
    "gauss": {
        "title": "Gaussian Blur",
        "summary": "Smooth the image with a Gaussian-weighted kernel.",
        "details": (
            "Neighbors closer to the center pixel weigh more than distant ones, "
            "so the blur looks more natural than a plain mean blur.\n\n"
            "What it does:\n"
            "• Reduces noise and high-frequency detail\n"
            "• Softens edges gently\n"
            "• Kernel size must be odd (3, 5, 7, …)\n\n"
            "Often a first step before edge detection or as augmentation.\n\n"
            "OpenCV: cv2.GaussianBlur"
        ),
    },
    "median": {
        "title": "Median Blur",
        "summary": "Replace each pixel with the median of its neighborhood.",
        "details": (
            "Excellent at removing salt-and-pepper noise — scattered bright "
            "(salt) and dark (pepper) pixels — while keeping edges sharper than "
            "a mean blur.\n\n"
            "What it does:\n"
            "• Sorts neighborhood values and picks the middle one\n"
            "• Outlier noisy pixels are discarded\n\n"
            "Workshop note: preferred cleanup for speckled noise.\n\n"
            "OpenCV: cv2.medianBlur"
        ),
    },
    "bilateral": {
        "title": "Bilateral Filter",
        "summary": "Smooth flat regions while preserving edges.",
        "details": (
            "Blurs similar colors near each pixel but avoids mixing across "
            "strong edges.\n\n"
            "What it does:\n"
            "• Reduces texture / noise in smooth areas\n"
            "• Keeps object boundaries clearer\n"
            "• σ (sigma) controls how strongly colors and space mix\n\n"
            "Workshop uses: edge-aware smoothing; helpful before edge / "
            "object-related steps.\n\n"
            "OpenCV: cv2.bilateralFilter"
        ),
    },
    "sharpen": {
        "title": "Sharpen",
        "summary": "Increase local contrast so edges look clearer.",
        "details": (
            "Applies a 3×3 convolutional kernel:\n\n"
            "    0  -1   0\n"
            "   -1   5  -1\n"
            "    0  -1   0\n\n"
            "The center weight boosts the pixel; neighbors are subtracted, "
            "which exaggerates edges (unsharp-style effect).\n\n"
            "Use for clearer detail; too much sharpening can create halos "
            "or amplify noise.\n\n"
            "OpenCV: cv2.filter2D"
        ),
    },
    "noise": {
        "title": "Gaussian Noise",
        "summary": "Add random Gaussian noise to every pixel.",
        "details": (
            "Samples noise from a normal distribution (mean 0, σ = Noise σ) "
            "and adds it to the image.\n\n"
            "What it does:\n"
            "• Simulates sensor / capture noise\n"
            "• Higher σ = stronger grain\n\n"
            "Workshop uses: data augmentation so models tolerate imperfect "
            "photos; also to test denoise filters (e.g. Median).\n\n"
            "OpenCV helper: add noise array to the image"
        ),
    },
    "random": {
        "title": "Random Pipeline",
        "summary": "Apply a random sequence of workshop filters.",
        "details": (
            "Randomly may apply blur, Gaussian blur, sharpen, median, flip, "
            "rotation, and/or noise — similar to the session “pipeline” exercise.\n\n"
            "What it does:\n"
            "• Builds varied augmented versions of one image\n"
            "• Each run can produce a different combination\n"
            "• Steps are listed in History\n\n"
            "Great for demos and for generating diverse training samples."
        ),
    },
    "compare": {
        "title": "Compare Original",
        "summary": "Toggle canvas between original and current image.",
        "details": (
            "Does not change pixels. Switches the view so you can judge "
            "before vs after an effect.\n\n"
            "Also available from View ▸ Compare Original and the toolbar."
        ),
    },
    "faces": {
        "title": "Detect Faces",
        "summary": "Find faces with OpenCV Haar cascades and draw boxes.",
        "details": (
            "Uses haarcascade_frontalface_default.xml (classic CV, no GPU).\n\n"
            "Works best on frontal faces with reasonable lighting.\n"
            "From session5 webcam exercises — here applied to a still image.\n\n"
            "Logic lives in core.vision_classic.faces (UI-free)."
        ),
    },
    "faces_eyes": {
        "title": "Detect Faces + Eyes",
        "summary": "Draw face boxes (green) and eye boxes (blue).",
        "details": (
            "Runs frontal-face and eye Haar cascades, then annotates the image.\n\n"
            "Eye detections can false-positive on textured regions; use as a "
            "learning demo, not production biometrics.\n\n"
            "OpenCV: CascadeClassifier + detectMultiScale"
        ),
    },
}


def get_help(key: str) -> dict[str, str]:
    return HELP.get(
        key,
        {
            "title": key,
            "summary": "No help text yet.",
            "details": "Help for this action will be added in a later session.",
        },
    )
