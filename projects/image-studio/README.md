# Image Studio

Desktop image-processing app for the **AI Training Workshop**, built with **Python**, **Tkinter**, and **OpenCV**.

Implements the operations covered in early sessions and is structured so new filters can be added as the course continues.

## Features (from workshop sessions)

| Area | Operations |
|------|------------|
| Geometry | Resize, Flip (H / V / Both), Rotate (angle + fit canvas) |
| Blur / Smooth | Mean, Gaussian, Median, Bilateral |
| Detail | Sharpen (3×3 convolution kernel) |
| Augmentation | Gaussian noise, Random filter pipeline |
| Workflow | Before / After preview, Undo, Reset, Save |

## Setup

```bash
cd projects/image-studio
py -m pip install -r requirements.txt
```

## Run

```bash
cd projects/image-studio
py main.py
```

Shortcuts: `Ctrl+O` open · `Ctrl+S` save · `Ctrl+Z` undo · `Ctrl+R` reset

## Project layout

```
image-studio/
├── main.py           # entry point
├── app.py            # Tkinter UI
├── filters.py        # OpenCV operations (easy to extend)
├── requirements.txt
└── images/           # optional sample images
```

## Extending in later sessions

1. Add a new function in `filters.py` that returns `(image, label)`.
2. Wire a button (or slider) in `app.py` that calls `_commit(result, label)`.

That keeps UI and image math separated as the app grows.
