# Image Studio

Desktop image-processing app for the **AI Learning Lab**, built with **Python**, **Tkinter**, and **OpenCV**.

Filter and vision logic live in the shared UI-free package `core` (so a web
adapter can reuse the same code later). This app remains the default UI.

## App chrome (Paint.NET–inspired)

- **Menu bar** — File, Edit, View, Image, Effects, Window, Help  
- **Toolbar** — Open / Save, Undo / Redo / Reset, Zoom, Compare  
- **Tools** (F5) — filter & transform tool palette + options  
- **History** (F6) — applied steps with Undo / Redo  
- **Layers** (F7) — background layer (ready to grow later)  
- **Colors** (F8) — primary/secondary swatches  
- **Canvas** — gray workspace, zoom, scrollbars  

## Features (from workshop sessions)

| Area | Operations |
|------|------------|
| Geometry | Resize, Flip (H / V / Both), Rotate (angle + fit canvas) |
| Blur / Smooth | Mean, Gaussian, Median, Bilateral |
| Detail | Sharpen (3×3 convolution kernel) |
| Augmentation | Gaussian noise, Random filter pipeline |
| Classic Vision | Haar face detect, faces + eyes |
| Workflow | Before / After preview, Undo, Reset, Save |

## Setup

From the repository root (preferred):

```bash
uv sync
```

## Run

```bash
uv run python projects/image-studio/main.py
```

Or from this folder after root `uv sync`:

```bash
py main.py
```

Shortcuts: `Ctrl+O` open · `Ctrl+S` save · `Ctrl+Z` undo · `Ctrl+R` reset

## Project layout

```
image-studio/
├── main.py           # entry point (adds repo root to sys.path)
├── app.py            # Tkinter UI
├── filters.py        # shim → core.image.filters
├── help_text.py
└── images/           # optional sample images
```

Shared logic: `core/image/filters.py`, `core/vision_classic/faces.py`

## Extending in later sessions

1. Add a function in `core/…` that returns `(image, label)`.
2. Wire a button in `app.py` that calls `_commit(result, label)`.
3. Only if asked for web: add a thin adapter under `lab/adapters/` importing the same `core` function.
