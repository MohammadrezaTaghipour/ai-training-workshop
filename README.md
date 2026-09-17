# AI Learning Lab

Personal lab for the AI Training Workshop course: keep session experiments,
promote reusable logic into a shared `core`, and run desktop apps by default.

**UI policy:** Tkinter desktop first. Streamlit/web only when a feature is
explicitly requested — adapters go under `lab/adapters/` and must call `core`
(or the vision CLI), never reimplement algorithms.

## Layout

```
sessions/                 Course journal (raw exercises — keep as learning record)
core/                     UI-free shared logic (filters, classic vision, paths)
projects/
  image-studio/           Desktop image filters + Haar face/eye detect
  detection-studio/       Desktop YOLO UI → shells out to vision runtime
runtimes/
  vision/                 Isolated Ultralytics env (OpenCV 5 / Torch)
lab/adapters/             Optional web adapters (empty until you ask for web)
```

## Setup (main Lab — OpenCV 4)

```bash
uv sync
```

Optional Streamlit deps (not required for desktop):

```bash
uv sync --group web
```

## Run desktop apps

```bash
uv run python projects/image-studio/main.py
uv run python projects/detection-studio/main.py
```

## Vision runtime (YOLO — separate env)

Ultralytics pulls OpenCV 5.x; the main Lab stays on OpenCV 4.x. Use a second environment:

```bash
cd runtimes/vision
uv sync
uv run ai-lab-vision info
uv run ai-lab-vision detect-image path\to\image.jpg -o output\out.jpg
```

**Session 6** course code lives in `sessions/session6/`. Always run it with the
vision runtime — see `sessions/session6/README.md`.

Detection Studio calls this worker via the vision `.venv` Python.
Weights are under `sessions/session6/models/` and `runtimes/vision/models/`.

## Adding a new course capability

1. Experiment in `sessions/<name>/` (any style).
2. When worth reusing, move pure functions into `core/…`.
3. Wire a **Tkinter** UI under `projects/` (default).
4. If deps conflict (e.g. new ML stack), add `runtimes/<capability>/`.
5. Add `lab/adapters/…` **only** when you explicitly want a web face.

## Projects

| Project | Stack | Description |
|---------|-------|-------------|
| [Image Studio](projects/image-studio) | Tkinter · OpenCV · `core` | Filters, transforms, Haar face/eye detection |
| [Detection Studio](projects/detection-studio) | Tkinter · vision CLI | YOLO image **and video** detection via isolated runtime |
