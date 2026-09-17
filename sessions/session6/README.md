# Session 6 — YOLO / Ultralytics

Course scratchpad for YOLO / Ultralytics (formerly a separate workshop repo).

## Compatibility (important)

| Environment | OpenCV | Use for Session 6? |
|-------------|--------|--------------------|
| Main Lab (repo root `.venv`) | **4.13** | **No** — Ultralytics conflicts |
| `runtimes/vision` | **5.x** + Torch + Ultralytics | **Yes** |

Do **not** `uv add ultralytics` to the root project.

## Layout

```
session6/
  main.py          # commented course exercises (unchanged paths)
  YOLO_o1.py       # notes / links
  images/          # sample images & videos (was sibling images/)
  models/          # local *.pt weights
  output/          # optional save targets
```

## Run

```bash
# from repo root — uses vision runtime only
uv run --project runtimes/vision python sessions/session6/main.py
```

Or:

```bash
cd sessions/session6
..\..\runtimes\vision\.venv\Scripts\python.exe main.py
```

Uncomment the region you want to try inside `main.py`. Paths like `images/...` and `models/...` are relative to this folder.

## Lab apps

- Desktop UI: `uv run python projects/detection-studio/main.py`
- CLI: `uv run --project runtimes/vision ai-lab-vision detect-image ...`
