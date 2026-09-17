# Detection Studio

Desktop launcher for YOLO **image and video** detection (Session 6 capabilities).

Runs in the **main Lab** env (Tkinter + OpenCV 4) and shells out to the
isolated **vision** runtime (`runtimes/vision`) so Ultralytics / OpenCV 5
never mix into the main environment.

## Features

| Action | Behavior |
|--------|----------|
| Open Image / Open Video | Load media; video shows first-frame preview |
| Session 6 sample… | Quick pick from `sessions/session6/images/` |
| Run Detection | Image → annotated still; Video → annotated MP4 (offline) |
| Max frames | Limit video length (default 90; `0` = entire file) |
| Play Result | Open annotated output in the OS default player |
| Save Result… | Copy output to a chosen path |

## Setup

```bash
uv sync
cd runtimes/vision
uv sync
```

## Run

```bash
uv run python projects/detection-studio/main.py
```

Video tip: start with a short clip or **Max frames = 60–120**. Full HD videos
on CPU can take a long time if Max frames is 0.
