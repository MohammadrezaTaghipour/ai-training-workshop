"""Desktop YOLO launcher — image + video via the isolated vision runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

_ROOT = Path(__file__).resolve().parents[2]
_VISION = _ROOT / "runtimes" / "vision"
_SESSION6_IMAGES = _ROOT / "sessions" / "session6" / "images"

_BUILTIN_MODELS = (
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _vision_python() -> Path | None:
    if os.name == "nt":
        candidate = _VISION / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = _VISION / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _discover_models() -> list[str]:
    found: dict[str, Path] = {}
    for folder in (
        _VISION / "models",
        _VISION,
        _ROOT / "sessions" / "session6" / "models",
    ):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.pt")):
            found.setdefault(path.name, path)
    names = list(found.keys())
    for name in _BUILTIN_MODELS:
        if name not in found:
            names.append(name)
    return names or list(_BUILTIN_MODELS)


def _resolve_model(name: str) -> str:
    name = name.strip()
    if not name:
        return "yolo12n.pt"
    as_path = Path(name)
    if as_path.is_file():
        return str(as_path.resolve())
    for folder in (
        _VISION / "models",
        _VISION,
        _ROOT / "sessions" / "session6" / "models",
    ):
        candidate = folder / name
        if candidate.is_file():
            return str(candidate.resolve())
    return name


def _video_first_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0.0
    ok, frame = cap.read()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return (frame if ok else None), frame_count, fps


class DetectionStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Detection Studio — AI Learning Lab")
        self.geometry("1000x680")
        self.minsize(780, 520)

        self._source: Path | None = None
        self._result: Path | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._busy = False
        self._media_kind = "image"  # image | video
        self._video_frames = 0

        models = _discover_models()
        default_model = "yolo12n.pt" if "yolo12n.pt" in models else models[0]

        self.model_var = tk.StringVar(value=default_model)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.max_frames_var = tk.IntVar(value=90)
        self.status_var = tk.StringVar(
            value="Open an image or video, then Run Detection."
        )

        self._build(models)

    def _build(self, models: list[str]) -> None:
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill=tk.X)

        ttk.Button(bar, text="Open Image…", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open Video…", command=self.open_video).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        if _SESSION6_IMAGES.is_dir():
            ttk.Button(
                bar, text="Session 6 sample…", command=self.open_session6_sample
            ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Run Detection", command=self.run_detection).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(bar, text="Save Result…", command=self.save_result).pack(side=tk.LEFT)
        ttk.Button(bar, text="Play Result", command=self.play_result).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        opts = ttk.Frame(self, padding=(8, 0))
        opts.pack(fill=tk.X)
        ttk.Label(opts, text="Model").pack(side=tk.LEFT)
        ttk.Combobox(
            opts,
            textvariable=self.model_var,
            values=models,
            state="readonly",
            width=16,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(opts, text="Conf").pack(side=tk.LEFT)
        ttk.Spinbox(
            opts, from_=0.05, to=0.95, increment=0.05, textvariable=self.conf_var, width=5
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(opts, text="Max frames").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Spinbox(
            opts, from_=0, to=10000, increment=30, textvariable=self.max_frames_var, width=6
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(opts, text="(0 = entire video)", foreground="#666").pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg="#606060", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(self, textvariable=self.status_var, anchor=tk.W).pack(
            fill=tk.X, padx=8, pady=(0, 8)
        )

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.jfif"), ("All", "*.*")],
        )
        if path:
            self._load_source(Path(path))

    def open_video(self) -> None:
        initial = str(_SESSION6_IMAGES) if _SESSION6_IMAGES.is_dir() else None
        path = filedialog.askopenfilename(
            title="Open video",
            initialdir=initial,
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")],
        )
        if path:
            self._load_source(Path(path))

    def open_session6_sample(self) -> None:
        if not _SESSION6_IMAGES.is_dir():
            messagebox.showinfo("Samples", "sessions/session6/images not found.")
            return
        path = filedialog.askopenfilename(
            title="Session 6 sample",
            initialdir=str(_SESSION6_IMAGES),
            filetypes=[
                ("Media", "*.jpg *.jpeg *.png *.mp4 *.jfif"),
                ("All", "*.*"),
            ],
        )
        if path:
            self._load_source(Path(path))

    def _load_source(self, path: Path) -> None:
        self._source = path
        self._result = None
        ext = path.suffix.lower()
        if ext in _VIDEO_EXTS:
            self._media_kind = "video"
            frame, count, fps = _video_first_frame(path)
            self._video_frames = count
            self._show_bgr(frame)
            tip = f"Loaded video {path.name} ({count} frames @ {fps:.1f} fps)"
            if count > 90:
                tip += " — consider Max frames for a quicker demo"
            self.status_var.set(tip)
        elif ext in _IMAGE_EXTS:
            self._media_kind = "image"
            self._video_frames = 0
            self._show_bgr(cv2.imread(str(path)))
            self.status_var.set(f"Loaded image {path.name}")
        else:
            messagebox.showwarning("Open", f"Unsupported file type: {ext}")
            self._source = None

    def save_result(self) -> None:
        if self._result is None or not self._result.is_file():
            messagebox.showinfo("Save", "Run detection first.")
            return
        if self._media_kind == "video":
            dest = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("All", "*.*")],
            )
        else:
            dest = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            )
        if dest:
            Path(dest).write_bytes(self._result.read_bytes())
            self.status_var.set(f"Saved → {dest}")

    def play_result(self) -> None:
        if self._result is None or not self._result.is_file():
            messagebox.showinfo("Play", "Run detection first.")
            return
        try:
            os.startfile(self._result)  # type: ignore[attr-defined]
        except AttributeError:
            messagebox.showinfo("Play", f"Open this file manually:\n{self._result}")

    def run_detection(self) -> None:
        if self._busy:
            return
        if self._source is None:
            messagebox.showwarning("Detection", "Open an image or video first.")
            return
        if not (_VISION / "pyproject.toml").is_file():
            messagebox.showerror("Vision runtime", f"Missing {_VISION}")
            return

        source = self._source
        kind = self._media_kind
        model = _resolve_model(self.model_var.get())
        try:
            conf = float(self.conf_var.get())
        except (TypeError, ValueError, tk.TclError):
            conf = 0.25
        try:
            max_frames = int(self.max_frames_var.get())
        except (TypeError, ValueError, tk.TclError):
            max_frames = 90

        venv_py = _vision_python()
        if venv_py is None:
            messagebox.showerror(
                "Vision runtime",
                "Vision venv not found.\n\nRun once:\n  cd runtimes/vision\n  uv sync",
            )
            return

        self._busy = True
        if kind == "video":
            limit = f", max {max_frames} frames" if max_frames else ", entire video"
            self.status_var.set(
                f"Processing video{limit}… (CPU YOLO can take a while)"
            )
        else:
            self.status_var.set("Running detection…")
        self.update_idletasks()
        threading.Thread(
            target=self._worker_thread,
            args=(source, kind, model, conf, max_frames, venv_py),
            daemon=True,
        ).start()

    def _worker_thread(
        self,
        source: Path,
        kind: str,
        model: str,
        conf: float,
        max_frames: int,
        venv_py: Path,
    ) -> None:
        tmp = Path(tempfile.gettempdir())
        if kind == "video":
            out = tmp / "ai-lab-detect-out.mp4"
            cmd = [
                str(venv_py),
                "-m",
                "worker",
                "detect-video",
                str(source),
                "-o",
                str(out),
                "-m",
                model,
                "--conf",
                str(conf),
                "--max-frames",
                str(max_frames),
            ]
        else:
            out = tmp / "ai-lab-detect-out.jpg"
            cmd = [
                str(venv_py),
                "-m",
                "worker",
                "detect-image",
                str(source),
                "-o",
                str(out),
                "-m",
                model,
                "--conf",
                str(conf),
            ]

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(_VISION),
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
        except OSError as exc:
            self.after(0, lambda: self._on_fail(str(exc)))
            return

        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "unknown error").strip()
            self.after(0, lambda e=err: self._on_fail(e))
            return

        if not out.is_file():
            self.after(
                0,
                lambda: self._on_fail(
                    "Worker finished but output was not created.\n"
                    + (completed.stdout or "")
                    + "\n"
                    + (completed.stderr or "")
                ),
            )
            return

        message = (completed.stdout or "").strip() or "Done."
        preview = None
        if kind == "video":
            preview, _, _ = _video_first_frame(out)
        else:
            preview = cv2.imread(str(out))
        self.after(0, lambda: self._on_ok(preview, out, kind, message))

    def _on_ok(self, preview, out: Path, kind: str, message: str) -> None:
        self._busy = False
        self._result = out
        self._media_kind = kind
        self._show_bgr(preview)
        last = message.splitlines()[-1] if message else "Done."
        if kind == "video":
            last += "  — use Play Result or Save Result…"
        self.status_var.set(last)

    def _on_fail(self, message: str) -> None:
        self._busy = False
        self.status_var.set("Detection failed.")
        text = message.strip()
        if len(text) > 1200:
            text = text[:1200] + "\n…"
        messagebox.showerror(
            "Vision worker failed",
            text
            + "\n\nSetup:\n  cd runtimes/vision\n  uv sync\n\n"
            "For long videos, set Max frames (e.g. 60–120) first.",
        )

    def _show_bgr(self, image) -> None:
        self.canvas.delete("all")
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.update_idletasks()
        max_w = max(self.canvas.winfo_width(), 400)
        max_h = max(self.canvas.winfo_height(), 300)
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            rgb = cv2.resize(
                rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)


def main() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    app = DetectionStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
