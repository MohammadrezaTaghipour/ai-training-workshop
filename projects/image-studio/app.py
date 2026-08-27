"""
AI Workshop — Image Studio
Desktop image processing app (Python + Tkinter + OpenCV).

Extensible base for filters learned in class; new session features
can be plugged into the sidebar without rewriting the canvas logic.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import filters as fx


APP_TITLE = "AI Workshop — Image Studio"
MAX_PREVIEW = 520
HISTORY_LIMIT = 30


class ImageStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x720")
        self.minsize(960, 600)
        self.configure(bg="#1e1f24")

        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.history: list[tuple[np.ndarray, str]] = []
        self._photo_before: ImageTk.PhotoImage | None = None
        self._photo_after: ImageTk.PhotoImage | None = None

        self._build_style()
        self._build_layout()
        self._bind_shortcuts()
        self._set_status("Open an image to get started.")

    # ── UI construction ──────────────────────────────────────────────

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Root.TFrame", background="#1e1f24")
        style.configure("Panel.TFrame", background="#2a2c33")
        style.configure("Canvas.TFrame", background="#15161a")
        style.configure(
            "Title.TLabel",
            background="#2a2c33",
            foreground="#f2f3f5",
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "Muted.TLabel",
            background="#2a2c33",
            foreground="#a8adb8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Section.TLabel",
            background="#2a2c33",
            foreground="#8ab4f8",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "CanvasTitle.TLabel",
            background="#15161a",
            foreground="#c5cad3",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#15161a",
            foreground="#9aa0a6",
            font=("Segoe UI", 9),
            padding=8,
        )
        style.configure("Accent.TButton", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=4)
        style.configure(
            "TLabelframe",
            background="#2a2c33",
            foreground="#f2f3f5",
        )
        style.configure(
            "TLabelframe.Label",
            background="#2a2c33",
            foreground="#8ab4f8",
            font=("Segoe UI Semibold", 9),
        )
        style.configure("TScale", background="#2a2c33")
        style.configure("TCheckbutton", background="#2a2c33", foreground="#e8eaed")
        style.configure("TRadiobutton", background="#2a2c33", foreground="#e8eaed")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(root, style="Panel.TFrame", padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="Session tools: blur · flip · rotate · sharpen · noise · pipeline",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(16, 0))

        body = ttk.Frame(root, style="Root.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sidebar
        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=12, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Preview area
        preview = ttk.Frame(body, style="Canvas.TFrame", padding=12)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        canvases = ttk.Frame(preview, style="Canvas.TFrame")
        canvases.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(canvases, style="Canvas.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        ttk.Label(left, text="Before (original)", style="CanvasTitle.TLabel").pack(anchor=tk.W)
        self.before_label = tk.Label(left, bg="#0f1013", fg="#6b7280", text="No image")
        self.before_label.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        right = ttk.Frame(canvases, style="Canvas.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        ttk.Label(right, text="After (processed)", style="CanvasTitle.TLabel").pack(anchor=tk.W)
        self.after_label = tk.Label(right, bg="#0f1013", fg="#6b7280", text="No image")
        self.after_label.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # History / status
        bottom = ttk.Frame(preview, style="Canvas.TFrame")
        bottom.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(bottom, text="Applied steps", style="CanvasTitle.TLabel").pack(anchor=tk.W)
        self.history_box = tk.Listbox(
            bottom,
            height=4,
            bg="#0f1013",
            fg="#d1d5db",
            selectbackground="#3b82f6",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9),
        )
        self.history_box.pack(fill=tk.X, pady=(4, 0))

        self.status = ttk.Label(root, text="", style="Status.TLabel", anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="File", style="Section.TLabel").pack(anchor=tk.W)
        file_row = ttk.Frame(parent, style="Panel.TFrame")
        file_row.pack(fill=tk.X, pady=(4, 10))
        ttk.Button(file_row, text="Open…", command=self.open_image).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(file_row, text="Save…", command=self.save_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0)
        )

        edit_row = ttk.Frame(parent, style="Panel.TFrame")
        edit_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(edit_row, text="Undo", command=self.undo).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(edit_row, text="Reset", command=self.reset_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0)
        )

        # Resize
        resize_box = ttk.LabelFrame(parent, text="Resize", padding=8)
        resize_box.pack(fill=tk.X, pady=4)
        self.width_var = tk.IntVar(value=400)
        self.height_var = tk.IntVar(value=250)
        self._spin(resize_box, "Width", self.width_var, 16, 4000)
        self._spin(resize_box, "Height", self.height_var, 16, 4000)
        ttk.Button(resize_box, text="Apply Resize", command=self.apply_resize).pack(fill=tk.X, pady=(6, 0))

        # Blur
        blur_box = ttk.LabelFrame(parent, text="Blur / Smooth", padding=8)
        blur_box.pack(fill=tk.X, pady=4)
        self.kernel_var = tk.IntVar(value=5)
        self._scale(blur_box, "Kernel (odd)", self.kernel_var, 1, 21)
        ttk.Button(blur_box, text="Mean Blur", command=self.apply_mean_blur).pack(fill=tk.X, pady=2)
        ttk.Button(blur_box, text="Gaussian Blur", command=self.apply_gaussian).pack(fill=tk.X, pady=2)
        ttk.Button(blur_box, text="Median Blur", command=self.apply_median).pack(fill=tk.X, pady=2)

        bi_row = ttk.Frame(blur_box, style="Panel.TFrame")
        bi_row.pack(fill=tk.X, pady=(4, 0))
        self.sigma_var = tk.DoubleVar(value=75)
        self._scale(blur_box, "Bilateral σ", self.sigma_var, 10, 200)
        ttk.Button(blur_box, text="Bilateral Filter", command=self.apply_bilateral).pack(fill=tk.X, pady=2)

        # Sharpen
        sharp_box = ttk.LabelFrame(parent, text="Sharpen", padding=8)
        sharp_box.pack(fill=tk.X, pady=4)
        ttk.Button(sharp_box, text="Sharpen Kernel", command=self.apply_sharpen).pack(fill=tk.X)

        # Flip
        flip_box = ttk.LabelFrame(parent, text="Flip", padding=8)
        flip_box.pack(fill=tk.X, pady=4)
        for mode, label in (
            ("horizontal", "Horizontal (Y)"),
            ("vertical", "Vertical (X)"),
            ("both", "Both axes"),
        ):
            ttk.Button(
                flip_box,
                text=label,
                command=lambda m=mode: self.apply_flip(m),
            ).pack(fill=tk.X, pady=2)

        # Rotate
        rot_box = ttk.LabelFrame(parent, text="Rotate", padding=8)
        rot_box.pack(fill=tk.X, pady=4)
        self.angle_var = tk.DoubleVar(value=45)
        self.fit_var = tk.BooleanVar(value=True)
        self._scale(rot_box, "Angle °", self.angle_var, -180, 180)
        ttk.Checkbutton(rot_box, text="Fit canvas (no crop)", variable=self.fit_var).pack(anchor=tk.W)
        quick = ttk.Frame(rot_box, style="Panel.TFrame")
        quick.pack(fill=tk.X, pady=4)
        for angle in (90, -90, 180):
            ttk.Button(
                quick,
                text=f"{angle}°",
                width=6,
                command=lambda a=angle: self.apply_rotate(a),
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(rot_box, text="Apply Angle", command=lambda: self.apply_rotate()).pack(fill=tk.X)

        # Noise + pipeline
        misc = ttk.LabelFrame(parent, text="Noise & Pipeline", padding=8)
        misc.pack(fill=tk.X, pady=4)
        self.noise_var = tk.DoubleVar(value=15)
        self._scale(misc, "Noise σ", self.noise_var, 1, 50)
        ttk.Button(misc, text="Add Gaussian Noise", command=self.apply_noise).pack(fill=tk.X, pady=2)
        ttk.Button(misc, text="Random Pipeline", command=self.apply_random).pack(fill=tk.X, pady=2)

    def _spin(self, parent: ttk.Frame, label: str, variable: tk.Variable, from_: int, to: int) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel", width=8).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=from_, to=to, textvariable=variable, width=8).pack(side=tk.RIGHT)

    def _scale(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Scale(row, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL).pack(fill=tk.X)

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self.open_image())
        self.bind("<Control-s>", lambda _e: self.save_image())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-r>", lambda _e: self.reset_image())

    # ── File ops ─────────────────────────────────────────────────────

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Open failed", f"Could not read:\n{path}")
            return

        self.original = image
        self.current = image.copy()
        self.history.clear()
        self.history_box.delete(0, tk.END)
        h, w = image.shape[:2]
        self.width_var.set(w)
        self.height_var.set(h)
        self._refresh_previews()
        self._set_status(f"Loaded {Path(path).name}  ({w}×{h})")

    def save_image(self) -> None:
        if self.current is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        ok = cv2.imwrite(path, self.current)
        if not ok:
            messagebox.showerror("Save failed", f"Could not write:\n{path}")
            return
        self._set_status(f"Saved → {Path(path).name}")

    # ── Edit helpers ─────────────────────────────────────────────────

    def _require_image(self) -> bool:
        if self.current is None:
            messagebox.showinfo("No image", "Open an image first.")
            return False
        return True

    def _commit(self, result: np.ndarray, label: str) -> None:
        assert self.current is not None
        self.history.append((self.current.copy(), label))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
        self.current = result
        self.history_box.insert(tk.END, f"{self.history_box.size() + 1}. {label}")
        self.history_box.see(tk.END)
        self._refresh_previews()
        self._set_status(f"Applied: {label}")

    def undo(self) -> None:
        if not self.history:
            self._set_status("Nothing to undo.")
            return
        previous, label = self.history.pop()
        self.current = previous
        if self.history_box.size():
            self.history_box.delete(tk.END)
        self._refresh_previews()
        self._set_status(f"Undid: {label}")

    def reset_image(self) -> None:
        if self.original is None:
            return
        self.current = self.original.copy()
        self.history.clear()
        self.history_box.delete(0, tk.END)
        self._refresh_previews()
        self._set_status("Reset to original.")

    # ── Filter actions ───────────────────────────────────────────────

    def apply_resize(self) -> None:
        if not self._require_image():
            return
        result, label = fx.resize(self.current, self.width_var.get(), self.height_var.get())
        self._commit(result, label)

    def apply_mean_blur(self) -> None:
        if not self._require_image():
            return
        result, label = fx.mean_blur(self.current, self.kernel_var.get())
        self._commit(result, label)

    def apply_gaussian(self) -> None:
        if not self._require_image():
            return
        result, label = fx.gaussian_blur(self.current, self.kernel_var.get())
        self._commit(result, label)

    def apply_median(self) -> None:
        if not self._require_image():
            return
        result, label = fx.median_blur(self.current, self.kernel_var.get())
        self._commit(result, label)

    def apply_bilateral(self) -> None:
        if not self._require_image():
            return
        sigma = float(self.sigma_var.get())
        result, label = fx.bilateral_filter(self.current, 9, sigma, sigma)
        self._commit(result, label)

    def apply_sharpen(self) -> None:
        if not self._require_image():
            return
        result, label = fx.sharpen(self.current)
        self._commit(result, label)

    def apply_flip(self, mode: str) -> None:
        if not self._require_image():
            return
        result, label = fx.flip(self.current, mode)
        self._commit(result, label)

    def apply_rotate(self, angle: float | None = None) -> None:
        if not self._require_image():
            return
        degrees = float(self.angle_var.get() if angle is None else angle)
        result, label = fx.rotate(self.current, degrees, fit=self.fit_var.get())
        self._commit(result, label)

    def apply_noise(self) -> None:
        if not self._require_image():
            return
        result, label = fx.add_gaussian_noise(self.current, float(self.noise_var.get()))
        self._commit(result, label)

    def apply_random(self) -> None:
        if not self._require_image():
            return
        result, labels = fx.apply_random_pipeline(self.current)
        label = "Random → " + ", ".join(labels)
        self._commit(result, label)

    # ── Preview ──────────────────────────────────────────────────────

    def _refresh_previews(self) -> None:
        if self.original is not None:
            self._photo_before = self._to_photo(self.original)
            self.before_label.configure(image=self._photo_before, text="")
        if self.current is not None:
            self._photo_after = self._to_photo(self.current)
            self.after_label.configure(image=self._photo_after, text="")

    def _to_photo(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(MAX_PREVIEW / max(w, 1), MAX_PREVIEW / max(h, 1), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def main() -> None:
    app = ImageStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
