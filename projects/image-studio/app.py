"""
AI Workshop — Image Studio
Paint.NET–inspired layout: menu, toolbar, Tools / History / Layers / Colors
utility windows, and a center canvas for workshop image filters.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import filters as fx
import help_text as docs
from core.vision_classic import faces as vision_faces


APP_TITLE = "Image Studio"
HISTORY_LIMIT = 40

# Paint.NET–like Windows chrome
CHROME = "#F0F0F0"
TOOLBAR = "#E8E8E8"
PANEL = "#F7F7F7"
PANEL_BORDER = "#A0A0A0"
CANVAS_BG = "#808080"
TITLEBAR = "#3B6EA5"
TITLE_TEXT = "#FFFFFF"
TEXT = "#1A1A1A"
MUTED = "#666666"
ACCENT = "#3B6EA5"
BTN = "#E1E1E1"
BTN_ACTIVE = "#CDE3F8"


class UtilityWindow(ttk.Frame):
    """Docked utility panel styled like Paint.NET child windows."""

    def __init__(self, master: tk.Misc, title: str, **kwargs) -> None:
        super().__init__(master, style="Utility.TFrame", **kwargs)
        header = tk.Frame(self, bg=TITLEBAR, height=24)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text=title,
            bg=TITLEBAR,
            fg=TITLE_TEXT,
            font=("Segoe UI", 9, "bold"),
            padx=8,
        ).pack(side=tk.LEFT, fill=tk.Y)
        self.body = tk.Frame(self, bg=PANEL, highlightbackground=PANEL_BORDER, highlightthickness=1)
        self.body.pack(fill=tk.BOTH, expand=True)


class ImageStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} — AI Workshop")
        self.geometry("1280x800")
        self.minsize(1024, 680)
        self.configure(bg=CHROME)

        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.history: list[tuple[np.ndarray, str]] = []
        self.redo_stack: list[tuple[np.ndarray, str]] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._image_path: str | None = None
        self._zoom = 1.0
        self._compare_mode = False
        self._active_tool = tk.StringVar(value="pointer")

        self.kernel_var = tk.IntVar(value=5)
        self.sigma_var = tk.DoubleVar(value=75)
        self.noise_var = tk.DoubleVar(value=15)
        self.fit_var = tk.BooleanVar(value=True)
        self.show_tools = tk.BooleanVar(value=True)
        self.show_history = tk.BooleanVar(value=True)
        self.show_layers = tk.BooleanVar(value=True)
        self.show_colors = tk.BooleanVar(value=True)

        self._build_style()
        self._build_menubar()
        self._build_layout()
        self._bind_shortcuts()
        self._set_status("Ready.")
        self._select_tool("pointer")

    # ── Style ────────────────────────────────────────────────────────

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Chrome.TFrame", background=CHROME)
        style.configure("Toolbar.TFrame", background=TOOLBAR)
        style.configure("Utility.TFrame", background=PANEL)
        style.configure("CanvasHost.TFrame", background=CANVAS_BG)
        style.configure(
            "Tool.TButton",
            font=("Segoe UI", 9),
            padding=(8, 4),
            background=BTN,
        )
        style.map("Tool.TButton", background=[("active", BTN_ACTIVE), ("pressed", ACCENT)])
        style.configure("TButton", font=("Segoe UI", 9), padding=4)
        style.configure("TCheckbutton", background=PANEL, font=("Segoe UI", 8))
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 8))
        style.configure(
            "Status.TLabel",
            background=CHROME,
            foreground=TEXT,
            font=("Segoe UI", 9),
            padding=(8, 4),
        )

    # ── Menu (Paint.NET–like) ────────────────────────────────────────

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self, tearoff=0)

        file_m = tk.Menu(menubar, tearoff=0)
        file_m.add_command(label="Open…", accelerator="Ctrl+O", command=self.open_image)
        file_m.add_command(label="Save As…", accelerator="Ctrl+S", command=self.save_image)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_m)

        edit_m = tk.Menu(menubar, tearoff=0)
        edit_m.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_m.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit_m.add_separator()
        edit_m.add_command(label="Reset to Original", accelerator="Ctrl+R", command=self.reset_image)
        menubar.add_cascade(label="Edit", menu=edit_m)

        view_m = tk.Menu(menubar, tearoff=0)
        view_m.add_command(label="Zoom In", accelerator="Ctrl+=", command=lambda: self._nudge_zoom(1.25))
        view_m.add_command(label="Zoom Out", accelerator="Ctrl+-", command=lambda: self._nudge_zoom(0.8))
        view_m.add_command(label="Zoom to Window", accelerator="Ctrl+0", command=self._zoom_to_fit)
        view_m.add_command(label="Actual Size", command=lambda: self._set_zoom(1.0))
        view_m.add_separator()
        view_m.add_checkbutton(label="Compare Original", command=self.toggle_compare)
        menubar.add_cascade(label="View", menu=view_m)

        image_m = tk.Menu(menubar, tearoff=0)
        image_m.add_command(label="Resize…", command=self.dialog_resize)
        image_m.add_separator()
        image_m.add_command(label="Flip Horizontal", command=lambda: self.apply_flip("horizontal"))
        image_m.add_command(label="Flip Vertical", command=lambda: self.apply_flip("vertical"))
        image_m.add_separator()
        image_m.add_command(label="Rotate 90° Clockwise", command=lambda: self.apply_rotate(-90))
        image_m.add_command(label="Rotate 90° Counter-Clockwise", command=lambda: self.apply_rotate(90))
        image_m.add_command(label="Rotate 180°", command=lambda: self.apply_rotate(180))
        image_m.add_command(label="Rotate Arbitrary…", command=self.dialog_rotate)
        menubar.add_cascade(label="Image", menu=image_m)

        effects_m = tk.Menu(menubar, tearoff=0)
        blur_m = tk.Menu(effects_m, tearoff=0)
        blur_m.add_command(label="Mean Blur", command=self.apply_mean_blur)
        blur_m.add_command(label="Gaussian Blur", command=self.apply_gaussian)
        blur_m.add_command(label="Median Blur", command=self.apply_median)
        blur_m.add_command(label="Bilateral Filter", command=self.apply_bilateral)
        effects_m.add_cascade(label="Blurs", menu=blur_m)
        effects_m.add_command(label="Sharpen", command=self.apply_sharpen)
        effects_m.add_command(label="Add Noise…", command=self.dialog_noise)
        effects_m.add_separator()
        effects_m.add_command(label="Random Pipeline", command=self.apply_random)
        effects_m.add_separator()
        vision_m = tk.Menu(effects_m, tearoff=0)
        vision_m.add_command(label="Detect Faces", command=self.apply_face_detect)
        vision_m.add_command(label="Detect Faces + Eyes", command=self.apply_face_eye_detect)
        effects_m.add_cascade(label="Classic Vision", menu=vision_m)
        effects_m.add_separator()
        help_effects = tk.Menu(effects_m, tearoff=0)
        for key, label in (
            ("mean", "Mean Blur"),
            ("gauss", "Gaussian Blur"),
            ("median", "Median Blur"),
            ("bilateral", "Bilateral Filter"),
            ("sharpen", "Sharpen"),
            ("noise", "Gaussian Noise"),
            ("random", "Random Pipeline"),
            ("faces", "Detect Faces"),
            ("faces_eyes", "Detect Faces + Eyes"),
        ):
            help_effects.add_command(
                label=f"?  {label}",
                command=lambda k=key: self.show_tool_help(k),
            )
        effects_m.add_cascade(label="Explain Effect…", menu=help_effects)
        menubar.add_cascade(label="Effects", menu=effects_m)

        window_m = tk.Menu(menubar, tearoff=0)
        window_m.add_checkbutton(
            label="Tools",
            variable=self.show_tools,
            accelerator="F5",
            command=self._sync_utility_visibility,
        )
        window_m.add_checkbutton(
            label="History",
            variable=self.show_history,
            accelerator="F6",
            command=self._sync_utility_visibility,
        )
        window_m.add_checkbutton(
            label="Layers",
            variable=self.show_layers,
            accelerator="F7",
            command=self._sync_utility_visibility,
        )
        window_m.add_checkbutton(
            label="Colors",
            variable=self.show_colors,
            accelerator="F8",
            command=self._sync_utility_visibility,
        )
        menubar.add_cascade(label="Window", menu=window_m)

        help_m = tk.Menu(menubar, tearoff=0)
        help_m.add_command(
            label="Help for Selected Tool",
            command=lambda: self.show_tool_help(self._active_tool.get()),
        )
        help_m.add_separator()
        help_m.add_command(label="About Image Studio", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_m)

        self.config(menu=menubar)

    # ── Layout ───────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Chrome.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        self._build_toolbar(root)

        self.workspace = ttk.Frame(root, style="Chrome.TFrame")
        self.workspace.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: Tools
        self.tools_win = UtilityWindow(self.workspace, "Tools")
        self.tools_win.configure(width=168)
        self._build_tools_window(self.tools_win.body)

        # Right column: History / Layers / Colors
        self.right_column = ttk.Frame(self.workspace, style="Chrome.TFrame")

        self.history_win = UtilityWindow(self.right_column, "History")
        self._build_history_window(self.history_win.body)

        self.layers_win = UtilityWindow(self.right_column, "Layers")
        self._build_layers_window(self.layers_win.body)

        self.colors_win = UtilityWindow(self.right_column, "Colors")
        self._build_colors_window(self.colors_win.body)

        # Center canvas
        canvas_host = ttk.Frame(self.workspace, style="CanvasHost.TFrame")
        self.canvas_host = canvas_host

        self.canvas = tk.Canvas(
            canvas_host,
            bg=CANVAS_BG,
            highlightthickness=0,
            cursor="arrow",
        )
        self.h_scroll = ttk.Scrollbar(canvas_host, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_host, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_host.rowconfigure(0, weight=1)
        canvas_host.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", lambda _e: self._refresh_canvas())
        self.placeholder = self.canvas.create_text(
            20,
            20,
            anchor="nw",
            fill="#D0D0D0",
            font=("Segoe UI", 14),
            text="Open an image to start editing\nFile ▸ Open   or   Ctrl+O",
        )

        self._sync_utility_visibility()

        status = ttk.Frame(root, style="Chrome.TFrame")
        status.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Separator(status, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.status = ttk.Label(status, text="", style="Status.TLabel", anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.zoom_label = ttk.Label(status, text="100%", style="Status.TLabel")
        self.zoom_label.pack(side=tk.RIGHT)
        self.size_label = ttk.Label(status, text="", style="Status.TLabel")
        self.size_label.pack(side=tk.RIGHT, padx=(0, 12))

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        bar = tk.Frame(parent, bg=TOOLBAR, height=40, highlightbackground=PANEL_BORDER, highlightthickness=1)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=TOOLBAR)
        inner.pack(side=tk.LEFT, padx=6, pady=4)

        def add_btn(text: str, cmd) -> None:
            tk.Button(
                inner,
                text=text,
                command=cmd,
                bg=BTN,
                relief="flat",
                padx=10,
                pady=3,
                font=("Segoe UI", 9),
                activebackground=BTN_ACTIVE,
            ).pack(side=tk.LEFT, padx=2)

        def sep() -> None:
            tk.Frame(inner, bg=PANEL_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        add_btn("Open", self.open_image)
        add_btn("Save", self.save_image)
        sep()
        add_btn("Undo", self.undo)
        add_btn("Redo", self.redo)
        add_btn("Reset", self.reset_image)
        sep()
        add_btn("Zoom −", lambda: self._nudge_zoom(0.8))
        add_btn("Zoom +", lambda: self._nudge_zoom(1.25))
        add_btn("Fit", self._zoom_to_fit)
        sep()
        add_btn("Compare", self.toggle_compare)

        # Utility toggles (Paint.NET menu-bar icons)
        utils = tk.Frame(bar, bg=TOOLBAR)
        utils.pack(side=tk.RIGHT, padx=8)
        for label, var, tip in (
            ("Tools", self.show_tools, "F5"),
            ("History", self.show_history, "F6"),
            ("Layers", self.show_layers, "F7"),
            ("Colors", self.show_colors, "F8"),
        ):
            tk.Checkbutton(
                utils,
                text=label,
                variable=var,
                command=self._sync_utility_visibility,
                bg=TOOLBAR,
                font=("Segoe UI", 8),
                activebackground=TOOLBAR,
                selectcolor=BTN_ACTIVE,
            ).pack(side=tk.LEFT, padx=2)

    def _build_tools_window(self, body: tk.Frame) -> None:
        tip = tk.Label(
            body,
            text="Tool  ·  click ? for help",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        )
        tip.pack(fill=tk.X, padx=6, pady=(6, 2))

        list_frame = tk.Frame(body, bg=PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        tools = [
            ("pointer", "↖", "Pointer"),
            ("hand", "✋", "Pan"),
            ("resize", "▭", "Resize"),
            ("flip_h", "⇔", "Flip H"),
            ("flip_v", "⇕", "Flip V"),
            ("rotate", "↻", "Rotate"),
            ("mean", "▒", "Mean Blur"),
            ("gauss", "░", "Gaussian"),
            ("median", "▦", "Median"),
            ("bilateral", "◉", "Bilateral"),
            ("sharpen", "✦", "Sharpen"),
            ("noise", "⁘", "Noise"),
            ("random", "🎲", "Random"),
            ("faces", "☺", "Faces"),
            ("faces_eyes", "◉", "Faces+Eyes"),
            ("compare", "⧉", "Compare"),
        ]
        self._tool_buttons: dict[str, tk.Button] = {}
        for key, icon, name in tools:
            row = tk.Frame(list_frame, bg=PANEL)
            row.pack(fill=tk.X, pady=1)

            btn = tk.Button(
                row,
                text=icon,
                width=3,
                font=("Segoe UI", 11),
                bg=BTN,
                relief="raised",
                bd=1,
                command=lambda k=key: self._select_tool(k),
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self._tool_buttons[key] = btn

            name_btn = tk.Button(
                row,
                text=name,
                anchor="w",
                bg=PANEL,
                relief="flat",
                font=("Segoe UI", 8),
                activebackground=BTN_ACTIVE,
                command=lambda k=key: self._select_tool(k),
            )
            name_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

            help_btn = tk.Button(
                row,
                text="?",
                width=2,
                font=("Segoe UI", 9, "bold"),
                bg="#FFF8DC",
                fg=ACCENT,
                relief="raised",
                bd=1,
                activebackground="#FFE4A0",
                command=lambda k=key: self.show_tool_help(k),
            )
            help_btn.pack(side=tk.RIGHT)
            self._tooltip(help_btn, f"Help: {name}")

        opts = tk.LabelFrame(body, text="Tool options", bg=PANEL, font=("Segoe UI", 8), fg=TEXT)
        opts.pack(fill=tk.X, padx=6, pady=8)

        row = tk.Frame(opts, bg=PANEL)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text="Kernel", bg=PANEL, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Spinbox(row, from_=1, to=21, textvariable=self.kernel_var, width=4).pack(side=tk.RIGHT)

        row2 = tk.Frame(opts, bg=PANEL)
        row2.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row2, text="Bilateral σ", bg=PANEL, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Spinbox(row2, from_=10, to=200, textvariable=self.sigma_var, width=4).pack(side=tk.RIGHT)

        row3 = tk.Frame(opts, bg=PANEL)
        row3.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row3, text="Noise σ", bg=PANEL, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Spinbox(row3, from_=1, to=50, textvariable=self.noise_var, width=4).pack(side=tk.RIGHT)

        tk.Checkbutton(
            opts,
            text="Fit canvas on rotate",
            variable=self.fit_var,
            bg=PANEL,
            font=("Segoe UI", 8),
            activebackground=PANEL,
        ).pack(anchor=tk.W, padx=4, pady=(4, 6))

        help_row = tk.Frame(body, bg=PANEL)
        help_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Button(
            help_row,
            text="? Help for selected tool",
            command=lambda: self.show_tool_help(self._active_tool.get()),
            bg="#FFF8DC",
            fg=ACCENT,
            relief="raised",
            font=("Segoe UI", 8, "bold"),
            activebackground="#FFE4A0",
        ).pack(fill=tk.X)

        self.tool_hint = tk.Label(
            body,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
            wraplength=140,
            justify=tk.LEFT,
        )
        self.tool_hint.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _build_history_window(self, body: tk.Frame) -> None:
        self.history_list = tk.Listbox(
            body,
            bg="white",
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="white",
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
            height=12,
        )
        self.history_list.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.history_list.insert(tk.END, "• Open image")
        btns = tk.Frame(body, bg=PANEL)
        btns.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(btns, text="Undo", command=self.undo, bg=BTN, relief="flat", font=("Segoe UI", 8)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )
        tk.Button(btns, text="Redo", command=self.redo, bg=BTN, relief="flat", font=("Segoe UI", 8)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=1
        )

    def _build_layers_window(self, body: tk.Frame) -> None:
        self.layers_list = tk.Listbox(
            body,
            bg="white",
            fg=TEXT,
            height=3,
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
            selectbackground=ACCENT,
            selectforeground="white",
        )
        self.layers_list.pack(fill=tk.X, padx=1, pady=1)
        self.layers_list.insert(tk.END, "👁  Background")
        self.layers_list.selection_set(0)
        note = tk.Label(
            body,
            text="Single-layer editing\n(layers expand later)",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
            justify=tk.LEFT,
        )
        note.pack(anchor=tk.W, padx=6, pady=4)

    def _build_colors_window(self, body: tk.Frame) -> None:
        swatches = tk.Frame(body, bg=PANEL)
        swatches.pack(padx=8, pady=8)
        self.primary_swatch = tk.Frame(swatches, bg="#000000", width=48, height=48, highlightthickness=1, highlightbackground="#333")
        self.primary_swatch.grid(row=0, column=0, padx=(0, 4))
        self.secondary_swatch = tk.Frame(swatches, bg="#FFFFFF", width=48, height=48, highlightthickness=1, highlightbackground="#333")
        self.secondary_swatch.grid(row=0, column=1)
        tk.Label(body, text="Primary / Secondary", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(
            padx=8, pady=(0, 4)
        )
        self.color_info = tk.Label(body, text="RGB —", bg=PANEL, fg=TEXT, font=("Consolas", 9))
        self.color_info.pack(padx=8, pady=(0, 8))

    # ── Tools ────────────────────────────────────────────────────────

    def _select_tool(self, key: str) -> None:
        self._active_tool.set(key)
        for k, btn in self._tool_buttons.items():
            btn.configure(bg=BTN_ACTIVE if k == key else BTN, relief="sunken" if k == key else "raised")

        actions = {
            "pointer": None,
            "hand": None,
            "resize": self.dialog_resize,
            "flip_h": lambda: self.apply_flip("horizontal"),
            "flip_v": lambda: self.apply_flip("vertical"),
            "rotate": self.dialog_rotate,
            "mean": self.apply_mean_blur,
            "gauss": self.apply_gaussian,
            "median": self.apply_median,
            "bilateral": self.apply_bilateral,
            "sharpen": self.apply_sharpen,
            "noise": self.dialog_noise,
            "random": self.apply_random,
            "faces": self.apply_face_detect,
            "faces_eyes": self.apply_face_eye_detect,
            "compare": self.toggle_compare,
        }
        info = docs.get_help(key)
        self.tool_hint.configure(text=info["summary"])
        self.canvas.configure(cursor="fleur" if key == "hand" else "arrow")

        action = actions.get(key)
        if action and key not in ("pointer", "hand"):
            action()

    def show_tool_help(self, key: str) -> None:
        info = docs.get_help(key)
        dialog = tk.Toplevel(self)
        dialog.title(f"Help — {info['title']}")
        dialog.configure(bg=PANEL)
        dialog.transient(self)
        dialog.resizable(False, False)

        header = tk.Frame(dialog, bg=TITLEBAR)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"?  {info['title']}",
            bg=TITLEBAR,
            fg=TITLE_TEXT,
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        ).pack(anchor=tk.W)

        body = tk.Frame(dialog, bg=PANEL, padx=14, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            body,
            text=info["summary"],
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 8))

        tk.Label(
            body,
            text=info["details"],
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 9),
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        tk.Button(
            dialog,
            text="Close",
            command=dialog.destroy,
            bg=BTN,
            relief="raised",
            font=("Segoe UI", 9),
            padx=16,
            pady=4,
        ).pack(pady=(4, 12))

        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + 80
        dialog.geometry(f"+{x}+{y}")
        dialog.focus_set()
        dialog.bind("<Escape>", lambda _e: dialog.destroy())

    def _tooltip(self, widget: tk.Widget, text: str) -> None:
        tip: list[tk.Toplevel | None] = [None]

        def enter(_e=None) -> None:
            if tip[0] is not None:
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            x = widget.winfo_rootx() + 30
            y = widget.winfo_rooty() + 20
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(
                win,
                text=text,
                bg="#FFFFE1",
                fg=TEXT,
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 8),
                padx=4,
                pady=2,
            ).pack()
            tip[0] = win

        def leave(_e=None) -> None:
            if tip[0] is not None:
                tip[0].destroy()
                tip[0] = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def _sync_utility_visibility(self) -> None:
        self.tools_win.pack_forget()
        self.right_column.pack_forget()
        self.canvas_host.pack_forget()
        self.history_win.pack_forget()
        self.layers_win.pack_forget()
        self.colors_win.pack_forget()

        if self.show_tools.get():
            self.tools_win.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
            self.tools_win.pack_propagate(False)
            self.tools_win.configure(width=168)

        right_visible = (
            self.show_history.get() or self.show_layers.get() or self.show_colors.get()
        )
        if right_visible:
            self.right_column.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
            if self.show_history.get():
                self.history_win.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
            if self.show_layers.get():
                self.layers_win.pack(fill=tk.X, pady=(0, 4))
            if self.show_colors.get():
                self.colors_win.pack(fill=tk.X)

        self.canvas_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── Dialogs ──────────────────────────────────────────────────────

    def dialog_resize(self) -> None:
        if not self._require_image():
            return
        assert self.current is not None
        h, w = self.current.shape[:2]
        width = simpledialog.askinteger("Resize", "Width (px):", initialvalue=w, minvalue=16, maxvalue=8000, parent=self)
        if width is None:
            return
        height = simpledialog.askinteger("Resize", "Height (px):", initialvalue=h, minvalue=16, maxvalue=8000, parent=self)
        if height is None:
            return
        result, label = fx.resize(self.current, width, height)
        self._commit(result, label)

    def dialog_rotate(self) -> None:
        if not self._require_image():
            return
        angle = simpledialog.askfloat(
            "Rotate",
            "Angle (degrees, positive = CCW):",
            initialvalue=45.0,
            minvalue=-360,
            maxvalue=360,
            parent=self,
        )
        if angle is None:
            return
        self.apply_rotate(angle)

    def dialog_noise(self) -> None:
        if not self._require_image():
            return
        sigma = simpledialog.askfloat(
            "Add Noise",
            "Gaussian noise σ:",
            initialvalue=float(self.noise_var.get()),
            minvalue=1,
            maxvalue=80,
            parent=self,
        )
        if sigma is None:
            return
        self.noise_var.set(sigma)
        self.apply_noise()

    # ── Zoom / view ──────────────────────────────────────────────────

    def _nudge_zoom(self, factor: float) -> None:
        self._set_zoom(max(0.1, min(8.0, self._zoom * factor)))

    def _set_zoom(self, value: float) -> None:
        self._zoom = value
        self.zoom_label.configure(text=f"{int(self._zoom * 100)}%")
        self._refresh_canvas()

    def _zoom_to_fit(self) -> None:
        if self.current is None:
            return
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = self.current.shape[:2]
        scale = min(cw / w, ch / h, 1.0) * 0.96
        self._set_zoom(scale)

    def toggle_compare(self) -> None:
        if self.original is None:
            return
        self._compare_mode = not self._compare_mode
        mode = "Original" if self._compare_mode else "Current"
        self._set_status(f"View: {mode}")
        self._refresh_canvas()

    # ── File ─────────────────────────────────────────────────────────

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Error", f"Could not open:\n{path}")
            return

        self._image_path = path
        self.original = image
        self.current = image.copy()
        self.history.clear()
        self.redo_stack.clear()
        self._compare_mode = False
        self.history_list.delete(0, tk.END)
        self.history_list.insert(tk.END, f"• Open {Path(path).name}")
        self._update_color_from_image()
        self._zoom_to_fit()
        self._update_size_label()
        self.title(f"{APP_TITLE} — {Path(path).name}")
        self._set_status(f"Opened {Path(path).name}")

    def save_image(self) -> None:
        if self.current is None:
            messagebox.showinfo("Save", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        if not cv2.imwrite(path, self.current):
            messagebox.showerror("Error", f"Could not save:\n{path}")
            return
        self._set_status(f"Saved {Path(path).name}")

    # ── History ──────────────────────────────────────────────────────

    def _require_image(self) -> bool:
        if self.current is None:
            messagebox.showinfo("Image Studio", "Open an image first.")
            return False
        return True

    def _commit(self, result: np.ndarray, label: str) -> None:
        assert self.current is not None
        self.history.append((self.current.copy(), label))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
        self.redo_stack.clear()
        self.current = result
        self.history_list.insert(tk.END, label)
        self.history_list.see(tk.END)
        self._compare_mode = False
        self._refresh_canvas()
        self._update_size_label()
        self._update_color_from_image()
        self._set_status(label)

    def undo(self) -> None:
        if not self.history:
            self._set_status("Nothing to undo.")
            return
        previous, label = self.history.pop()
        assert self.current is not None
        self.redo_stack.append((self.current.copy(), label))
        self.current = previous
        if self.history_list.size() > 1:
            self.history_list.delete(tk.END)
        self._refresh_canvas()
        self._update_size_label()
        self._set_status(f"Undo: {label}")

    def redo(self) -> None:
        if not self.redo_stack:
            self._set_status("Nothing to redo.")
            return
        image, label = self.redo_stack.pop()
        assert self.current is not None
        self.history.append((self.current.copy(), label))
        self.current = image
        self.history_list.insert(tk.END, label)
        self.history_list.see(tk.END)
        self._refresh_canvas()
        self._update_size_label()
        self._set_status(f"Redo: {label}")

    def reset_image(self) -> None:
        if self.original is None:
            return
        self.current = self.original.copy()
        self.history.clear()
        self.redo_stack.clear()
        self.history_list.delete(0, tk.END)
        name = Path(self._image_path).name if self._image_path else "image"
        self.history_list.insert(tk.END, f"• Reset {name}")
        self._compare_mode = False
        self._refresh_canvas()
        self._update_size_label()
        self._set_status("Reset to original.")

    # ── Effects ──────────────────────────────────────────────────────

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

    def apply_rotate(self, angle: float) -> None:
        if not self._require_image():
            return
        result, label = fx.rotate(self.current, float(angle), fit=self.fit_var.get())
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
        self._commit(result, "Random → " + ", ".join(labels))

    def apply_face_detect(self) -> None:
        if not self._require_image():
            return
        result, label = vision_faces.draw_faces(self.current)
        self._commit(result, label)

    def apply_face_eye_detect(self) -> None:
        if not self._require_image():
            return
        result, label = vision_faces.draw_faces_and_eyes(self.current)
        self._commit(result, label)

    # ── Canvas ───────────────────────────────────────────────────────

    def _display_image(self) -> np.ndarray | None:
        if self.current is None:
            return None
        if self._compare_mode and self.original is not None:
            return self.original
        return self.current

    def _refresh_canvas(self) -> None:
        image = self._display_image()
        self.canvas.delete("img")
        if image is None:
            return
        if self.placeholder:
            self.canvas.delete(self.placeholder)
            self.placeholder = None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        zw, zh = max(1, int(w * self._zoom)), max(1, int(h * self._zoom))
        if (zw, zh) != (w, h):
            interp = cv2.INTER_AREA if self._zoom < 1 else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (zw, zh), interpolation=interp)

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="img")
        self.canvas.configure(scrollregion=(0, 0, zw, zh))

    def _update_size_label(self) -> None:
        if self.current is None:
            self.size_label.configure(text="")
            return
        h, w = self.current.shape[:2]
        self.size_label.configure(text=f"{w} × {h} px")

    def _update_color_from_image(self) -> None:
        if self.current is None:
            return
        h, w = self.current.shape[:2]
        b, g, r = [int(x) for x in self.current[h // 2, w // 2]]
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        self.primary_swatch.configure(bg=hex_color)
        self.color_info.configure(text=f"RGB ({r}, {g}, {b})")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Image Studio",
            "Image Studio — AI Learning Lab\n\n"
            "Layout inspired by Paint.NET\n"
            "Python · Tkinter · OpenCV\n"
            "Logic from shared `core` package (web-ready)\n\n"
            "Click the ? next to any tool for a description\n"
            "of what that filter or transform does.\n\n"
            "Utility windows: Tools (F5), History (F6),\n"
            "Layers (F7), Colors (F8).",
        )

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self.open_image())
        self.bind("<Control-s>", lambda _e: self.save_image())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())
        self.bind("<Control-r>", lambda _e: self.reset_image())
        self.bind("<Control-equal>", lambda _e: self._nudge_zoom(1.25))
        self.bind("<Control-minus>", lambda _e: self._nudge_zoom(0.8))
        self.bind("<Control-0>", lambda _e: self._zoom_to_fit())
        self.bind("<F5>", lambda _e: self._toggle_var(self.show_tools))
        self.bind("<F6>", lambda _e: self._toggle_var(self.show_history))
        self.bind("<F7>", lambda _e: self._toggle_var(self.show_layers))
        self.bind("<F8>", lambda _e: self._toggle_var(self.show_colors))

    def _toggle_var(self, var: tk.BooleanVar) -> None:
        var.set(not var.get())
        self._sync_utility_visibility()


def main() -> None:
    app = ImageStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
