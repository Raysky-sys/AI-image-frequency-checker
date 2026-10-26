"""
AI_image_frequency_checker.py
=============================

Operator GUI for extracting and amplifying the high-frequency residual of
an image.

Purpose
-------
The low-frequency content of an image IS the picture. What is left after
removing it is the processing fingerprint: compression traces, resampling
grid artefacts, upscaling patterns, sensor noise, and the structured
patterns that some generative pipelines leave behind.

This tool isolates that residual so it can be inspected visually.

Pipeline
--------
    1. Load image -> grayscale float32 in [0, 1]
    2. np.fft.fft2 -> np.fft.fftshift      (DC component to centre)
    3. Circular high-pass mask              (low frequencies blocked)
    4. np.fft.ifftshift -> np.fft.ifft2 -> abs()
    5. Contrast stretch + CLAHE + gain
    6. Four-panel Matplotlib visualisation

Usage
-----
    python AI_image_frequency_checker.py                # empty, use the GUI
    python AI_image_frequency_checker.py photo.png      # preload an image

Author: Raysky-sys
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import List, Optional

import matplotlib

# The TkAgg backend MUST be selected before the FigureCanvasTkAgg import.
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from residual_core import (
    Params,
    analyze_image,
    load_gray,
    save_gray,
)

from make_version_info import VERSION_STRING as __version__

APP_NAME = "AI Image Frequency Checker"


class ResidualViewer(tk.Tk):
    """Main application window."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()

        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1480x940")
        self.minsize(1100, 720)

        # --- state -------------------------------------------------------
        self.image_paths: List[str] = []
        self.image_index: int = -1
        self.current_path: Optional[str] = None
        self.result = None

        self.params = Params()

        self._work_queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._busy = False
        self._pending_recompute = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(80, self._drain_queue)
        self._log(f"{APP_NAME} v{__version__} ready.")
        self._log(
            "Purpose: expose the HIGH-FREQUENCY RESIDUAL of an image, i.e. "
            "the processing\nfingerprint left by compression, resampling, "
            "upscaling and generative pipelines."
        )
        self._log(
            "Reminder: a visible residual proves the file was PROCESSED. "
            "It does not by itself\nprove the file was AI-generated, nor "
            "which model produced it."
        )

        if initial_path:
            self.after(200, lambda: self._load_paths([initial_path]))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)

        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=3)
        outer.rowconfigure(2, weight=1)

        # ---------------- toolbar ----------------
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self.btn_open = ttk.Button(toolbar, text="Open Image(s)",
                                   command=self._open_images)
        self.btn_prev = ttk.Button(toolbar, text="<< Previous",
                                   command=lambda: self._step(-1))
        self.btn_next = ttk.Button(toolbar, text="Next >>",
                                   command=lambda: self._step(1))
        self.btn_save_res = ttk.Button(toolbar, text="Save Residual",
                                        command=self._save_residual)
        self.btn_save_fig = ttk.Button(toolbar, text="Save Figure",
                                        command=self._save_figure)
        self.btn_reset = ttk.Button(toolbar, text="Reset Parameters",
                                     command=self._reset_params)

        self.btn_open.pack(side="left")
        self.btn_prev.pack(side="left", padx=(6, 0))
        self.btn_next.pack(side="left", padx=(6, 0))
        self.btn_save_res.pack(side="left", padx=(18, 0))
        self.btn_save_fig.pack(side="left", padx=(6, 0))
        self.btn_reset.pack(side="left", padx=(6, 0))

        self.lbl_image = ttk.Label(toolbar, text="No image loaded")
        self.lbl_image.pack(side="right")

        # ---------------- control panel ----------------
        panel = ttk.LabelFrame(outer, text="Parameters", padding=8)
        panel.grid(row=1, column=0, sticky="nsw", padx=(0, 8))

        self.var_radius = tk.DoubleVar(value=self.params.radius_frac)
        self.var_feather = tk.DoubleVar(value=self.params.feather_px)
        self.var_gain = tk.DoubleVar(value=self.params.gain)
        self.var_clip = tk.DoubleVar(value=self.params.clahe_clip)
        self.var_tiles = tk.IntVar(value=self.params.clahe_tiles)
        self.var_clahe = tk.BooleanVar(value=self.params.use_clahe)
        self.var_robust = tk.BooleanVar(value=self.params.robust_stretch)
        self.var_real = tk.BooleanVar(value=self.params.use_real_part)
        self.var_cmap = tk.StringVar(value="gray")
        self.var_auto = tk.BooleanVar(value=True)

        row = 0

        ttk.Label(panel, text="High-pass radius",
                  font=("Segoe UI", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.sld_radius = ttk.Scale(
            panel, from_=0.005, to=0.45, variable=self.var_radius,
            command=self._on_param_change)
        self.sld_radius.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1
        self.lbl_radius = ttk.Label(panel, text="")
        self.lbl_radius.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2,
                                  sticky="ew", pady=8)
        row += 1

        ttk.Label(panel, text="Feather width (px)").grid(
            row=row, column=0, sticky="w")
        self.sld_feather = ttk.Scale(
            panel, from_=0, to=60, variable=self.var_feather,
            command=self._on_param_change)
        self.sld_feather.grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(panel, text="Linear gain").grid(row=row, column=0, sticky="w")
        self.sld_gain = ttk.Scale(
            panel, from_=0.1, to=8.0, variable=self.var_gain,
            command=self._on_param_change)
        self.sld_gain.grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2,
                                  sticky="ew", pady=8)
        row += 1

        ttk.Checkbutton(panel, text="CLAHE contrast enhancement",
                        variable=self.var_clahe,
                        command=self._on_param_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(panel, text="CLAHE clip limit").grid(
            row=row, column=0, sticky="w")
        self.sld_clip = ttk.Scale(
            panel, from_=1.0, to=8.0, variable=self.var_clip,
            command=self._on_param_change)
        self.sld_clip.grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(panel, text="CLAHE tiles").grid(row=row, column=0, sticky="w")
        self.sld_tiles = ttk.Scale(
            panel, from_=2, to=16, variable=self.var_tiles,
            command=self._on_param_change)
        self.sld_tiles.grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Checkbutton(panel, text="Robust percentile stretch",
                        variable=self.var_robust,
                        command=self._on_param_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Checkbutton(panel, text="Use signed real part (not abs)",
                        variable=self.var_real,
                        command=self._on_param_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2,
                                  sticky="ew", pady=8)
        row += 1

        ttk.Label(panel, text="Residual colormap").grid(
            row=row, column=0, sticky="w")
        self.cmb_cmap = ttk.Combobox(
            panel, textvariable=self.var_cmap,
            values=("gray", "viridis", "magma", "inferno", "hot", "jet", "turbo"),
            state="readonly", width=12)
        self.cmb_cmap.grid(row=row, column=1, sticky="ew")
        self.cmb_cmap.bind("<<ComboboxSelected>>", self._on_param_change)
        row += 1

        ttk.Checkbutton(panel, text="Auto-recompute on change",
                        variable=self.var_auto,
                        command=self._on_param_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Button(panel, text="Recompute Now",
                   command=self._trigger_recompute).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        panel.columnconfigure(1, weight=1)

        # ---------------- figure canvas ----------------
        canvas_frame = ttk.Frame(outer)
        canvas_frame.grid(row=1, column=1, sticky="nsew")

        self.figure = Figure(figsize=(12, 9), dpi=100)
        self.axes = self.figure.subplots(2, 2)
        self.figure.suptitle(
            "High-Frequency Residual Extraction Pipeline",
            fontsize=13, fontweight="bold")

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status = ttk.Label(canvas_frame, anchor="w", text="Idle")
        self.status.pack(fill="x", pady=(4, 0))

        # ---------------- log ----------------
        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=7, wrap="word",
                                font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical",
                                   command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.bind("<Left>", lambda _e: self._step(-1))
        self.bind("<Right>", lambda _e: self._step(1))

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def _read_params(self) -> Params:
        return Params(
            radius_frac=float(self.var_radius.get()),
            feather_px=float(self.var_feather.get()),
            gain=float(self.var_gain.get()),
            use_clahe=bool(self.var_clahe.get()),
            clahe_clip=float(self.var_clip.get()),
            clahe_tiles=int(self.var_tiles.get()),
            robust_stretch=bool(self.var_robust.get()),
            use_real_part=bool(self.var_real.get()),
        )

    def _reset_params(self) -> None:
        defaults = Params()
        self.var_radius.set(defaults.radius_frac)
        self.var_feather.set(defaults.feather_px)
        self.var_gain.set(defaults.gain)
        self.var_clahe.set(defaults.use_clahe)
        self.var_clip.set(defaults.clahe_clip)
        self.var_tiles.set(defaults.clahe_tiles)
        self.var_robust.set(defaults.robust_stretch)
        self.var_real.set(defaults.use_real_part)
        self.var_cmap.set("gray")
        self._log("Parameters reset to defaults.")
        self._trigger_recompute()

    def _on_param_change(self, _event=None) -> None:
        if self.var_auto.get():
            self._schedule_recompute()
        self._update_radius_label()

    def _update_radius_label(self) -> None:
        self.lbl_radius.configure(
            text=f"Fraction of min(H,W): {float(self.var_radius.get()):.3f}")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _open_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._load_paths(list(paths))

    def _load_paths(self, paths: List[str]) -> None:
        self.image_paths = paths
        self.image_index = 0
        self._load_current()

    def _step(self, delta: int) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index + delta) % len(self.image_paths)
        self._load_current()

    def _load_current(self) -> None:
        if not self.image_paths:
            return

        path = self.image_paths[self.image_index]
        self.current_path = path
        self.lbl_image.configure(
            text=f"[{self.image_index + 1}/{len(self.image_paths)}] "
                 f"{os.path.basename(path)}")
        self._log(f"Loading: {path}")
        self._trigger_recompute()

    # ------------------------------------------------------------------
    # Background analysis
    # ------------------------------------------------------------------
    def _schedule_recompute(self) -> None:
        """Debounce so slider drags do not queue dozens of jobs."""
        if self._pending_recompute:
            return
        self._pending_recompute = True
        self.after(220, self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._pending_recompute = False
        self._trigger_recompute()

    def _trigger_recompute(self) -> None:
        if self.current_path is None:
            return

        if self._busy:
            self._pending_recompute = True
            return

        path = self.current_path
        params = self._read_params()

        self._busy = True
        self.status.configure(text=f"Analysing {os.path.basename(path)} ...")
        self._set_buttons(busy=True)

        self._worker = threading.Thread(
            target=self._analysis_worker, args=(path, params), daemon=True)
        self._worker.start()

    def _analysis_worker(self, path: str, params: Params) -> None:
        try:
            gray = load_gray(path)
            result = analyze_image(gray, params=params, path=path)
            self._work_queue.put(("ok", result))
        except Exception as error:  # noqa: BLE001
            self._work_queue.put(
                ("error", f"{error}\n{traceback.format_exc()}"))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._work_queue.get_nowait()

                if kind == "ok":
                    self.result = payload
                    self._render(payload)
                    self._busy = False
                    self._set_buttons(busy=False)
                    self.status.configure(
                        text=f"Done in {payload.elapsed_ms:.0f} ms | "
                             f"radius {payload.radius_px} px "
                             f"({payload.radius_frac:.3f} x min(H,W))")
                    self._log(
                        f"Analysis complete: {os.path.basename(payload.path)} | "
                        f"{payload.gray.shape[1]}x{payload.gray.shape[0]} px | "
                        f"radius {payload.radius_px} px | "
                        f"{payload.elapsed_ms:.0f} ms")

                elif kind == "error":
                    self._busy = False
                    self._set_buttons(busy=False)
                    self.status.configure(text="Analysis failed")
                    self._log(f"[!] ERROR: {payload}")
                    messagebox.showerror("Analysis Error",
                                         payload.splitlines()[0])

        except queue.Empty:
            pass

        self.after(80, self._drain_queue)

    def _set_buttons(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in (self.btn_open, self.btn_prev, self.btn_next,
                       self.btn_save_res, self.btn_save_fig, self.btn_reset):
            button.configure(state=state)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self, result) -> None:
        axes = self.axes.ravel()
        for axis in axes:
            axis.clear()

        # (1) Original grayscale image.
        axes[0].imshow(result.gray, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(
            f"1. Original (grayscale, 0-1)\n{result.gray.shape[1]} x "
            f"{result.gray.shape[0]} px", fontsize=10)
        axes[0].set_axis_off()

        # (2) Magnitude spectrum, log-scaled.
        axes[1].imshow(result.spectrum_log, cmap="magma")
        axes[1].set_title(
            "2. Magnitude spectrum |FFT|, log1p scaled\n"
            "(DC at centre after fftshift)", fontsize=10)
        axes[1].set_axis_off()

        # (3) Masked magnitude spectrum with the blocked disc outlined.
        axes[2].imshow(result.masked_log, cmap="magma")
        centre_y, centre_x = result.center
        axes[2].add_patch(Circle(
            (centre_x, centre_y), result.radius_px,
            fill=False, edgecolor="cyan", linewidth=1.4, linestyle="--"))
        axes[2].set_title(
            f"3. Masked spectrum (centre blocked, r = {result.radius_px} px)\n"
            "0 = blocked low frequencies, 1 = passed high frequencies",
            fontsize=10)
        axes[2].set_axis_off()

        # (4) Final amplified residual.
        axes[3].imshow(result.residual, cmap=self.var_cmap.get(),
                       vmin=0.0, vmax=1.0)
        axes[3].set_title(
            "4. Amplified high-frequency residual\n"
            "(contrast stretch + CLAHE + gain)", fontsize=10)
        axes[3].set_axis_off()

        try:
            self.figure.tight_layout(rect=(0, 0, 1, 0.97))
        except Exception:  # noqa: BLE001
            pass

        self.canvas.draw_idle()


    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _save_residual(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return

        default_name = (os.path.splitext(
            os.path.basename(self.result.path))[0] + "_residual.png")

        path = filedialog.asksaveasfilename(
            title="Save residual image", defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return

        try:
            save_gray(path, self.result.residual)
            self._log(f"Residual saved: {path}")
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Save Error", str(error))

    def _save_figure(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return

        default_name = (os.path.splitext(
            os.path.basename(self.result.path))[0] + "_pipeline.png")

        path = filedialog.asksaveasfilename(
            title="Save figure", defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return

        try:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved: {path}")
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Save Error", str(error))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno(
                    "Analysis running",
                    "An analysis is still running. Close anyway?"):
                return
        self.destroy()


def main() -> None:
    initial_path = None

    if len(sys.argv) > 1:
        candidate = os.path.abspath(sys.argv[1])
        if os.path.isfile(candidate):
            initial_path = candidate
        else:
            print(f"Warning: '{sys.argv[1]}' is not a file. "
                  f"Starting with an empty viewer.")

    app = ResidualViewer(initial_path=initial_path)
    app.mainloop()


if __name__ == "__main__":
    main()
