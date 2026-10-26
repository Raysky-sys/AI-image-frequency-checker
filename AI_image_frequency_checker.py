"""
AI_image_frequency_checker.py
=============================

Operator GUI for analysing the processing history of an image.

Three analysis methods are available:

    FFT Residual     - frequency-domain high-pass extraction
    ELA              - error level analysis against a JPEG recompression
    Noise Analysis   - reverse denoising, isolates the sensor noise

The default view shows two panels: the original image and the final
result. The spectrum diagnostics can be switched on for operators who
want to inspect the frequency domain.

Every slider carries a live numeric readout, so the readings quoted in
the operator guide can be followed exactly. The advanced options are
collapsed by default to keep the panel uncluttered for new operators.

The full operator guide is built into the application and is available
offline through the "Read Before You Try" button.

Usage
-----
    python AI_image_frequency_checker.py
    python AI_image_frequency_checker.py photo.png

Author: Raysky-sys
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from typing import Callable, List, Optional, Tuple

import cv2
import matplotlib

# The TkAgg backend MUST be selected before the FigureCanvasTkAgg import.
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import help_content
from analysis_cache import (
    KIND_FFT,
    KIND_GRAY,
    KIND_MAGMA,
    KIND_RGB,
    CachedResult,
    ResultCache,
    build_cache_entry,
    downscale,
)
from forensic_tools import (

    MODE_ELA,
    MODE_NOISE,
    AnalysisCancelled,
    ForensicsParams,
    analyze_ela,
    analyze_noise,
    load_color,
)
from image_ops import (
    MAGNIFIER_CHOICES,
    magnifier_key_from_label,
)
from residual_core import (
    Params,
    analyze_image,
    load_gray,
    save_gray,
)

from make_version_info import VERSION_STRING as __version__

APP_NAME = "AI Image Frequency Checker"

MODE_FFT = "FFT Residual"
MODE_ELA_LABEL = "ELA (Error Level Analysis)"
MODE_NOISE_LABEL = "Noise Analysis"

MODE_CHOICES = (MODE_FFT, MODE_ELA_LABEL, MODE_NOISE_LABEL)

#: The mode value carried by a ResidualResult, which is the FFT method.
#: ForensicsResult uses "ela" and "noise" for the other two methods.
RESULT_MODE_FFT = "fft"

ZOOM_LEVELS = ("1x", "2x", "3x", "4x", "6x", "8x", "12x", "16x")

#: The longest edge, in pixels, used for the on-screen panel previews.
#: The full-resolution data is retained for the magnifier lens and for
#: saving, so nothing is lost, but the redraw stays fast.
MAX_PREVIEW_DIM = 1800

#: The literal bullet used in the guide, since the Tk 8.6 text widget
#: does not support a bullet tag option.
BULLET = "\u2022  "

#: The section header arrows.
ARROW_EXPANDED = "\u25bc"
ARROW_COLLAPSED = "\u25b6"


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------
def app_directory() -> str:
    """Return the folder that contains the application."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative: str) -> str:
    """Resolve a bundled resource, working frozen and unfrozen."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidate = os.path.join(base, relative)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(app_directory(), relative)


def config_directory() -> str:
    """Return a writable per-user settings directory."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "AIImageFrequencyChecker")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        path = os.path.expanduser("~")
    return path


def load_config() -> dict:
    """Load the saved settings, tolerating a missing or corrupt file."""
    path = os.path.join(config_directory(), "settings.json")
    defaults = {
        "show_help_on_start": False,
        "fft_advanced_expanded": False,
    }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            defaults.update(data)
    except (OSError, ValueError):
        pass
    return defaults


def save_config(config: dict) -> None:
    """Save the settings, ignoring any write failure."""
    path = os.path.join(config_directory(), "settings.json")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    except OSError:
        pass


def install_crash_handler() -> None:
    """
    Write unhandled exceptions to a log file.

    A --windowed build has no console, so without this a crash would
    vanish the moment the error dialog is dismissed.
    """

    def handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        try:
            log_path = os.path.join(
                config_directory(), "crash_log.txt")

            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 60 + "\n")
                handle.write(
                    f"Crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                handle.write("=" * 60 + "\n")
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=handle)
        except Exception:
            pass

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handler



# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
class CollapsibleSection(ttk.Frame):
    """
    A titled section whose contents can be collapsed.

    Used to keep the advanced analysis options out of the way for
    operators who only need the primary control.

    Collapsing hides the widgets but changes no parameter value, so the
    analysis result is identical whether the section is open or closed.
    """

    def __init__(
        self,
        master: tk.Misc,
        title: str,
        collapsed: bool = True,
        on_toggle: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)

        self.section_title = title
        self._expanded = not collapsed
        self._on_toggle = on_toggle

        # ttk.Button does not accept the -anchor option. That option
        # belongs to the classic tk.Button, and passing an unknown
        # option raises a TclError when the widget is constructed, which
        # stops the application before it can start. The header text is
        # centred instead.
        style = ttk.Style(self)
        style.configure(
            "SectionHeader.TButton",
            font=("Segoe UI", 9, "bold"),
        )

        self.header = ttk.Button(
            self,
            text="",
            command=self.toggle,
            style="SectionHeader.TButton",
        )

        self.header.grid(row=0, column=0, sticky="ew")

        self.body = ttk.Frame(self)
        self.body.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.columnconfigure(0, weight=1)

        if collapsed:
            self.body.grid_remove()

        self._update_header()

    def toggle(self) -> None:
        """Show or hide the contents of the section."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Set the expanded state and notify the listener."""
        self._expanded = bool(expanded)

        if self._expanded:
            self.body.grid()
        else:
            self.body.grid_remove()

        self._update_header()

        if self._on_toggle:
            self._on_toggle(self._expanded)

    @property
    def expanded(self) -> bool:
        return self._expanded

    def _update_header(self) -> None:
        arrow = ARROW_EXPANDED if self._expanded else ARROW_COLLAPSED
        self.header.configure(text=f"{arrow}  {self.section_title}")


# ---------------------------------------------------------------------------
# Help window
# ---------------------------------------------------------------------------
class HelpWindow(tk.Toplevel):
    """
    The offline operator guide.

    Deliberately NON-MODAL. A Tk grab_set() on this window would block
    every other window in the application, including the main window,
    which makes the whole program appear frozen. The guide is a
    reference, so the operator should be able to read it while the
    tool is in use.
    """

    def __init__(
        self,
        master: tk.Misc,
        on_show_startup_change=None,
        show_on_start: bool = False,
        on_close=None,
    ) -> None:
        super().__init__(master)

        self.title("Read Before You Try - Operator Guide")
        self.geometry("1000x780")
        self.minsize(720, 560)

        # transient() keeps the guide always above the main window
        # without taking any input away from it.
        self.transient(master)

        self._on_show_startup_change = on_show_startup_change
        self._on_close_callback = on_close

        self._build(show_on_start)
        self._centre_on(master)

        # Closing the window with the X button must run the same
        # cleanup as the Close button.
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _centre_on(self, master: tk.Misc) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _close(self) -> None:
        """Release everything this window owns, then destroy it."""
        try:
            # Release any grab this window may hold. Without this, the
            # main window can be left unable to receive input.
            self.grab_release()
        except tk.TclError:
            pass

        self.destroy()

        if self._on_close_callback:
            self._on_close_callback()

    def _build(self, show_on_start: bool) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))

        ttk.Label(
            header,
            text="READ BEFORE YOU TRY",
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")

        ttk.Button(
            header, text="Copy All Text", command=self._copy_all
        ).pack(side="right", padx=(6, 0))
        ttk.Button(
            header, text="Save Guide to File", command=self._save_guide
        ).pack(side="right")

        banner = tk.Text(
            outer,
            height=12,
            wrap="word",
            font=("Segoe UI", 9),
            background="#fff8e1",
            foreground="#4a3b00",
            relief="flat",
            padx=8,
            pady=6,
        )
        banner.insert("1.0", help_content.INTRO)
        banner.configure(state="disabled")
        banner.pack(fill="x", pady=(0, 8))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        for section in help_content.SECTIONS:
            frame = ttk.Frame(self.notebook, padding=6)
            self.notebook.add(frame, text=section["label"])

            text = ScrolledText(
                frame,
                wrap="word",
                font=("Segoe UI", 10),
                background="#fbfbfb",
                foreground="#1c1c1c",
                relief="flat",
                padx=10,
                pady=8,
            )
            text.pack(fill="both", expand=True)

            # Each section is rendered independently. A failure in one
            # section must not abort the rest of the guide, and the
            # reason is shown in the section itself.
            try:
                self._style_text(text)
                self._render_section(text, section["blocks"])
            except Exception as error:  # noqa: BLE001
                text.configure(state="normal")
                text.insert(
                    tk.END,
                    f"This section could not be rendered.\n\n{error}\n",
                )
                text.configure(state="disabled")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))

        self.show_on_start = tk.BooleanVar(value=bool(show_on_start))
        ttk.Checkbutton(
            footer,
            text="Show this guide automatically when the application starts",
            variable=self.show_on_start,
            command=self._toggle_startup,
        ).pack(side="left")

        ttk.Button(
            footer, text="Close", command=self._close
        ).pack(side="right")

    def _style_text(self, text: tk.Text) -> None:
        """
        Apply the shared tag styles.

        The styles live in help_content so the pre-build self-check
        can verify that every option is supported by this Tk build.
        """
        for tag, options in help_content.TAG_STYLES.items():
            text.tag_configure(tag, **options)

    def _render_section(self, text: tk.Text, blocks: List[tuple]) -> None:
        """Render the guide blocks and leave the page read-only."""
        text.configure(state="normal")

        try:
            for kind, payload in blocks:
                if kind == "h2":
                    text.insert(tk.END, f"{str(payload).upper()}\n", "h2")
                elif kind == "p":
                    text.insert(tk.END, f"{payload}\n", "p")
                elif kind == "b":
                    text.insert(tk.END, f"{BULLET}{payload}\n", "b")
                elif kind == "b2":
                    text.insert(tk.END, f"     - {payload}\n", "b")
                elif kind == "note":
                    text.insert(tk.END, f"NOTE:  {payload}\n", "note")
                elif kind == "warn":
                    text.insert(tk.END, f"CAUTION:  {payload}\n", "warn")
                elif kind == "code":
                    text.insert(tk.END, f"{payload}\n", "p")
                elif kind == "table":
                    left, right = payload
                    text.insert(tk.END, f"{left}\n", "tablehead")
                    text.insert(tk.END, f"{right}\n", "table")
        finally:
            # Guarantees the page cannot be edited, even if rendering
            # raises part-way through.
            text.configure(state="disabled")

    def _copy_all(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(help_content.full_text())
        messagebox.showinfo(
            "Copied",
            "The complete operator guide has been copied to the clipboard.",
            parent=self,
        )

    def _save_guide(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save operator guide",
            defaultextension=".txt",
            initialfile="AI_Image_Frequency_Checker_Operator_Guide.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(help_content.full_text())
        except OSError as error:
            messagebox.showerror("Save Error", str(error), parent=self)
            return

        messagebox.showinfo(
            "Guide Saved", f"Saved to:\n\n{path}", parent=self)

    def _toggle_startup(self) -> None:
        if self._on_show_startup_change:
            self._on_show_startup_change(bool(self.show_on_start.get()))


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class ResidualViewer(tk.Tk):
    """Main application window."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()

        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1520x980")
        self.minsize(1140, 760)

        # --- state -------------------------------------------------------
        self.image_paths: List[str] = []
        self.image_index: int = -1
        self.current_path: Optional[str] = None
        self.result = None

        # The lens position, in pixel coordinates of the source image.
        self.zoom_center: Optional[Tuple[float, float]] = None

        # The full-resolution final panel, retained so the magnifier
        # lens can crop from the original data rather than a preview.
        self._panel4_full = None
        self._panel4_is_float = False

        # Named "settings", not "config". Tk provides a built-in
        # config() method, so assigning a dict to self.config would
        # shadow it and break self.configure(menu=...).
        self.settings = load_config()
        self._help_window: Optional[HelpWindow] = None

        # Readouts that mirror a control's value, as (label, variable,
        # format string) tuples.
        self._value_readouts: List[Tuple[ttk.Label, tk.Variable, str]] = []

        # Whether the advanced FFT options are expanded.
        self._advanced_expanded = bool(
            self.settings.get("fft_advanced_expanded", False))

        # Keeps analysis results in RAM so switching between images is
        # instant. The budget bounds how much memory is held.
        self.cache = ResultCache(
            max_bytes=300 * 1024 * 1024,
            max_entries=10,
        )

        self._work_queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._busy = False
        self._pending_recompute = False
        self._cancel_requested = False
        self._closing = False

        # True while the operator is dragging a slider. No analysis is
        # started until the drag ends.
        self._dragging = False

        # True when a parameter has changed since the last analysis ran.
        self._params_dirty = False

        # The handle of the pending debounced recompute, so that a new
        # change restarts the timer instead of firing on the first one.
        self._recompute_job: Optional[str] = None


        # Widgets that must be disabled while an analysis runs, and
        # the state each one returns to when it is enabled again.
        self._busy_widgets: List[Tuple[tk.Widget, str]] = []

        self._build_ui()
        self._apply_window_icon()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(80, self._drain_queue)
        self._update_readouts()

        self._log(f"{APP_NAME} v{__version__} ready.")
        self._log(
            "Three methods available: FFT Residual, ELA, and Noise Analysis."
        )
        self._log(
            "Reminder: a visible trace proves the file was PROCESSED. It "
            "does not by itself\nprove the file was AI-generated, nor which "
            "model produced it."
        )
        self._log(
            "Press the 'Read Before You Try' button for the full operator "
            "guide before interpreting results."
        )

        if initial_path:
            self.after(200, lambda: self._load_paths([initial_path]))

        if self.settings.get("show_help_on_start"):
            self.after(500, self._show_help)

    # ------------------------------------------------------------------
    # Numeric readouts
    # ------------------------------------------------------------------
    def _add_slider_row(
        self,
        parent: tk.Misc,
        row: int,
        label_text: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        formatter: str,
        attribute: Optional[str] = None,
    ) -> int:
        """
        Add a labelled slider with a live numeric readout.

        The readout is driven by a trace on the variable rather than by
        the slider's command callback, because a trace also fires when
        the value is set programmatically, for example by the reset
        button. The readout therefore always shows the true value.

        Returns the next free row.
        """
        ttk.Label(parent, text=label_text).grid(
            row=row, column=0, sticky="w", pady=1)

        slider = ttk.Scale(
            parent,
            from_=from_,
            to=to,
            variable=variable,
            command=self._on_param_change,
        )
        slider.grid(row=row, column=1, sticky="ew", pady=1)

        # The press and release events tell the application when a drag
        # starts and ends, so that no analysis is started while the
        # operator is still moving the slider.
        slider.bind("<ButtonPress-1>", self._on_slider_press)
        slider.bind("<ButtonRelease-1>", self._on_slider_release)

        readout = ttk.Label(
            parent,
            text=formatter.format(variable.get()),
            width=8,
            anchor="e",
        )
        readout.grid(row=row, column=2, sticky="e", padx=(6, 0), pady=1)

        self._value_readouts.append((readout, variable, formatter))
        variable.trace_add("write", self._on_value_trace)

        if attribute is not None:
            setattr(self, attribute, slider)

        self._register_busy_widget(slider)

        return row + 1

    def _update_readouts(self) -> None:
        """
        Refresh every numeric readout.

        Called once after the interface is built, and again whenever a
        watched variable changes.
        """
        for readout, variable, formatter in self._value_readouts:
            try:
                value = variable.get()
                readout.configure(text=formatter.format(value))
            except (tk.TclError, ValueError, TypeError):
                # The variable was momentarily empty or invalid.
                continue

    def _on_value_trace(self, *_args) -> None:
        """
        Called whenever any watched variable changes, whether from the
        operator, the reset button, or any other code path.
        """
        self._update_readouts()


    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------
    def _register_busy_widget(
        self,
        widget: tk.Widget,
        idle_state: str = "normal",
    ) -> None:
        """Register a widget that must be disabled during an analysis."""
        self._busy_widgets.append((widget, idle_state))

    def _set_busy(self, busy: bool) -> None:
        """
        Enable or disable the interface.

        Every widget returns to the exact state it was registered with,
        so a combobox comes back as "readonly" rather than "normal",
        and a checkbutton comes back as "normal" rather than "readonly".
        """
        self._busy = busy

        for widget, idle_state in self._busy_widgets:
            try:
                widget.configure(
                    state="disabled" if busy else idle_state)
            except tk.TclError:
                # The widget was destroyed. Nothing to do.
                pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)

        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=3)
        outer.rowconfigure(2, weight=1)

        self._build_menu()

        # ---------------- toolbar ----------------
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self.btn_open = ttk.Button(
            toolbar, text="Open Image(s)", command=self._open_images)
        self.btn_prev = ttk.Button(
            toolbar, text="<< Previous", command=lambda: self._step(-1))
        self.btn_next = ttk.Button(
            toolbar, text="Next >>", command=lambda: self._step(1))
        self.btn_save_res = ttk.Button(
            toolbar, text="Save Result", command=self._save_residual)
        self.btn_save_fig = ttk.Button(
            toolbar, text="Save Figure", command=self._save_figure)
        self.btn_reset = ttk.Button(
            toolbar, text="Reset Parameters", command=self._reset_params)
        self.btn_cancel = ttk.Button(
            toolbar, text="Cancel", command=self._cancel_analysis)

        self.style = ttk.Style(self)
        self.style.configure("Help.TButton", font=("Segoe UI", 9, "bold"))

        self.btn_help = ttk.Button(
            toolbar,
            text="Read Before You Try",
            style="Help.TButton",
            command=self._show_help,
        )

        self.btn_open.pack(side="left")
        self.btn_prev.pack(side="left", padx=(6, 0))
        self.btn_next.pack(side="left", padx=(6, 0))
        self.btn_save_res.pack(side="left", padx=(18, 0))
        self.btn_save_fig.pack(side="left", padx=(6, 0))
        self.btn_reset.pack(side="left", padx=(6, 0))
        self.btn_cancel.pack(side="left", padx=(6, 0))

        self.var_diagnostics = tk.BooleanVar(value=False)
        self.chk_diagnostics = ttk.Checkbutton(
            toolbar,
            text="Show spectrum diagnostics",
            variable=self.var_diagnostics,
            command=self._on_diagnostics_toggle,
        )
        self.chk_diagnostics.pack(side="left", padx=(18, 0))
        self._register_busy_widget(self.chk_diagnostics)

        self.lbl_image = ttk.Label(toolbar, text="No image loaded")
        self.lbl_image.pack(side="right")

        self.btn_help.pack(side="right", padx=(10, 0))

        for button in (self.btn_open, self.btn_prev, self.btn_next,
                       self.btn_save_res, self.btn_save_fig, self.btn_reset):
            self._register_busy_widget(button)

        # The cancel button is only useful while an analysis is running.
        self.btn_cancel.configure(state="disabled")

        # ---------------- control panel ----------------
        panel = ttk.LabelFrame(outer, text="Parameters", padding=8)
        panel.grid(row=1, column=0, sticky="nsw", padx=(0, 8))
        panel.columnconfigure(1, weight=1)

        # Variables
        self.var_mode = tk.StringVar(value=MODE_FFT)
        self.var_radius = tk.DoubleVar(value=0.08)
        self.var_feather = tk.DoubleVar(value=0.0)
        self.var_gain = tk.DoubleVar(value=1.0)
        self.var_clip = tk.DoubleVar(value=2.0)
        self.var_tiles = tk.IntVar(value=8)
        self.var_clahe = tk.BooleanVar(value=True)
        self.var_robust = tk.BooleanVar(value=True)
        self.var_real = tk.BooleanVar(value=False)
        self.var_cmap = tk.StringVar(value="gray")
        self.var_auto = tk.BooleanVar(value=True)

        self.var_ela_quality = tk.IntVar(value=75)
        self.var_ela_scale = tk.DoubleVar(value=10.0)

        self.var_noise_ksize = tk.IntVar(value=3)
        self.var_noise_amp = tk.DoubleVar(value=4.0)

        self.var_magnifier = tk.StringVar(value="Auto Contrast")
        self.var_opacity = tk.DoubleVar(value=1.0)
        self.var_zoom = tk.StringVar(value="1x")

        row = 0

        # --- method selector ---
        ttk.Label(
            panel, text="Analysis method",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        self.cmb_mode = ttk.Combobox(
            panel,
            textvariable=self.var_mode,
            values=MODE_CHOICES,
            state="readonly",
            width=28,
        )
        self.cmb_mode.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(2, 10))
        self.cmb_mode.bind("<<ComboboxSelected>>", self._on_mode_change)
        self._register_busy_widget(self.cmb_mode, "readonly")
        row += 1

        # --- the three method frames share one grid cell ---
        self.frame_fft = ttk.LabelFrame(
            panel, text="FFT Residual options", padding=6)
        self.frame_fft.grid(
            row=row, column=0, columnspan=3, sticky="ew")
        self.frame_fft.columnconfigure(1, weight=1)

        self.frame_ela = ttk.LabelFrame(
            panel, text="ELA options", padding=6)
        self.frame_ela.grid(
            row=row, column=0, columnspan=3, sticky="ew")
        self.frame_ela.columnconfigure(1, weight=1)

        self.frame_noise = ttk.LabelFrame(
            panel, text="Noise Analysis options", padding=6)
        self.frame_noise.grid(
            row=row, column=0, columnspan=3, sticky="ew")
        self.frame_noise.columnconfigure(1, weight=1)

        self._build_fft_controls(self.frame_fft)
        self._build_ela_controls(self.frame_ela)
        self._build_noise_controls(self.frame_noise)

        row += 1

        # --- shared display frame ---
        self.frame_shared = ttk.LabelFrame(
            panel, text="Display options", padding=6)
        self.frame_shared.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.frame_shared.columnconfigure(1, weight=1)

        srow = 0

        ttk.Label(self.frame_shared, text="Magnifier enhancement").grid(
            row=srow, column=0, sticky="w", pady=1)
        self.cmb_magnifier = ttk.Combobox(
            self.frame_shared,
            textvariable=self.var_magnifier,
            values=[label for _key, label in MAGNIFIER_CHOICES],
            state="readonly",
            width=22,
        )
        self.cmb_magnifier.grid(
            row=srow, column=1, sticky="ew", pady=1)
        self.cmb_magnifier.bind(
            "<<ComboboxSelected>>", self._on_param_change)
        self._register_busy_widget(self.cmb_magnifier, "readonly")
        srow += 1

        srow = self._add_slider_row(
            self.frame_shared,
            srow,
            "Opacity",
            self.var_opacity,
            0.0,
            1.0,
            "{:.2f}",
            attribute="sld_opacity",
        )

        ttk.Label(self.frame_shared, text="Zoom lens").grid(
            row=srow, column=0, sticky="w", pady=1)
        self.cmb_zoom = ttk.Combobox(
            self.frame_shared, textvariable=self.var_zoom,
            values=ZOOM_LEVELS, state="readonly", width=9)
        self.cmb_zoom.grid(row=srow, column=1, sticky="ew", pady=1)
        self.cmb_zoom.bind("<<ComboboxSelected>>", self._on_zoom_change)
        self._register_busy_widget(self.cmb_zoom, "readonly")
        srow += 1

        ttk.Label(
            self.frame_shared,
            text="Click the result panel to move the lens. "
                 "Scroll to zoom.",
            wraplength=250, foreground="#555555",
        ).grid(row=srow, column=0, columnspan=3, sticky="w", pady=(4, 0))

        row += 1

        self.chk_auto = ttk.Checkbutton(
            panel, text="Auto-recompute on change",
            variable=self.var_auto, command=self._on_auto_toggle)
        self.chk_auto.grid(row=row, column=0, columnspan=3, sticky="w")
        self._register_busy_widget(self.chk_auto)
        row += 1

        self.btn_recompute = ttk.Button(
            panel, text="Recompute Now", command=self._trigger_recompute)
        self.btn_recompute.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self._register_busy_widget(self.btn_recompute)
        row += 1

        # The hotkey is written under the button rather than inside it,
        # because a ttk.Button does not render a newline in its text.
        ttk.Label(
            panel,
            text="(hotkey: Spacebar)",
            foreground="#555555",
        ).grid(row=row, column=0, columnspan=3, sticky="ew")
        row += 1


        self._update_mode_visibility()

        # ---------------- figure ----------------
        canvas_frame = ttk.Frame(outer)
        canvas_frame.grid(row=1, column=1, sticky="nsew")

        self.figure = Figure(
            figsize=(12, 9), dpi=100, layout="constrained")

        # The default layout is two panels: the original image and the
        # final result. The spectrum diagnostics are opt-in.
        self._build_axes(show_diagnostics=False)

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status = ttk.Label(
            canvas_frame, anchor="w", text="Idle", wraplength=1200)
        self.status.pack(fill="x", pady=(4, 0))

        # ---------------- log ----------------
        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame, height=7, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        self.log_text.tag_configure("error", foreground="#b00020")
        self.log_text.tag_configure("good", foreground="#1f6b1f")

        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.bind("<Left>", lambda _e: self._step(-1))
        self.bind("<Right>", lambda _e: self._step(1))
        self.bind("<F1>", lambda _e: self._show_help())

        # Safety net. If a drag ends outside the slider the widget never
        # sees the release, which would leave the drag flag set and
        # stop any further recompute from being scheduled.
        self.bind("<ButtonRelease-1>", self._on_global_release, add="+")

        # The space bar is the shortcut for Recompute Now. A binding on
        # the toplevel applies to every widget in the main window, but
        # not to the separate guide window.
        self.bind("<KeyPress-space>", self._on_space, add="+")


        # Magnifier lens interaction on the result panel.
        self.canvas.mpl_connect("button_press_event", self._on_panel_click)
        self.canvas.mpl_connect("scroll_event", self._on_panel_scroll)

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    def _build_axes(self, show_diagnostics: bool) -> None:
        """
        Create the plot axes.

        The default layout is a single row of two panels: the original
        image and the final result. With diagnostics enabled the
        frequency-domain views are added back in a two by two grid.
        """
        self.figure.clear()

        if show_diagnostics:
            grid = self.figure.subplots(2, 2)
            self.ax_original = grid[0][0]
            self.ax_spectrum = grid[0][1]
            self.ax_masked = grid[1][0]
            self.ax_result = grid[1][1]
        else:
            grid = self.figure.subplots(1, 2)
            self.ax_original = grid[0]
            self.ax_spectrum = None
            self.ax_masked = None
            self.ax_result = grid[1]

        self.figure.suptitle(
            "AI Image Frequency Checker",
            fontsize=12,
            fontweight="bold",
        )

    def _on_diagnostics_toggle(self) -> None:
        """Switch between the two panel and the four panel layout."""
        self._build_axes(bool(self.var_diagnostics.get()))

        if self.result is not None:
            self._render(self.result)
        else:
            self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Control builders
    # ------------------------------------------------------------------
    def _build_fft_controls(self, parent) -> None:
        """
        Build the FFT controls.

        The high-pass radius is the primary control and stays visible.
        Everything else lives in a collapsible section so a new operator
        sees a single slider rather than a wall of options.
        """
        row = 0

        ttk.Label(
            parent, text="High-pass radius",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        row = self._add_slider_row(
            parent,
            row,
            "Fraction of min(H,W)",
            self.var_radius,
            0.005,
            0.45,
            "{:.3f}",
            attribute="sld_radius",
        )

        self.lbl_radius = ttk.Label(parent, text="")
        self.lbl_radius.grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # --- collapsible advanced section ---
        self.section_fft_advanced = CollapsibleSection(
            parent,
            "Advanced FFT options",
            collapsed=not self._advanced_expanded,
            on_toggle=self._on_advanced_toggle,
        )
        self.section_fft_advanced.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        body = self.section_fft_advanced.body
        body.columnconfigure(1, weight=1)

        brow = 0

        brow = self._add_slider_row(
            body,
            brow,
            "Feather width (px)",
            self.var_feather,
            0,
            60,
            "{:.0f}",
            attribute="sld_feather",
        )

        brow = self._add_slider_row(
            body,
            brow,
            "Linear gain",
            self.var_gain,
            0.1,
            8.0,
            "{:.2f}",
            attribute="sld_gain",
        )

        self.chk_clahe = ttk.Checkbutton(
            body, text="CLAHE contrast enhancement",
            variable=self.var_clahe, command=self._on_param_change)
        self.chk_clahe.grid(
            row=brow, column=0, columnspan=3, sticky="w", pady=1)
        self._register_busy_widget(self.chk_clahe)
        brow += 1

        brow = self._add_slider_row(
            body,
            brow,
            "CLAHE clip limit",
            self.var_clip,
            1.0,
            8.0,
            "{:.2f}",
            attribute="sld_clip",
        )

        brow = self._add_slider_row(
            body,
            brow,
            "CLAHE tiles",
            self.var_tiles,
            2,
            16,
            "{:.0f}",
            attribute="sld_tiles",
        )

        self.chk_robust = ttk.Checkbutton(
            body, text="Robust percentile stretch",
            variable=self.var_robust, command=self._on_param_change)
        self.chk_robust.grid(
            row=brow, column=0, columnspan=3, sticky="w", pady=1)
        self._register_busy_widget(self.chk_robust)
        brow += 1

        self.chk_real = ttk.Checkbutton(
            body, text="Use signed real part (not abs)",
            variable=self.var_real, command=self._on_param_change)
        self.chk_real.grid(
            row=brow, column=0, columnspan=3, sticky="w", pady=1)
        self._register_busy_widget(self.chk_real)
        brow += 1

        ttk.Label(body, text="Residual colormap").grid(
            row=brow, column=0, sticky="w", pady=1)
        self.cmb_cmap = ttk.Combobox(
            body, textvariable=self.var_cmap,
            values=("gray", "viridis", "magma", "inferno",
                    "hot", "jet", "turbo"),
            state="readonly", width=12)
        self.cmb_cmap.grid(row=brow, column=1, sticky="ew", pady=1)
        self.cmb_cmap.bind("<<ComboboxSelected>>", self._on_param_change)
        self._register_busy_widget(self.cmb_cmap, "readonly")

    def _build_ela_controls(self, parent) -> None:
        row = 0

        ttk.Label(parent, text="JPEG quality").grid(
            row=row, column=0, sticky="w", pady=1)
        self.spin_ela_quality = ttk.Spinbox(
            parent, from_=1, to=100, increment=1,
            textvariable=self.var_ela_quality, width=9,
            command=self._on_param_change)
        self.spin_ela_quality.grid(row=row, column=1, sticky="ew", pady=1)
        self.spin_ela_quality.bind("<Return>", self._on_param_change)
        self.spin_ela_quality.bind("<FocusOut>", self._on_param_change)
        self._register_busy_widget(self.spin_ela_quality)
        row += 1

        ttk.Label(parent, text="Error scale").grid(
            row=row, column=0, sticky="w", pady=1)
        self.spin_ela_scale = ttk.Spinbox(
            parent, from_=0.5, to=50.0, increment=0.5,
            textvariable=self.var_ela_scale, width=9,
            command=self._on_param_change)
        self.spin_ela_scale.grid(row=row, column=1, sticky="ew", pady=1)
        self.spin_ela_scale.bind("<Return>", self._on_param_change)
        self.spin_ela_scale.bind("<FocusOut>", self._on_param_change)
        self._register_busy_widget(self.spin_ela_scale)
        row += 1

        ttk.Label(
            parent,
            text="Match the original JPEG quality for a clean background.",
            wraplength=250, foreground="#555555",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_noise_controls(self, parent) -> None:
        row = 0

        ttk.Label(parent, text="Median filter size").grid(
            row=row, column=0, sticky="w", pady=1)
        self.cmb_noise_ksize = ttk.Combobox(
            parent, textvariable=self.var_noise_ksize,
            values=("3", "5"), state="readonly", width=9)
        self.cmb_noise_ksize.grid(row=row, column=1, sticky="ew", pady=1)
        self.cmb_noise_ksize.bind(
            "<<ComboboxSelected>>", self._on_param_change)
        self._register_busy_widget(self.cmb_noise_ksize, "readonly")
        row += 1

        ttk.Label(parent, text="Noise amplitude").grid(
            row=row, column=0, sticky="w", pady=1)
        self.spin_noise_amp = ttk.Spinbox(
            parent, from_=0.5, to=30.0, increment=0.5,
            textvariable=self.var_noise_amp, width=9,
            command=self._on_param_change)
        self.spin_noise_amp.grid(row=row, column=1, sticky="ew", pady=1)
        self.spin_noise_amp.bind("<Return>", self._on_param_change)
        self.spin_noise_amp.bind("<FocusOut>", self._on_param_change)
        self._register_busy_widget(self.spin_noise_amp)
        row += 1

        ttk.Label(
            parent,
            text="Larger amplitude makes the noise brighter.",
            wraplength=250, foreground="#555555",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Open Image(s)...", accelerator="Ctrl+O",
            command=self._open_images)
        file_menu.add_command(
            label="Save Result...", accelerator="Ctrl+S",
            command=self._save_residual)
        file_menu.add_command(
            label="Save Figure...", command=self._save_figure)
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit", accelerator="Alt+F4", command=self._on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(
            label="Read Before You Try...", accelerator="F1",
            command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menu_bar)

        self.bind_all("<Control-o>", lambda _e: self._open_images())
        self.bind_all("<Control-s>", lambda _e: self._save_residual())

    def _apply_window_icon(self) -> None:
        icon = resource_path("icon.ico")
        if not os.path.isfile(icon):
            return
        try:
            self.iconbitmap(default=icon)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def _show_help(self) -> None:
        if self._help_window is not None:
            try:
                if self._help_window.winfo_exists():
                    self._help_window.deiconify()
                    self._help_window.lift()
                    return
            except tk.TclError:
                self._help_window = None

        self._help_window = HelpWindow(
            self,
            on_show_startup_change=self._set_show_help_on_start,
            show_on_start=bool(
                self.settings.get("show_help_on_start", False)),
            on_close=self._on_help_closed,
        )

    def _on_help_closed(self) -> None:
        """Called when the guide window is closed."""
        self._help_window = None

    def _set_show_help_on_start(self, enabled: bool) -> None:
        self.settings["show_help_on_start"] = bool(enabled)
        save_config(self.settings)
        self._log(
            f"Operator guide will "
            f"{'be shown' if enabled else 'not be shown'} at startup."
        )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            f"{APP_NAME}\n"
            f"Version {__version__}\n\n"
            "Three analysis methods:\n"
            "  FFT Residual   - 2D FFT with a circular high-pass filter\n"
            "  ELA            - error level analysis against a JPEG "
            "recompression\n"
            "  Noise Analysis - reverse denoising, isolates the sensor "
            "noise\n\n"
            "This is a visual triage aid. It does not verify provenance "
            "and it is\nnot a forensic-grade tool.\n\n"
            "Author: Raysky-sys",
            parent=self,
        )

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    def _on_mode_change(self, _event=None) -> None:
        self._update_mode_visibility()
        self._log(f"Analysis method: {self.var_mode.get()}")
        self._trigger_recompute()

    def _update_mode_visibility(self) -> None:
        """
        Show only the parameter group that belongs to the active method.

        All three frames occupy the same grid cell, so switching methods
        does not change the height of the panel and the window does not
        jump around.
        """
        mode = self.var_mode.get()

        self.frame_fft.grid_remove()
        self.frame_ela.grid_remove()
        self.frame_noise.grid_remove()

        if mode == MODE_FFT:
            self.frame_fft.grid()
        elif mode == MODE_ELA_LABEL:
            self.frame_ela.grid()
        else:
            self.frame_noise.grid()

    def _magnifier_key(self) -> str:
        return magnifier_key_from_label(self.var_magnifier.get())

    def _on_advanced_toggle(self, expanded: bool) -> None:
        """Remember whether the advanced FFT options are open."""
        self._advanced_expanded = bool(expanded)
        self.settings["fft_advanced_expanded"] = bool(expanded)
        save_config(self.settings)

    def _on_space(self, event=None):
        """
        The space bar is the shortcut for Recompute Now.

        The event is passed through while the operator is typing in a
        text field, and while a button has the focus, because the
        keyboard already activates a focused button with the space bar.
        """
        focused = self.focus_get()

        if focused is not None:
            widget_class = focused.winfo_class()

            if widget_class in (
                "TEntry", "Entry", "Text", "TCombobox", "Combobox"
            ):
                # Let the space bar be typed.
                return None

            if widget_class in ("TButton", "Button"):
                # The keyboard already activates a focused button.
                return None

        self._trigger_recompute()
        return "break"


    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _read_params(self):
        """Return the settings for the active method."""
        mode = self.var_mode.get()

        magnifier = self._magnifier_key()
        opacity = float(self.var_opacity.get())

        if mode == MODE_FFT:
            return (
                MODE_FFT,
                Params(
                    radius_frac=float(self.var_radius.get()),
                    feather_px=float(self.var_feather.get()),
                    gain=float(self.var_gain.get()),
                    use_clahe=bool(self.var_clahe.get()),
                    clahe_clip=float(self.var_clip.get()),
                    clahe_tiles=int(self.var_tiles.get()),
                    robust_stretch=bool(self.var_robust.get()),
                    use_real_part=bool(self.var_real.get()),
                    magnifier=magnifier,
                    opacity=opacity,
                ),
                None,
            )

        forensics = ForensicsParams(
            mode=MODE_ELA if mode == MODE_ELA_LABEL else MODE_NOISE,
            ela_quality=int(self.var_ela_quality.get()),
            ela_error_scale=float(self.var_ela_scale.get()),
            noise_ksize=int(self.var_noise_ksize.get()),
            noise_amplitude=float(self.var_noise_amp.get()),
            magnifier=magnifier,
            opacity=opacity,
        )

        return (mode, None, forensics)

    def _reset_params(self) -> None:
        self.var_mode.set(MODE_FFT)
        self.var_radius.set(0.08)
        self.var_feather.set(0.0)
        self.var_gain.set(1.0)
        self.var_clip.set(2.0)
        self.var_tiles.set(8)
        self.var_clahe.set(True)
        self.var_robust.set(True)
        self.var_real.set(False)
        self.var_cmap.set("gray")
        self.var_ela_quality.set(75)
        self.var_ela_scale.set(10.0)
        self.var_noise_ksize.set(3)
        self.var_noise_amp.set(4.0)
        self.var_magnifier.set("Auto Contrast")
        self.var_opacity.set(1.0)
        self.var_zoom.set("1x")

        self._update_mode_visibility()
        self._log("Parameters reset to defaults.")
        self._trigger_recompute()

    def _on_param_change(self, _event=None) -> None:
        """
        A parameter has changed.

        While a slider is being dragged the value updates live but no
        analysis is started. The analysis begins when the operator
        releases the slider, so one drag produces one result instead of
        one result per pixel of movement.
        """
        was_dirty = self._params_dirty
        self._params_dirty = True

        if self._dragging:
            self._update_radius_label()
            return

        if self.var_auto.get():
            self._schedule_recompute()
        else:
            if not was_dirty:
                self._log(
                    "Parameter changed. Automatic recompute is off, "
                    "press Recompute Now (Space) to apply it."
                )
            self._update_radius_label()

    def _on_auto_toggle(self) -> None:
        """
        Turn the automatic recompute on or off.

        Toggling this never starts an analysis on its own. If a
        parameter was changed while automatic recompute was off, the
        analysis is brought up to date once, because the displayed
        result would otherwise no longer match the settings.
        """
        enabled = bool(self.var_auto.get())

        self._log(
            f"Auto-recompute on change: {'ON' if enabled else 'OFF'}"
        )

        if enabled and self._params_dirty:
            self._trigger_recompute()


    def _on_slider_press(self, _event=None) -> None:
        """The operator has started dragging a slider."""
        self._dragging = True

    def _on_slider_release(self, _event=None) -> None:
        """The operator has finished dragging a slider."""
        if not self._dragging:
            return

        self._dragging = False

        if self.var_auto.get() and self._params_dirty:
            self._schedule_recompute()

    def _on_global_release(self, _event=None) -> None:
        """
        Safety net for a drag that ended outside the slider.

        The slider never sees the release event in that case, which
        would leave the drag flag set and stop any further recompute
        from ever being scheduled.
        """
        if not self._dragging:
            return

        self._dragging = False

        if self.var_auto.get() and self._params_dirty:
            self._schedule_recompute()


    def _update_radius_label(self) -> None:
        self.lbl_radius.configure(
            text=f"Blocked disc: {float(self.var_radius.get()):.3f} "
                 f"of the short side")

    # ------------------------------------------------------------------
    # Cache support
    # ------------------------------------------------------------------
    @staticmethod
    def _signature_for(mode, fft_params, forensics_params) -> tuple:
        """
        A hashable description of the settings used for an analysis.

        Two results are comparable when their signatures match. The
        colour map and the spectrum diagnostics are display options
        and are deliberately not included, because they do not change
        the analysis.
        """
        if mode == MODE_FFT:
            return (
                mode,
                round(fft_params.radius_frac, 6),
                round(fft_params.feather_px, 3),
                round(fft_params.gain, 4),
                bool(fft_params.use_clahe),
                round(fft_params.clahe_clip, 4),
                int(fft_params.clahe_tiles),
                bool(fft_params.robust_stretch),
                bool(fft_params.use_real_part),
                fft_params.magnifier,
                round(fft_params.opacity, 4),
            )

        return (
            mode,
            int(forensics_params.ela_quality),
            round(forensics_params.ela_error_scale, 4),
            int(forensics_params.noise_ksize),
            round(forensics_params.noise_amplitude, 4),
            forensics_params.magnifier,
            round(forensics_params.opacity, 4),
        )

    def _current_signature(self) -> tuple:
        """The signature of the settings currently selected."""
        mode, fft_params, forensics_params = self._read_params()
        return self._signature_for(mode, fft_params, forensics_params)

    def _update_status(self, cached) -> None:
        """Refresh the status bar for a displayed result."""
        self.status.configure(
            text=f"Done in {cached.elapsed_ms:.0f} ms | "
                 f"{cached.width}x{cached.height} px | {cached.detail} | "
                 f"{self.cache.describe()}"
        )


    def _on_zoom_change(self, _event=None) -> None:
        # We removed self.zoom_center = None
        # Now the focal point persists when you scroll the wheel
        if self.result is not None:
            self._render(self.result)

    # ------------------------------------------------------------------
    # Magnifier lens
    # ------------------------------------------------------------------
    def _zoom_factor(self) -> float:
        try:
            return float(str(self.var_zoom.get()).rstrip("x"))
        except ValueError:
            return 1.0

    def _on_panel_click(self, event) -> None:
        """Move the lens to the clicked point on the result panel."""
        if event.inaxes is not self.ax_result:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.zoom_center = (float(event.xdata), float(event.ydata))

        if self.result is not None:
            self._render(self.result)

    def _on_panel_scroll(self, event) -> None:
        """Change the zoom level with the mouse wheel."""
        if event.inaxes is not self.ax_result:
            return

        try:
            index = ZOOM_LEVELS.index(self.var_zoom.get())
        except ValueError:
            index = 0

        if event.button == "up":
            index = min(index + 1, len(ZOOM_LEVELS) - 1)
        else:
            index = max(index - 1, 0)

        self.var_zoom.set(ZOOM_LEVELS[index])
        self._on_zoom_change()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _open_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more images",
            filetypes=[
                ("Image files",
                 "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._load_paths(list(paths))

    def _load_paths(self, paths: List[str]) -> None:
        self.image_paths = paths
        self.image_index = 0
        self.zoom_center = None
        self._load_current()

    def _step(self, delta: int) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index + delta) % len(self.image_paths)
        self.zoom_center = None
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

        self.zoom_center = None

        # Show the cached result straight away. There is no waiting and
        # no flicker, even on a large photograph.
        cached = self.cache.get(path)
        self.cache.protect(path)

        if cached is not None:
            self.result = cached
            self._render(cached)
            self._update_status(cached)

            if cached.signature != self._current_signature():
                if self.var_auto.get():
                    self._log(
                        "Cached result is out of date, refreshing in the "
                        "background..."
                    )
                    self._trigger_recompute()
                else:
                    self._log(
                        "Cached result is out of date. Automatic recompute "
                        "is off, press Recompute Now (Space) to refresh it."
                    )
            return

        # Nothing cached for this image yet, so it has to be analysed.
        self._trigger_recompute()


    # ------------------------------------------------------------------
    # Background analysis
    # ------------------------------------------------------------------
    def _schedule_recompute(self) -> None:
        """
        Ask for a recompute, restarting the timer on every change.

        The previous timer is cancelled first, so the analysis runs a
        short moment after the LAST change rather than a fixed time
        after the first one. That is what makes a slider drag produce a
        single result instead of a queue of them.
        """
        if self._recompute_job is not None:
            try:
                self.after_cancel(self._recompute_job)
            except tk.TclError:
                pass
            self._recompute_job = None

        self._pending_recompute = True
        self._recompute_job = self.after(250, self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._recompute_job = None
        self._trigger_recompute()


    def _trigger_recompute(self) -> None:
        if self.current_path is None:
            return

        if self._busy:
            # Remember that another run is needed. It is started as soon
            # as the running job finishes.
            self._pending_recompute = True
            return

        # The analysis is starting now, so the current settings are
        # about to be reflected in the result.
        self._pending_recompute = False
        self._params_dirty = False

        mode, fft_params, forensics_params = self._read_params()
        signature = self._signature_for(
            mode, fft_params, forensics_params)
        path = self.current_path



        self._busy = True
        self._cancel_requested = False
        self._set_busy(True)
        self.btn_cancel.configure(state="normal")
        self.status.configure(
            text=f"Analysing {os.path.basename(path)} ...")

        self._worker = threading.Thread(
            target=self._analysis_worker,
            args=(path, mode, fft_params, forensics_params, signature),
            daemon=True,
        )
        self._worker.start()

    def _cancel_analysis(self) -> None:
        """Ask the running analysis to stop at the next safe point."""
        if not self._busy:
            return

        self._cancel_requested = True
        self._log("Cancellation requested...")
        self.status.configure(text="Cancelling...")

    def _analysis_worker(
        self,
        path,
        mode,
        fft_params,
        forensics_params,
        signature,
    ):
        """
        Run the analysis off the main thread.

        This function must never touch a Tk widget. It communicates
        with the GUI only through the queue. The result is pruned into a
        display-ready cache entry here, so the large float64
        intermediates are released as soon as they are no longer needed.
        """

        def progress(message):
            self._work_queue.put(("progress", message))

        def should_cancel():
            return self._cancel_requested

        try:
            if mode == MODE_FFT:
                progress("Loading the image...")
                gray = load_gray(path)

                if should_cancel():
                    raise AnalysisCancelled("Cancelled by the operator.")

                progress("Computing the FFT...")
                result = analyze_image(
                    gray, params=fft_params, path=path)
            else:
                progress("Loading the image...")
                colour = load_color(path)

                if should_cancel():
                    raise AnalysisCancelled("Cancelled by the operator.")

                if mode == MODE_ELA_LABEL:
                    result = analyze_ela(
                        colour,
                        params=forensics_params,
                        progress=progress,
                        should_cancel=should_cancel,
                        path=path,
                    )
                else:
                    result = analyze_noise(
                        colour,
                        params=forensics_params,
                        progress=progress,
                        should_cancel=should_cancel,
                        path=path,
                    )

            progress("Preparing the cached result...")
            cached = build_cache_entry(result, signature)

            self._work_queue.put(("ok", (cached, path)))

        except AnalysisCancelled as error:
            self._work_queue.put(("cancelled", str(error)))

        except Exception as error:  # noqa: BLE001
            self._work_queue.put(
                ("error", f"{error}\n{traceback.format_exc()}"))


    def _drain_queue(self) -> None:
        """
        Process the results posted by the worker thread.

        The whole body is guarded. An exception here must never stop
        the polling loop, because that would leave the interface
        permanently unresponsive with no explanation.
        """
        try:
            while True:
                kind, payload = self._work_queue.get_nowait()

                if kind == "ok":
                    cached, path = payload

                    # Always cache it, even if the operator has moved on
                    # to another image in the meantime.
                    self.cache.put(cached)

                    if path == self.current_path:
                        self.cache.protect(path)
                        self.result = cached
                        self._render(cached)
                        self._update_status(cached)

                        self._log(
                            f"Analysis complete: "
                            f"{os.path.basename(path)} | "
                            f"{cached.width}x{cached.height} px | "
                            f"{cached.detail} | "
                            f"{cached.elapsed_ms:.0f} ms", tag="good")
                    else:
                        self._log(
                            f"Background analysis of "
                            f"{os.path.basename(path)} finished and cached."
                        )

                    self._finish_job()


                elif kind == "progress":
                    self.status.configure(text=str(payload))

                elif kind == "cancelled":
                    self._log(str(payload))
                    self.status.configure(text="Cancelled")
                    self._finish_job()

                elif kind == "error":
                    # Log the error rather than opening a modal dialog.
                    # A dialog raised from this callback can land behind
                    # the main window and look exactly like a frozen
                    # application.
                    self._log(f"[!] ERROR: {payload}", tag="error")
                    self.status.configure(
                        text="Analysis failed. See the log for details.")
                    self._finish_job()

        except queue.Empty:
            pass

        except Exception as error:  # noqa: BLE001
            self._log(
                f"[!] Internal error while processing results: {error}",
                tag="error")
            traceback.print_exc()
            self._finish_job()

        finally:
            # Always reschedule, even if something above raised.
            if not self._closing:
                try:
                    self.after(80, self._drain_queue)
                except tk.TclError:
                    pass

    def _finish_job(self) -> None:
        """Release the interface and start any queued recompute."""
        self._set_busy(False)
        self.btn_cancel.configure(state="disabled")

        if self._pending_recompute:
            self._pending_recompute = False
            self.after(60, self._trigger_recompute)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self, cached) -> None:
        """Draw a cached result."""
        for axis in (self.ax_original, self.ax_spectrum,
                     self.ax_masked, self.ax_result):
            if axis is not None:
                axis.clear()

        try:
            self._render_panels(cached)
        except Exception as error:  # noqa: BLE001
            self._log(f"[!] Render error: {error}", tag="error")
            traceback.print_exc()

        self.canvas.draw_idle()

    def _render_panels(self, cached) -> None:
        """Draw the panels, hiding the diagnostics when they are off."""
        self._draw_panel(self.ax_original, cached.panels[0])
        self.ax_original.set_title(
            self._title_original(cached), fontsize=10, pad=7)
        self.ax_original.set_axis_off()

        if self.ax_spectrum is not None:
            self._draw_panel(self.ax_spectrum, cached.panels[1])
            self.ax_spectrum.set_title(
                self._title_spectrum(cached), fontsize=10, pad=7)
            self.ax_spectrum.set_axis_off()

        if self.ax_masked is not None:
            self._draw_panel(self.ax_masked, cached.panels[2])
            self.ax_masked.set_title(
                self._title_masked(cached), fontsize=10, pad=7)
            self.ax_masked.set_axis_off()

        self._draw_result_panel(self.ax_result, cached)
        self.ax_result.set_title(
            self._title_result(cached), fontsize=10, pad=7)
        self.ax_result.set_axis_off()

    def _draw_panel(self, axis, spec) -> None:
        """Draw one panel using the display kind it was stored with."""
        data, kind = spec

        if kind == KIND_GRAY:
            axis.imshow(
                data, cmap="gray", vmin=0, vmax=255,
                interpolation="nearest")
        elif kind == KIND_FFT:
            axis.imshow(
                data, cmap=self.var_cmap.get(), vmin=0, vmax=255,
                interpolation="nearest")
        elif kind == KIND_MAGMA:
            axis.imshow(
                data, cmap="magma", vmin=0, vmax=255,
                interpolation="nearest")
        else:
            axis.imshow(data, interpolation="nearest")

    def _title_original(self, cached) -> str:
        if cached.mode == RESULT_MODE_FFT:
            return "Original grayscale"
        return "Original"

    def _title_spectrum(self, cached) -> str:
        if cached.mode == MODE_ELA:
            return "Recompressed JPEG"
        if cached.mode == MODE_NOISE:
            return "Median filtered (denoised)"
        return "FFT magnitude (log scale)"

    def _title_masked(self, cached) -> str:
        if cached.mode == MODE_ELA:
            return "Error map (amplified)"
        if cached.mode == MODE_NOISE:
            return "Noise map (amplified)"
        return "Masked FFT"

    def _title_result(self, cached) -> str:
        if cached.mode == RESULT_MODE_FFT:
            base = "Amplified high-frequency residual"
        elif cached.mode == MODE_ELA:
            base = "ELA result"
        else:
            base = "Noise result"

        opacity = float(self.var_opacity.get())

        if opacity < 0.999:
            base += f"\n(blended with the original at {opacity:.2f})"

        return base

    def _draw_result_panel(self, axis, cached) -> None:
        """
        Display the result panel, optionally through the magnifier lens.

        The lens crops from the full resolution data, so the operator
        always inspects real pixels rather than an enlarged preview.
        """
        zoom = self._zoom_factor()
        full = cached.full_result

        if full is None or zoom <= 1.001:
            self._draw_panel(axis, cached.panels[3])
            return

        height, width = full.shape[:2]

        if self.zoom_center is None:
            centre_x = width / 2.0
            centre_y = height / 2.0
        else:
            centre_x, centre_y = self.zoom_center

        half_width = width / (2.0 * zoom)
        half_height = height / (2.0 * zoom)

        x0 = int(max(0, min(width - 2 * half_width,
                            centre_x - half_width)))
        y0 = int(max(0, min(height - 2 * half_height,
                            centre_y - half_height)))
        x1 = int(min(width, x0 + 2 * half_width))
        y1 = int(min(height, y0 + 2 * half_height))

        crop = full[y0:y1, x0:x1]

        if cached.full_kind == KIND_FFT:
            axis.imshow(
                crop,
                cmap=self.var_cmap.get(),
                vmin=0,
                vmax=255,
                interpolation="nearest",
                extent=(x0, x1, y1, y0),
            )
        else:
            axis.imshow(
                crop,
                interpolation="nearest",
                extent=(x0, x1, y1, y0),
            )

        # A crosshair marks the centre of the lens.
        axis.axvline(centre_x, color="cyan", linewidth=0.7, linestyle="--")
        axis.axhline(centre_y, color="cyan", linewidth=0.7, linestyle="--")


    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _save_residual(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return

        base = os.path.splitext(os.path.basename(self.result.path))[0]
        image = self.result.full_result

        if image is None:
            messagebox.showerror(
                "Save Error",
                "No full resolution result is available for this image.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save result image", defaultextension=".png",
            initialfile=f"{base}_result.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return

        try:
            success, buffer = cv2.imencode(
                ".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

            if not success:
                raise IOError("Could not encode the PNG file.")

            buffer.tofile(path)
            self._log(f"Result saved: {path}")
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Save Error", str(error))


        path = filedialog.asksaveasfilename(
            title="Save result image", defaultextension=".png",
            initialfile=f"{base}_result.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return

        try:
            if self.result.mode == RESULT_MODE_FFT:
                save_gray(path, image)
            else:
                success, buffer = cv2.imencode(
                    ".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                if not success:
                    raise IOError("Could not encode the PNG file.")
                buffer.tofile(path)

            self._log(f"Result saved: {path}")
        except Exception as error:  # noqa: BLE001
            message_
            messagebox.showerror("Save Error", str(error))

    def _save_figure(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return

        base = os.path.splitext(os.path.basename(self.result.path))[0]

        path = filedialog.asksaveasfilename(
            title="Save figure", defaultextension=".png",
            initialfile=f"{base}_analysis.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return

        try:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved: {path}")
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Save Error", str(error))

    def _log(self, message: str, tag: str = "") -> None:
        self.log_text.configure(state="normal")
        start = self.log_text.index("end-1c")
        self.log_text.insert(tk.END, message.rstrip() + "\n", tag)
        end = self.log_text.index("end-1c")

        if tag == "error":
            self.log_text.tag_add("error", start, end)

        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        self._closing = True

        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno(
                    "Analysis running",
                    "An analysis is still running. Close anyway?"):
                self._closing = False
                return
            self._cancel_requested = True

        if self._help_window is not None:
            try:
                self._help_window.grab_release()
                self._help_window.destroy()
            except tk.TclError:
                pass

        self.destroy()


def main() -> None:
    install_crash_handler()

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
