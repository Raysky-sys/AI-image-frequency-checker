"""
selfcheck.py
=============

Pre-build validation for the AI Image Frequency Checker.

Runs before PyInstaller so a broken project is caught at the source,
rather than producing an executable that fails on the operator's
machine.

Verifies that:

    1. The third-party packages are importable.
    2. Every required source file exists.
    3. Every required module imports cleanly.
    4. The operator guide has all nine sections, and none is empty.
    5. The core analysis functions work on synthetic data.
    6. All three result types expose the same public attributes.
    7. The guide text styles are supported by this Tk build.
    8. The GUI module compiles without syntax errors.

Run directly:
    python selfcheck.py

Exit code 0 means everything is good.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = [
    "AI_image_frequency_checker.py",
    "residual_core.py",
    "forensic_tools.py",
    "image_ops.py",
    "analysis_cache.py",
    "help_content.py",
    "make_version_info.py",
]


REQUIRED_SECTIONS = 9

REQUIRED_PACKAGES = [
    "numpy",
    "cv2",
    "matplotlib",
    "tkinter",
]

#: Attributes the GUI reads from every result object. If a field is
#: added to one result type it must be added to all of them.
REQUIRED_RESULT_ATTRIBUTES = ("path", "mode", "detail", "elapsed_ms")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fail(message):
    """Print a failure and exit with a non-zero code."""
    print(f"          [FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"          [OK]   {message}")


def skip(message):
    """Print a skipped check."""
    print(f"          [SKIP] {message}")


def show_environment():
    """Print the interpreter details, so a mismatch is obvious."""
    print("           --- environment ---")
    print(f"           Interpreter: {sys.executable}")
    print(f"           Version:     {sys.version.split()[0]}")
    print(f"           Working dir: {os.getcwd()}")
    print(f"           Script dir:  {HERE}")
    print()


def print_source_context(module_name, lineno, context=6):
    """
    Print the source lines around a syntax error.

    A syntax error message gives a line number but no context, which
    makes it hard to find the problem in a long file.
    """
    path = os.path.join(HERE, f"{module_name}.py")

    if not os.path.isfile(path) or not lineno:
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return

    start = max(0, lineno - 1 - context)
    end = min(len(lines), lineno + context)

    print()
    print(f"           Context around {module_name}.py line {lineno}:")
    print()

    for index in range(start, end):
        marker = ">>" if index == lineno - 1 else "  "
        print(f"           {marker} {index + 1:5d}: {lines[index]}")


# ---------------------------------------------------------------------------
# The individual checks
# ---------------------------------------------------------------------------
def check_dependencies():
    """Confirm the third-party packages are importable."""
    print("           Checking third-party packages...")

    missing = []

    for name in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "n/a")
            print(f"             {name:12s} {version}")
        except Exception as error:
            missing.append(name)
            print(f"             {name:12s} MISSING ({error})")

    if missing:
        fail(
            "These packages are not available to this interpreter: "
            + ", ".join(missing)
        )
        print()
        print("           Fix it by running:")
        print(
            f"               {sys.executable} -m pip install "
            f"--upgrade -r requirements.txt"
        )
        print()

    print()
    ok("All required packages are importable.")


def check_files():
    """Confirm every required source file is present."""
    print("           Checking required files...")

    for name in REQUIRED_FILES:
        path = os.path.join(HERE, name)

        if not os.path.isfile(path):
            fail(f"{name} was not found in {HERE}")

    ok(f"All {len(REQUIRED_FILES)} required source files are present.")


def check_modules():
    """Confirm the project modules import cleanly."""
    print("           Importing modules...")

    for name in ("residual_core", "forensic_tools", "help_content"):
        try:
            __import__(name)
        except SyntaxError as error:
            fail(f"{name} has a syntax error on line {error.lineno}.")
            print_source_context(name, error.lineno)
            traceback.print_exc()
        except Exception as error:
            fail(f"Could not import {name}: {error}")
            traceback.print_exc()

    ok("Core modules import cleanly.")


def check_guide():
    """Confirm the operator guide is complete."""
    print("           Validating the operator guide...")

    import help_content

    sections = help_content.SECTIONS

    if len(sections) != REQUIRED_SECTIONS:
        fail(
            f"Expected {REQUIRED_SECTIONS} guide sections, "
            f"found {len(sections)}."
        )

    empty = [
        section.get("label", "<unnamed>")
        for section in sections
        if not section.get("blocks")
    ]

    if empty:
        fail(
            "These guide sections have no content: "
            + ", ".join(empty)
        )

    counts = [len(section["blocks"]) for section in sections]

    print(f"           Guide sections: {len(sections)}")
    print(f"           Block counts:   {counts}")

    ok("The operator guide is complete.")

    try:
        text = help_content.full_text()
    except Exception as error:
        fail(f"help_content.full_text() raised {error}")
        traceback.print_exc()

    if len(text) < 2000:
        fail(
            "The rendered guide text is suspiciously short "
            f"({len(text)} characters). It is probably truncated."
        )

    ok(f"The rendered guide is {len(text):,} characters.")


def check_analysis():
    """Run each analysis function against synthetic data."""
    print("           Testing the analysis functions...")

    import numpy as np

    import forensic_tools as ft
    import residual_core as rc

    rng = np.random.default_rng(1234)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)

    try:
        result = rc.analyze_image(
            image[:, :, 0].astype(np.float32) / 255.0,
            params=rc.Params(),
            path="<selfcheck>",
        )
    except Exception as error:
        fail(f"residual_core.analyze_image failed: {error}")
        traceback.print_exc()

    if result.residual.shape != (128, 128):
        fail(
            f"Unexpected residual shape {result.residual.shape}, "
            f"expected (128, 128)."
        )

    ok("FFT analysis produces a result of the correct shape.")

    try:
        ela = ft.analyze_ela(
            image,
            ft.ForensicsParams(mode=ft.MODE_ELA),
            path="<selfcheck>",
        )
    except Exception as error:
        fail(f"forensic_tools.analyze_ela failed: {error}")
        traceback.print_exc()

    if ela.panel4.shape != (128, 128, 3):
        fail(
            f"Unexpected ELA shape {ela.panel4.shape}, "
            f"expected (128, 128, 3)."
        )

    ok("ELA produces a result of the correct shape.")

    try:
        noise = ft.analyze_noise(
            image,
            ft.ForensicsParams(mode=ft.MODE_NOISE),
            path="<selfcheck>",
        )
    except Exception as error:
        fail(f"forensic_tools.analyze_noise failed: {error}")
        traceback.print_exc()

    if noise.panel4.shape != (128, 128, 3):
        fail(
            f"Unexpected noise shape {noise.panel4.shape}, "
            f"expected (128, 128, 3)."
        )

    ok("Noise Analysis produces a result of the correct shape.")

    for key, label in ft.MAGNIFIER_CHOICES:
        try:
            ft.apply_magnifier(image, key)
        except Exception as error:
            fail(f"apply_magnifier({label}) failed: {error}")

    ok(f"All {len(ft.MAGNIFIER_CHOICES)} magnifier modes work.")


def check_result_contract():
    """
    Confirm all three result types expose the same public attributes.

    The GUI logs the FFT, ELA, and Noise results through identical code,
    so a field that exists on one type but not another causes a runtime
    error at the moment the result is displayed.
    """
    print("           Checking the result attribute contract...")

    import numpy as np

    import forensic_tools as ft
    import residual_core as rc

    rng = np.random.default_rng(4321)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    fft_result = rc.analyze_image(
        image[:, :, 0].astype(np.float32) / 255.0,
        params=rc.Params(),
        path="<selfcheck>",
    )

    ela_result = ft.analyze_ela(
        image,
        ft.ForensicsParams(mode=ft.MODE_ELA),
        path="<selfcheck>",
    )

    noise_result = ft.analyze_noise(
        image,
        ft.ForensicsParams(mode=ft.MODE_NOISE),
        path="<selfcheck>",
    )

    for name, result in (
        ("FFT", fft_result),
        ("ELA", ela_result),
        ("Noise", noise_result),
    ):
        missing = [
            attribute
            for attribute in REQUIRED_RESULT_ATTRIBUTES
            if not hasattr(result, attribute)
        ]

        if missing:
            fail(
                f"The {name} result is missing these attributes: "
                + ", ".join(missing)
            )

        if not isinstance(getattr(result, "path", None), str):
            fail(f"The {name} result has a non-string path attribute.")

    ok("All three result types expose the same attributes.")


def check_guide_styles():
    """
    Confirm the guide text tag options are supported by this Tk build.

    An unsupported option raises a TclError inside the guide window,
    which aborts the rendering of every section and leaves the guide
    blank with no visible error in a windowed build. This check uses a
    hidden Tk window to test the options against the real Tk
    installation on this machine.
    """
    print("           Checking the guide text styles...")

    import help_content

    if not hasattr(help_content, "TAG_STYLES"):
        fail(
            "help_content.py does not define TAG_STYLES. Append the "
            "style definitions to the end of that file."
        )

    try:
        import tkinter as tk
    except Exception as error:
        skip(f"tkinter is unavailable: {error}")
        return

    try:
        root = tk.Tk()
    except Exception as error:
        skip(f"No display available: {error}")
        return

    try:
        text = tk.Text(root)

        for tag, options in help_content.TAG_STYLES.items():
            try:
                text.tag_configure(tag, **options)
            except Exception as error:
                fail(
                    f"The guide style '{tag}' uses an option this Tk "
                    f"build does not support: {error}"
                )
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    ok(
        f"All {len(help_content.TAG_STYLES)} guide text styles are "
        f"supported by this Tk build."
    )


def check_magnifier_effect():
    """
    Confirm that each magnifier mode measurably changes the layer.

    A contrast stretch based on the absolute minimum and maximum is
    defeated by a handful of outlier pixels. If the layer already
    contains a few pure black and pure white pixels, the stretch maps
    0 to 0 and 255 to 255, the lookup table becomes the identity, and
    the enhancement produces output identical to the input. The faint
    structure in the bulk of the layer then stays invisible.

    The percentile limits used by the current implementation are immune
    to that, and this check enforces it.
    """
    print("           Checking the magnifier enhancements...")

    import numpy as np

    from image_ops import (
        MAGNIFIER_CHOICES,
        MAGNIFIER_NONE,
        apply_magnifier,
    )

    # A layer that mimics a real analysis layer: a narrow band of
    # structure with a handful of extreme outliers, which is exactly
    # what image edges produce in a real noise or error map.
    layer = np.full((128, 128), 128, dtype=np.uint8)
    layer[::2, ::2] = 126
    layer[1::2, 1::2] = 130
    layer[0, 0] = 0
    layer[0, 1] = 255
    layer[1, 0] = 255
    layer[1, 1] = 0

    for key, label in MAGNIFIER_CHOICES:
        enhanced = apply_magnifier(layer, key)

        if key == MAGNIFIER_NONE:
            if not np.array_equal(enhanced, layer):
                fail(
                    f"The magnifier mode '{label}' altered the layer, "
                    f"but it is meant to leave it untouched."
                )
            continue

        if np.array_equal(enhanced, layer):
            fail(
                f"The magnifier mode '{label}' did not change the layer "
                f"at all. A contrast stretch that is a no-op means the "
                f"outlier pixels are defeating it."
            )

        spread = (
            int(np.percentile(enhanced, 99.5))
            - int(np.percentile(enhanced, 0.5))
        )

        if spread < 100:
            fail(
                f"The magnifier mode '{label}' only produced a spread of "
                f"{spread} levels. The bulk of the structure should be "
                f"spread across most of the tonal range."
            )

    ok("Every magnifier mode measurably changes the layer.")
def check_cache():
    """
    Confirm the result cache stores, retrieves, and bounds its entries.

    A cache that grows without limit will exhaust memory on a large
    collection, and a cache that returns the wrong entry would show the
    operator the wrong image.
    """
    print("           Checking the result cache...")

    import numpy as np

    from analysis_cache import CachedResult, ResultCache, KIND_GRAY

    def make_entry(name, size):
        """A small entry. Each one is roughly size squared bytes."""
        return CachedResult(
            path=name,
            mode="fft",
            signature=("fft",),
            detail="test",
            elapsed_ms=1.0,
            width=8,
            height=8,
            panels=[
                (np.zeros((4, 4), dtype=np.uint8), KIND_GRAY)
            ] * 4,
            full_result=np.zeros((size, size), dtype=np.uint8),
        )

    # --- a normal entry is stored and returned ------------------------
    cache = ResultCache(max_bytes=1024 * 1024, max_entries=4)

    cache.put(make_entry("a", 8))
    cache.put(make_entry("b", 8))

    if cache.count != 2:
        fail(f"The cache should hold 2 entries, it holds {cache.count}.")

    found = cache.get("a")

    if found is None or found.path != "a":
        fail("The cache did not return the entry it was given.")

    # --- the entry limit is respected ---------------------------------
    cache = ResultCache(max_bytes=1024 * 1024, max_entries=2)
    cache.put(make_entry("a", 8))
    cache.put(make_entry("b", 8))
    cache.put(make_entry("c", 8))

    if cache.count > 2:
        fail(f"The cache exceeded its entry limit, it holds {cache.count}.")

    # --- the protected entry survives an eviction ----------------------
    cache = ResultCache(max_bytes=1024 * 1024, max_entries=2)
    cache.put(make_entry("keep", 8))
    cache.protect("keep")
    cache.put(make_entry("other", 8))
    cache.put(make_entry("third", 8))

    if cache.get("keep") is None:
        fail("The cache evicted the entry that was on screen.")

    # --- an entry larger than the whole budget is still kept ----------
    cache = ResultCache(max_bytes=64, max_entries=4)
    cache.put(make_entry("huge", 64))

    if cache.get("huge") is None:
        fail("The cache discarded a newly added entry that was over budget.")

    # --- the memory budget forces an eviction -------------------------
    # Each entry is about 4160 bytes, so two of them do not fit in 1024.
    cache = ResultCache(max_bytes=1024, max_entries=100)
    cache.put(make_entry("a", 64))
    cache.put(make_entry("b", 64))

    if cache.count >= 2:
        fail(
            "The cache did not evict an entry when it went over its "
            "memory budget."
        )

    ok("The result cache stores, retrieves, and bounds its entries.")




def check_gui_api():
    """
    Confirm the GUI module defines the expected public interface.

    A file truncated by a copy and paste can still be syntactically
    valid if the cut lands on a function boundary. Importing the module
    and checking for the key classes catches that case, which a syntax
    check alone would miss.
    """
    print("           Checking the GUI public interface...")

    import importlib.util

    source = os.path.join(HERE, "AI_image_frequency_checker.py")

    spec = importlib.util.spec_from_file_location(
        "_aifc_gui_check", source)

    if spec is None or spec.loader is None:
        fail("Could not load the GUI module for inspection.")

    module = importlib.util.module_from_spec(spec)

    try:
        # Importing the module is safe. main() only runs under the
        # __main__ guard, so no window is created.
        spec.loader.exec_module(module)
    except Exception as error:
        fail(f"The GUI module failed to import: {error}")
        traceback.print_exc()

    required = (
        "ResidualViewer",
        "HelpWindow",
        "CollapsibleSection",
        "main",
    )

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        fail(
            "The GUI module is missing these definitions: "
            + ", ".join(missing)
            + ". The file may have been truncated."
        )

    ok("The GUI module defines the expected public interface.")

def check_gui_construction():
    """
    Build the real GUI in a hidden window.

    This is the only check that exercises every widget and every method
    call the application makes during start-up. A widget option that
    ttk does not accept, such as passing -anchor to a ttk.Button,
    raises a TclError at construction time. A method that is called but
    never defined raises an AttributeError. In both cases the module
    compiles and imports perfectly, so a syntax check and an import
    check both miss it, and the application dies on the first window
    instead of starting.
    """
    print("           Building the GUI in a hidden window...")

    import importlib.util

    source = os.path.join(HERE, "AI_image_frequency_checker.py")

    spec = importlib.util.spec_from_file_location(
        "_aifc_gui_build", source)

    if spec is None or spec.loader is None:
        fail("Could not load the GUI module.")

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as error:
        fail(f"The GUI module failed to import: {error}")
        traceback.print_exc()

    app = None

    try:
        app = module.ResidualViewer()
        app.withdraw()
        app.update_idletasks()
    except Exception as error:
        fail(
            f"The GUI could not be constructed: {error}\n"
            f"      This is usually an invalid widget option or a "
            f"method that is called but not defined."
        )
        traceback.print_exc()
    finally:
        if app is not None:
            try:
                app.destroy()
            except Exception:
                pass

    ok("The GUI builds successfully.")



def check_gui():
    """Confirm the GUI module compiles without syntax errors."""
    print("           Checking the GUI module...")

    import py_compile

    source = os.path.join(HERE, "AI_image_frequency_checker.py")

    try:
        py_compile.compile(source, doraise=True)
    except py_compile.PyCompileError as error:
        fail(f"The GUI module has a syntax error: {error}")
        traceback.print_exc()

    ok("The GUI module compiles without syntax errors.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: The checks to run, in order. The names are looked up at runtime so
#: a partially edited file reports a clear message instead of a
#: NameError deep inside main().
CHECK_NAMES = [
    "check_dependencies",
    "check_files",
    "check_modules",
    "check_guide",
    "check_analysis",
    "check_result_contract",
    "check_guide_styles",
    "check_magnifier_effect",
    "check_cache",
    "check_gui",
    "check_gui_api",
    "check_gui_construction",
]



def main():
    """Run every check. Returns 0 on success."""
    print()
    show_environment()

    try:
        for name in CHECK_NAMES:
            function = globals().get(name)

            if function is None:
                fail(
                    f"The self-check function '{name}' is not defined in "
                    f"this file. The file is incomplete or was only "
                    f"partially edited."
                )

            function()

    except SystemExit:
        raise
    except Exception as error:
        print(f"          [FAIL] Unexpected error: {error}")
        traceback.print_exc()
        return 1

    print()
    print("           All self-checks passed.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
