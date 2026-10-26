"""
forensic_tools.py
=================

Error Level Analysis (ELA) and Noise Analysis for image forensics.

Both are visualisation aids for image triage. Neither proves that an
image was manipulated.

Error Level Analysis
--------------------
The image is recompressed with a JPEG of a chosen quality, and the
difference between the original and the recompressed copy is displayed.

Regions that were saved the same number of times as their surroundings
produce a uniform error level. A region that was pasted, cloned, warped,
or otherwise edited after the last save produces a different error,
because the recompression treats it differently from the pixels around
it.

Noise Analysis
--------------
A reverse-denoising operation.

A median filter removes the noise from the image. Subtracting that
result from the original leaves the noise, plus halos wherever the image
was resampled, warped, deformed, or airbrushed. Because a resampling
operation discards some of the high-frequency noise, the affected region
shows up as a dark halo in the noise map.

The magnifier enhancement and the opacity blend are shared with the FFT
method and live in image_ops.py.

Author: Raysky-sys
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from image_ops import (
    MAGNIFIER_AUTOCONTRAST,
    MAGNIFIER_CHOICES,
    MAGNIFIER_HISTOGRAM_EQUALIZATION,
    MAGNIFIER_NONE,
    apply_magnifier,
    blend,
)

__all__ = [
    "ForensicsParams",
    "ForensicsResult",
    "AnalysisCancelled",
    "MAGNIFIER_NONE",
    "MAGNIFIER_HISTOGRAM_EQUALIZATION",
    "MAGNIFIER_AUTOCONTRAST",
    "MAGNIFIER_CHOICES",
    "MODE_ELA",
    "MODE_NOISE",
    "load_color",
    "recompress_jpeg",
    "apply_magnifier",
    "blend",
    "analyze_ela",
    "analyze_noise",
]


MODE_ELA = "ela"
MODE_NOISE = "noise"

#: Median filter kernel sizes supported by OpenCV for 8-bit images.
NOISE_KERNEL_SIZES = (3, 5)


class AnalysisCancelled(Exception):
    """Raised when the operator cancels a running analysis."""


# ---------------------------------------------------------------------------
# Parameters and results
# ---------------------------------------------------------------------------
@dataclass
class ForensicsParams:
    """Operator settings for ELA and Noise Analysis."""

    mode: str = MODE_ELA

    # ----- Error Level Analysis -----
    #: JPEG quality used for the recompression. Lower values produce
    #: larger, more obvious error differences.
    ela_quality: int = 75

    #: Multiplier applied to the absolute difference between the original
    #: and the recompressed copy.
    ela_error_scale: float = 10.0

    # ----- Noise Analysis -----
    #: Size of the median filter used for the reverse-denoising step.
    noise_ksize: int = 3

    #: Multiplier applied to the extracted noise.
    noise_amplitude: float = 4.0

    # ----- Shared display controls -----
    #: Which magnifier enhancement to apply.
    magnifier: str = MAGNIFIER_AUTOCONTRAST

    #: Blend factor between the original image (0) and the analysis
    #: layer (1).
    opacity: float = 1.0


@dataclass
class ForensicsResult:
    """The four panels produced by an ELA or Noise analysis."""

    mode: str
    original: np.ndarray      # BGR uint8
    panel2: np.ndarray        # BGR uint8
    panel3: np.ndarray        # BGR uint8
    panel4: np.ndarray        # BGR uint8
    detail: str
    elapsed_ms: float

    #: The source file path. The FFT result carries the same field, so
    #: the GUI can log either result type through identical code.
    path: str = ""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_color(path: str) -> np.ndarray:
    """
    Load an image as a BGR uint8 array.

    np.fromfile() and cv2.imdecode() are used because cv2.imread() can
    fail silently on Windows paths containing non-ASCII characters.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    raw = np.fromfile(path, dtype=np.uint8)

    if raw.size == 0:
        raise IOError(f"Could not read image data from: {path}")

    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        raise IOError(f"Unsupported or corrupt image file: {path}")

    return image


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def recompress_jpeg(bgr_u8: np.ndarray, quality: int) -> np.ndarray:
    """
    Recompress an image with JPEG at the given quality and decode it back.

    The image is encoded through a genuine JPEG encoder rather than
    simulated, so the recompression includes all the real artefacts:
    8x8 block quantisation, DCT rounding, and chroma subsampling. Those
    artefacts are the entire basis of the error level analysis.
    """
    quality = int(np.clip(quality, 1, 100))

    encoded_ok, buffer = cv2.imencode(
        ".jpg",
        bgr_u8,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not encoded_ok:
        raise IOError(
            "JPEG encoding failed during the error level analysis.")

    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if decoded is None:
        raise IOError(
            "JPEG decoding failed during the error level analysis.")

    return decoded


# ---------------------------------------------------------------------------
# Error Level Analysis
# ---------------------------------------------------------------------------
def analyze_ela(
    bgr_u8: np.ndarray,
    params: Optional[ForensicsParams] = None,
    progress=None,
    should_cancel=None,
    path: str = "",
) -> ForensicsResult:
    """
    Run the error level analysis.

    Panels produced:

    1. The original image.
    2. The recompressed copy, at the chosen JPEG quality.
    3. The amplified error map, after the magnifier enhancement.
    4. The error map blended over the original at the chosen opacity.
    """
    start = time.perf_counter()
    params = params or ForensicsParams()

    original = np.ascontiguousarray(bgr_u8)

    def report(message):
        if progress:
            progress(message)

    def check_cancel():
        if should_cancel and should_cancel():
            raise AnalysisCancelled("Cancelled by the operator.")

    check_cancel()
    report(f"Recompressing with JPEG quality {params.ela_quality}...")

    recompressed = recompress_jpeg(original, params.ela_quality)

    check_cancel()
    report("Computing the error map...")

    difference = cv2.absdiff(original, recompressed).astype(np.float32)

    scaled = np.clip(
        difference * float(params.ela_error_scale),
        0,
        255,
    ).astype(np.uint8)

    report(f"Applying {params.magnifier} enhancement...")

    enhanced = apply_magnifier(scaled, params.magnifier)

    check_cancel()
    report("Blending with the original image...")

    final = blend(original, enhanced, params.opacity)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    report("Done.")

    return ForensicsResult(
        mode=MODE_ELA,
        original=original,
        panel2=recompressed,
        panel3=enhanced,
        panel4=final,
        detail=(
            f"ELA | JPEG quality={params.ela_quality} | "
            f"error scale={params.ela_error_scale:.1f} | "
            f"magnifier={params.magnifier} | "
            f"opacity={params.opacity:.2f}"
        ),
        elapsed_ms=elapsed_ms,
        path=path,
    )


# ---------------------------------------------------------------------------
# Noise Analysis
# ---------------------------------------------------------------------------
def analyze_noise(
    bgr_u8: np.ndarray,
    params: Optional[ForensicsParams] = None,
    progress=None,
    should_cancel=None,
    path: str = "",
) -> ForensicsResult:
    """
    Run the noise analysis.

    A median filter removes the noise from the image. The result is
    subtracted from the original, which leaves the noise, plus halos
    wherever the image was resampled, warped, deformed, or airbrushed.

    Panels produced:

    1. The original image.
    2. The median-filtered image, which is the picture with the noise
       removed.
    3. The amplified noise map, after the magnifier enhancement.
    4. The noise map blended over the original at the chosen opacity.
    """
    start = time.perf_counter()
    params = params or ForensicsParams()

    original = np.ascontiguousarray(bgr_u8)

    def report(message):
        if progress:
            progress(message)

    def check_cancel():
        if should_cancel and should_cancel():
            raise AnalysisCancelled("Cancelled by the operator.")

    kernel = int(params.noise_ksize)

    if kernel not in NOISE_KERNEL_SIZES:
        kernel = 3

    check_cancel()
    report(f"Applying the {kernel}x{kernel} median filter...")

    denoised = cv2.medianBlur(original, kernel)

    check_cancel()
    report("Extracting and amplifying the noise...")

    signed = (
        original.astype(np.float32) - denoised.astype(np.float32)
    )

    amplified = np.clip(
        128.0 + signed * float(params.noise_amplitude),
        0,
        255,
    ).astype(np.uint8)

    report(f"Applying {params.magnifier} enhancement...")

    enhanced = apply_magnifier(amplified, params.magnifier)

    check_cancel()
    report("Blending with the original image...")

    final = blend(original, enhanced, params.opacity)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    report("Done.")

    return ForensicsResult(
        mode=MODE_NOISE,
        original=original,
        panel2=denoised,
        panel3=enhanced,
        panel4=final,
        detail=(
            f"Noise | median k={kernel} | "
            f"amplitude={params.noise_amplitude:.1f} | "
            f"magnifier={params.magnifier} | "
            f"opacity={params.opacity:.2f}"
        ),
        elapsed_ms=elapsed_ms,
        path=path,
    )
