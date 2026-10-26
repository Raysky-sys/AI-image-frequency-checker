"""
residual_core.py
================

Core DSP pipeline for the AI Image Frequency Checker.

The analysis is a frequency-domain high-pass extraction:

    1. Load image -> grayscale float32 in [0, 1]
    2. np.fft.fft2 -> np.fft.fftshift      (DC component to centre)
    3. Circular high-pass mask              (low frequencies blocked)
    4. np.fft.ifftshift -> np.fft.ifft2 -> abs()
    5. Contrast stretch + CLAHE + gain
    6. Magnifier enhancement + opacity blend

The magnifier enhancement and the opacity blend are shared with the
error level and noise analyses, and live in image_ops.py.

Author: Raysky-sys
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from image_ops import (
    MAGNIFIER_AUTOCONTRAST,
    MAGNIFIER_NONE,
    apply_magnifier,
)

__all__ = [
    "Params",
    "ResidualResult",
    "load_gray",
    "save_gray",
    "minmax_normalize",
    "percentile_stretch",
    "high_pass_mask",
    "analyze_image",
]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """Operator-controlled settings for one analysis."""

    #: Centre-block radius as a FRACTION of min(height, width).
    #: Expressed as a fraction so the same setting is meaningful on a
    #: 512x512 crop and on a 6000x4000 photo from a 50 MP camera.
    radius_frac: float = 0.08

    #: Width (in pixels) of the smooth cosine transition at the edge of
    #: the blocked low-frequency disc. 0 = hard-edged (ringing), > 0 = softer.
    feather_px: float = 0.0

    #: Linear gain applied after the contrast stretch.
    gain: float = 1.0

    #: Enable CLAHE on the residual.
    use_clahe: bool = True

    #: CLAHE clip limit (1.0 = no clipping, 8.0 = very aggressive).
    clahe_clip: float = 2.0

    #: CLAHE tile grid size.
    clahe_tiles: int = 8

    #: Use a percentile-based stretch instead of a pure min-max stretch.
    #: More robust to a single hot pixel, and usually better for revealing
    #: faint structured watermarks.
    robust_stretch: bool = True

    #: Percentile limits used by the robust stretch.
    pct_low: float = 0.5
    pct_high: float = 99.5

    #: If True, use real(ifft) instead of abs(ifft).
    #: abs() folds negative excursions onto positive ones, which hides
    #: signed structure. The real part keeps the true sign of the residual.
    use_real_part: bool = False

    #: Which magnifier enhancement to apply to the final residual.
    #: Auto Contrast is the default, matching the other two methods.
    magnifier: str = MAGNIFIER_AUTOCONTRAST


    #: Blend factor between the original image (0) and the analysis
    #: layer (1). Shared with the ELA and noise analyses.
    opacity: float = 1.0


@dataclass
class ResidualResult:
    """All intermediate products of the pipeline."""

    path: str
    gray: np.ndarray            # (H, W)   float32, 0..1
    spectrum_log: np.ndarray    # (H, W)   float64, log1p(|fftshift(FFT)|)
    spectrum_lin: np.ndarray    # (H, W)   float64, |fftshift(FFT)|
    mask: np.ndarray            # (H, W)   float64, 1 = pass, 0 = block
    masked_log: np.ndarray      # (H, W)   float64, log1p(|masked FFT|)
    residual_raw: np.ndarray    # (H, W)   float64, |ifft2| of masked spectrum
    residual: np.ndarray        # (H, W)   float32, 0..1, amplified
    radius_px: int
    radius_frac: float
    center: Tuple[int, int]
    elapsed_ms: float

    #: Identifies the analysis that produced this result. The
    #: ForensicsResult carries the same field, so the GUI can treat
    #: both result types identically.
    mode: str = "fft"

    #: A one-line summary of the settings used, shown in the log.
    detail: str = ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    """
    Load an image as a float32 grayscale array normalised to [0, 1].

    np.fromfile() + cv2.imdecode() is used instead of cv2.imread() because
    cv2.imread() silently fails on Windows paths containing non-ASCII
    characters.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise IOError(f"Could not read image data from: {path}")

    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Unsupported or corrupt image file: {path}")

    # Normalise the 0..255 integer range down to 0..1 float.
    return image.astype(np.float32) / 255.0


def save_gray(path: str, array01: np.ndarray) -> None:
    """Write a float array in [0, 1] out as an 8-bit image (Unicode-safe)."""
    data = np.clip(np.asarray(array01) * 255.0, 0, 255).astype(np.uint8)
    extension = os.path.splitext(path)[1] or ".png"

    success, buffer = cv2.imencode(extension, data)
    if not success:
        raise IOError(f"Could not encode output image: {path}")

    buffer.tofile(path)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def minmax_normalize(array: np.ndarray) -> np.ndarray:
    """Stretch an array so its minimum maps to 0.0 and its maximum to 1.0."""
    low = float(np.min(array))
    high = float(np.max(array))
    span = high - low

    if span < 1e-12:
        # Completely flat residual: nothing to stretch.
        return np.zeros_like(array, dtype=np.float32)

    return ((array - low) / span).astype(np.float32)


def percentile_stretch(
    array: np.ndarray,
    low_percent: float = 0.5,
    high_percent: float = 99.5,
) -> np.ndarray:
    """
    Stretch using robust percentiles instead of the absolute min/max.

    A watermark residual is typically orders of magnitude weaker than the
    compression and sensor noise floor, so the min/max stretch is dominated
    by outliers. Percentile stretching amplifies the faint structured
    component far more effectively.
    """
    low = float(np.percentile(array, low_percent))
    high = float(np.percentile(array, high_percent))
    span = high - low

    if span < 1e-12:
        return np.zeros_like(array, dtype=np.float32)

    stretched = (array - low) / span

    # Clip the tails so a handful of hot pixels cannot dominate the display.
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Frequency-domain filtering
# ---------------------------------------------------------------------------
def high_pass_mask(
    shape: Tuple[int, int],
    radius_px: float,
    feather_px: float = 0.0,
) -> np.ndarray:
    """
    Build a circular high-pass mask.

    The mask blocks a disc of radius ``radius_px`` centred on the DC
    component (the low-frequency image content) and passes everything
    outside it (the high-frequency residual).

    Parameters
    ----------
    shape : (height, width) of the spectrum.
    radius_px : radius of the blocked disc, in pixels.
    feather_px : width of a smooth cosine roll-off centred on ``radius_px``.
                 0 produces a hard edge, which causes ringing artefacts in
                 the spatial domain. A small feather (2-5 % of the image's
                 short side) gives a cleaner residual.

    Returns
    -------
    mask : float64 array of the same shape, values in {0.0, 1.0} for a hard
           edge or smoothly varying in [0, 1] for a feathered edge.
           1.0 = frequency passes, 0.0 = frequency is blocked.
    """
    height, width = shape
    center_y = height // 2
    center_x = width // 2

    # np.ogrid builds the coordinate grid without allocating two full
    # (H, W) integer arrays, which matters on large images.
    yy, xx = np.ogrid[:height, :width]

    # Euclidean distance from the DC component at the spectrum centre.
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    if feather_px <= 0.0:
        # Hard-edged circular block.
        return (distance > radius_px).astype(np.float64)

    half = feather_px / 2.0
    inner = max(radius_px - half, 0.0)
    outer = radius_px + half

    # Normalised position inside the transition band, 0..1.
    t = np.clip((distance - inner) / max(outer - inner, 1e-6), 0.0, 1.0)

    # Raised cosine: 0 at the inner edge, 1 at the outer edge. Its first
    # derivative is zero at both ends, which avoids visible banding.
    mask = 0.5 * (1.0 - np.cos(np.pi * t))

    mask[distance <= inner] = 0.0
    mask[distance >= outer] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def analyze_image(
    gray: np.ndarray,
    params: Optional[Params] = None,
    path: str = "",
) -> ResidualResult:
    """
    Run the complete high-frequency residual extraction pipeline.

    Parameters
    ----------
    gray : (H, W) float array in [0, 1].
    params : operator settings; defaults are used when omitted.
    path : source path, carried through for display and logging.

    Returns
    -------
    ResidualResult with every intermediate stage retained.
    """
    start = time.perf_counter()
    params = params or Params()

    gray = np.asarray(gray, dtype=np.float64)
    height, width = gray.shape

    # Reference dimension for the radius: the short side. Using min(H, W)
    # keeps the blocked disc circular in real pixel terms for any aspect
    # ratio, which is what makes the setting resolution independent.
    reference = float(min(height, width))
    radius_px = max(1.0, params.radius_frac * reference)

    # -- Step 1: forward 2D FFT -------------------------------------------
    # FFT decomposes the image into spatial frequencies. The result is
    # complex because each coefficient carries both a magnitude and a phase.
    spectrum = np.fft.fft2(gray)

    # fftshift rolls the array by half so the zero-frequency (DC) term,
    # which sits in the corner [0, 0] of an unshifted FFT, is moved to the
    # geometric centre of the array.
    spectrum_shifted = np.fft.fftshift(spectrum)

    # Magnitude spectrum, linear and log-scaled for display.
    magnitude_linear = np.abs(spectrum_shifted)
    magnitude_log = np.log1p(magnitude_linear)

    # -- Step 2: high-pass mask -------------------------------------------
    mask = high_pass_mask(
        (height, width),
        radius_px=radius_px,
        feather_px=params.feather_px,
    )

    # Multiply in the frequency domain = convolution in the spatial domain.
    masked_spectrum = spectrum_shifted * mask
    masked_log = np.log1p(np.abs(masked_spectrum))

    # -- Step 3: inverse transform ----------------------------------------
    # ifftshift undoes fftshift, then ifft2 returns to the spatial domain.
    reconstructed = np.fft.ifft2(np.fft.ifftshift(masked_spectrum))

    # The result is complex. abs() keeps the magnitude of the residual.
    # Optionally the signed real part is used instead.
    if params.use_real_part:
        residual_raw = np.real(reconstructed)
    else:
        residual_raw = np.abs(reconstructed)

    # -- Step 4: amplification ---------------------------------------------
    if params.robust_stretch:
        residual = percentile_stretch(
            residual_raw,
            low_percent=params.pct_low,
            high_percent=params.pct_high,
        )
    else:
        residual = minmax_normalize(residual_raw)

    if params.use_clahe:
        # CLAHE operates on 8-bit or 16-bit integer images, so the float
        # residual is mapped to 0..255 first and converted back afterwards.
        as_uint8 = np.clip(residual * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=float(params.clahe_clip),
            tileGridSize=(int(params.clahe_tiles), int(params.clahe_tiles)),
        )
        residual = clahe.apply(as_uint8).astype(np.float32) / 255.0

    # Linear gain on top of the contrast stretch, then clip back to [0, 1].
    residual = np.clip(residual * float(params.gain), 0.0, 1.0).astype(np.float32)

    # -- Step 5: magnifier enhancement and opacity blend -------------------
    # These are the same controls the ELA and noise analyses use, so all
    # three methods behave identically from the operator's point of view.
    if params.magnifier and params.magnifier != MAGNIFIER_NONE:
        as_uint8 = np.clip(residual * 255.0, 0, 255).astype(np.uint8)
        enhanced = apply_magnifier(as_uint8, params.magnifier)
        residual = enhanced.astype(np.float32) / 255.0

    if params.opacity < 1.0:
        residual = np.clip(
            gray.astype(np.float32) * (1.0 - float(params.opacity))
            + residual.astype(np.float32) * float(params.opacity),
            0.0,
            1.0,
        ).astype(np.float32)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return ResidualResult(
        path=path,
        gray=gray.astype(np.float32),
        spectrum_log=magnitude_log,
        spectrum_lin=magnitude_linear,
        mask=mask,
        masked_log=masked_log,
        residual_raw=residual_raw,
        residual=residual,
        radius_px=int(round(radius_px)),
        radius_frac=float(params.radius_frac),
        center=(height // 2, width // 2),
        elapsed_ms=elapsed_ms,
        mode="fft",
        detail=(
            f"FFT | radius {int(round(radius_px))} px "
            f"({params.radius_frac:.3f} x min(H,W)) | "
            f"magnifier={params.magnifier} | "
            f"opacity={params.opacity:.2f}"
        ),
    )
