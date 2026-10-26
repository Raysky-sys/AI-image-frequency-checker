"""
image_ops.py
=============

Shared image display operations.

These are used by every analysis method, so the magnifier enhancement
and the opacity blend behave identically no matter which method is
selected.

Author: Raysky-sys
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "MAGNIFIER_NONE",
    "MAGNIFIER_HISTOGRAM_EQUALIZATION",
    "MAGNIFIER_AUTOCONTRAST",
    "MAGNIFIER_AUTOCONTRAST_BY_CHANNEL",
    "MAGNIFIER_CHOICES",
    "AUTO_CONTRAST_LOW_PERCENT",
    "AUTO_CONTRAST_HIGH_PERCENT",
    "magnifier_key_from_label",
    "apply_magnifier",
    "blend",
]


# ---------------------------------------------------------------------------
# Magnifier enhancement modes
# ---------------------------------------------------------------------------
MAGNIFIER_NONE = "none"
MAGNIFIER_HISTOGRAM_EQUALIZATION = "histogram_equalization"
MAGNIFIER_AUTOCONTRAST = "autocontrast"
MAGNIFIER_AUTOCONTRAST_BY_CHANNEL = "autocontrast_by_channel"

#: (internal key, display label) pairs.
MAGNIFIER_CHOICES: List[Tuple[str, str]] = [
    (MAGNIFIER_NONE, "None"),
    (MAGNIFIER_HISTOGRAM_EQUALIZATION, "Histogram Equalization"),
    (MAGNIFIER_AUTOCONTRAST, "Auto Contrast"),
    (MAGNIFIER_AUTOCONTRAST_BY_CHANNEL, "Auto Contrast by Channel"),
]

#: Percentile limits used by the robust auto contrast.
#:
#: A plain minimum and maximum stretch is defeated by a small number of
#: outlier pixels. The analysis layers produced by this application are
#: faint by nature: a typical noise map occupies only about sixteen
#: levels out of 255, and the strongest differences come from image
#: edges rather than from the signal of interest.
#:
#: If the layer already contains a few pure black and pure white pixels,
#: a minimum and maximum stretch maps 0 to 0 and 255 to 255. The lookup
#: table becomes the identity, the enhancement does nothing at all, and
#: the faint structure stays invisible.
#:
#: Percentile limits discard those outliers, so the stretch is spent on
#: the bulk of the signal instead.
AUTO_CONTRAST_LOW_PERCENT = 0.5
AUTO_CONTRAST_HIGH_PERCENT = 99.5


def magnifier_key_from_label(label: str) -> str:
    """Map a display label back to its internal key."""
    for key, name in MAGNIFIER_CHOICES:
        if name == label:
            return key
    return MAGNIFIER_AUTOCONTRAST


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def _build_lut(low: float, high: float) -> Optional[np.ndarray]:
    """
    Build a 256 entry lookup table that maps low..high onto 0..255.

    Returns None when the range is too narrow to be worth stretching.
    """
    if not np.isfinite(low) or not np.isfinite(high):
        return None

    if high - low < 1.0:
        return None

    scale = 255.0 / float(high - low)
    table = np.clip(
        (np.arange(256) - low) * scale,
        0,
        255,
    ).astype(np.uint8)

    return table


def _stretch_limits(
    values: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the stretch limits for a layer, using robust percentiles.

    Falls back to the absolute minimum and maximum when the percentile
    range is degenerate, for example on a perfectly flat image.
    """
    low = float(np.percentile(values, AUTO_CONTRAST_LOW_PERCENT))
    high = float(np.percentile(values, AUTO_CONTRAST_HIGH_PERCENT))

    if high - low < 1.0:
        low = float(np.min(values))
        high = float(np.max(values))

    if high - low < 1e-6:
        return None, None

    return low, high


def apply_magnifier(image_bgr_u8: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply one of the four magnifier enhancements.

    Parameters
    ----------
    image_bgr_u8 :
        The layer to enhance. Any channel count is accepted, including
        a single grayscale channel.
    mode :
        One of the MAGNIFIER_* constants.

    Notes
    -----
    Auto Contrast applies a single lookup table built from robust
    percentile limits taken across all channels. Because every channel
    is stretched by the same amount, the colour relationships are
    preserved. This is the option that reveals a faint noise map or
    error map, and it should be the default.

    Auto Contrast by Channel computes separate limits for each channel,
    which stretches each channel independently and therefore shifts the
    colour balance. On a single channel image it behaves identically to
    Auto Contrast.

    Histogram Equalisation redistributes the tonal range rather than
    stretching it. It is the most robust option for finding faint
    structure inside an already enhanced layer, and it can shift colour.
    """
    mode = str(mode or MAGNIFIER_NONE).strip().lower()

    if mode == MAGNIFIER_NONE:
        return image_bgr_u8.copy()

    if mode == MAGNIFIER_HISTOGRAM_EQUALIZATION:
        channels = [
            cv2.equalizeHist(channel)
            for channel in cv2.split(image_bgr_u8)
        ]
        return cv2.merge(channels)

    if mode == MAGNIFIER_AUTOCONTRAST:
        low, high = _stretch_limits(image_bgr_u8)

        if low is None:
            return image_bgr_u8.copy()

        table = _build_lut(low, high)

        if table is None:
            return image_bgr_u8.copy()

        # One shared table for every channel keeps the colours intact.
        return cv2.LUT(image_bgr_u8, table)

    if mode == MAGNIFIER_AUTOCONTRAST_BY_CHANNEL:
        merged = []

        for channel in cv2.split(image_bgr_u8):
            low, high = _stretch_limits(channel)

            if low is None:
                merged.append(channel)
                continue

            table = _build_lut(low, high)

            if table is None:
                merged.append(channel)
                continue

            merged.append(cv2.LUT(channel, table))

        return cv2.merge(merged)

    return image_bgr_u8.copy()


def blend(
    original_bgr_u8: np.ndarray,
    layer_bgr_u8: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """
    Blend the analysis layer over the original image.

    An opacity of 0 shows only the original image. An opacity of 1 shows
    only the analysis layer. The default magnifier behaviour in
    photoForensics is the same: lowering the opacity reveals more of the
    original photograph underneath.
    """
    factor = float(np.clip(opacity, 0.0, 1.0))

    blended = (
        original_bgr_u8.astype(np.float32) * (1.0 - factor)
        + layer_bgr_u8.astype(np.float32) * factor
    )

    return np.clip(blended, 0, 255).astype(np.uint8)
