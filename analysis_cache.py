"""
analysis_cache.py
=================

Keeps analysis results in memory so that switching between images is
instant.

Without a cache, every time the operator changes image the analysis is
recomputed from scratch. On a large photograph that takes a couple of
seconds, and the common workflow of selecting a method and then
flipping back and forth between two images to compare them becomes
frustrating.

This module keeps a pruned, display-ready copy of each result together
with a signature of the settings that produced it. When an image is
revisited:

    * the cached result is displayed immediately, with no waiting
    * if the settings have changed since it was computed, a refresh is
      started in the background

Memory is bounded by a budget and a maximum number of entries. The least
recently used entry is discarded first, and the entry currently on screen
is never discarded.

Author: Raysky-sys
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "KIND_GRAY",
    "KIND_RGB",
    "KIND_FFT",
    "KIND_MAGMA",
    "PanelSpec",
    "CachedResult",
    "ResultCache",
    "downscale",
    "build_cache_entry",
]


# ---------------------------------------------------------------------------
# Panel display kinds
#
# A panel is stored as a uint8 array plus a kind, so the interface knows
# how to draw it without having to guess from the channel count.
# ---------------------------------------------------------------------------
KIND_GRAY = "gray"       # single channel, fixed grey colour map
KIND_RGB = "rgb"         # three channel colour
KIND_FFT = "fft"         # single channel, operator colour map
KIND_MAGMA = "magma"     # single channel, fixed colour map

PanelSpec = Tuple[np.ndarray, str]


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------
def downscale(array: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """
    Reduce an array so its longest edge is at most max_dim pixels.

    The reduction uses INTER_AREA, which averages the source pixels
    rather than dropping them, so downscaling does not invent or destroy
    high frequency structure.
    """
    height, width = array.shape[:2]
    longest = max(height, width)

    if longest <= max_dim:
        return array

    scale = max_dim / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        array,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _to_u8(array: np.ndarray) -> np.ndarray:
    """Convert a float array in the range 0..1 to uint8."""
    return np.clip(
        np.asarray(array, dtype=np.float32) * 255.0, 0, 255
    ).astype(np.uint8)


def _normalise_u8(array: np.ndarray) -> np.ndarray:
    """
    Scale a float array across the full uint8 range.

    This reproduces what Matplotlib does automatically when it displays
    a float array, so the result looks the same as before the cache was
    introduced.
    """
    array = np.asarray(array, dtype=np.float32)
    low = float(np.min(array))
    high = float(np.max(array))

    if high - low < 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _bgr_to_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a BGR array to RGB, leaving single channel arrays alone."""
    if array.ndim == 2:
        return array
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# The cached result
# ---------------------------------------------------------------------------
@dataclass
class CachedResult:
    """
    A pruned, display-ready copy of one analysis result.

    Only what the interface actually needs is kept. The large float64
    intermediates are discarded, which is what makes it practical to
    hold a dozen of these at once.
    """

    path: str
    mode: str
    signature: tuple
    detail: str
    elapsed_ms: float
    width: int
    height: int

    #: The four display panels, each an array plus its display kind.
    panels: List[PanelSpec]

    #: The full resolution result, used by the magnifier lens and by
    #: Save Result. Two dimensional for the FFT method, three channel for
    #: the colour methods.
    full_result: Optional[np.ndarray] = None
    full_kind: str = KIND_RGB

    @property
    def size_bytes(self) -> int:
        total = 0
        for data, _kind in self.panels:
            total += data.nbytes
        if self.full_result is not None:
            total += self.full_result.nbytes
        return total


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class ResultCache:
    """
    A bounded, least recently used cache of analysis results.

    The entry for the image currently on screen is protected, so it is
    never the one that gets thrown away when the budget is reached.
    """

    def __init__(
        self,
        max_bytes: int = 300 * 1024 * 1024,
        max_entries: int = 10,
    ) -> None:
        self._entries: "OrderedDict[str, CachedResult]" = OrderedDict()
        self._max_bytes = int(max_bytes)
        self._max_entries = int(max_entries)
        self._bytes = 0
        self._protected: Optional[str] = None

    # -- access --------------------------------------------------------
    def get(self, path: str) -> Optional[CachedResult]:
        """Return the cached result for a path, or None."""
        entry = self._entries.get(path)

        if entry is not None:
            self._entries.move_to_end(path)

        return entry

    def put(self, entry: CachedResult) -> None:
        """Store a result, replacing any previous entry for the same path."""
        existing = self._entries.pop(entry.path, None)

        if existing is not None:
            self._bytes -= existing.size_bytes

        self._entries[entry.path] = entry
        self._bytes += entry.size_bytes

        # The entry that was just added is never the one evicted, even
        # if it is on its own larger than the whole budget. Otherwise a
        # single oversized image would disappear the moment it was
        # stored.
        self._evict(keep=entry.path)


    def protect(self, path: Optional[str]) -> None:
        """Mark a path as the one currently on screen."""
        self._protected = path

        if path is not None and path in self._entries:
            self._entries.move_to_end(path)

    def clear(self) -> None:
        """Discard every cached result."""
        self._entries.clear()
        self._bytes = 0
        self._protected = None

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def describe(self) -> str:
        """A short human readable summary for the status bar."""
        return (
            f"{self.count} cached, "
            f"{self._bytes / (1024 * 1024):.0f} MB"
        )

    # -- eviction ------------------------------------------------------
    def _evict(self, keep: Optional[str] = None) -> None:
        """
        Discard the least recently used entries over the budget.

        The entry currently on screen and the entry that was just added
        are both protected.
        """
        while (
            len(self._entries) > self._max_entries
            or self._bytes > self._max_bytes
        ):
            victim = None

            for path in self._entries:
                if path != self._protected and path != keep:
                    victim = path
                    break

            if victim is None:
                # Nothing can be evicted without losing a protected
                # entry, so keep what is left.
                break

            entry = self._entries.pop(victim)
            self._bytes -= entry.size_bytes



# ---------------------------------------------------------------------------
# Building a cache entry from an analysis result
# ---------------------------------------------------------------------------
def build_cache_entry(result, signature: tuple) -> CachedResult:
    """
    Build a pruned, display-ready copy of an analysis result.

    The large intermediate arrays are discarded. What is kept is one
    preview per panel, plus a full resolution copy of the final result
    for the magnifier lens and for Save Result.
    """
    if result.mode == "fft":
        # --- FFT method, single channel ---------------------------------
        original_full = result.gray
        result_full = result.residual

        original = _to_u8(downscale(original_full))
        result_preview = _to_u8(downscale(result_full))
        spectrum = _normalise_u8(downscale(result.spectrum_log))
        masked = _normalise_u8(downscale(result.masked_log))

        panels = [
            (original, KIND_GRAY),
            (spectrum, KIND_MAGMA),
            (masked, KIND_MAGMA),
            (result_preview, KIND_FFT),
        ]

        full = _to_u8(result_full)
        full_kind = KIND_FFT

        height, width = original_full.shape[:2]

    else:
        # --- ELA and Noise methods, three channel -----------------------
        original = _bgr_to_rgb(downscale(result.original))
        result_preview = _bgr_to_rgb(downscale(result.panel4))
        spectrum = _bgr_to_rgb(downscale(result.panel2))
        masked = _bgr_to_rgb(downscale(result.panel3))

        panels = [
            (original, KIND_RGB),
            (spectrum, KIND_RGB),
            (masked, KIND_RGB),
            (result_preview, KIND_RGB),
        ]

        full = _bgr_to_rgb(result.panel4)
        full_kind = KIND_RGB

        height, width = result.original.shape[:2]

    return CachedResult(
        path=result.path,
        mode=result.mode,
        signature=signature,
        detail=result.detail,
        elapsed_ms=result.elapsed_ms,
        width=width,
        height=height,
        panels=panels,
        full_result=full,
        full_kind=full_kind,
    )
