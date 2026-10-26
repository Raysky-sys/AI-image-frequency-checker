# AI Image Frequency Checker

A desktop application that extracts and amplifies the **high-frequency
residual** of an image, so that faint structured patterns left behind by
compression, resampling, upscaling, and generative pipelines become
visible to the human eye.

Written in Python using OpenCV, NumPy, and Matplotlib, with a Tkinter
operator interface. Ships as a standalone Windows executable.


---

## Table of Contents

- [What the application does](#what-the-application-does)
- [The mathematics](#the-mathematics)
- [The four-panel visualisation](#the-four-panel-visualisation)
- [Features](#features)
- [Requirements](#requirements)
- [Running from source](#running-from-source)
- [Building the standalone executable](#building-the-standalone-executable)
- [Running the application](#running-the-application)
- [Parameter reference](#parameter-reference)
- [Operator notes and tips](#operator-notes-and-tips)
- [How to read the output](#how-to-read-the-output)
- [What this tool can and cannot prove](#what-this-tool-can-and-cannot-prove)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Related tools](#related-tools)

---

## What the application does

Every digital image carries two overlapping layers of information:

1. **Low-frequency content** — the picture itself. Broad shapes, colour
   regions, lighting gradients.
2. **High-frequency content** — the fine detail and, crucially, a record of
   *how the file was processed*. JPEG quantization blocks, resampling
   grids, sharpening halos, sensor noise, and the periodic lattice
   patterns that upscalers and diffusion models leave behind.

The picture dominates. The processing fingerprint is orders of magnitude
weaker, so it is effectively invisible in a normal viewer.

This application removes the picture and amplifies what is left.

### Intended use

- **Triage.** Find images with an unusual or inconsistent processing
  history in a large collection, then escalate those for deeper analysis.
- **Visual comparison.** Put a suspected image and a known reference
  side by side and compare their residual structure.
- **Education.** Demonstrate why a photograph that has been re-saved,
  rescaled, or edited looks nothing like a camera original in the
  frequency domain.
- **Support a separate metadata scan.** The high-frequency residual and
  the string-based marker scanner in this repository answer different
  questions. Use both.

---

## The mathematics

### 1. Pre-processing

The image is decoded to a single grayscale channel and divided by 255, so
pixel values occupy the float range `[0.0, 1.0]`. Grayscale is used
because luminance carries the perceptual structure; colour channels would
triple the computation without adding forensic value.

### 2. Frequency domain transformation

The 2D Discrete Fourier Transform decomposes the image into its
constituent spatial frequencies. The output is complex: every coefficient
carries both a magnitude (how much energy at that frequency) and a phase
(where that frequency sits in space).

An unshifted FFT places the zero-frequency component, the DC term, in the
corner at index `[0, 0]`. `fftshift` rolls the array by half its length
in each dimension so DC lands in the geometric centre. This makes the
spectrum symmetric and lets a simple radial mask work.

### 3. High-pass filtering (residual extraction)

A circular mask is built with:

Multiplying the spectrum by this mask is, by the convolution theorem,
equivalent to convolving the image with the corresponding spatial
kernel. The result is a band-stop operation in the low band, leaving the
high band intact.

The radius is expressed as a **fraction of `min(height, width)`** rather
than a fixed pixel count, so the same setting behaves identically on a
512 × 512 crop and a 6000 × 4000 photograph.

An optional **feather** applies a raised-cosine roll-off across the edge
of the blocked disc. A hard-edged mask produces ringing in the spatial
domain, and that ringing can be mistaken for a suspicious pattern. A
feather of 2–4 % of the short side removes the artefact.

### 4. Inverse transformation

`ifftshift` undoes `fftshift`, restoring the original frequency ordering,
and `ifft2` returns to the spatial domain. The result is complex, so
`np.abs` is applied to keep the magnitude.

An optional **signed real part** mode uses `np.real()` instead. `abs()`
folds negative excursions onto positive ones, which destroys polarity
information — that matters when a suspected pattern has a clear sign.

### 5. Signal amplification

The residual is orders of magnitude weaker than the noise floor, so it
must be stretched before it is visible.

- **Robust percentile stretch** (default) maps the 0.5th and 99.5th
  percentiles onto 0 and 1, then clips. This is far more effective than
  a plain min-max stretch, because a single hot pixel would otherwise
  dominate the range and flatten everything else to black.
- **Min-max stretch** is available as a comparison.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) is applied
  to the stretched residual. Its tile grid is local, so faint structured
  regions are enhanced without being drowned out by brighter areas.
- A **linear gain** multiplier is applied last.

---

## The four-panel visualisation

| Panel | Content |
| --- | --- |
| **1** | Original image, grayscale, normalised to 0–1 |
| **2** | Magnitude spectrum of the FFT, `log1p` scaled, DC at centre |
| **3** | The same spectrum after masking, with the blocked disc outlined in cyan |
| **4** | The final amplified high-frequency residual |

Panel 2 is where the classic "this image has a strong high-frequency
cross-hatch" observation comes from. Panel 4 is the actionable output.

---

## Features

- **Multi-image loading** — select one or many files and step through them.
- **Live parameter tuning** — every control triggers an automatic
  re-analysis in a background thread, so the operator can sweep the
  radius and watch the residual change.
- **Debounced updates** — dragging a slider does not queue dozens of
  jobs.
- **Non-blocking** — analysis runs on a worker thread; the window stays
  responsive.
- **Two export options** — save the residual as an 8-bit PNG, or save the
  full four-panel figure at 150 DPI.
- **Resolution independence** — the radius is a fraction of the short side.
- **Unicode-safe file paths** — uses `np.fromfile` / `cv2.imdecode` rather
  than `cv2.imread`, which silently fails on Windows paths containing
  non-ASCII characters.
- **Headless use** — `residual_core.py` has no GUI dependency and can be
  imported into a batch script.

---

## Requirements

**Running from source:**

- Python 3.10 or newer
- `numpy`, `opencv-python`, `matplotlib`
- Tkinter, which ships with the standard Windows and macOS installers.
  On some Linux distributions: `sudo apt install python3-tk`

**Building the executable:**

- Windows
- `pyinstaller` (see `requirements-build.txt`)

---

## Running from source

```bash
cd AI_image_frequency_checker
pip install -r requirements.txt
python AI_image_frequency_checker.py

python AI_image_frequency_checker.py "C:\images\photo.png"


Operator notes and tips
Start with the radius sweep
The default radius of 0.08 blocks the lowest 8 % of frequencies. Rather than trusting a single setting, sweep it:

Below ~0.02 — almost nothing survives. The residual is almost pure sensor noise.
0.03 – 0.10 — the working range. Compression traces and fine texture are visible.
0.10 – 0.20 — only strong structure survives. Good for spotting gross upscaling lattices.
Above 0.25 — the image is effectively destroyed and you are looking at very little meaningful signal.
A pattern that changes shape, brightness, or position as the radius slider moves is not a reliable indicator. Real structured artefacts tend to persist, or to scale predictably with the cut-off.

Always check the feather
Before escalating any finding, re-run with feather = 0 and then with feather ≈ 8. If the suspicious pattern appears only in the hard-edged run, it is ringing artefact, not evidence. This is the single most common false positive in this kind of analysis.

Prefer the robust stretch
Min-max stretching is dominated by outliers. A single hot pixel can flatten the entire residual to near-black, hiding a real pattern in plain sight. The percentile stretch is the better default; use min-max only for comparison.

Use the signed real part when polarity matters
abs() folds negative excursions onto positive ones. If a suspected pattern has an obvious light/dark alternation, switch on the signed real part. A true structured artefact keeps its polarity; noise does not.

Compare against a reference
The fastest way to interpret a residual is to put it next to a residual from an image you already trust — a known camera original, or a straightforward screenshot. Differences in the residual structure are far more informative than any single residual on its own.

Save the figure, not just the residual
The four-panel figure is the defensible artefact. It shows the original, the spectrum, the mask that was applied, and the result, so a reviewer can see exactly what was done and reproduce it. An isolated residual PNG with no parameter record invites the question "what settings produced this?".

Write down the parameters
The log line records the resolution, the effective pixel radius, and the timing. Screenshot or copy that line alongside any finding.

How to read the output
Panel 2 — magnitude spectrum
A bright cross along the horizontal and vertical axes is normal. It reflects the strong horizontal and vertical correlations that most natural images possess.
A bright, compact blob at the centre is the low-frequency image content.
A distinct high-frequency cross-hatch or grid pattern spread across the spectrum indicates structured high-frequency content, which is the classic signature of an upscaling or tiling operation.
Panel 3 — masked spectrum
The outlined circle shows exactly which frequencies were discarded. The visible content outside the circle is what produced panel 4. If the circle covers almost the whole spectrum, the radius is too large and the result is meaningless.

Panel 4 — the residual
Observation	Likely explanation
Fine random noise, no structure	Normal sensor and compression noise.
Regular grid or cross-hatch	Upscaling, tiling, or a periodic generator artefact.
Blocky 8 × 8 or 16 × 16 pattern	JPEG quantization at that quality level.
Bright edges only, content shape preserved	The image was sharpened or had a strong edge-preserving filter applied.
One region markedly different from the rest	Partial regeneration, inpainting, or compositing within that region.
Pattern that shifts when the radius changes	Ringing artefact. Not evidence.
Pattern identical in geometry to visible content	Content leakage. Increase the radius.
What this tool can and cannot prove
This section matters more than any of the above.

It reliably exposes
GAN and diffusion upscaling artefacts — periodic lattices in the residual
JPEG quantization traces and re-compression history
Resampling signatures from any editing pipeline
Sharpening and denoising halos
Inconsistencies between regenerated and original regions
Camera-app re-processing that has destroyed original metadata
It does not establish
That the image was AI-generated. A photograph saved twice by two different apps produces a strong, structured residual too.
That a specific vendor's watermark is present. SynthID is embedded in pixel values with a keyed detector and is not necessarily concentrated in the high-frequency band. The FFT residual is far more likely to reveal that a file was processed than which model made it.
Authenticity or provenance of any kind. This is a visual triage aid, not a verification tool.
The correct workflow
Use this tool to triage — find the images with unusual processing history.
Inspect metadata with the string scanner in this repository.
Check for a C2PA / Content Credentials credential and validate it properly.
Only then draw conclusions, and record every parameter used.
A strong, spatially uniform, high-frequency periodic pattern that survives a feathered filter is worth escalating. A noisy residual that changes character with the radius slider is almost certainly just compression.

Limitations
Visualisation only. Nothing here is a statistical test. There is no p-value, no classifier, and no reference corpus.
Marker-free by design. This is a frequency-domain tool and finds nothing that the byte-level scanner in the sibling folder does not.
Large images use a lot of memory. A 50 MP image produces a complex array of roughly 200 MB for the FFT alone.
No batch mode in the GUI. Use the residual_core module directly for scripted analysis.
Grayscale only. Chroma subsampling and per-channel processing are not examined.
JPEG artefacts are expected, not suspicious. Most images on the internet have been through JPEG at some point.
Not a forensic-grade tool. For evidence-grade work, use established forensic imaging software and a documented methodology.