"""
help_content.py
===============

Offline operator guide for the AI Image Frequency Checker.

Contains only structured text and no GUI imports, so it can be verified
independently before the executable is built.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

Block = Tuple[str, Any]

TITLE = 'AI Image Frequency Checker - Operator Guide'

INTRO = (
    'This guide is stored inside the application and works offline.\n'
    'Everything needed to interpret the output is available here.\n'
    '\n'
    'BEFORE RELYING ON A FINDING:\n'
    '  1. Read the Operator Notes and Tips section.\n'
    '  2. Read the Can / Cannot Prove section.\n'
    '  3. Re-run the image with different settings.\n'
    '  4. Compare against a trusted reference image.\n'
    '\n'
    'Three analysis methods are available:\n'
    '  FFT Residual     - the frequency-domain high-frequency residual\n'
    '  ELA              - error level analysis, compares against a\n'
    '                     JPEG recompression of the same image\n'
    '  Noise Analysis   - reverse denoising, isolates the sensor noise\n'
    '\n'
    'Short version: these tools show HOW AN IMAGE WAS PROCESSED, not who '
    'or what created it.'
)


QUICKSTART: List[Block] = [
    ('h2', 'Purpose'),
    ('p', 'This application removes or isolates the parts of an image '
          'that record how the file was processed, so that faint traces '
          'become visible to the human eye.'),
    ('p', 'The processing history of an image is orders of magnitude '
          'weaker than the picture itself, so it is invisible in a '
          'normal viewer until it is extracted and amplified.'),

    ('h2', 'The three analysis methods'),
    ('table', ('FFT Residual',
               'Removes the low-frequency content and amplifies what '
               'remains. Reveals compression traces, resampling grids, '
               'upscaling lattices, and sharpening halos.')),
    ('table', ('ELA',
               'Error Level Analysis. Recompresses the image with JPEG '
               'and displays the difference. Reveals regions that were '
               'edited after the last save.')),
    ('table', ('Noise Analysis',
               'Reverse denoising. Removes the picture and keeps the '
               'noise. Reveals airbrushing, warping, deformation, and '
               'perspective-corrected cloning.')),

    ('h2', 'Basic workflow'),
    ('b', 'Open one or more images and step through them with the arrow '
          'keys.'),
    ('b', 'Choose the analysis method from the top of the parameter '
          'panel.'),
    ('b', 'Start with the default settings, then change one parameter at '
          'a time.'),
    ('b', 'Compare the result against a reference image you trust.'),
    ('b', 'Save the complete four-panel figure, not only the final '
          'image.'),
    ('b', 'Record the image dimensions and all parameter values.'),
]


TIPS: List[Block] = [
    ('h2', 'Start with a radius sweep on the FFT method'),
    ('p', 'The default radius of 0.08 blocks the lowest 8 percent of '
          'frequencies. Treat the radius as a sweep rather than a '
          'setting.'),
    ('b', 'Below 0.02 - almost nothing survives. The residual is almost '
          'pure sensor noise.'),
    ('b', '0.03 to 0.10 - the working range. Compression traces and fine '
          'texture are visible.'),
    ('b', '0.10 to 0.20 - only strong structure survives.'),
    ('b', 'Above 0.25 - the image is effectively destroyed.'),
    ('note', 'A pattern that changes shape, brightness, or position as the '
             'radius moves is NOT a reliable indicator. Real structured '
             'artefacts tend to persist, or to scale predictably with the '
             'cut-off.'),

    ('h2', 'Always check the feather before escalating'),
    ('p', 'A hard-edged mask with a feather of zero causes ringing in the '
          'spatial domain, and that ringing can look exactly like a '
          'suspicious periodic pattern.'),
    ('b', 'Run once with a feather of zero.'),
    ('b', 'Run again with a feather of 8 to 12 pixels.'),
    ('b', 'If the pattern only appears in the hard-edged run, it is an '
          'artefact of your own filter, not evidence of anything.'),
    ('warn', 'This is the single most common false positive in this kind '
             'of analysis. Check it every single time.'),

    ('h2', 'Prefer the robust percentile stretch'),
    ('p', 'Min-max stretching is dominated by outliers. One hot pixel can '
          'flatten the entire residual to near-black and hide a real '
          'pattern in plain sight.'),
    ('b', 'Use min-max only for comparison, and never as your only run.'),

    ('h2', 'The magnifier enhancement applies to every method'),
    ('p', 'The analysis layers are faint by nature. A typical noise map '
          'occupies only about sixteen levels out of 255, so the Auto '
          'Contrast enhancement is what makes it visible. Set the '
          'enhancement to None only when you want to judge the raw '
          'signal level instead.'),
    ('b', 'On the FFT result the residual is already contrast stretched, '
          'so Auto Contrast has little effect and Histogram Equalization '
          'is the more useful option.'),
    ('b', 'The noise amplitude multiplies the raw noise before the '
          'enhancement, so it controls how harsh the result looks. A high '
          'amplitude saturates the extremes and produces a harder, more '
          'binary map. A low amplitude gives a smoother, more graded one.'),

    ('h2', 'Every slider shows its current value'),
    ('b', 'Each slider carries a live numeric readout, so the readings '
          'quoted in this guide can be followed exactly. To set a feather '
          'of 8, drag the feather slider until the number beside it reads '
          '8.'),

    ('h2', 'The advanced options are collapsed by default'),
    ('b', 'Everything below the high-pass radius sits in a collapsible '
          'section. Click the header to open the feather, gain, CLAHE and '
          'colormap controls. Collapsing a section hides the controls but '
          'changes no value, so the result is identical either way.'),


    ('h2', 'Use the signed real part when polarity matters'),
    ('p', 'The absolute value folds negative excursions onto positive '
          'ones, which destroys polarity information.'),
    ('b', 'Enable the signed real part option when a suspected pattern '
          'has an obvious light and dark alternation.'),

    ('h2', 'Use the magnifier lens for pixel-level inspection'),
    ('b', 'Set the zoom to 4x or 8x and click anywhere on the final panel '
          'to move the lens to that point.'),
    ('b', 'Use the mouse wheel over the final panel to change the zoom '
          'level.'),
    ('b', 'Return the zoom to 1x before saving the figure if you want a '
          'full-frame record.'),

    ('h2', 'Compare against a reference'),
    ('p', 'The fastest way to interpret any of these results is to place '
          'it next to the same analysis of an image you already trust, '
          'such as a camera original or a straight screenshot. Differences '
          'in the analysis layers are far more informative than any '
          'single result viewed in isolation.'),

    ('h2', 'Save the figure, not just the final image'),
    ('p', 'The four-panel figure records the original, the intermediate '
          'stage, the analysis layer, and the blended result, so a '
          'reviewer can see what was done and reproduce it. An isolated '
          'image with no parameter record invites the question of what '
          'settings produced this.'),

    ('h2', 'Write down the parameters'),
    ('p', 'The status bar and the log record the method, the resolution, '
          'every parameter value, and the elapsed time. Copy that line '
          'into your notes alongside any finding.'),

    ('h2', 'Feed the results into a wider process'),
    ('b', 'Use these tools to triage, not to conclude.'),
    ('b', 'Then inspect metadata with the string scanner in the sibling '
          'folder of this project.'),
    ('b', 'Then check for a C2PA or Content Credentials credential and '
          'validate it properly.'),
    ('b', 'Only then draw a conclusion, and record every parameter used.'),
]


READING: List[Block] = [
    ('h2', 'FFT method'),
    ('b', 'Panel 1 is the source image in grayscale.'),
    ('b', 'Panel 2 is the full magnitude spectrum, log scaled. The bright '
          'central blob is the low-frequency image content. A bright '
          'cross along the horizontal and vertical axes is common in '
          'natural images.'),
    ('b', 'Panel 3 is the spectrum after the circular high-pass mask is '
          'applied. The cyan circle shows exactly which frequencies were '
          'discarded.'),
    ('b', 'Panel 4 is the amplified residual.'),
    ('table', ('Fine random noise',
               'Usually normal sensor or compression noise.')),
    ('table', ('Regular grid or cross-hatch',
               'Upscaling, tiling, or a periodic generator artefact.')),
    ('table', ('8x8 or 16x16 blocks',
               'JPEG quantization structure. Not suspicious on its own.')),
    ('table', ('Only edges remain visible',
               'Sharpening or edge-preserving filtering was applied.')),
    ('table', ('Residual resembles the original picture',
               'Content leakage. Increase the radius.')),

    ('h2', 'ELA method'),
    ('b', 'Panel 1 is the original image.'),
    ('b', 'Panel 2 is the recompressed copy. It should look almost '
          'identical to the original. If it does not, the file was '
          'already heavily compressed, or it has unusual colour '
          'subsampling.'),
    ('b', 'Panel 3 is the error map, amplified and enhanced.'),
    ('b', 'Panel 4 is the error map blended over the original.'),

    ('h2', 'Noise method'),
    ('b', 'Panel 1 is the original image.'),
    ('b', 'Panel 2 is the median-filtered image, which is the picture with '
          'the noise removed. It looks almost like the original.'),
    ('b', 'Panel 3 is the extracted noise, amplified and enhanced.'),
    ('b', 'Panel 4 is the noise map blended over the original.'),
]


ELA: List[Block] = [
    ('h2', 'How error level analysis works'),
    ('p', 'The image is recompressed with a JPEG of a chosen quality. The '
          'difference between the original and the recompressed copy is '
          'then displayed and amplified.'),
    ('p', 'Pixels that have been compressed the same number of times '
          'produce the same error. A region that was pasted, cloned, '
          'warped, or retouched after the last save was compressed a '
          'different number of times, so it produces a different error. '
          'That difference is what stands out.'),
    ('p', 'This is why the JPEG quality setting matters so much. If the '
          'recompression quality does not match the quality the file was '
          'originally saved at, the whole image lights up and the '
          'manipulated region is lost in the noise.'),

    ('h2', 'Choosing the JPEG quality'),
    ('b', 'Start at 75. It is a good general-purpose value.'),
    ('b', 'If the whole image shows heavy error, raise the quality. Try '
          '85, then 90, then 95.'),
    ('b', 'The correct quality is the one where the background becomes '
          'flat and only the interesting regions remain bright.'),
    ('b', 'If the image was saved as PNG, or has been through a chain of '
          're-saves, no quality value will give a clean background. '
          'Treat the result as unreliable.'),

    ('h2', 'Setting the error scale'),
    ('b', 'The error scale multiplies the difference between the original '
          'and the recompressed copy.'),
    ('b', 'The default of 10 makes typical quantization errors visible '
          'without saturating.'),
    ('b', 'Raise it if the manipulated region is too dim to see. Lower it '
          'if the whole panel is blown out to white.'),
    ('b', 'Change the scale and the quality together if needed. The goal '
          'is a flat background with the region of interest standing out.'),

    ('h2', 'Reading the error map'),
    ('table', ('Uniform flat field',
               'The whole image was processed consistently. This is the '
               'desired baseline.')),
    ('table', ('Bright patches or blocks',
               'A region was recompressed differently from its '
               'surroundings. Consistent with editing after the last '
               'save.')),
    ('table', ('Bright halo around an edge',
               'The classic signature of a pasted or cloned region: the '
               'seam between the two compression histories.')),
    ('table', ('8x8 blocky structure',
               'Normal JPEG quantization grid. Not interesting on its '
               'own.')),
    ('table', ('Sky brighter than the subject',
               'Different regions were compressed a different number of '
               'times, which can happen with layered or exported images.')),
    ('table', ('Text or flat areas look very bright',
               'Expected. Sharp edges produce larger quantization errors '
               'than soft gradients.')),

    ('h2', 'Operator notes'),
    ('warn', 'The results of this method can be misleading. A photograph '
             'that has simply been re-saved several times will show '
             'differences everywhere, and a clean photograph can show '
             'more structure than a manipulated one.'),
    ('b', 'Always compare against a reference image processed with the '
          'same settings.'),
    ('b', 'Text, screenshots, and graphic content produce false '
          'positives. Check the source type before interpreting.'),
    ('b', 'A matching JPEG quality is the single biggest factor in '
          'getting a usable result.'),
]


NOISE: List[Block] = [
    ('h2', 'How noise analysis works'),
    ('p', 'Natural images are full of sensor noise. When an image is '
          'modified, that modification often leaves visible traces in the '
          'noise.'),
    ('p', 'A median filter removes the noise from the image while keeping '
          'the edges. Subtracting that filtered result from the original '
          'reverses the process: instead of removing the noise, it '
          'removes everything except the noise.'),
    ('p', 'What is left is the noise, plus halos wherever the image was '
          'resampled, warped, deformed, or airbrushed. A resampling '
          'operation discards some of the high-frequency noise, so the '
          'affected region shows a dark halo in the noise map.'),

    ('h2', 'Where it is most useful'),
    ('b', 'Airbrushing and retouching, which replace the noise entirely.'),
    ('b', 'Warping and warping tools, which leave a halo of missing '
          'noise around the region.'),
    ('b', 'Deformations and elastic transforms.'),
    ('b', 'Perspective-corrected cloning, which can evade the clone '
          'detection and the error level analysis.'),

    ('h2', 'Choosing the median filter size'),
    ('b', 'Size 3 is subtle and preserves more genuine detail. Start '
          'here.'),
    ('b', 'Size 5 removes more noise, which makes the signal stronger but '
          'also creates more false halos around genuine edges.'),
    ('b', 'Do not go larger. A large filter starts removing real image '
          'detail, and the resulting halos become indistinguishable from '
          'the real thing.'),

    ('h2', 'Setting the noise amplitude'),
    ('b', 'The amplitude multiplies the extracted noise.'),
    ('b', 'The default of 4 makes typical sensor noise clearly visible '
          'without saturating.'),
    ('b', 'Raise it to reveal very faint structure. Lower it if the whole '
          'panel is blown out.'),
    ('b', 'Because the noise is signed and mapped around the middle of '
          'the tonal range, positive noise reads brighter and negative '
          'noise reads darker. The colour shifts are genuine, not a '
          'rendering artefact.'),

    ('h2', 'Reading the noise map'),
    ('table', ('Fine uniform grain everywhere',
               'Normal. The image has not been heavily processed.')),
    ('table', ('Dark halo around a region',
               'Resampling, warping, or deformation. This is the '
               'signature of a warp tool.')),
    ('table', ('A region with no noise at all',
               'The pixels there were synthesised. Airbrushing or '
               'generative fill.')),
    ('table', ('A region with visibly different noise character',
               'The region was processed by a different pipeline, or was '
               'copied from elsewhere.')),
    ('table', ('Straight edges or rectangular blocks',
               'A pasted or cloned region.')),
    ('table', ('Structure following the subject',
               'Sharpening or denoising was applied to the whole image.')),

    ('h2', 'Operator notes'),
    ('warn', 'This method works best on high quality, high resolution '
             'images. Smaller images contain too little noise information '
             'for it to work.'),
    ('b', 'Try the result both with and without histogram equalisation. '
          'Equalisation can reveal structure, but it can also hide it.'),
    ('b', 'Compare against a reference image processed with identical '
          'settings.'),
    ('b', 'A heavily compressed image has had its noise destroyed already, '
          'so there is nothing left to analyse.'),
]


PROOF: List[Block] = [
    ('h2', 'What it can expose'),
    ('b', 'JPEG quantization and re-compression history.'),
    ('b', 'Resampling and upscaling patterns.'),
    ('b', 'Sharpening and denoising halos.'),
    ('b', 'Regions recompressed a different number of times from their '
          'surroundings.'),
    ('b', 'Airbrushed, warped, or deformed regions, and the halos they '
          'leave behind.'),
    ('b', 'Differences between regenerated and original regions.'),
    ('b', 'Periodic lattice artefacts associated with some upscaling and '
          'generative pipelines.'),

    ('h2', 'What it cannot prove'),
    ('b', 'That an image was AI-generated. A normally edited photograph '
          'can produce a stronger residual than a generated image.'),
    ('b', 'That a particular vendor watermark is present. SynthID uses a '
          'keyed detector and is not necessarily concentrated in the '
          'high-frequency band. These tools are far more likely to reveal '
          'THAT a file was processed than WHICH model made it.'),
    ('b', 'Authenticity, integrity, or provenance.'),
    ('b', 'The exact tool or model that processed an image.'),
    ('b', 'Intentional deception. Most traces have benign causes.'),

    ('h2', 'Correct workflow'),
    ('b', 'Use these tools to triage unusual processing history.'),
    ('b', 'Inspect metadata using the separate byte-level marker scanner.'),
    ('b', 'Check for a C2PA or Content Credentials credential and validate '
          'it using an appropriate credential validator.'),
    ('b', 'Compare against a known reference.'),
    ('b', 'Only then form a conclusion, and document every parameter.'),

    ('note', 'A strong, spatially uniform periodic pattern that survives '
             'an alternate filter is worth escalating. A residual that '
             'changes dramatically with every setting is usually ordinary '
             'compression or noise.'),

    ('h2', 'An honest example'),
    ('p', 'A genuine AI-generated image, saved once as PNG, will often '
          'show a very weak, fairly uniform residual with no obvious '
          'structure. A genuine photograph that has been through a social '
          'media application, resized, and re-saved as JPEG will often '
          'show a much stronger and more interesting looking residual. The '
          'visual impressiveness of the result is not correlated with the '
          'likelihood of AI generation.'),
]


LIMITS: List[Block] = [
    ('h2', 'Technical limitations'),
    ('b', 'These are visualisation aids, not statistical classifiers.'),
    ('b', 'There is no confidence score, p-value, or trained reference '
          'database.'),
    ('b', 'The FFT method analyses grayscale only. Colour-channel and '
          'chroma artefacts are not shown.'),
    ('b', 'Large images require considerable memory because the complete '
          'FFT is held in RAM.'),
    ('b', 'There is no tiled or streaming analysis mode.'),
    ('b', 'The tools find nothing that the byte-level marker scanner does '
          'not also find. They answer different questions and are '
          'complementary, not overlapping.'),

    ('h2', 'Interpretive limitations'),
    ('b', 'No causal inference. A pattern tells you something happened, '
          'not what tool did it or whether it was malicious.'),
    ('b', 'No baseline normalisation. Each result is stretched to fill '
          'the display range, so a clean image and a heavily processed '
          'one both look equally strong. Absolute amplitude is not '
          'meaningful unless you compare with a reference.'),
    ('b', 'Many different editing operations produce similar traces.'),
    ('b', 'JPEG artefacts are extremely common and are not automatically '
          'evidence of AI generation.'),
    ('b', 'The noise method uses a true two-dimensional median rather than '
          'the separable variant used by photoForensics. The visual '
          'difference is minor.'),
    ('b', 'A professionally processed AI image may have a weaker residual '
          'than an ordinary photograph.'),

    ('h2', 'Use limitations'),
    ('b', 'Do not use these tools as the sole basis for legal, employment, '
          'disciplinary, or publishing decisions.'),
    ('b', 'Always corroborate suspicious images with independent evidence.'),
    ('b', 'For evidence-grade work, use established forensic procedures and '
          'documented methodologies.'),
]


TROUBLE: List[Block] = [
    ('h2', 'The guide is empty or only a few tabs are shown'),
    ('p', 'The build contains an incomplete help_content.py. Run the '
          'pre-build self-check and rebuild after replacing the file.'),
    ('code', 'python selfcheck.py'),
    ('p', 'The self-check reports the number of guide sections and the '
          'block count of each one. All nine sections must be present and '
          'none may be empty.'),

    ('h2', 'The ELA panel is uniformly bright'),
    ('b', 'The recompression quality does not match the original file. '
          'Raise the JPEG quality until the background flattens.'),
    ('b', 'The image has been re-saved many times, so no quality will '
          'produce a clean background.'),
    ('b', 'The image is not a JPEG at all, for example a PNG. The error '
          'level analysis is not meaningful in that case.'),

    ('h2', 'The ELA panel is completely black'),
    ('b', 'The error scale is too low. Raise it.'),
    ('b', 'The JPEG quality is 100, so almost no error is produced. Lower '
          'it to around 75 to 90.'),

    ('h2', 'The noise panel is nearly flat'),
    ('b', 'The image is small. Noise analysis needs a reasonably large '
          'source image.'),
    ('b', 'The image has been heavily compressed, so its noise has already '
          'been destroyed.'),
    ('b', 'The noise amplitude is too low. Raise it.'),

    ('h2', 'The noise panel shows halos everywhere'),
    ('b', 'The median filter is too large for this image. Use size 3.'),
    ('b', 'The image has heavy sharpening or denoising applied, which '
          'produces genuine halos around every edge.'),

    ('h2', 'The noise map has strong colour casts'),
    ('b', 'This is expected. Each colour channel carries its own noise, '
          'and the signs differ between channels.'),
    ('b', 'Switch the magnifier enhancement to Auto Contrast if the '
          'colours become distracting.'),

    ('h2', 'Panel titles overlap one another'),
    ('p', 'The figure must use constrained layout and short subplot '
          'titles. Replace AI_image_frequency_checker.py with the '
          'supplied version and rebuild.'),

    ('h2', 'The final image is all black or all white'),
    ('b', 'On the FFT method, the radius is too large and everything was '
          'blocked. Reduce it to around 0.05.'),
    ('b', 'On the ELA method, the error scale is too high. Lower it.'),
    ('b', 'On the noise method, the noise amplitude is too high. Lower it.'),

    ('h2', 'TclError on startup'),
    ('p', 'Tkinter is not installed correctly. Reinstall Python and enable '
          'Tcl and Tk, or use the packaged executable.'),

    ('h2', 'The executable is large'),
    ('p', 'Expected. NumPy, OpenCV, and Matplotlib are bundled, producing '
          'an executable of roughly 60 to 90 MB.'),

    ('h2', 'The analysis is slow'),
    ('p', 'Normal for very large images. A 50-megapixel image can require '
          'several seconds and significant memory.'),
]


PARAMETERS: List[Block] = [
    ('h2', 'Shared'),
    ('table', ('Analysis method',
               'FFT Residual, ELA, or Noise Analysis.')),
    ('table', ('Magnifier enhancement',
               'None, Histogram Equalization, Auto Contrast, or Auto '
               'Contrast by Channel. Applied to the result of every '
               'method.')),
    ('table', ('Opacity',
               '0 to 1. Blend between the original image and the analysis '
               'layer. Lower values reveal more of the original.')),
    ('table', ('Zoom lens',
               '1x to 16x. Click the final panel to move the lens, scroll '
               'to zoom.')),

    ('h2', 'FFT Residual'),
    ('table', ('High-pass radius',
               '0.005 to 0.45, default 0.08. Fraction of min(H, W).')),
    ('table', ('Feather width',
               '0 to 60, default 0. Smooth transition around the blocked '
               'region.')),
    ('table', ('Linear gain',
               '0.1 to 8.0, default 1.0.')),
    ('table', ('CLAHE',
               'Enabled by default. Adaptive contrast enhancement.')),
    ('table', ('CLAHE clip limit',
               '1.0 to 8.0, default 2.0.')),
    ('table', ('CLAHE tiles',
               '2 to 16, default 8.')),
    ('table', ('Robust percentile stretch',
               'Enabled by default. Percentile-based stretching.')),
    ('table', ('Signed real part',
               'Disabled by default. Preserves the sign of the residual.')),
    ('table', ('Residual colormap',
               'Default gray.')),

    ('h2', 'Error Level Analysis'),
    ('table', ('JPEG quality',
               '1 to 100, default 75. The quality used for the '
               'recompression.')),
    ('table', ('Error scale',
               '0.5 to 50, default 10. Multiplies the difference between '
               'the original and the recompressed copy.')),

    ('h2', 'Noise Analysis'),
    ('table', ('Median filter size',
               '3 or 5, default 3. The noise reduction filter size.')),
    ('table', ('Noise amplitude',
               '0.5 to 30, default 4. Multiplies the extracted noise.')),
]


SECTIONS: List[Dict[str, Any]] = [
    {
        'id': 'quickstart',
        'label': 'Quick Start',
        'blocks': QUICKSTART,
    },
    {
        'id': 'tips',
        'label': 'Operator Tips',
        'blocks': TIPS,
    },
    {
        'id': 'reading',
        'label': 'Reading Output',
        'blocks': READING,
    },
    {
        'id': 'ela',
        'label': 'Error Level Analysis',
        'blocks': ELA,
    },
    {
        'id': 'noise',
        'label': 'Noise Analysis',
        'blocks': NOISE,
    },
    {
        'id': 'proof',
        'label': 'Can / Cannot Prove',
        'blocks': PROOF,
    },
    {
        'id': 'limits',
        'label': 'Limitations',
        'blocks': LIMITS,
    },
    {
        'id': 'trouble',
        'label': 'Troubleshooting',
        'blocks': TROUBLE,
    },
    {
        'id': 'params',
        'label': 'Parameters',
        'blocks': PARAMETERS,
    },
]


def section_text(section: Dict[str, Any]) -> str:
    """Render one section as plain text."""
    label = str(section['label'])

    lines = [
        label,
        '=' * len(label),
        '',
    ]

    for kind, payload in section['blocks']:
        if kind == 'h2':
            lines.extend([
                '',
                str(payload).upper(),
                '-' * len(str(payload)),
            ])
        elif kind == 'p':
            lines.extend(['', str(payload)])
        elif kind == 'b':
            lines.append('  * ' + str(payload))
        elif kind == 'b2':
            lines.append('      - ' + str(payload))
        elif kind == 'note':
            lines.extend(['', '[ NOTE ]  ' + str(payload)])
        elif kind == 'warn':
            lines.extend(['', '[ CAUTION ]  ' + str(payload)])
        elif kind == 'code':
            lines.extend(['', '    ' + str(payload)])
        elif kind == 'table':
            left, right = payload
            lines.append('  ' + str(left))
            lines.append('      ' + str(right))

    lines.append('')

    return '\n'.join(lines)


def full_text() -> str:
    """Render the complete guide for clipboard copy or file export."""
    chunks = [
        TITLE,
        '=' * len(TITLE),
        '',
        INTRO,
        '',
    ]

    for section in SECTIONS:
        chunks.append(section_text(section))

    return '\n'.join(chunks)



# ---------------------------------------------------------------------------
# Text tag styles
#
# These are used to render the guide in the GUI. They live here rather
# than in the GUI module so the pre-build self-check can validate every
# option against the actual Tk build on the build machine. An option
# that this Tk version does not support raises a TclError, which would
# abort the rendering of the whole guide and leave it blank.
#
# Note: there is deliberately no "bullet" option here. It is not
# supported by the Tk 8.6 text widget. Bullets are inserted as literal
# characters instead.
# ---------------------------------------------------------------------------
TAG_STYLES = {
    "h2": {
        "font": ("Segoe UI", 10, "bold"),
        "foreground": "#123a6b",
        "spacing1": 10,
        "spacing3": 4,
    },
    "p": {
        "font": ("Segoe UI", 10),
        "spacing3": 6,
        "lmargin1": 6,
    },
    "b": {
        "font": ("Segoe UI", 10),
        "lmargin1": 18,
        "lmargin2": 10,
        "spacing3": 2,
    },
    "note": {
        "font": ("Segoe UI", 10, "italic"),
        "background": "#eaf3ea",
        "foreground": "#1f4d1f",
        "lmargin1": 10,
        "lmargin2": 10,
        "spacing1": 8,
        "spacing3": 8,
    },
    "warn": {
        "font": ("Segoe UI", 10, "bold"),
        "background": "#fdecec",
        "foreground": "#8a1c1c",
        "lmargin1": 10,
        "lmargin2": 10,
        "spacing1": 8,
        "spacing3": 8,
    },
    "table": {
        "font": ("Segoe UI", 10),
        "lmargin1": 18,
        "lmargin2": 10,
        "spacing3": 4,
    },
    "tablehead": {
        "font": ("Segoe UI", 10, "bold"),
        "lmargin1": 18,
        "lmargin2": 10,
        "spacing3": 4,
    },
}
