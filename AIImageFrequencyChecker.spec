# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['matplotlib.backends.backend_tkagg', 'help_content', 'forensic_tools', 'analysis_cache', 'image_ops']
hiddenimports += collect_submodules('residual_core')
hiddenimports += collect_submodules('forensic_tools')
hiddenimports += collect_submodules('analysis_cache')
hiddenimports += collect_submodules('image_ops')


a = Analysis(
    ['AI_image_frequency_checker.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide', 'PySide6', 'pytest', 'IPython', 'notebook', 'tkinter.test'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AIImageFrequencyChecker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon=['icon.ico'],
)
