# -*- mode: python ; coding: utf-8 -*-
import sys
import os
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

# Cross-platform icon handling
icon_file = None
if os.path.exists('MyIcon.icns'):
    icon_file = 'MyIcon.icns'
elif os.path.exists('MyIcon.ico'):
    icon_file = 'MyIcon.ico'

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FittingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=icon_file,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FittingApp',
)

# Only create macOS app bundle on macOS
if sys.platform == 'darwin' and icon_file:
    app = BUNDLE(
        exe,
        name='FittingApp.app',
        icon=icon_file,
        bundle_identifier=None,
    )
