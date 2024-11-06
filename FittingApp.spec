# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_interface.py'],
    pathex=[],
    binaries=[],
    datas=[('interface_DyeAlone_fitting.py', '.'), ('interface_GDA_fitting.py', '.'), ('interface_IDA_fitting.py', '.'), ('pltstyle.py', '.')],
    hiddenimports=['matplotlib', 'tkinter'],
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
    a.binaries,
    a.datas,
    [],
    name='FittingApp',
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
    icon=['MyIcon.icns'],
)
app = BUNDLE(
    exe,
    name='FittingApp.app',
    icon='MyIcon.icns',
    bundle_identifier=None,
)
