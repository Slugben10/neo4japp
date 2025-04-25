# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('config.json', '.'), ('.env', '.')],
    hiddenimports=['wx', 'wx.lib.scrolledpanel', 'wx.lib.newevent', 'json', 'threading', 'requests', 'shutil', 'traceback', 'dotenv', 'altgraph', 'docx', 'pypdf', 'wx.adv', 'wx.html', 'wx.grid', 'wx.xrc', 'wx._xml', 'wx._html', 'wx._adv', 'wx._core', 'wx._controls'],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/hook-app.py', 'hooks/hook-macos-paths.py'],
    excludes=['tkinter', 'PySide', 'PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RA',
)
app = BUNDLE(
    coll,
    name='RA.app',
    icon=None,
    bundle_identifier='com.researchassistant.app',
)
