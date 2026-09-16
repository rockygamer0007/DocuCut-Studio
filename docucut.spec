
from pathlib import Path

ROOT = Path(r"D:\DocuCut-Studio-release")

a = Analysis(
    [str(ROOT / "docucut" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "docucut" / "profiles" / "profiles.json"), "docucut/profiles"),
        (str(ROOT / "assets" / "docucut_icon.ico"), "assets"),
    ],
    hiddenimports=[
        "cv2",
        "numpy",
        "pymupdf",
        "PySide6",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="DocuCutStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "assets" / "docucut_icon.ico"),
)

