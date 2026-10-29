# simulador_3camaras.spec
# ========================
# Modo onefile: genera UN SOLO .exe portable (~320 MB).
# Se extrae en una carpeta temporal al ejecutarse y se limpia al cerrar.
#
# Uso:
#   python -m PyInstaller simulador_3camaras.spec
#
# El ejecutable resultante estará en:  dist\SimuladorBioelectroquimico.exe

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SITE_PACKAGES = os.path.join(
    os.path.dirname(sys.executable), "Lib", "site-packages"
)

# ── Metadatos de paquetes (.dist-info) necesarios para importlib.metadata ─────
# Streamlit llama a importlib.metadata.version("streamlit") al arrancar.
# PyInstaller NO los incluye automáticamente — hay que agregarlos a mano.
DIST_INFO_PKGS = [
    "streamlit-1.66.1.dist-info",
    "click-8.5.0.dist-info",
    "altair-6.3.0.dist-info",
    "pandas-3.0.3.dist-info",
    "numpy-2.4.6.dist-info",
    "scipy-1.18.1.dist-info",
    "plotly-7.1.0.dist-info",
    "pyarrow-26.0.0.dist-info",
    "pydeck-0.9.3.dist-info",
    "packaging-26.3.dist-info",
    "attrs-26.1.0.dist-info",
    "jsonschema-4.26.0.dist-info",
    "watchdog-6.0.0.dist-info",
    "requests-2.34.2.dist-info",
    "pillow-13.0.0.dist-info",
]

# ── Archivos de datos ─────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("streamlit", include_py_files=False)
datas += collect_data_files("plotly")
datas += [("app_simulador_3camaras.py", ".")]

# Incluir cada carpeta .dist-info en la raíz del bundle
for dist_dir in DIST_INFO_PKGS:
    full_path = os.path.join(SITE_PACKAGES, dist_dir)
    if os.path.isdir(full_path):
        datas += [(full_path, dist_dir)]

# ── Módulos ocultos ───────────────────────────────────────────────────────────
hiddenimports = (
    collect_submodules("streamlit")
    + collect_submodules("streamlit.web")
    + collect_submodules("streamlit.runtime")
    + collect_submodules("streamlit.components")
    + collect_submodules("altair")
    + collect_submodules("plotly")
    + collect_submodules("pandas")
    + collect_submodules("numpy")
    + collect_submodules("scipy")
    + collect_submodules("scipy.integrate")
    + collect_submodules("scipy.special")
    + collect_submodules("scipy.sparse")
    + collect_submodules("scipy.optimize")
    + [
        "streamlit.web.cli",
        "streamlit.runtime.scriptrunner",
        "click",
        "tomli",
        "PIL",
        "importlib.metadata",
        "importlib.resources",
    ]
)

# ── Análisis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "unittest",
        "test",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ── ONE FILE ─────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SimuladorBioelectroquimico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
