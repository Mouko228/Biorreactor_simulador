"""
launcher.py – Arranca la app Streamlit desde el ejecutable .exe
================================================================
Punto de entrada para PyInstaller.
Lanza Streamlit en proceso interno y abre el navegador una sola vez.
"""

import os
import sys
import threading
import time
import webbrowser


def resource_path(relative_path: str) -> str:
    """Resuelve rutas correctamente dentro del .exe (PyInstaller _MEIPASS)."""
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def open_browser(port: int, delay: float = 3.5):
    """Abre el navegador UNA sola vez después de que Streamlit esté listo."""
    time.sleep(delay)
    webbrowser.open_new(f"http://localhost:{port}")


def main():
    port = 8501
    app_path = resource_path("app_simulador_3camaras.py")

    # Abre el navegador en un hilo separado (una sola vez)
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Importar y lanzar Streamlit directamente (sin subprocess)
    # Esto funciona porque PyInstaller empaqueta streamlit completo
    from streamlit.web import cli as stcli

    # Forzar developmentMode=false antes de que Streamlit lea su config
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
