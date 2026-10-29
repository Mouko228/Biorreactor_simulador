@echo off
REM ============================================================
REM  instalar_y_ejecutar.bat
REM  Instala las dependencias y lanza el simulador.
REM  Solo necesitas Python instalado (python.org).
REM  Doble clic para usar.
REM ============================================================

echo.
echo  ================================================
echo   Gemelo Digital - Sistema Bioelectroquimico
echo   Iniciando simulador...
echo  ================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en PATH.
    echo.
    echo  Descarga Python desde: https://www.python.org/downloads/
    echo  Asegurate de marcar "Add Python to PATH" al instalar.
    echo.
    pause
    exit /b 1
)

echo [1/2] Instalando dependencias (solo la primera vez)...
python -m pip install streamlit numpy scipy plotly pandas --quiet
if errorlevel 1 (
    echo [ERROR] No se pudieron instalar las dependencias.
    pause
    exit /b 1
)

echo [2/2] Lanzando el simulador...
echo.
echo  Abre tu navegador en: http://localhost:8501
echo  Para cerrar: presiona Ctrl+C en esta ventana.
echo.

python -m streamlit run "%~dp0app_simulador_3camaras.py" ^
    --server.headless=false ^
    --browser.gatherUsageStats=false ^
    --server.port=8501

pause
