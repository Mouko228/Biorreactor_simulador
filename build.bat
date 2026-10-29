@echo off
REM ============================================================
REM  build.bat – Construye el ejecutable del simulador
REM  Doble clic para compilar. El .exe quedará en:
REM      dist\SimuladorBioelectroquimico\
REM ============================================================

echo.
echo  ========================================
echo   Gemelo Digital – Build del ejecutable
echo  ========================================
echo.

REM Verificar que Python está disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado en PATH.
    pause
    exit /b 1
)

REM Instalar/actualizar dependencias necesarias
echo [1/3] Verificando dependencias...
python -m pip install pyinstaller streamlit numpy scipy plotly pandas --quiet --upgrade
if errorlevel 1 (
    echo [ERROR] No se pudieron instalar las dependencias.
    pause
    exit /b 1
)

REM Limpiar builds anteriores
echo [2/3] Limpiando builds anteriores...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

REM Ejecutar PyInstaller
echo [3/3] Compilando con PyInstaller...
python -m PyInstaller simulador_3camaras.spec --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] La compilacion fallo. Revisa los mensajes arriba.
    pause
    exit /b 1
)

echo.
echo  ============================================
echo   BUILD EXITOSO
echo   Ejecutable en: dist\
echo   Archivo:        SimuladorBioelectroquimico.exe  (archivo unico)
echo  ============================================
echo.
pause
