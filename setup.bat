@echo off
REM ============================================================================
REM  setup.bat — Setup script for the Dataiku Model API
REM
REM  This script will:
REM  1. Delete any existing virtual environment (guarantees a clean state).
REM  2. Create a fresh Python virtual environment.
REM  3. Install all required packages OFFLINE from the local 'packages' folder.
REM
REM  No internet connection is required — all wheels are in the 'packages' folder.
REM ============================================================================

@echo off
setlocal

echo Changing directory to the script's location...
cd /d "%~dp0"

REM ── Step 1: Remove existing venv ─────────────────────────────────────────────
echo.
echo [1/3] Removing existing virtual environment (if any)...
if exist venv (
    rmdir /s /q venv
    if %errorlevel% neq 0 (
        echo ERROR: Could not remove existing 'venv' folder.
        echo        Make sure no process is using it, then try again.
        pause
        exit /b 1
    )
    echo Removed old venv.
) else (
    echo No existing venv found. Skipping removal.
)

REM ── Step 2: Create fresh venv ────────────────────────────────────────────────
echo.
echo [2/3] Creating a fresh Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo        Please ensure Python 3.12 is installed and on the PATH.
    pause
    exit /b 1
)
echo Virtual environment created.

REM ── Step 3: Install packages offline ─────────────────────────────────────────
echo.
echo [3/3] Installing dependencies OFFLINE from 'packages\' folder...
if not exist packages (
    echo ERROR: 'packages' folder not found.
    echo        Run 'download_packages.bat' on an internet-connected machine first,
    echo        then copy the 'packages' folder here.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
pip install --no-index --find-links=packages -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python packages from local 'packages' folder.
    echo        Ensure all required wheels are present. Re-run download_packages.bat
    echo        if needed.
    pause
    exit /b 1
)

echo.
echo =================================================================
echo  Setup Complete!
echo =================================================================
echo.
echo  Next steps for IIS deployment:
echo  1. In IIS Manager, create a new Site:
echo       Physical path : C:\edms\Dataiku Model API
echo       Binding       : http, port 5009
echo.
echo  (Note: FastCGI registration is NO LONGER needed. IIS will automatically
echo   start the API using HttpPlatformHandler and Waitress via web.config)
echo.
pause
endlocal
