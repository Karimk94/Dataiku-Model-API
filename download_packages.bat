@echo off
setlocal

echo Changing directory to the script's location...
cd /d "%~dp0"

echo ============================================================
echo  Downloading packages for OFFLINE installation
echo ============================================================
echo.

if exist packages (
    echo Cleaning old packages folder...
    rmdir /s /q packages
)
mkdir packages

echo Downloading all requirements specifically for Python 3.12 (64-bit) to ensure compatibility...
pip download -r requirements.txt --only-binary=:all: --platform win_amd64 --python-version 3.12 -d packages
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to download 64-bit wheels.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Download complete!
echo ============================================================
echo The 'packages\' folder now contains all necessary offline wheels.
echo Run 'create_archive.bat' to package everything for deployment.
echo.
pause
endlocal
