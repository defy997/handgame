@echo off
echo ============================================
echo  HandGame CFR - Build Standalone EXE
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install numpy pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple -q
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo [2/3] Building exe...
pyinstaller --onefile --name handgame_cfr --console cfr_play.py
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)

echo [3/3] Copying strategy file...
if not exist dist\cfr_output mkdir dist\cfr_output
if exist cfr_output\cfr_strategy.npz (
    copy cfr_output\cfr_strategy.npz dist\cfr_output\cfr_strategy.npz >nul
    echo     Done.
) else (
    echo     WARNING: cfr_output\cfr_strategy.npz not found.
    echo     Please train the model first.
)

echo.
echo ============================================
echo  Done! Send these to your friend:
echo    dist\handgame_cfr.exe
echo    dist\cfr_output\cfr_strategy.npz
echo ============================================
pause
