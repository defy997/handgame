@echo off
echo ============================================
echo  HandGame Nash - Build Standalone EXE
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install "numpy==1.26.4" pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo [2/3] Building exe...
if exist build\play_nash rmdir /s /q build\play_nash
pyinstaller --onefile --name handgame_nash --console --clean ^
    --exclude-module scipy ^
    --exclude-module torch ^
    --exclude-module tensorflow ^
    play_nash.py
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)

echo [3/3] Copying strategy file...
if not exist dist\cfr_output mkdir dist\cfr_output

if exist cfr_output\nash_finite.npz (
    copy cfr_output\nash_finite.npz dist\cfr_output\nash_finite.npz >nul
    echo     Copied nash_finite.npz  (finite-horizon Nash, correct round limit)
) else if exist cfr_output\nash_exact.json (
    copy cfr_output\nash_exact.json dist\cfr_output\nash_exact.json >nul
    echo     WARNING: nash_finite.npz not found, using nash_exact.json (fallback)
    echo     Run: python nash_finite.py   to generate the finite-horizon Nash strategy.
) else (
    echo     WARNING: No strategy file found!
    echo     Run: python nash_finite.py   to generate nash_finite.npz
)

echo.
echo ============================================
echo  Done! Send these to your friend:
echo    dist\handgame_nash.exe
echo    dist\cfr_output\nash_finite.npz
echo ============================================
pause
