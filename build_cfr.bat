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
pip install "numpy==1.26.4" pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo     Numpy version:
python -c "import numpy; print('   ', numpy.__version__)"

echo [2/3] Building exe...
if exist build\handgame_cfr rmdir /s /q build\handgame_cfr
pyinstaller --onefile --name handgame_cfr --console --clean cfr_play.py
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)

echo [3/3] Converting and copying strategy file...
if not exist dist\cfr_output mkdir dist\cfr_output
if exist cfr_output\cfr_strategy.json (
    copy cfr_output\cfr_strategy.json dist\cfr_output\cfr_strategy.json >nul
    echo     Done (json already exists).
) else if exist cfr_output\cfr_strategy.npz (
    python -c "import numpy as np, json; d=np.load('cfr_output/cfr_strategy.npz',allow_pickle=True); data={'R0':d['R0'].tolist(),'R1':d['R1'].tolist(),'S0':d['S0'].tolist(),'S1':d['S1'].tolist(),'hw':d['hw'].tolist(),'total_steps':int(d['total_steps'][0]),'episode_count':int(d['episode_count'][0]),'phase_steps':list(d['phase_steps'].astype(int)),'counter_hits':{str(k):int(v) for k,v in zip(d['ctr_keys'],d['ctr_vals'])},'cfg':list(d['cfg'].astype(int))}; json.dump(data,open('cfr_output/cfr_strategy.json','w',encoding='utf-8'))"
    copy cfr_output\cfr_strategy.json dist\cfr_output\cfr_strategy.json >nul
    echo     Done (converted from npz).
) else (
    echo     WARNING: strategy file not found. Please train first.
)

echo.
echo ============================================
echo  Done! Send these to your friend:
echo    dist\handgame_cfr.exe
echo    dist\cfr_output\cfr_strategy.npz
echo ============================================
pause
