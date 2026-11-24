@echo off
rem ---------------------------------------------------------------------------
rem  Lab PC launcher - stage 6.2 training. Double-click it, or run from cmd:
rem     lab.bat                      start or resume training
rem     lab.bat --status             show progress only
rem     lab.bat --backup E:\capstone also copy checkpoints/results to the USB on exit
rem  Ctrl+C pauses safely (finished epochs are saved). Run lab.bat again to resume.
rem  If Windows then asks "Terminate batch job (Y/N)?", either answer is fine.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"


echo [lab] updating code ...
git pull --ff-only || echo [lab] git pull failed - continuing with the code already here

if not exist ".venv\Scripts\python.exe" (
    echo [lab] first run: creating .venv and installing packages - about 10 minutes
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo [lab] Could not create .venv - is Python 3.10+ installed and on PATH?
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu130
)
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt

".venv\Scripts\python.exe" -m src.lab %*
echo.
pause
