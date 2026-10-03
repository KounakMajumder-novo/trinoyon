@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 (
    echo Could not create the Python environment. Install Python 3.10 or newer and enable the Python launcher.
    pause
    exit /b 1
  )
)

"%PYTHON%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
  echo Installing the required webcam dependencies...
  "%PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and INSTALL.md.
    pause
    exit /b 1
  )
)

"%PYTHON%" app.py %*
if errorlevel 1 (
  echo The app exited with an error. See the message above and INSTALL.md.
  pause
)
endlocal
