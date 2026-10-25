@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

:: Ensure working directory is always the script folder
cd /d "%~dp0"

echo =====================================================================
echo                 LIDL SPENDING ^& NUTRITION - LOCAL LAUNCHER
echo =====================================================================

:: 1. Verify Python Installation and Minimum Version
where python >nul 2>nul
if errorlevel 1 (
    echo [!] ERROR: Python is not installed or not found in system PATH.
    echo [!] Please install Python 3.10 or newer from https://www.python.org/
    echo [!] IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python -c "import sys; print('[i] Detected Python', sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [!] ERROR: Python 3.10 or newer is required.
    echo [!] Install a supported Python version and try again.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment if Missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment '.venv'...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Fast Dependency Check - Install only if missing
echo [*] Checking Python dependencies...
.\.venv\Scripts\python.exe -c "import duckdb, requests, lidlplus_api, zstandard, playwright, httpx, pandas, streamlit" >nul 2>&1
if errorlevel 1 (
    echo [*] Missing dependencies detected. Installing from requirements.txt...
    .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [!] ERROR: Package installation failed. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [+] Dependencies installed successfully.
)

:: 4. Verify .env and Lidl Refresh Token
set NEED_AUTH=0
call :CHECK_LIDL_TOKEN
if errorlevel 1 set NEED_AUTH=1

if "%NEED_AUTH%"=="1" (
    echo.
    echo [*] Valid Lidl Plus authentication token not found in '.env'.
    echo [*] Sync requires an authenticated Lidl Plus account.
    echo [*] The local review UI will not ask for your Lidl password.
    echo.

    :: Install a browser only when the authentication wizard is actually needed.
    if not exist "%LOCALAPPDATA%\ms-playwright" (
        echo [*] Installing Chromium for the official Lidl authentication flow...
        .\.venv\Scripts\python.exe -m playwright install chromium
        if errorlevel 1 (
            echo [!] ERROR: Failed to install Chromium for Lidl authentication.
            pause
            exit /b 1
        )
    )

    echo [*] Launching the existing Lidl Plus Authentication Wizard...
    .\.venv\Scripts\python.exe auth_login.py
    if errorlevel 1 (
        echo.
        echo [!] Authentication was aborted or failed. The control panel was not started.
        echo.
        pause
        exit /b 1
    )
    call :CHECK_LIDL_TOKEN
    if errorlevel 1 (
        echo [!] ERROR: Authentication finished without saving a Lidl refresh token.
        echo [!] Re-run the launcher or authenticate with 'python auth_login.py'.
        pause
        exit /b 1
    )
    echo.
)

:: 5. Start the local Streamlit control panel. Sync starts only after an explicit UI action.
:: Streamlit may search for another port when its configured port is occupied;
:: reject an existing listener first so normal startup stays on the promised URL.
.\.venv\Scripts\python.exe -c "import socket,sys; s=socket.socket(); result=s.connect_ex(('127.0.0.1', 8501)); s.close(); sys.exit(0 if result == 0 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [!] ERROR: Port 8501 is already in use on this computer.
    echo [!] Stop the other local app and run this launcher again.
    pause
    exit /b 1
)

echo [*] Starting the Lidl Spending ^& Nutrition control panel...
echo [*] Local address: http://127.0.0.1:8501
echo [*] No synchronization starts automatically. Use the Sync button when ready.
echo [*] Press Ctrl+C in this window to stop the control panel.
echo.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.showEmailPrompt false
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo =====================================================================
    echo [!] Streamlit stopped with exit code %APP_EXIT_CODE%.
    echo [!] If port 8501 is already in use, stop the other local app and try again.
    echo =====================================================================
    pause
    exit /b %APP_EXIT_CODE%
)

echo.
echo =====================================================================
echo [+] Local control panel stopped.
echo =====================================================================
pause
goto :EOF

:: Match the token condition used by sync_incremental.load_lidl_token without displaying it.
:CHECK_LIDL_TOKEN
.\.venv\Scripts\python.exe -c "from pathlib import Path; import sys; p=Path('.env'); tokens=[line.strip().split('=', 1)[1].strip() for line in p.read_text(encoding='utf-8').split('\n') if line.strip().startswith('LIDL_REFRESH_TOKEN=')] if p.is_file() else []; sys.exit(0 if tokens and tokens[-1] else 1)" >nul 2>&1
exit /b %ERRORLEVEL%
