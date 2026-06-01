@echo off
echo ============================================
echo     ResumeAI - Setup and Launch
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)

echo [1/3] Python found. Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo.
echo [2/3] Dependencies installed successfully!
echo.

:: Ask for API key if not set
if "%GROQ_API_KEY%"=="" (
    echo [3/3] Starting ResumeAI...
    echo.
    echo NOTE: You can enter your Groq API key directly in the app header.
    echo Get your key at: https://console.groq.com
    echo.
) else (
    echo [3/3] Groq API key found in environment. Starting...
    echo.
)

echo ============================================
echo  App is starting at: http://localhost:7860
echo  Open this URL in your browser!
echo  Press Ctrl+C to stop the app.
echo ============================================
echo.

python app.py
pause
