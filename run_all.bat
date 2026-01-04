@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ==================================================
echo Starting full SFA stack (Node + Simulators + Python APIs + Chatbot)
echo ==================================================
echo.
echo [0/7] Checking Node dependencies (if needed)...
REM --- npm install: skip if node_modules exists, otherwise ask ---
if exist node_modules (
    echo node_modules found - skipping npm install.
) else (
    if exist package.json (
    set /p RUN_NPM=Run npm install now? Y/N: 
        if /I "%RUN_NPM%"=="Y" (
            echo Running npm install...
            npm install
        ) else (
            echo Skipping npm install as requested.
        )
    ) else (
        echo package.json not found, skipping npm install.
    )
)

REM ensure logs directory
if not exist "%~dp0logs" mkdir "%~dp0logs"

echo [1/7] Starting Ollama Server (if required)...
echo If you don't need Ollama, you can skip this step in run_all.bat or comment it out.
START "Ollama Server" cmd /k "ollama serve 1>"%~dp0logs\ollama.log" 2>&1"
timeout /t 2 /nobreak > nul
echo.
echo [2/7] Starting Gateway Server (Node.js) on port 3000
START "SFA Gateway" cmd /k "node server.js 1>"%~dp0logs\node.log" 2>&1"
echo [3/7] Starting Simulators (each in its own window)
START "12 Simulator" cmd /k ""%~dp0\12계통\12 pratice.bat"" 1>"%~dp0logs\sim12.log" 2>&1
START "22 Simulator" cmd /k ""%~dp0\22계통\22 pratice.bat"" 1>"%~dp0logs\sim22.log" 2>&1
START "DC Simulator" cmd /k ""%~dp0\본선 시뮬레이션(최종)\dc pratice.bat"" 1>"%~dp0logs\simdc.log" 2>&1
echo [4/7] Starting Analysis API Server (Python) on port 8000
START "SFA Analysis API" cmd /k ""%~dp0myenv313\Scripts\python.exe" -m uvicorn analysis_api:app --host 0.0.0.0 --port 8000 1>"%~dp0logs\analysis.log" 2>&1"
echo [5/7] Starting Prediction API Server (Python) on port 8002
START "SFA Prediction API" cmd /k ""%~dp0myenv313\Scripts\python.exe" -m uvicorn prediction_api:app --host 0.0.0.0 --port 8002 1>"%~dp0logs\prediction.log" 2>&1"
echo [6/7] Starting Chatbot Server (Python Flask) on port 5000
START "DTRO Chatbot" cmd /k "cd /d "%~dp0chatbot" && python chatbot_qdrant.py"
echo Waiting for services to initialize (8 seconds)...
timeout /t 8 /nobreak > nul
echo.
echo [7/7] Launching SFA Dashboard in default browser...
start "" "http://localhost:3000"
echo.
echo All start commands issued. Check the opened windows for logs and errors.
echo If something failed, open a terminal and run the failing command manually to see error details.
echo ==================================================
pause
