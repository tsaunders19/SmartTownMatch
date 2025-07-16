@echo off

cd /d %~dp0

SET ENVFILE=%~dp0.env
REM Track missing keys so we can abort if any are blank
SET MISSING_KEYS=0

IF NOT EXIST "%ENVFILE%" (
    echo Creating .env template. Please add your API keys before next run.
    (
        echo CENSUS_API_KEY=
        echo WALK_SCORE_API_KEY=
        echo GOOGLE_PLACES_API_KEY=
    ) > "%ENVFILE%"
) ELSE (
    for %%K in (CENSUS_API_KEY WALK_SCORE_API_KEY GOOGLE_PLACES_API_KEY) do (
        call :ensureKey %%K
        call :checkKey %%K
    )
)

REM Abort if any required API keys are missing
IF "%MISSING_KEYS%"=="1" (
    echo.
    echo One or more API keys are missing. Please open %ENVFILE% and fill in the values before running this script again.
    pause
    exit /b 1
)

cd backend
IF NOT EXIST .venv (
    echo Creating Python virtualenv...
    python -m venv .venv || goto :venv_fail
)

call .venv\Scripts\activate.bat

pip freeze > _current_reqs.txt
findstr /i \c:"flask" _current_reqs.txt >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Installing backend requirements...
    pip install -r requirements.txt || goto :pip_fail
)
DEL _current_reqs.txt 2>nul

start "SmartTownMatch Backend" cmd /k "cd /d %cd% & call .venv\Scripts\activate.bat & flask run"

cd ..\frontend
IF NOT EXIST node_modules (
    echo Installing frontend npm packages...
    npm install || goto :npm_fail
)

echo Starting React frontend...
npm start

exit /b 0

:checkKey
findstr /r "^%1=.*" "%ENVFILE%" | findstr /r "^%1=$" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo ERROR: %1 is empty in .env. Application will not start until this key is provided.
    SET MISSING_KEYS=1
)
exit /b 0

:venv_fail
echo ERROR: Failed to create Python virtual environment.
exit /b 1
:pip_fail
echo ERROR: pip install failed.
exit /b 1
:npm_fail
echo ERROR: npm install failed.
exit /b 1 

:ensureKey
findstr /b /c:"%1=" "%ENVFILE%" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo %1=>> "%ENVFILE%"
)
exit /b 0 