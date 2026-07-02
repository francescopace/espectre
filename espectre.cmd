@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%SCRIPT_DIR%espectre" %*
    exit /b %ERRORLEVEL%
)

where py >NUL 2>NUL
if %ERRORLEVEL% EQU 0 (
    py -3 "%SCRIPT_DIR%espectre" %*
    exit /b %ERRORLEVEL%
)

python "%SCRIPT_DIR%espectre" %*
exit /b %ERRORLEVEL%
