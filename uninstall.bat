@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "INSTALL_DIR=%ROOT%\bin\windows"

set "REMOVED=0"

if exist "%INSTALL_DIR%\axllm.exe" (
  del /F /Q "%INSTALL_DIR%\axllm.exe"
  set "REMOVED=1"
)

if exist "%INSTALL_DIR%\llm_smoke.exe" (
  del /F /Q "%INSTALL_DIR%\llm_smoke.exe"
  set "REMOVED=1"
)

if exist "%INSTALL_DIR%\axcl_rt.dll" (
  del /F /Q "%INSTALL_DIR%\axcl_rt.dll"
  set "REMOVED=1"
)

if exist "%INSTALL_DIR%" (
  dir /b "%INSTALL_DIR%" >nul 2>nul && (
    rem keep directory if user put other files there
  ) || (
    rmdir "%INSTALL_DIR%" >nul 2>nul
  )
)

if "%REMOVED%"=="0" (
  echo Nothing to remove under %INSTALL_DIR%
) else (
  echo Removed Windows AXCL local install from %INSTALL_DIR%
)

exit /b 0
