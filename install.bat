@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BUILD_DIR=%ROOT%\build_windows_mingw"
set "INSTALL_DIR=%ROOT%\bin\windows"

if not defined AXCL_DIR set "AXCL_DIR=C:\Program Files\AXCL\axcl\out\axcl_win_x64"

where cmake >nul 2>nul || (
  echo Error: cmake not found in PATH.
  exit /b 1
)

where mingw32-make >nul 2>nul || (
  echo Error: mingw32-make not found in PATH.
  exit /b 1
)

where axcl-smi >nul 2>nul || (
  echo Error: axcl-smi not found in PATH.
  exit /b 1
)

if not exist "%AXCL_DIR%" (
  echo Error: AXCL_DIR not found: %AXCL_DIR%
  echo Hint: set AXCL_DIR to your AXCL SDK root, for example:
  echo   set AXCL_DIR=C:\Program Files\AXCL\axcl\out\axcl_win_x64
  exit /b 1
)

if not exist "%ROOT%\third_party\SimpleCV\CMakeLists.txt" (
  echo Error: missing third_party\SimpleCV. Please init submodules first.
  exit /b 1
)

if not exist "%ROOT%\third_party\openai-api.cpp\CMakeLists.txt" (
  echo Error: missing third_party\openai-api.cpp. Please init submodules first.
  exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 exit /b 1

echo ==^> Configuring Windows AXCL build with MinGW
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "MinGW Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%BUILD_DIR%\install" ^
  -DBUILD_AX650=OFF ^
  -DBUILD_AXCL=ON ^
  -DAXCL_DIR="%AXCL_DIR%"
if errorlevel 1 exit /b 1

echo ==^> Building axllm and llm_smoke
cmake --build "%BUILD_DIR%" --parallel 8
if errorlevel 1 exit /b 1

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if errorlevel 1 exit /b 1

copy /Y "%BUILD_DIR%\axllm.exe" "%INSTALL_DIR%\axllm.exe" >nul || exit /b 1
copy /Y "%BUILD_DIR%\llm_smoke.exe" "%INSTALL_DIR%\llm_smoke.exe" >nul || exit /b 1

if exist "%AXCL_DIR%\bin\axcl_rt.dll" copy /Y "%AXCL_DIR%\bin\axcl_rt.dll" "%INSTALL_DIR%\axcl_rt.dll" >nul
if exist "%AXCL_DIR%\axcl_rt.dll" copy /Y "%AXCL_DIR%\axcl_rt.dll" "%INSTALL_DIR%\axcl_rt.dll" >nul

echo Installed:
echo   %INSTALL_DIR%\axllm.exe
echo   %INSTALL_DIR%\llm_smoke.exe
exit /b 0
