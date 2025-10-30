@echo off
setlocal

set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
  echo ERROR: VsDevCmd.bat not found at %VSDEVCMD%
  exit /b 1
)

call %VSDEVCMD% -arch=x64 -host_arch=x64
if errorlevel 1 (
  echo ERROR: Failed to initialize VS Developer environment
  exit /b 1
)

set CC=cl
set CXX=cl
set NINJA_EXE=C:\Users\cisco\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe
if not exist "%NINJA_EXE%" (
  echo ERROR: Ninja not found at %NINJA_EXE%
  exit /b 1
)

cmake -S . -B build-ninja-msvc -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_POLICY_VERSION_MINIMUM=3.5
if errorlevel 1 exit /b 1

cmake --build build-ninja-msvc -j 8 --target vesper_tests
if errorlevel 1 exit /b 1

exit /b 0

