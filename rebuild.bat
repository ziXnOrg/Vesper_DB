@echo off
echo === Rebuilding Vesper with AVX2 Support ===

REM Change to build directory
cd /d "%~dp0build"

REM Reconfigure with CMake
echo.
echo Reconfiguring CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release -DVESPER_ENABLE_KERNEL_DISPATCH=ON

REM Build the project
echo.
echo Building project...
cmake --build . --config Release --parallel

REM Run the SIMD kernel test
echo.
echo Running SIMD kernel test...
Release\simd_kernels_test.exe

echo.
echo Build complete!
pause