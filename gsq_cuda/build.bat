@echo off
rem Build gsq_splat.dll (CUDA C++ blob renderer) for the RTX 5060 (sm_120).
rem Uses portable MSVC (%LOCALAPPDATA%\msvc-portable) + CUDA 12.8 (%LOCALAPPDATA%\cuda-12.8).
cd /d "%~dp0"
call "%LOCALAPPDATA%\msvc-portable\setup_x64.bat"
set "CUDA_PATH=%LOCALAPPDATA%\cuda-12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
nvcc -O3 -shared -o gsq_splat.dll splat.cu ^
  -gencode arch=compute_120,code=sm_120 ^
  -gencode arch=compute_120,code=compute_120 ^
  -Xcompiler "/MD /O2" -cudart static
