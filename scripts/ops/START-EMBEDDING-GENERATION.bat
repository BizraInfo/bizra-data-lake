@echo off
color 0A
title BIZRA Embedding Generator - RTX 4090 Beast Mode

echo.
echo  ======================================================================
echo                  BIZRA EMBEDDING GENERATOR - GPU ACCELERATED
echo  ======================================================================
echo.
echo   Target: 413,734 files from C:\BIZRA-NODE0\knowledge
echo   Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
echo   Hardware: RTX 4090 with 16GB VRAM
echo   Estimated Time: 2-4 hours
echo.
echo  ======================================================================
echo.

set /p confirm="Start embedding generation? This will use GPU heavily. (Y/N): "

if /i not "%confirm%"=="Y" (
    echo Operation cancelled.
    pause
    exit /b
)

echo.
echo [1/3] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing required packages...
pip install --break-system-packages sentence-transformers torch tqdm 2>&1 | findstr /C:"Requirement already satisfied" /C:"Successfully installed"

echo.
echo [3/3] Starting embedding generation...
echo.
echo NOTE: This will run for 2-4 hours. Safe to minimize window.
echo       Progress is saved every 100 files (resumable if interrupted).
echo       Check GPU usage with: nvidia-smi
echo.

python C:\BIZRA-DATA-LAKE\generate-embeddings.py

echo.
echo ======================================================================
echo   EMBEDDING GENERATION COMPLETE
echo ======================================================================
echo.
echo   Next steps:
echo   1. Check C:\BIZRA-NODE0\knowledge\embeddings\vectors\
echo   2. Review generation_stats.json for details
echo   3. Ready for Phase 3: Semantic search integration
echo.
pause
