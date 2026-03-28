@echo off
REM BIZRA Node0 Day Runner — handles Docker WSL conflicts
REM Usage: double-click or run from cmd

echo ============================================================
echo   BIZRA Node0 Day Runner
echo   Handles Docker WSL conflicts automatically
echo ============================================================
echo.

REM Step 1: Shutdown WSL cleanly (kills Docker WSL too)
echo [1/6] Shutting down WSL cleanly...
wsl --shutdown
timeout /t 5 /nobreak >nul

REM Step 2: Start Ubuntu specifically (not Docker)
echo [2/6] Starting Ubuntu WSL...
wsl -d Ubuntu -- echo "Ubuntu OK"
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: Ubuntu WSL did not start. Check WSL installation.
    pause
    exit /b 1
)

REM Step 3: Clean stale PIDs and start kernel
echo [3/6] Starting BIZRA kernel...
wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/boot_kernel.sh
echo.

REM Step 4: Wait for warmup
echo [4/6] Waiting 60s for FAISS + encoder warmup...
timeout /t 60 /nobreak >nul

REM Step 5: Check heartbeat
echo [5/6] Checking heartbeat...
wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/hb.sh
echo.

REM Step 6: Run missions
echo [6/6] Running 3 missions...
echo.
echo --- Mission 1 ---
wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/m1.sh
echo.
echo --- Mission 2 ---
wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/m2.sh
echo.
echo --- Mission 3 ---
wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/m3.sh
echo.

echo ============================================================
echo   Day complete. Generate manifest:
echo   wsl -d Ubuntu bash /mnt/c/BIZRA-DATA-LAKE/scripts/first_manifest.sh
echo ============================================================
pause
