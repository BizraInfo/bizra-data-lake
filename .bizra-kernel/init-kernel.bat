@echo off
setlocal enabledelayedexpansion
REM BIZRA Context Kernel Loader for Windows (High-SNR)

set "KERNEL_DIR=%~dp0"
REM Trim trailing backslash
if "!KERNEL_DIR:~-1!"=="\" set "KERNEL_DIR=!KERNEL_DIR:~0,-1!"
set "PROMPT_FILE=%TEMP%\bizra-kernel-init.txt"

echo.
echo ============================================================
echo BIZRA CONTEXT KERNEL v1.1 - INITIALIZATION (HIGH-SNR)
echo ============================================================
echo Location: !KERNEL_DIR!
echo.
echo This copies a concise activation prompt to your clipboard.
echo Paste it into Claude to load the kernel instantly.
echo.
echo Files that will be loaded:
echo   [1] core-context.json (essential, load first)
echo   [2] hooks.yaml (workflows)
echo   [3] memory.json (session persistence)
echo   [4] tools.yaml (tools and protocols)
echo   [5] standards.yaml (quality gates)
echo ============================================================
echo.

(
  echo Activate BIZRA Context Kernel (High-SNR).
  echo Load files in order:
  echo   1. !KERNEL_DIR!\core-context.json
  echo   2. !KERNEL_DIR!\hooks.yaml
  echo   3. !KERNEL_DIR!\memory.json
  echo   4. !KERNEL_DIR!\tools.yaml
  echo   5. !KERNEL_DIR!\standards.yaml
  echo.
  echo After loading, reply with:
  echo - Current local time
  echo - Loaded context summary
  echo - Available workflow hooks
  echo - Ready for autonomous execution
) > "!PROMPT_FILE!"

type "!PROMPT_FILE!" | clip

echo [SUCCESS] High-SNR prompt copied to clipboard.
echo.
echo Next Steps:
echo   1. Open Claude (web, desktop, or API)
echo   2. Paste the prompt (Ctrl+V)
echo   3. Claude will load full context in seconds
echo.
echo Alternative: run load-kernel.py --mode snr
echo.
echo ============================================================
echo.

endlocal
pause
