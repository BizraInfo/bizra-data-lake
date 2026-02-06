@echo off
REM BIZRA CLI Launcher
REM بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

setlocal EnableDelayedExpansion

REM LM Studio API Key - set via environment variable
if "%LM_STUDIO_API_KEY%"=="" (
    echo WARNING: LM_STUDIO_API_KEY not set. Set it before running:
    echo   set LM_STUDIO_API_KEY=your_key_here
)

REM Check if we have arguments
if "%~1"=="" (
    REM No args - start TUI
    wsl -- bash -c "export LM_STUDIO_API_KEY='%LM_STUDIO_API_KEY%' && /mnt/c/BIZRA-DATA-LAKE/bizra"
) else if "%~1"=="help" (
    wsl -- /mnt/c/BIZRA-DATA-LAKE/bizra --help
) else if "%~1"=="--help" (
    wsl -- /mnt/c/BIZRA-DATA-LAKE/bizra --help
) else if "%~1"=="-h" (
    wsl -- /mnt/c/BIZRA-DATA-LAKE/bizra --help
) else (
    REM Pass all arguments with API key
    wsl -- bash -c "export LM_STUDIO_API_KEY='%LM_STUDIO_API_KEY%' && /mnt/c/BIZRA-DATA-LAKE/bizra %*"
)

endlocal
