@echo off
REM BIZRA Sovereign Node — Windows CLI Launcher
REM Usage: bizra          → Launch Rust TUI (graphical dashboard)
REM        bizra repl     → Python interactive REPL
REM        bizra status   → System status
REM        bizra --help   → All commands

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "RUST_TUI=%PROJECT_ROOT%\bizra-omega\target\release\bizra.exe"
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

REM ── No arguments: launch TUI ──
if "%~1"=="" (
    if exist "%RUST_TUI%" (
        "%RUST_TUI%"
        exit /b %errorlevel%
    )
    if exist "%PROJECT_ROOT%\bizra.exe" (
        "%PROJECT_ROOT%\bizra.exe"
        exit /b %errorlevel%
    )
    if exist "%PYTHON%" (
        "%PYTHON%" -m core.sovereign
        exit /b %errorlevel%
    )
    echo Error: No Rust TUI binary or Python found.
    echo Build TUI:  cd bizra-omega ^&^& cargo build --release -p bizra-cli
    exit /b 1
)

REM ── TUI subcommand ──
if /i "%~1"=="tui" (
    if exist "%RUST_TUI%" (
        "%RUST_TUI%"
        exit /b %errorlevel%
    )
    echo Rust TUI not found. Build with:
    echo   cd bizra-omega ^&^& cargo build --release -p bizra-cli
    exit /b 1
)

REM ── Help ──
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

REM ── Version ──
if "%~1"=="--version" goto :show_version
if "%~1"=="-V" goto :show_version

REM ── All other commands → Python ──
if exist "%PYTHON%" (
    "%PYTHON%" -m core.sovereign %*
    exit /b %errorlevel%
)
echo Error: Python not found at %PYTHON%
exit /b 1

:show_help
echo BIZRA Sovereign Node — Unified CLI
echo.
echo Usage:
echo   bizra                Launch graphical TUI (Rust ratatui)
echo   bizra tui            Launch graphical TUI explicitly
echo   bizra repl           Python interactive REPL
echo   bizra status         Show system status
echo   bizra query "..."    Single query with SNR scoring
echo   bizra dashboard      Node identity dashboard
echo   bizra impact         Sovereignty progression
echo   bizra wallet         Token balances (SEED, BLOOM, IMPT)
echo   bizra tokens         Token supply and ledger
echo   bizra onboard        Create sovereign identity
echo   bizra setup          First-time setup wizard
echo   bizra start          Full system (API + dashboard + proactive)
echo   bizra live           Real-time Rich dashboard
echo   bizra serve          API server (default port 8080)
echo   bizra doctor         System health check
echo   bizra sel            Sovereign Experience Ledger
echo   bizra verify         Verify genesis/chain/tokens
echo   bizra --version      Show version
echo.
echo Standing on Giants: Shannon • Lamport • Besta • Vaswani • Anthropic
exit /b 0

:show_version
echo BIZRA Sovereign Engine v1.0.0
echo Codename: Genesis
echo Standing on Giants: Shannon • Lamport • Vaswani • Anthropic
exit /b 0
