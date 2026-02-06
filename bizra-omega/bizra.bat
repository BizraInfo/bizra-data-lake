@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM BIZRA CLI — Launch Script
REM ═══════════════════════════════════════════════════════════════════════════
REM
REM Usage:
REM   bizra           - Start TUI interface
REM   bizra status    - Show node status
REM   bizra info      - Show system info
REM   bizra agent list - List PAT agents
REM   bizra --help    - Show all commands
REM
REM ═══════════════════════════════════════════════════════════════════════════

"%~dp0target\release\bizra.exe" %*
