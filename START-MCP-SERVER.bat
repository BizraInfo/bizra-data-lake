@echo off
echo ========================================================
echo   BIZRA DDAGI OS v2.0.0 - MCP SERVER LAUNCHER
echo ========================================================
echo.
echo [1/2] Activating Environment...
call .venv\Scripts\activate.bat

echo [2/2] Starting BIZRA MCP Server...
echo       (Standard Input/Output Mode)
echo.

python bizra_mcp.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server crashed with exit code %errorlevel%
    pause
)
