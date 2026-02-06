@echo off
echo ===============================================================================
echo   BIZRA ECOSYSTEM MCP SERVER
echo   Unified Interface for Ultimate Engine, Orchestrator, Apex, and Peak
echo ===============================================================================
echo.
echo [1] Running in HTTP Mode (Port 8888)
echo     Access via: http://localhost:8888/
echo     Send JSON-RPC to: POST http://localhost:8888/
echo.
echo [2] For STDIO mode (Claude Desktop / Cursor):
echo     Use command: python ecosystem_mcp_server.py --stdio
echo.

cd /d "%~dp0"
python ecosystem_mcp_server.py --http --port 8888
pause