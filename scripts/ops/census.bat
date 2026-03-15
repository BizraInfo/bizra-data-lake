@echo off
echo === PAT DATA SOURCE CENSUS (native NTFS speed) ===
echo.

echo --- C:\BIZRA-DATA-LAKE ---
dir /s /a-d "C:\BIZRA-DATA-LAKE" 2>nul | find "File(s)"
echo.

echo --- C:\BIZRA-Dual-Agentic-system--main ---
dir /s /a-d "C:\BIZRA-Dual-Agentic-system--main" 2>nul | find "File(s)"
echo.

echo --- C:\BIZRA-NODE0 ---
dir /s /a-d "C:\BIZRA-NODE0" 2>nul | find "File(s)"
echo.

echo --- C:\Users\BIZRA-OS\Downloads ---
dir /s /a-d "C:\Users\BIZRA-OS\Downloads" 2>nul | find "File(s)"
echo.

echo --- B:\BIZRA-SOVEREIGN ---
dir /s /a-d "B:\BIZRA-SOVEREIGN" 2>nul | find "File(s)"
echo.

echo --- C:\BIZRA-PROJECTS ---
dir /s /a-d "C:\BIZRA-PROJECTS" 2>nul | find "File(s)"
echo.

echo --- C:\BIZRA-TaskMaster ---
dir /s /a-d "C:\BIZRA-TaskMaster" 2>nul | find "File(s)"
echo.

echo === DONE ===
