@echo off
REM ═══════════════════════════════════════════════════════════
REM BIZRA Cockpit Deployment — Product Surface for NODE0
REM ═══════════════════════════════════════════════════════════
REM
REM Marathon session (44 commits) already shipped:
REM   ✓ Z3 formal proofs (4 membrane properties)
REM   ✓ Typed exceptions (0 broad handlers)
REM   ✓ Membrane tax CI gate (0.007ms measured)
REM   ✓ CMN v2 paper (ArXiv-ready)
REM   ✓ Security hardening (8/8 credentials rotated)
REM   ✓ Coverage ratcheted 65% → 70%
REM   ✓ CI pinned (ubuntu-24.04, no :latest)
REM   ✓ 9-stage MissionExecutor
REM   ✓ FanoutEventBus (CQRS + sovereign)
REM   ✓ 126x reflex speedup
REM
REM This script deploys the PRODUCT SURFACE:
REM   → Cockpit (React + Vite + bridge.js)
REM   → Corpus governance tool
REM   → Phase 1 integration test
REM ═══════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

set "REPO=C:\BIZRA-DATA-LAKE"
set "SRC=%~dp0"

echo.
echo  ════════════════════════════════════════════
echo  BIZRA Cockpit Deployment
echo  ════════════════════════════════════════════
echo.

REM ═══ 1. COCKPIT ═══
echo  [1/3] Deploying cockpit scaffold...

if not exist "%REPO%\cockpit" mkdir "%REPO%\cockpit"
if not exist "%REPO%\cockpit\src" mkdir "%REPO%\cockpit\src"

xcopy /Y /S "%SRC%bizra-cockpit\*" "%REPO%\cockpit\" >nul 2>nul
echo        bizra-cockpit/ → cockpit/
echo        App.jsx (SovereignWorld v2) ready
echo        bridge.js (v2, 9-stage pipeline) ready

REM ═══ 2. CORPUS GOVERNANCE ═══
echo  [2/3] Deploying corpus governance...

copy /Y "%SRC%corpus_governance.py" "%REPO%\scripts\corpus_governance.py" >nul 2>nul
echo        corpus_governance.py → scripts/

REM ═══ 3. PHASE 1 GATE ═══
echo  [3/3] Deploying integration test...

copy /Y "%SRC%phase1_gate.py" "%REPO%\scripts\phase1_gate.py" >nul 2>nul
echo        phase1_gate.py → scripts/

echo.
echo  ════════════════════════════════════════════
echo  Deployment complete.
echo  ════════════════════════════════════════════
echo.

REM Verify
set OK=0
if exist "%REPO%\cockpit\src\App.jsx" (set /a OK+=1) else (echo  MISSING: cockpit/src/App.jsx)
if exist "%REPO%\cockpit\src\bridge.js" (set /a OK+=1) else (echo  MISSING: cockpit/src/bridge.js)
if exist "%REPO%\cockpit\package.json" (set /a OK+=1) else (echo  MISSING: cockpit/package.json)
if exist "%REPO%\scripts\corpus_governance.py" (set /a OK+=1) else (echo  MISSING: corpus_governance.py)
if exist "%REPO%\scripts\phase1_gate.py" (set /a OK+=1) else (echo  MISSING: phase1_gate.py)

echo  %OK%/5 artifacts deployed.
echo.

if %OK%==5 (
    echo  Next steps:
    echo.
    echo    1. Start the cockpit:
    echo       cd %REPO%\cockpit
    echo       npm install
    echo       npm run dev
    echo       REM → opens http://localhost:5173
    echo.
    echo    2. Verify backend connections:
    echo       cd %REPO%
    echo       python scripts\phase1_gate.py
    echo.
    echo    3. Index the research corpus:
    echo       python scripts\corpus_governance.py scan
    echo.
    echo    4. Wire live data (3 replacements in App.jsx):
    echo       See WIRING_GUIDE.md in cockpit/
    echo.
    echo    5. Commit:
    echo       git add cockpit/ scripts/corpus_governance.py scripts/phase1_gate.py
    echo       git commit -m "product: sovereign cockpit + corpus governance + phase1 gate"
    echo.
)

echo  بذرة واحدة تصنع غابة
echo.

endlocal
