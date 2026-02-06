@echo off
echo 🌌 ACTIVATING BIZRA SOVEREIGN PIPELINE (v2026.01)
echo --------------------------------------------------

set PY_EXE="C:\BIZRA-DATA-LAKE\.venv\Scripts\python.exe"

echo 🏛️ Phase 1: Canonizing Corpus (Layer 1)...
%PY_EXE% corpus_manager.py

echo 🧠 Phase 2: Generating Semantic Index (Layer 2)...
%PY_EXE% vector_engine.py

echo 🔍 Phase 3: High-Precision Extraction (Layer 4)...
%PY_EXE% langextract_engine.py

echo 🧬 Phase 4: Building Hypergraph Knowledge (Layer 3)...
%PY_EXE% build-hypergraph.py

echo --------------------------------------------------
echo ✅ BIZRA PIPELINE COMPLETE. READY FOR EXPLORATION.
pause
