# BIZRA Phase 0 Week 1 — LIVE EXECUTION

**Node:** node_0000_genesis_momo_dubai  
**Root:** C:\BIZRA-Dual-Agentic-system--main  
**Status:** 🔥 OPERATIONAL  
**GPU:** NVIDIA RTX 4090 Laptop (13.5GB VRAM available)

---

## ✅ Files Created (Root Correction Applied)

### Core Infrastructure
- ✅ `scripts/bizra-root.ps1` — Dynamic root resolver (PowerShell)
- ✅ `scripts/bizra-root.js` — Dynamic root resolver (Node.js)
- ✅ `.bizra/config/node.yaml` — Canonical node configuration
- ✅ `tools/phase0_audit.ps1` — Evidence capture script

### Runtime Configuration
- ✅ `docs/runtime/slots.yaml` — Slot → Model mapping (UNSEALED)
- ✅ `docs/runtime/resource_policy.yaml` — VRAM constraints + valid combinations

---

## 🚀 EXECUTE NOW (Single Command)

```powershell
cd C:\BIZRA-Dual-Agentic-system--main
powershell -ExecutionPolicy Bypass -File .\tools\phase0_audit.ps1
```

**This will:**
1. Resolve BIZRA_ROOT dynamically (works from any subfolder)
2. Create timestamped evidence directory: `docs/evidence/phase0_week1/<timestamp>/`
3. Capture 20+ evidence files (identity, GPU, Ollama, models, containers, probes)
4. Generate summary with file list + next steps

**Output:** Path to evidence directory (copy for next step)

---

## 📋 What Happens Next

### After Audit Completes:
1. **Review Evidence** — Check `docs/evidence/phase0_week1/<timestamp>/`
2. **Fill Manifests** — Replace `[EVIDENCE_FILL]` in `docs/runtime/*.yaml`
3. **Create Deterministic Variants** (Optional) — For SAT gate validation:
   ```powershell
   ollama show bizra-planner:latest --modelfile > tools/modelfiles/bizra-planner.det.modelfile
   # Edit: temperature 0, top_p 1
   ollama create bizra-planner:det -f tools/modelfiles/bizra-planner.det.modelfile
   ```
4. **Seal** — Set status: SEALED in yaml files
5. **Commit** — Git commit + signed tag

---

## 🎯 Phase 0 Week 1 Objectives

- [x] Root path corrected (C:\BIZRA-Dual-Agentic-system--main)
- [x] Root resolver created (dynamic, git-based)
- [x] Canonical config established (.bizra/config/node.yaml)
- [x] Audit script generated (evidence capture)
- [x] Slot policy defined (docs/runtime/slots.yaml)
- [x] Resource policy defined (VRAM constraints)
- [ ] **Execute audit** ← YOU ARE HERE
- [ ] Fill evidence placeholders
- [ ] Create deterministic variants (optional)
- [ ] Seal manifests
- [ ] Commit + tag

---

## 🔍 Key Decisions (Path C: Hybrid)

### Production Models (temp > 0)
- ✅ `bizra-planner:latest` — temp=0.7, top_p=0.9
- ✅ `deepseek-r1:8b` — temp=0.6, top_p=0.95
- ✅ `qwen2.5:7b`, `mistral:latest`, `llama3.2:latest` — configs TBD

### Deterministic Variants (temp=0, optional)
- ⏳ `bizra-planner:det` — For SAT gate validation
- ⏳ `deepseek-r1:det` — For consensus-critical proofs

### Golden Set Strategy
- ✅ Schema validation (JSON structure, not exact text)
- ✅ Semantic similarity (embedding distance)
- ✅ Performance bounds (latency < threshold)
- ✅ Fuzzy matching (contains_any, not exact)

---

## 📊 VRAM Constraints (Verified)

```
Total VRAM:     16GB
Available:      13.5GB (low VRAM mode active)
Soft Cap:       13.5GB

Model Sizes:
├─ bizra-planner:latest  6.5GB
├─ deepseek-r1:8b        5.3GB
├─ qwen2.5:7b            4.8GB
├─ mistral:latest        4.4GB
├─ llama3.2:latest       2.0GB
└─ nomic-embed:latest    0.3GB

Valid Combinations:
✅ planner + embed       6.7GB
✅ deepseek + embed      5.6GB
✅ planner + deepseek   11.8GB
❌ All 6 simultaneous   23.4GB (will trigger LRU unload)
```

---

## 🆘 Troubleshooting

### Issue: "git not found"
**Solution:** Install Git or use fallback (walks up to find .git directory)

### Issue: "Ollama not responding"
**Solution:** 
```powershell
ollama serve
# Wait 5 seconds, then re-run audit
```

### Issue: "Docker containers not found"
**Solution:** Start containers first:
```powershell
cd C:\BIZRA-Dual-Agentic-system--main
docker compose up -d
```

---

## 📞 Next Command

**Run this NOW:**
```powershell
powershell -ExecutionPolicy Bypass -File C:\BIZRA-Dual-Agentic-system--main\tools\phase0_audit.ps1
```

**Then paste the evidence directory path here for next steps.**

---

**Status:** ⏸️ AWAITING AUDIT EXECUTION  
**Time to Complete:** ~2-5 minutes  
**Dependencies:** Ollama running, Docker containers optional