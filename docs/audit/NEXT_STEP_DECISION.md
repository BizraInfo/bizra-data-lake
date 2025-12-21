# BIZRA Next Step Decision
**Decision Date**: 2025-12-21 | **Auditor**: Claude Opus 4.5

---

## ✅ CHOSEN NEXT STEP: Fix CI YAML Syntax Error

### Why This Step

| Criterion | Rationale |
|-----------|-----------|
| **Highest Leverage** | Unblocks ALL downstream CI gates (security, quality, ihsān, performance, container) |
| **Lowest Effort** | Single character fix (remove 8 spaces of indentation at line 180) |
| **Zero Dependencies** | No prerequisites; can be done immediately |
| **Ethical Alignment** | Restores Ihsān gate enforcement; without it, unethical code could merge |
| **Evidence-Based** | YAML parser error confirmed in `get_errors` output |

### The Fix

**File**: `.github/workflows/elite-ci-cd.yml`  
**Line**: 180  
**Problem**: Step `- name: Verify HG-RTP compliance artifacts` is indented 8 spaces too many

**Before** (broken):
```yaml
      - name: Install deps
        run: python -m pip install pyyaml

        - name: Verify HG-RTP compliance artifacts  # ❌ 8 extra spaces
        run: |
```

**After** (fixed):
```yaml
      - name: Install deps
        run: python -m pip install pyyaml

      - name: Verify HG-RTP compliance artifacts  # ✅ Correct indentation
        run: |
```

---

## Acceptance Criteria

| # | Criterion | Verification Command |
|---|-----------|---------------------|
| 1 | YAML is valid | `python -c "import yaml; yaml.safe_load(open('.github/workflows/elite-ci-cd.yml'))"` |
| 2 | Workflow triggers | Push to branch; observe GitHub Actions run |
| 3 | Ihsān gate executes | CI log shows "☪️ Ihsān Gate" job |
| 4 | No new errors | `get_errors` on workflow file returns empty |

---

## Rollback Plan

If fix causes unexpected issues:

1. `git checkout HEAD~1 -- .github/workflows/elite-ci-cd.yml`
2. Push revert commit
3. Investigate in branch before re-attempting

**Risk Level**: MINIMAL - this is a pure formatting fix with no logic change

---

## Execution Sequence

```powershell
# Step 1: Fix the YAML
# (Apply the indentation fix to line 180)

# Step 2: Validate locally
python -c "import yaml; yaml.safe_load(open('.github/workflows/elite-ci-cd.yml')); print('YAML OK')"

# Step 3: Commit
git add .github/workflows/elite-ci-cd.yml
git commit -m "fix(ci): correct YAML indentation at line 180 - restore Ihsān gate"

# Step 4: Push and verify
git push origin feature/coderabbit-integration-dual-system

# Step 5: Monitor GitHub Actions
# Expected: All gates should now run (security → quality → ihsān → performance → container)
```

---

## Post-Fix: Next Priority

Once CI is green, proceed to **FACT_BACKLOG #2**: Fix Dockerfile.refinery CMD variable expansion to restore refinery service.

---

## Ihsān Attestation

This decision:
- ✅ Uses only verified evidence (YAML parser errors, docker logs, cargo test output)
- ✅ Proposes restorative action (restores ethics gate)
- ✅ Has explicit rollback plan
- ✅ Minimizes blast radius (single file, no logic change)
