# prep_shrink.ps1 — Run as Administrator
# TARGET: Free enough space to create 2 TB B:\BIZRA partition
# Strategy: Delete B:, clean waste, clear blockers, defrag, then shrink C:

try {
    Write-Host "=== BIZRA Partition Prep — Target: 2 TB B: Drive ===" -ForegroundColor Cyan
    Write-Host ""

    # ================================================================
    # PHASE 1: DELETE EMPTY B: PARTITION
    # ================================================================
    Write-Host "[PHASE 1] Deleting empty B: partition..." -ForegroundColor Yellow
    $bVol = Get-Volume -DriveLetter B -ErrorAction SilentlyContinue
    if ($bVol) {
        $bSizeGB = [math]::Round($bVol.Size / 1GB, 2)
        # Find the partition object for B:
        $bPart = Get-Partition -DriveLetter B -ErrorAction SilentlyContinue
        if ($bPart) {
            Remove-Partition -DiskNumber $bPart.DiskNumber -PartitionNumber $bPart.PartitionNumber -Confirm:$false
            Write-Host "  B: deleted ($bSizeGB GB returned to unallocated)" -ForegroundColor Green
        } else {
            Write-Host "  B: volume found but partition lookup failed — delete manually in Disk Management" -ForegroundColor DarkYellow
        }
    } else {
        Write-Host "  B: not found (already deleted or not mounted)" -ForegroundColor Green
    }

    # ================================================================
    # PHASE 2: CLEAN BUILD ARTIFACTS (safe to delete — all rebuildable)
    # ================================================================
    Write-Host ""
    Write-Host "[PHASE 2] Cleaning build artifacts and caches..." -ForegroundColor Yellow
    $totalCleaned = 0

    # --- bizra-omega target/ (~85 GB) ---
    $targetDir = "C:\BIZRA-DATA-LAKE\bizra-omega\target"
    if (Test-Path $targetDir) {
        $targetSize = [math]::Round((Get-ChildItem $targetDir -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
        Write-Host "  Deleting bizra-omega/target/ ($targetSize GB)..." -ForegroundColor White
        Remove-Item $targetDir -Recurse -Force -ErrorAction SilentlyContinue
        $totalCleaned += $targetSize
        Write-Host "    -> Deleted" -ForegroundColor Green
    } else {
        Write-Host "  bizra-omega/target/ not found (already clean)" -ForegroundColor Green
    }

    # --- Redundant venvs (keep .venv, delete .venv-linux, .venv-wsl) ---
    $venvDirs = @(
        "C:\BIZRA-DATA-LAKE\.venv-linux",
        "C:\BIZRA-DATA-LAKE\.venv-wsl"
    )
    foreach ($vd in $venvDirs) {
        if (Test-Path $vd) {
            $vdSize = [math]::Round((Get-ChildItem $vd -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
            Write-Host "  Deleting $(Split-Path $vd -Leaf) ($vdSize GB)..." -ForegroundColor White
            Remove-Item $vd -Recurse -Force -ErrorAction SilentlyContinue
            $totalCleaned += $vdSize
            Write-Host "    -> Deleted" -ForegroundColor Green
        }
    }

    # --- __pycache__ directories across all BIZRA repos ---
    $bizraRoots = @(
        "C:\BIZRA-DATA-LAKE",
        "C:\BIZRA-NODE0",
        "C:\BIZRA-PROJECTS",
        "C:\BIZRA-Dual-Agentic-system--main"
    )
    $pycacheCount = 0
    $pycacheSize = 0
    foreach ($root in $bizraRoots) {
        if (Test-Path $root) {
            $caches = Get-ChildItem $root -Directory -Recurse -Filter "__pycache__" -Force -ErrorAction SilentlyContinue
            foreach ($cache in $caches) {
                $sz = (Get-ChildItem $cache.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
                $pycacheSize += $sz
                Remove-Item $cache.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $pycacheCount++
            }
        }
    }
    $pycacheSizeGB = [math]::Round($pycacheSize / 1GB, 2)
    Write-Host "  Deleted $pycacheCount __pycache__ dirs ($pycacheSizeGB GB)" -ForegroundColor Green
    $totalCleaned += $pycacheSizeGB

    # --- .mypy_cache directories ---
    $mypyCount = 0
    $mypySize = 0
    foreach ($root in $bizraRoots) {
        if (Test-Path $root) {
            $caches = Get-ChildItem $root -Directory -Recurse -Filter ".mypy_cache" -Force -ErrorAction SilentlyContinue
            foreach ($cache in $caches) {
                $sz = (Get-ChildItem $cache.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
                $mypySize += $sz
                Remove-Item $cache.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $mypyCount++
            }
        }
    }
    $mypySizeGB = [math]::Round($mypySize / 1GB, 2)
    Write-Host "  Deleted $mypyCount .mypy_cache dirs ($mypySizeGB GB)" -ForegroundColor Green
    $totalCleaned += $mypySizeGB

    # --- .ruff_cache directories ---
    foreach ($root in $bizraRoots) {
        if (Test-Path $root) {
            $ruffCaches = Get-ChildItem $root -Directory -Recurse -Filter ".ruff_cache" -Force -ErrorAction SilentlyContinue
            foreach ($cache in $ruffCaches) {
                Remove-Item $cache.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # --- Windows Temp ---
    $tempDir = $env:TEMP
    $tempSize = 0
    if (Test-Path $tempDir) {
        $tempSize = [math]::Round((Get-ChildItem $tempDir -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
        Get-ChildItem $tempDir -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Cleaned Windows Temp ($tempSize GB)" -ForegroundColor Green
        $totalCleaned += $tempSize
    }

    # --- User .cache (pip, npm, etc) ---
    $userCache = "$env:USERPROFILE\.cache"
    if (Test-Path $userCache) {
        $ucSize = [math]::Round((Get-ChildItem $userCache -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
        Write-Host "  User .cache found: $ucSize GB" -ForegroundColor White
        # Only delete pip/npm/huggingface caches (safe to rebuild)
        $safeCacheDirs = @("pip", "npm", "yarn", "nuget", "uv", "ruff", "pre-commit")
        foreach ($sc in $safeCacheDirs) {
            $scPath = Join-Path $userCache $sc
            if (Test-Path $scPath) {
                $scSize = [math]::Round((Get-ChildItem $scPath -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
                Remove-Item $scPath -Recurse -Force -ErrorAction SilentlyContinue
                $totalCleaned += $scSize
                Write-Host "    Deleted .cache/$sc ($scSize GB)" -ForegroundColor Green
            }
        }
    }

    Write-Host ""
    Write-Host "  TOTAL BUILD ARTIFACTS CLEANED: $([math]::Round($totalCleaned, 2)) GB" -ForegroundColor Cyan

    # ================================================================
    # PHASE 3: CLEAR UNMOVABLE FILE BLOCKERS
    # ================================================================
    Write-Host ""
    Write-Host "[PHASE 3] Clearing unmovable file blockers..." -ForegroundColor Yellow

    # --- Hibernation ---
    $hibFile = "$env:SystemDrive\hiberfil.sys"
    $hibSizeGB = 0
    if (Test-Path $hibFile -ErrorAction SilentlyContinue) {
        $hibSizeGB = [math]::Round((Get-Item $hibFile -Force).Length / 1GB, 2)
    }
    & powercfg /hibernate off 2>&1 | Out-Null
    $hiberPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power"
    Set-ItemProperty -Path $hiberPath -Name "HiberbootEnabled" -Value 0 -Type DWord -Force -ErrorAction SilentlyContinue
    Write-Host "  Hibernation OFF, Fast Startup OFF (freed ~$hibSizeGB GB)" -ForegroundColor Green

    # --- Pagefile → minimum ---
    $pfOldSize = 0
    $pageFiles = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
    if ($pageFiles) {
        foreach ($pf in $pageFiles) {
            $pfOldSize += $pf.AllocatedBaseSize
        }
    }
    $pfOldSizeGB = [math]::Round($pfOldSize / 1024, 2)

    $cs = Get-CimInstance Win32_ComputerSystem
    if ($cs.AutomaticManagedPagefile) {
        Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false}
    }
    $existingPF = Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue
    if ($existingPF) {
        $existingPF | Remove-CimInstance -ErrorAction SilentlyContinue
    }
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{
        Name = "C:\pagefile.sys"
        InitialSize = 4096
        MaximumSize = 4096
    } -ErrorAction SilentlyContinue | Out-Null
    $pfSaved = [math]::Round($pfOldSizeGB - 4, 2)
    if ($pfSaved -lt 0) { $pfSaved = 0 }
    Write-Host "  Pagefile: $pfOldSizeGB GB -> 4 GB (saves ~$pfSaved GB after restart)" -ForegroundColor Green

    # --- System Restore ---
    Disable-ComputerRestore -Drive "C:\" -ErrorAction SilentlyContinue
    vssadmin delete shadows /for=C: /all /quiet 2>&1 | Out-Null
    Write-Host "  System Restore OFF, shadow copies cleared" -ForegroundColor Green

    # --- Disable Search Indexer temporarily (holds unmovable files) ---
    Stop-Service WSearch -Force -ErrorAction SilentlyContinue
    Set-Service WSearch -StartupType Disabled -ErrorAction SilentlyContinue
    Write-Host "  Windows Search Indexer stopped + disabled" -ForegroundColor Green

    # ================================================================
    # PHASE 4: DEFRAG / CONSOLIDATE FREE SPACE
    # ================================================================
    Write-Host ""
    Write-Host "[PHASE 4] Running free-space consolidation on C:..." -ForegroundColor Yellow
    Write-Host "  This moves files toward the beginning of disk so shrink works better." -ForegroundColor DarkGray
    Write-Host "  This may take 10-30 minutes on a 3.5 TB drive..." -ForegroundColor DarkGray
    Write-Host ""

    # /D = full defrag + consolidate, /U = progress output
    defrag C: /D /U

    Write-Host ""
    Write-Host "  Defrag complete." -ForegroundColor Green

    # ================================================================
    # SUMMARY
    # ================================================================
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  PREP COMPLETE — SUMMARY" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $cFreeNow = [math]::Round((Get-Volume -DriveLetter C).SizeRemaining / 1GB, 2)
    $cSizeNow = [math]::Round((Get-Volume -DriveLetter C).Size / 1GB, 2)
    $afterRestartFree = [math]::Round($cFreeNow + $hibSizeGB + $pfSaved, 0)

    Write-Host "  C: size:              $cSizeNow GB" -ForegroundColor White
    Write-Host "  C: free now:          $cFreeNow GB" -ForegroundColor White
    Write-Host "  After restart free:   ~$afterRestartFree GB (hiberfil + pagefile shrink)" -ForegroundColor Cyan
    Write-Host "  Build artifacts:      $([math]::Round($totalCleaned, 2)) GB cleaned" -ForegroundColor White
    Write-Host "  B: partition:         Deleted (space returned)" -ForegroundColor White
    Write-Host ""
    Write-Host "=== NEXT STEPS ===" -ForegroundColor Green
    Write-Host "  1. RESTART the computer" -ForegroundColor White
    Write-Host "  2. Open Disk Management" -ForegroundColor White
    Write-Host "  3. Right-click C: -> Shrink Volume" -ForegroundColor White
    Write-Host "  4. Enter 2097152 MB (= 2 TB exactly) for shrink amount" -ForegroundColor White
    Write-Host "  5. Create new B: partition with all unallocated space" -ForegroundColor White
    Write-Host ""
    Write-Host "  If shrink still offers less than 2 TB:" -ForegroundColor DarkYellow
    Write-Host "  -> The iterative approach: shrink what you can, create B:," -ForegroundColor DarkGray
    Write-Host "     move BIZRA data C:->B:, then shrink C: more + extend B:" -ForegroundColor DarkGray
    Write-Host "  -> OR use MiniTool Partition Wizard Free (can force-move files)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  After partition is done, re-enable:" -ForegroundColor DarkGray
    Write-Host "    Set-Service WSearch -StartupType Automatic" -ForegroundColor DarkGray
    Write-Host "    Start-Service WSearch" -ForegroundColor DarkGray
}
catch {
    Write-Host "`n=== ERROR ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkYellow
}

Write-Host ""
Read-Host "Press Enter to close"
