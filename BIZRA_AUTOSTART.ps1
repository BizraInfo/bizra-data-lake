# BIZRA AUTO-START v1.0 [LOCAL-FIRST SOVEREIGN BOOT]
# Unified startup: LM Studio, Ollama, Docker services, kernel autoconfig, PAT memory restore.
# Register as Windows Startup: see Install-Autostart at bottom.
#
# Run manually: powershell -ExecutionPolicy Bypass -File C:\BIZRA-Dual-Agentic-system--main\BIZRA_AUTOSTART.ps1

$ErrorActionPreference = "Continue"
$BIZRA_ROOT = "C:\BIZRA-Dual-Agentic-system--main"
$BIZRA_HOME = "$env:USERPROFILE\.bizra"
$LOG_FILE = "$BIZRA_HOME\autostart.log"

# Ensure home directory exists
if (-not (Test-Path $BIZRA_HOME)) { New-Item -ItemType Directory -Path $BIZRA_HOME -Force | Out-Null }

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG_FILE -Value $line
}

function Log-OK($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [OK] $msg"
    Write-Host $line -ForegroundColor Green
    Add-Content -Path $LOG_FILE -Value $line
}

function Log-Warn($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [WARN] $msg"
    Write-Host $line -ForegroundColor Yellow
    Add-Content -Path $LOG_FILE -Value $line
}

Log "========== BIZRA AUTO-START SEQUENCE =========="
Log "Node: NODE0-GENESIS | Root: $BIZRA_ROOT"

# ═══════════════════════════════════════════════════
# PHASE 1: LLM Backends (Local-First)
# ═══════════════════════════════════════════════════
Log "PHASE 1: Starting Local LLM Backends..."

# --- Ollama ---
$ollamaRunning = Get-Process "ollama" -ErrorAction SilentlyContinue
if ($null -eq $ollamaRunning) {
    $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -ne $ollamaPath) {
        Log "Starting Ollama serve..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        Log-OK "Ollama started"
    } else {
        # Check common install locations
        $ollamaExe = "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
        if (Test-Path $ollamaExe) {
            Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
            Log-OK "Ollama started from AppData"
        } else {
            Log-Warn "Ollama not found. Install from https://ollama.com"
        }
    }
} else {
    Log-OK "Ollama already running"
}

# Verify Ollama and list models
try {
    $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    $modelNames = ($models.models | ForEach-Object { $_.name }) -join ", "
    Log-OK "Ollama models: $modelNames"
} catch {
    Log-Warn "Ollama not responding on :11434"
}

# --- LM Studio ---
$lmsRunning = Get-Process "LM Studio" -ErrorAction SilentlyContinue
if ($null -eq $lmsRunning) {
    $lmsPath = "C:\Users\$env:USERNAME\AppData\Local\Programs\LM Studio\LM Studio.exe"
    if (-not (Test-Path $lmsPath)) {
        $lmsPath = "C:\Program Files\LM Studio\LM Studio.exe"
    }
    if (Test-Path $lmsPath) {
        Log "Starting LM Studio..."
        Start-Process -FilePath $lmsPath -WindowStyle Minimized
        Start-Sleep -Seconds 5
        Log-OK "LM Studio started"
    } else {
        Log-Warn "LM Studio not found"
    }
} else {
    Log-OK "LM Studio already running"
}

# Verify LM Studio API
try {
    $lmsModels = Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -TimeoutSec 5 -ErrorAction Stop
    $lmsCount = $lmsModels.data.Count
    Log-OK "LM Studio: $lmsCount models loaded"
} catch {
    Log-Warn "LM Studio API not responding on :1234 (may need manual model loading)"
}

# ═══════════════════════════════════════════════════
# PHASE 2: Docker Services
# ═══════════════════════════════════════════════════
Log "PHASE 2: Starting Docker Services..."

# Check Docker Desktop
$dockerProc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
$dockerPipe = Test-Path "\\.\pipe\dockerDesktopLinuxEngine"

if ($null -eq $dockerProc -or -not $dockerPipe) {
    Log "Starting Docker Desktop..."
    if (Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe") {
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    }

    Log "Waiting for Docker engine (up to 120s)..."
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 3
        $info = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            break
        }
        Write-Host "." -NoNewline
    }
    Write-Host ""

    if ($ready) {
        Log-OK "Docker engine ready"
    } else {
        Log-Warn "Docker engine timeout - services may not start"
    }
} else {
    Log-OK "Docker Desktop running"
}

# Start BIZRA services
Push-Location $BIZRA_ROOT
try {
    Log "Starting Docker Compose services..."
    docker compose up -d 2>&1 | ForEach-Object { Log $_ }

    # Wait for critical services
    $criticalServices = @("synapse", "postgres", "wisdom", "vectors")
    foreach ($svc in $criticalServices) {
        $container = "bizra-dual-agentic-system--main-${svc}-1"
        $healthy = $false
        for ($i = 0; $i -lt 10; $i++) {
            $state = docker inspect --format="{{.State.Status}}" $container 2>$null
            if ($state -eq "running") { $healthy = $true; break }
            Start-Sleep -Seconds 2
        }
        if ($healthy) { Log-OK "$svc: running" } else { Log-Warn "$svc: not ready" }
    }
} finally {
    Pop-Location
}

# ═══════════════════════════════════════════════════
# PHASE 3: Kernel Auto-Configuration
# ═══════════════════════════════════════════════════
Log "PHASE 3: Running Kernel Auto-Configuration..."

# Run autoconfig in WSL
$autoconfigResult = wsl -d Ubuntu -- bash -c "cd /mnt/c/BIZRA-Dual-Agentic-system--main && python3 -c 'import asyncio; from core.autoconfig import auto_configure; import json; print(json.dumps(asyncio.run(auto_configure()), indent=2))' 2>/dev/null" 2>$null

if ($null -ne $autoconfigResult -and $autoconfigResult -ne "") {
    $configPath = "$BIZRA_HOME\autoconfig.json"
    $autoconfigResult | Out-File -FilePath $configPath -Encoding utf8
    Log-OK "Auto-config saved to $configPath"
} else {
    Log-Warn "Auto-config skipped (module not loaded or WSL unavailable)"
}

# ═══════════════════════════════════════════════════
# PHASE 4: PAT Memory Restore
# ═══════════════════════════════════════════════════
Log "PHASE 4: Restoring PAT Memory..."

$memFile = "$BIZRA_HOME\pat_memory.json"
if (Test-Path $memFile) {
    $memSize = (Get-Item $memFile).Length
    Log-OK "PAT memory found: $([math]::Round($memSize/1024, 1))KB"

    # Restore to Redis via WSL
    $restored = wsl -d Ubuntu -- bash -c "cd /mnt/c/BIZRA-Dual-Agentic-system--main && python3 -c 'import asyncio; from core.pat_memory import get_pat_memory; m=asyncio.run(get_pat_memory().__aenter__()); asyncio.run(m.load_from_disk()); print(\"restored\")' 2>/dev/null" 2>$null
    if ($restored -match "restored") {
        Log-OK "PAT memory restored to Redis"
    } else {
        Log-Warn "PAT memory restore skipped (Redis may not be ready)"
    }
} else {
    Log "No previous PAT memory found (first boot)"
}

# ═══════════════════════════════════════════════════
# PHASE 5: Health Verification
# ═══════════════════════════════════════════════════
Log "PHASE 5: System Health Verification..."

$healthChecks = @(
    @{Name="Ollama";   URL="http://localhost:11434/api/tags"},
    @{Name="LM Studio"; URL="http://localhost:1234/v1/models"},
    @{Name="Kernel";   URL="http://localhost:8010/healthz"},
    @{Name="Elite";    URL="http://localhost:8080/health"},
    @{Name="Synapse";  Check={redis-cli -h localhost -p 6380 -a bizra_synapse_secure ping 2>$null}}
)

$online = 0
$total = $healthChecks.Count

foreach ($check in $healthChecks) {
    try {
        if ($check.URL) {
            $null = Invoke-RestMethod -Uri $check.URL -TimeoutSec 3 -ErrorAction Stop
        }
        Log-OK "$($check.Name): ONLINE"
        $online++
    } catch {
        Log-Warn "$($check.Name): OFFLINE"
    }
}

Log "========== BIZRA BOOT COMPLETE =========="
Log "Services: $online/$total online"
Log "Mode: LOCAL-FIRST SOVEREIGN"
Log "Log: $LOG_FILE"
Log "========================================="

# ═══════════════════════════════════════════════════
# INSTALLER: Register as Windows Startup Task
# ═══════════════════════════════════════════════════
function Install-Autostart {
    param([switch]$OnLogin, [switch]$OnBoot)

    $scriptPath = "$BIZRA_ROOT\BIZRA_AUTOSTART.ps1"
    $taskName = "BIZRA-AutoStart"

    # Remove existing task
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""

    if ($OnBoot) {
        # Run at system startup (requires admin)
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest
    } else {
        # Run at user login (default, no admin needed)
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME"
    }

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "BIZRA Node0 Auto-Start: Local LLMs, Docker services, kernel autoconfig, PAT memory restore"

    Write-Host "`n[BIZRA] Auto-start registered as scheduled task: $taskName" -ForegroundColor Green
    Write-Host "[BIZRA] BIZRA will start automatically on every Windows login." -ForegroundColor Green
}

# To install: Run in PowerShell as Admin:
#   . C:\BIZRA-Dual-Agentic-system--main\BIZRA_AUTOSTART.ps1
#   Install-Autostart
#
# To install for system boot (requires admin):
#   Install-Autostart -OnBoot
