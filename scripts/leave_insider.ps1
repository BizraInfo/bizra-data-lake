# leave_insider.ps1 v2 — Run as Administrator
# Forces exit from Canary by disabling SelfHost service BEFORE setting registry

try {
    Write-Host "=== Leave Windows Insider v2 (Canary -> Stable) ===" -ForegroundColor Cyan
    Write-Host "Build: $([System.Environment]::OSVersion.Version.Build)" -ForegroundColor Yellow

    # --- Step 1: Stop and disable the Windows Insider services ---
    Write-Host "`n[1/6] Stopping Windows Self-Host services..." -ForegroundColor Yellow
    $services = @("WaaSMedicSvc", "wisvc")
    foreach ($svc in $services) {
        $s = Get-Service $svc -ErrorAction SilentlyContinue
        if ($s) {
            Stop-Service $svc -Force -ErrorAction SilentlyContinue
            Set-Service $svc -StartupType Disabled -ErrorAction SilentlyContinue
            Write-Host "  $svc -> Stopped + Disabled" -ForegroundColor Green
        } else {
            Write-Host "  $svc -> Not found (OK)" -ForegroundColor DarkYellow
        }
    }

    # --- Step 2: Take ownership of SelfHost registry keys ---
    Write-Host "`n[2/6] Securing registry keys..." -ForegroundColor Yellow
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\UI\Selection",
        "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\Applicability"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Write-Host "  Found: $p" -ForegroundColor Green
        } else {
            New-Item -Path $p -Force | Out-Null
            Write-Host "  Created: $p" -ForegroundColor Yellow
        }
    }

    # --- Step 3: Set Insider to ReleasePreview ---
    Write-Host "`n[3/6] Setting channel to ReleasePreview..." -ForegroundColor Yellow
    $selPath = "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\UI\Selection"
    $appPath = "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\Applicability"

    # UI Selection
    Set-ItemProperty -Path $selPath -Name "UIBranch" -Value "ReleasePreview" -Type String -Force
    Set-ItemProperty -Path $selPath -Name "UIContentType" -Value "Mainline" -Type String -Force
    Set-ItemProperty -Path $selPath -Name "UIRing" -Value "External" -Type String -Force

    # Applicability
    Set-ItemProperty -Path $appPath -Name "BranchName" -Value "ReleasePreview" -Type String -Force
    Set-ItemProperty -Path $appPath -Name "ContentType" -Value "Mainline" -Type String -Force
    Set-ItemProperty -Path $appPath -Name "Ring" -Value "External" -Type String -Force
    Write-Host "  Channel set." -ForegroundColor Green

    # --- Step 4: Disable flighting ---
    Write-Host "`n[4/6] Disabling preview builds..." -ForegroundColor Yellow
    Set-ItemProperty -Path $appPath -Name "IsBuildFlightingEnabled" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $appPath -Name "EnablePreviewBuilds" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $appPath -Name "IsRetailOS" -Value 1 -Type DWord -Force
    Write-Host "  Flighting disabled, IsRetailOS=1." -ForegroundColor Green

    # --- Step 5: Clear backup values + block re-enrollment ---
    Write-Host "`n[5/6] Clearing backup refs and blocking re-enrollment..." -ForegroundColor Yellow
    Set-ItemProperty -Path $appPath -Name "BranchBackup" -Value "" -Type String -Force
    Set-ItemProperty -Path $appPath -Name "RingBackupV2" -Value "" -Type String -Force

    # Block Windows from auto-enrolling back
    $blockPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate"
    if (!(Test-Path $blockPath)) { New-Item -Path $blockPath -Force | Out-Null }
    Set-ItemProperty -Path $blockPath -Name "ManagePreviewBuildsPolicyValue" -Value 1 -Type DWord -Force
    Write-Host "  Group policy block set." -ForegroundColor Green

    # --- Step 6: Restart Windows Update service ---
    Write-Host "`n[6/6] Restarting Windows Update..." -ForegroundColor Yellow
    Restart-Service wuauserv -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
    $wuStatus = (Get-Service wuauserv).Status
    Write-Host "  Windows Update: $wuStatus" -ForegroundColor Green

    # --- Verify ---
    Write-Host "`n=== VERIFICATION ===" -ForegroundColor Cyan
    $app = Get-ItemProperty $appPath
    $sel = Get-ItemProperty $selPath
    Write-Host "  BranchName:      $($app.BranchName)"
    Write-Host "  Ring:            $($app.Ring)"
    Write-Host "  Flighting:       $($app.IsBuildFlightingEnabled)"
    Write-Host "  EnablePreview:   $($app.EnablePreviewBuilds)"
    Write-Host "  IsRetailOS:      $($app.IsRetailOS)"
    Write-Host "  UIBranch:        $($sel.UIBranch)"
    Write-Host "  GP Block:        $((Get-ItemProperty $blockPath -ErrorAction SilentlyContinue).ManagePreviewBuildsPolicyValue)"

    if ($app.BranchName -eq "ReleasePreview" -and $app.IsBuildFlightingEnabled -eq 0) {
        Write-Host "`n=== SUCCESS ===" -ForegroundColor Green
        Write-Host "Services disabled so they can't reset values." -ForegroundColor Cyan
        Write-Host "Group policy blocks re-enrollment." -ForegroundColor Cyan
        Write-Host "`nNext steps:" -ForegroundColor Yellow
        Write-Host "  1. Settings > Windows Update > Check for updates" -ForegroundColor White
        Write-Host "  2. If no updates appear, restart once more" -ForegroundColor White
        Write-Host "  3. After stable build installs, re-enable services:" -ForegroundColor White
        Write-Host "     Set-Service wisvc -StartupType Manual" -ForegroundColor DarkGray
        Write-Host "     Set-Service WaaSMedicSvc -StartupType Manual" -ForegroundColor DarkGray
    } else {
        Write-Host "`n=== CHECK VALUES ABOVE ===" -ForegroundColor DarkYellow
    }
}
catch {
    Write-Host "`n=== ERROR ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkYellow
}

Write-Host ""
Read-Host "Press Enter to close"
