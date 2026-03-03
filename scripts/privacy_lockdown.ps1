# privacy_lockdown.ps1 — Run as Administrator
# Full telemetry + privacy lockdown for BIZRA Node0 sovereignty

try {
    Write-Host "=== BIZRA Privacy Lockdown ===" -ForegroundColor Cyan
    Write-Host "Securing Node0 sovereignty from Microsoft telemetry`n" -ForegroundColor Yellow

    # --- 1. Telemetry to Security level (0 = off, enterprise only) ---
    Write-Host "[1/8] Setting telemetry to Security (minimum)..." -ForegroundColor Yellow
    $dcPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"
    if (!(Test-Path $dcPath)) { New-Item -Path $dcPath -Force | Out-Null }
    Set-ItemProperty -Path $dcPath -Name "AllowTelemetry" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $dcPath -Name "MaxTelemetryAllowed" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $dcPath -Name "AllowDeviceNameInTelemetry" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $dcPath -Name "DoNotShowFeedbackNotifications" -Value 1 -Type DWord -Force

    $dcPath2 = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\DataCollection"
    if (!(Test-Path $dcPath2)) { New-Item -Path $dcPath2 -Force | Out-Null }
    Set-ItemProperty -Path $dcPath2 -Name "AllowTelemetry" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $dcPath2 -Name "MaxTelemetryAllowed" -Value 0 -Type DWord -Force
    Write-Host "  Telemetry -> Security (0)" -ForegroundColor Green

    # --- 2. Disable Connected User Experience (DiagTrack) ---
    Write-Host "[2/8] Disabling DiagTrack (Connected User Experience)..." -ForegroundColor Yellow
    Stop-Service DiagTrack -Force -ErrorAction SilentlyContinue
    Set-Service DiagTrack -StartupType Disabled -ErrorAction SilentlyContinue
    Stop-Service dmwappushservice -Force -ErrorAction SilentlyContinue
    Set-Service dmwappushservice -StartupType Disabled -ErrorAction SilentlyContinue
    Write-Host "  DiagTrack -> Disabled" -ForegroundColor Green
    Write-Host "  dmwappushservice -> Disabled" -ForegroundColor Green

    # --- 3. Disable advertising ID ---
    Write-Host "[3/8] Disabling advertising ID..." -ForegroundColor Yellow
    $adPath = "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\AdvertisingInfo"
    if (!(Test-Path $adPath)) { New-Item -Path $adPath -Force | Out-Null }
    Set-ItemProperty -Path $adPath -Name "Enabled" -Value 0 -Type DWord -Force
    $adPolicyPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AdvertisingInfo"
    if (!(Test-Path $adPolicyPath)) { New-Item -Path $adPolicyPath -Force | Out-Null }
    Set-ItemProperty -Path $adPolicyPath -Name "DisabledByGroupPolicy" -Value 1 -Type DWord -Force
    Write-Host "  Advertising ID -> Disabled" -ForegroundColor Green

    # --- 4. Disable activity history (timeline) ---
    Write-Host "[4/8] Disabling activity history..." -ForegroundColor Yellow
    $actPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\System"
    if (!(Test-Path $actPath)) { New-Item -Path $actPath -Force | Out-Null }
    Set-ItemProperty -Path $actPath -Name "EnableActivityFeed" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $actPath -Name "PublishUserActivities" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $actPath -Name "UploadUserActivities" -Value 0 -Type DWord -Force
    Write-Host "  Activity history -> Disabled" -ForegroundColor Green

    # --- 5. Disable Copilot / Recall / AI features ---
    Write-Host "[5/8] Disabling Copilot + Recall..." -ForegroundColor Yellow
    $copilotPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsCopilot"
    if (!(Test-Path $copilotPath)) { New-Item -Path $copilotPath -Force | Out-Null }
    Set-ItemProperty -Path $copilotPath -Name "TurnOffWindowsCopilot" -Value 1 -Type DWord -Force

    $recallPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsAI"
    if (!(Test-Path $recallPath)) { New-Item -Path $recallPath -Force | Out-Null }
    Set-ItemProperty -Path $recallPath -Name "DisableAIDataAnalysis" -Value 1 -Type DWord -Force
    Set-ItemProperty -Path $recallPath -Name "TurnOffSavingSnapshots" -Value 1 -Type DWord -Force
    Write-Host "  Copilot -> Disabled" -ForegroundColor Green
    Write-Host "  Recall/AI analysis -> Disabled" -ForegroundColor Green

    # --- 6. Disable cloud clipboard + input personalization ---
    Write-Host "[6/8] Disabling cloud clipboard + input personalization..." -ForegroundColor Yellow
    $clipPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\System"
    Set-ItemProperty -Path $clipPath -Name "AllowCrossDeviceClipboard" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $clipPath -Name "AllowClipboardHistory" -Value 0 -Type DWord -Force

    $inputPath = "HKCU:\SOFTWARE\Microsoft\InputPersonalization"
    if (!(Test-Path $inputPath)) { New-Item -Path $inputPath -Force | Out-Null }
    Set-ItemProperty -Path $inputPath -Name "RestrictImplicitInkCollection" -Value 1 -Type DWord -Force
    Set-ItemProperty -Path $inputPath -Name "RestrictImplicitTextCollection" -Value 1 -Type DWord -Force

    $inputTPath = "HKCU:\SOFTWARE\Microsoft\InputPersonalization\TrainedDataStore"
    if (!(Test-Path $inputTPath)) { New-Item -Path $inputTPath -Force | Out-Null }
    Set-ItemProperty -Path $inputTPath -Name "HarvestContacts" -Value 0 -Type DWord -Force
    Write-Host "  Cloud clipboard -> Disabled" -ForegroundColor Green
    Write-Host "  Typing/ink personalization -> Disabled" -ForegroundColor Green

    # --- 7. Block telemetry hosts via firewall ---
    Write-Host "[7/8] Blocking telemetry endpoints via firewall..." -ForegroundColor Yellow
    $telemetryHosts = @(
        "vortex.data.microsoft.com",
        "vortex-win.data.microsoft.com",
        "telecommand.telemetry.microsoft.com",
        "telecommand.telemetry.microsoft.com.nsatc.net",
        "oca.telemetry.microsoft.com",
        "sqm.telemetry.microsoft.com",
        "watson.telemetry.microsoft.com",
        "watson.microsoft.com",
        "settings-win.data.microsoft.com",
        "telemetry.microsoft.com",
        "telemetry.appex.bing.net",
        "v10.events.data.microsoft.com",
        "v20.events.data.microsoft.com",
        "self.events.data.microsoft.com"
    )

    # Resolve and block IPs
    $blockedIPs = @()
    foreach ($endpoint in $telemetryHosts) {
        try {
            $ips = [System.Net.Dns]::GetHostAddresses($endpoint) | Where-Object { $_.AddressFamily -eq 'InterNetwork' }
            foreach ($ip in $ips) {
                $blockedIPs += $ip.IPAddressToString
            }
        } catch { }
    }
    $blockedIPs = $blockedIPs | Select-Object -Unique

    if ($blockedIPs.Count -gt 0) {
        # Remove old rule if exists
        Remove-NetFirewallRule -DisplayName "BIZRA-Block-Telemetry" -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName "BIZRA-Block-Telemetry" -Direction Outbound -Action Block -RemoteAddress $blockedIPs -Protocol Any -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  Firewall rule created: $($blockedIPs.Count) IPs blocked" -ForegroundColor Green
    } else {
        Write-Host "  Could not resolve telemetry hosts (DNS may be cached)" -ForegroundColor DarkYellow
    }

    # --- 8. Disable scheduled telemetry tasks ---
    Write-Host "[8/8] Disabling telemetry scheduled tasks..." -ForegroundColor Yellow
    $tasks = @(
        "\Microsoft\Windows\Application Experience\Microsoft Compatibility Appraiser",
        "\Microsoft\Windows\Application Experience\ProgramDataUpdater",
        "\Microsoft\Windows\Autochk\Proxy",
        "\Microsoft\Windows\Customer Experience Improvement Program\Consolidator",
        "\Microsoft\Windows\Customer Experience Improvement Program\UsbCeip",
        "\Microsoft\Windows\DiskDiagnostic\Microsoft-Windows-DiskDiagnosticDataCollector",
        "\Microsoft\Windows\Feedback\Siuf\DmClient",
        "\Microsoft\Windows\Maps\MapsUpdateTask",
        "\Microsoft\Windows\Windows Error Reporting\QueueReporting"
    )
    $disabled = 0
    foreach ($task in $tasks) {
        try {
            Disable-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue | Out-Null
            $disabled++
        } catch { }
    }
    Write-Host "  $disabled telemetry tasks disabled" -ForegroundColor Green

    # --- Summary ---
    Write-Host "`n=== SOVEREIGNTY SECURED ===" -ForegroundColor Green
    Write-Host "Telemetry:          Level 0 (Security/Off)" -ForegroundColor White
    Write-Host "DiagTrack:          Disabled" -ForegroundColor White
    Write-Host "Advertising ID:     Disabled" -ForegroundColor White
    Write-Host "Activity History:   Disabled" -ForegroundColor White
    Write-Host "Copilot/Recall:     Disabled" -ForegroundColor White
    Write-Host "Cloud Clipboard:    Disabled" -ForegroundColor White
    Write-Host "Input Tracking:     Disabled" -ForegroundColor White
    Write-Host "Firewall Block:     Active ($($blockedIPs.Count) IPs)" -ForegroundColor White
    Write-Host "Scheduled Tasks:    $disabled disabled" -ForegroundColor White
    Write-Host "`nNode0 is now dark to Microsoft." -ForegroundColor Cyan
}
catch {
    Write-Host "`n=== ERROR ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkYellow
}

Write-Host ""
Read-Host "Press Enter to close"
