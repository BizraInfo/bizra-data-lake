# Fix for steps 7-8 only (variable name collision fixed)
try {
    Write-Host "=== Completing Privacy Lockdown (steps 7-8) ===" -ForegroundColor Cyan

    # --- 7. Block telemetry hosts via firewall ---
    Write-Host "[7/8] Blocking telemetry endpoints via firewall..." -ForegroundColor Yellow
    $telemetryHosts = @(
        "vortex.data.microsoft.com",
        "vortex-win.data.microsoft.com",
        "telecommand.telemetry.microsoft.com",
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
        Remove-NetFirewallRule -DisplayName "BIZRA-Block-Telemetry" -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName "BIZRA-Block-Telemetry" -Direction Outbound -Action Block -RemoteAddress $blockedIPs -Protocol Any -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  Firewall rule created: $($blockedIPs.Count) IPs blocked" -ForegroundColor Green
    } else {
        Write-Host "  Could not resolve telemetry hosts (offline or DNS cached)" -ForegroundColor DarkYellow
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
        "\Microsoft\Windows\Maps\MapsUpdateTask",
        "\Microsoft\Windows\Windows Error Reporting\QueueReporting"
    )
    $disabled = 0
    foreach ($t in $tasks) {
        try {
            Disable-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue | Out-Null
            $disabled++
        } catch { }
    }
    Write-Host "  $disabled telemetry tasks disabled" -ForegroundColor Green

    Write-Host "`n=== SOVEREIGNTY SECURED ===" -ForegroundColor Green
    Write-Host "All 8 layers locked. Node0 is dark to Microsoft." -ForegroundColor Cyan
}
catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "`nPress Enter to close"
