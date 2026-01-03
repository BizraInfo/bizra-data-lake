<#
.SYNOPSIS
    BIZRA Quality Gate - CI/CD Integration Script
    
.DESCRIPTION
    Runs the Quality Radar Elite and enforces Ihsān threshold gates.
    Exits with code 0 on success, 1 on failure.
    
.PARAMETER SkipTests
    Skip cargo tests for faster validation
    
.PARAMETER Threshold
    Override the Ihsān threshold (default: environment-based)
    
.PARAMETER Env
    Set environment (development, ci, production)
    
.PARAMETER OutputDir
    Directory for output files
    
.EXAMPLE
    .\run-quality-gate.ps1 -SkipTests -Env ci
    
.NOTES
    Part of BIZRA Elite CI Integrity Gates
#>

param(
    [switch]$SkipTests,
    [double]$Threshold = 0,
    [ValidateSet('development', 'ci', 'production')]
    [string]$Env = 'ci',
    [string]$OutputDir = 'evidence'
)

$ErrorActionPreference = 'Stop'
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# Colors
function Write-Color {
    param([string]$Text, [ConsoleColor]$Color = 'White')
    $old = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Text
    $Host.UI.RawUI.ForegroundColor = $old
}

Write-Color "`n╔══════════════════════════════════════════════════════════════════════╗" Cyan
Write-Color "║           BIZRA QUALITY GATE - CI INTEGRITY CHECK                  ║" Cyan
Write-Color "╚══════════════════════════════════════════════════════════════════════╝" Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$radarScript = Join-Path $repoRoot "scripts" "quality_radar_elite.py"
$outputPath = Join-Path $repoRoot $OutputDir "quality_gate"

# Set environment
$env:BIZRA_ENV = $Env

# Default thresholds by environment
$thresholds = @{
    'development' = 0.80
    'ci' = 0.90
    'production' = 0.95
}

$actualThreshold = if ($Threshold -gt 0) { $Threshold } else { $thresholds[$Env] }

Write-Host "`n📋 Configuration:"
Write-Host "   Environment: $Env"
Write-Host "   Threshold:   $actualThreshold"
Write-Host "   Skip Tests:  $SkipTests"
Write-Host "   Output:      $outputPath"

# Build command
$pythonArgs = @(
    $radarScript,
    "--json",
    "--prometheus",
    "--ci",
    "--threshold", $actualThreshold,
    "-o", $outputPath
)

if ($SkipTests) {
    $pythonArgs += "--skip-tests"
}

Write-Host "`n🚀 Running Quality Radar Elite...`n"
$startTime = Get-Date

try {
    # Run Python script
    $result = & python @pythonArgs
    $exitCode = $LASTEXITCODE
    
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    if ($exitCode -eq 0) {
        Write-Color "`n╔══════════════════════════════════════════════════════════════════════╗" Green
        Write-Color "║                    ✅ QUALITY GATE PASSED                            ║" Green
        Write-Color "╚══════════════════════════════════════════════════════════════════════╝" Green
        Write-Host "`n   Environment: $Env"
        Write-Host "   Threshold:   $actualThreshold"
        Write-Host "   Duration:    $([math]::Round($elapsed, 1))s"
        Write-Host "   Output:      $outputPath.*"
        
        # GitHub Actions format
        if ($env:GITHUB_ACTIONS) {
            Write-Host "::notice title=Quality Gate::Ihsān threshold passed (>= $actualThreshold)"
        }
        
        exit 0
    } else {
        Write-Color "`n╔══════════════════════════════════════════════════════════════════════╗" Red
        Write-Color "║                    ❌ QUALITY GATE FAILED                            ║" Red
        Write-Color "╚══════════════════════════════════════════════════════════════════════╝" Red
        Write-Host "`n   Environment: $Env"
        Write-Host "   Threshold:   $actualThreshold"
        Write-Host "   Duration:    $([math]::Round($elapsed, 1))s"
        
        # GitHub Actions format
        if ($env:GITHUB_ACTIONS) {
            Write-Host "::error title=Quality Gate Failed::Ihsān score below threshold ($actualThreshold)"
        }
        
        exit 1
    }
}
catch {
    Write-Color "`n❌ Error running quality radar: $_" Red
    
    if ($env:GITHUB_ACTIONS) {
        Write-Host "::error title=Quality Gate Error::$_"
    }
    
    exit 1
}
