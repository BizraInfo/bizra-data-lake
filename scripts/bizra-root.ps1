# BIZRA Root Resolver (PowerShell)
# Returns canonical repo root (works from any subfolder)
# Usage: $BIZRA_ROOT = & "$PSScriptRoot\bizra-root.ps1"

$ErrorActionPreference = "Stop"

# Method 1: Git toplevel (most reliable)
try {
    $root = (git rev-parse --show-toplevel 2>$null)
    if ($root) {
        $root = $root.Trim() -replace '/', '\'
        Write-Output $root
        exit 0
    }
} catch {
    # Git not available or not in a git repo
}

# Method 2: Walk up until .git found
$here = Resolve-Path "."
while ($true) {
    if (Test-Path (Join-Path $here.Path ".git")) {
        Write-Output $here.Path
        exit 0
    }
    $parent = Split-Path $here.Path -Parent
    if (-not $parent -or $parent -eq $here.Path) {
        throw "BIZRA root not found (no .git directory in hierarchy)"
    }
    $here = Resolve-Path $parent
}
