# BIZRA CLOUD INGESTION ENGINE
# Pulls data from OneDrive/Google Drive into Data Lake
# Preserves structure, handles large files, resumable

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('OneDrive', 'GoogleDrive', 'Both')]
    [string]$Source = 'Both',
    
    [switch]$DryRun,
    [switch]$SkipLargeFiles,  # Skip files > 1GB
    [int]$MaxFileSizeMB = 5000  # Maximum file size to copy (5GB default)
)

$script:DataLakePath = "C:\BIZRA-DATA-LAKE\00_INTAKE"
$script:CopiedCount = 0
$script:SkippedCount = 0
$script:ErrorCount = 0
$script:TotalBytesCopied = 0

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"

    switch($Level) {
        "ERROR" { Write-Host "[$timestamp] ❌ $Message" -ForegroundColor Red }
        "SUCCESS" { Write-Host "[$timestamp] ✅ $Message" -ForegroundColor Green }
        "WARN" { Write-Host "[$timestamp] ⚠️  $Message" -ForegroundColor Yellow }
        "PROGRESS" { Write-Host "[$timestamp] 🔄 $Message" -ForegroundColor Cyan }
        default { Write-Host "[$timestamp] ℹ️  $Message" }
    }
}

function Test-SafePath {
    # SECURITY: Validate path stays within allowed directory (prevents path traversal)
    param(
        [string]$Path,
        [string]$BaseDirectory
    )

    try {
        # Resolve both paths to absolute canonical form
        $resolvedPath = [System.IO.Path]::GetFullPath($Path)
        $resolvedBase = [System.IO.Path]::GetFullPath($BaseDirectory)

        # Check if resolved path starts with base directory
        if ($resolvedPath.StartsWith($resolvedBase, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }

        Write-Status "SECURITY: Path traversal attempt blocked: $Path" -Level "ERROR"
        return $false
    } catch {
        Write-Status "SECURITY: Invalid path detected: $Path" -Level "ERROR"
        return $false
    }
}

function Get-SafeRelativePath {
    # SECURITY: Sanitize relative path to remove traversal sequences
    param([string]$RelativePath)

    # Remove any path traversal sequences
    $sanitized = $RelativePath -replace '\.\.[\\/]', ''
    $sanitized = $sanitized -replace '^[\\/]+', ''

    # Remove any remaining suspicious patterns
    $sanitized = $sanitized -replace '[\x00-\x1f]', ''  # Control characters

    return $sanitized
}

function Get-OneDrivePath {
    # Try to detect OneDrive path
    $possiblePaths = @(
        "$env:OneDrive",
        "$env:OneDriveConsumer",
        "$env:OneDriveCommercial",
        "$env:USERPROFILE\OneDrive",
        "C:\Users\$env:USERNAME\OneDrive"
    )
    
    foreach($path in $possiblePaths) {
        if(Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

function Get-GoogleDrivePath {
    # Try to detect Google Drive path
    $possiblePaths = @(
        "$env:USERPROFILE\Google Drive",
        "G:\My Drive",
        "G:\",
        "C:\Users\$env:USERNAME\Google Drive"
    )
    
    foreach($path in $possiblePaths) {
        if(Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

function Copy-FileWithProgress {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [string]$RelativePath
    )
    
    try {
        $fileInfo = Get-Item $SourcePath
        $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
        
        # Check file size limits
        if($SkipLargeFiles -and $fileSizeMB -gt 1000) {
            Write-Status "SKIPPED (too large): $RelativePath ($fileSizeMB MB)" -Level "WARN"
            $script:SkippedCount++
            return
        }
        
        if($fileSizeMB -gt $MaxFileSizeMB) {
            Write-Status "SKIPPED (exceeds $MaxFileSizeMB MB): $RelativePath ($fileSizeMB MB)" -Level "WARN"
            $script:SkippedCount++
            return
        }
        
        if($DryRun) {
            Write-Status "DRY-RUN would copy: $RelativePath ($fileSizeMB MB)" -Level "PROGRESS"
            $script:CopiedCount++
            return
        }
        
        # Create destination directory if needed
        $destDir = Split-Path $DestinationPath -Parent
        if(-not (Test-Path $destDir)) {
            New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        }
        
        # Copy with progress for large files
        if($fileSizeMB -gt 50) {
            Write-Status "Copying large file: $RelativePath ($fileSizeMB MB)..." -Level "PROGRESS"
        }
        
        Copy-Item -Path $SourcePath -Destination $DestinationPath -Force
        
        $script:CopiedCount++
        $script:TotalBytesCopied += $fileInfo.Length
        
        Write-Status "Copied: $RelativePath ($fileSizeMB MB)" -Level "SUCCESS"
        
    } catch {
        Write-Status "ERROR copying $RelativePath : $_" -Level "ERROR"
        $script:ErrorCount++
    }
}

function Ingest-CloudDrive {
    param(
        [string]$SourcePath,
        [string]$SourceName
    )
    
    if(-not $SourcePath) {
        Write-Status "$SourceName path not found. Skipping." -Level "WARN"
        return
    }
    
    if(-not (Test-Path $SourcePath)) {
        Write-Status "$SourceName path does not exist: $SourcePath" -Level "ERROR"
        return
    }
    
    Write-Status "=" * 70 -Level "SUCCESS"
    Write-Status "📦 Ingesting from $SourceName" -Level "SUCCESS"
    Write-Status "Source: $SourcePath" -Level "SUCCESS"
    Write-Status "=" * 70 -Level "SUCCESS"
    
    # Get all files recursively
    Write-Status "Scanning files..." -Level "PROGRESS"
    $files = Get-ChildItem -Path $SourcePath -File -Recurse -ErrorAction SilentlyContinue
    
    Write-Status "Found $($files.Count) files to process" -Level "SUCCESS"
    
    $currentFile = 0
    foreach($file in $files) {
        $currentFile++
        $relativePath = $file.FullName.Substring($SourcePath.Length + 1)
        
        # Show progress every 100 files
        if($currentFile % 100 -eq 0) {
            $percent = [math]::Round(($currentFile / $files.Count) * 100, 1)
            Write-Status "Progress: $currentFile / $($files.Count) files ($percent%)" -Level "PROGRESS"
        }
        
        # SECURITY: Sanitize relative path before use
        $safeRelativePath = Get-SafeRelativePath -RelativePath $relativePath

        # Create mirrored path in intake
        $destPath = Join-Path $script:DataLakePath "$SourceName\$safeRelativePath"

        # SECURITY: Validate destination stays within data lake
        if (-not (Test-SafePath -Path $destPath -BaseDirectory $script:DataLakePath)) {
            Write-Status "SKIPPED (security): $relativePath" -Level "WARN"
            $script:SkippedCount++
            continue
        }

        Copy-FileWithProgress -SourcePath $file.FullName -DestinationPath $destPath -RelativePath $safeRelativePath
    }
}

# Main execution
Clear-Host
Write-Status "=" * 70 -Level "SUCCESS"
Write-Status "🌊 BIZRA CLOUD INGESTION ENGINE" -Level "SUCCESS"
Write-Status "=" * 70 -Level "SUCCESS"

if($DryRun) {
    Write-Status "🧪 DRY-RUN MODE (no files will be copied)" -Level "WARN"
}

# Detect cloud drive paths
$oneDrivePath = Get-OneDrivePath
$googleDrivePath = Get-GoogleDrivePath

if(-not $oneDrivePath -and -not $googleDrivePath) {
    Write-Status "No cloud drives detected!" -Level "ERROR"
    Write-Status "Please ensure OneDrive or Google Drive is configured." -Level "ERROR"
    exit 1
}

# Process based on source selection
switch($Source) {
    'OneDrive' {
        Ingest-CloudDrive -SourcePath $oneDrivePath -SourceName "OneDrive"
    }
    'GoogleDrive' {
        Ingest-CloudDrive -SourcePath $googleDrivePath -SourceName "GoogleDrive"
    }
    'Both' {
        if($oneDrivePath) {
            Ingest-CloudDrive -SourcePath $oneDrivePath -SourceName "OneDrive"
        }
        if($googleDrivePath) {
            Ingest-CloudDrive -SourcePath $googleDrivePath -SourceName "GoogleDrive"
        }
    }
}

# Final summary
Write-Status "=" * 70 -Level "SUCCESS"
Write-Status "📊 INGESTION COMPLETE" -Level "SUCCESS"
Write-Status "=" * 70 -Level "SUCCESS"
Write-Status "Files copied: $script:CopiedCount"
Write-Status "Files skipped: $script:SkippedCount"
Write-Status "Errors: $script:ErrorCount"

if($script:TotalBytesCopied -gt 0) {
    $totalGB = [math]::Round($script:TotalBytesCopied / 1GB, 2)
    Write-Status "Total data ingested: $totalGB GB" -Level "SUCCESS"
}

Write-Status ""
Write-Status "Next steps:" -Level "SUCCESS"
Write-Status "1. Run: .\DataLakeProcessor.ps1 -ProcessOnce" -Level "SUCCESS"
Write-Status "2. Files will be deduplicated and organized automatically" -Level "SUCCESS"
Write-Status ""
