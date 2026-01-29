# BIZRA Sovereign Data Lake Processor
# Handles cross-source ingestion (Intake, Downloads, Cloud)
# Context-Aware: Preserves provenance via directory structure

$DataLakeRoot = "C:\BIZRA-DATA-LAKE"
$Intake = Join-Path $DataLakeRoot "00_INTAKE"
$Processed = Join-Path $DataLakeRoot "02_PROCESSED"
$Downloads = "C:\Users\BIZRA-OS\Downloads"

$AllowedExtensions = @(".pdf", ".md", ".txt", ".docx", ".ipynb", ".py", ".js", ".ts", ".rs")

function Ingest-Folder {
    param(
        [string]$SourcePath,
        [string]$ContextTag
    )
    
    if (-not (Test-Path $SourcePath)) {
        Write-Host "WARNING: Source path not found: $SourcePath"
        return
    }

    $DestDir = Join-Path $Processed "text"
    if ($ContextTag) {
        $DestDir = Join-Path $DestDir $ContextTag
    }
    
    if (-not (Test-Path $DestDir)) { 
        New-Item -Path $DestDir -ItemType Directory -Force | Out-Null 
    }

    Write-Host "🌊 PIPELINE: Ingesting from [$SourcePath]..."
    
    $files = Get-ChildItem -Path $SourcePath -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
        $AllowedExtensions -contains $_.Extension.ToLower()
    }

    # Step 1: Stage files in 00_INTAKE for Agentic Cleaning
    $IntakeDir = Join-Path $Intake "text_staging"
    if (-not (Test-Path $IntakeDir)) { New-Item -Path $IntakeDir -ItemType Directory -Force | Out-Null }

    $count = 0
    foreach ($f in $files) {
        if ($ContextTag -eq "downloads" -and $count -ge 25) { 
            Write-Host "USAGE LIMIT: Stopping at 25 files for Downloads."
            break 
        }
        
        $dest = Join-Path $IntakeDir $f.Name
        Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
        $count++
    }
    
    Write-Host "🤖 AGENTIC: Triggering Code-Synthesis Cleaning on $count files..."
    
    # Step 2: Execute Agentic Cleaning (Reads INTAKE -> Writes PROCESSED)
    if ($count -gt 0) {
        python "C:\BIZRA-DATA-LAKE\agentic_cleaner.py"
        
        # Step 3: Archive Staged Files to RAW
        $RawDest = Join-Path $DataLakeRoot "01_RAW\archived_intake_$(Get-Date -Format 'yyyyMMdd')"
        if (-not (Test-Path $RawDest)) { New-Item -Path $RawDest -ItemType Directory -Force | Out-Null }
        
        Move-Item -Path "$IntakeDir\*" -Destination $RawDest -Force
        Write-Host "✅ ARCHIVE: Raw files moved to 01_RAW context."
    } else {
        Write-Host "SKIPPED: No files found to process."
    }
    
    Write-Host "RESULT: Ingestion cycle complete for $ContextTag."
}

# --- Execution ---

Write-Host "BIZRA SOVEREIGN INGESTION STARTING"
Write-Host "--------------------------------------"

# 1. Ingest Main Intake
Ingest-Folder -SourcePath $Intake -ContextTag ""

# 2. Ingest Downloads (Context-Aware)
Ingest-Folder -SourcePath $Downloads -ContextTag "downloads"

Write-Host "--------------------------------------"
Write-Host "COMPLETED: Ingestion sequence complete."
