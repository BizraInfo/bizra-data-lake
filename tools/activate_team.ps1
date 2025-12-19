# tools/activate_team.ps1
# Activates the local “personal agentic team” by producing an audit-grade evidence run:
# - Resolves workspace + vault contracts
# - Captures repo health (git status, tests)
# - Ingests chat export sample into the Data Lake (index-only by default)
# - Writes compact receipts (no raw chat content committed)

[CmdletBinding()]
param(
  [string]$ChatInputDir,
  [switch]$SkipChatIngest,
  [switch]$StoreFullText,
  [switch]$IngestCrashReports,
  [switch]$RunLLMTeam
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  return & "$PSScriptRoot\\..\\scripts\\bizra-root.ps1"
}

function Sha256File([string]$path) {
  (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
}

function New-RunDir([string]$repoRoot) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $dir = Join-Path $repoRoot ("docs\\evidence\\receipts\\activation_" + $ts)
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  return $dir
}

function Save-Text([string]$outPath, [scriptblock]$cmd) {
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $cmd 2>&1 | Out-File -FilePath $outPath -Encoding utf8
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
      "" | Out-File -FilePath $outPath -Append -Encoding utf8
      "EXITCODE: $LASTEXITCODE" | Out-File -FilePath $outPath -Append -Encoding utf8
    }
  } finally {
    $ErrorActionPreference = $old
  }
}

$repoRoot = Resolve-RepoRoot
Push-Location $repoRoot
try {
  $runDir = New-RunDir -repoRoot $repoRoot

  # Record invocation parameters (high-SNR, no secrets)
  @(
    "ChatInputDir=$ChatInputDir"
    "SkipChatIngest=$SkipChatIngest"
    "StoreFullText=$StoreFullText"
    "IngestCrashReports=$IngestCrashReports"
    "RunLLMTeam=$RunLLMTeam"
  ) | Out-File -FilePath (Join-Path $runDir "01_params.txt") -Encoding utf8

  # Resolve contracts (keep env vars in this session)
  . "$repoRoot\\scripts\\resolve-bizra-root.ps1" | Out-Null
  . "$repoRoot\\scripts\\resolve-bizra-vault.ps1" | Out-Null

  # [Agent: OPS] Environment snapshot (do not capture secrets)
  $envOut = Join-Path $runDir "00_env.txt"
  Save-Text $envOut {
    "BIZRA activation run (no secrets)"
    "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')"
    ""
    Get-ChildItem Env: |
      Where-Object { $_.Name -match '^(BIZRA_|KERNEL_|EVIDENCE_|OLLAMA_|POSTGRES_|REDIS_|COMPOSE_)' } |
      Where-Object { $_.Name -notmatch 'TOKEN|SECRET|PASSWORD|KEY' } |
      Sort-Object Name |
      Format-Table -AutoSize | Out-String
  }

  # [Agent: LIBRARIAN] Repo state
  Save-Text (Join-Path $runDir "10_git_status.txt") { git status --porcelain }
  Save-Text (Join-Path $runDir "11_git_branch.txt") { git rev-parse --abbrev-ref HEAD }
  Save-Text (Join-Path $runDir "12_git_head.txt") { git rev-parse HEAD }

  # [Agent: SECURITY] Lightweight secret scan (bounded, avoids scanning receipts/output)
  $secretPattern = "(password|secret|api[_-]?key|token\\s*[:=]|postgresql://[^\\s]+:[^@\\s]+@)"
  $secretGlobs = @(
    "--glob", "!.git/**",
    "--glob", "!target/**",
    "--glob", "!**/node_modules/**",
    "--glob", "!docs/evidence/**",
    "--glob", "!evidence/**"
  )

  Save-Text (Join-Path $runDir "20_secret_scan_files.txt") {
    rg -l $secretPattern -S . @secretGlobs --max-filesize 2M
  }
  Save-Text (Join-Path $runDir "21_secret_scan_snippet.txt") {
    rg -n $secretPattern -S . @secretGlobs --max-filesize 2M --stats
  }

  # [Agent: QA] Tests
  Save-Text (Join-Path $runDir "30_cargo_test.txt") { cargo test }

  # [Agent: ARCHITECT] Codebase inventory (high-SNR, evidence-first)
  $invJson = Join-Path $runDir "35_codebase_inventory.json"
  $invTxt = Join-Path $runDir "35_codebase_context.txt"
  Save-Text (Join-Path $runDir "35_codebase_inventory_run.txt") {
    python (Join-Path $repoRoot "tools\\codebase_inventory.py") --out $invJson --summary-out $invTxt
  }

  # [Agent: ARCHIVIST] Chat ingestion (index only by default)
  if (-not $SkipChatIngest) {
    if (-not $ChatInputDir) { $ChatInputDir = (Join-Path $repoRoot "chat data sample") }
    $py = Join-Path $repoRoot "tools\\ingest_chat_sample.py"
    if (-not (Test-Path $py)) { throw "Missing tool: $py" }

    $ingestOut = Join-Path $runDir "40_chat_ingest_summary.json"

    $args = @("--input-dir", $ChatInputDir)
    if ($StoreFullText) { $args += "--store-full-text" }

    Save-Text $ingestOut { python $py @args }
    $json = Get-Content -Raw $ingestOut

    # Parse out_dir from summary
    $summary = $null
    try { $summary = ($json | Out-String | ConvertFrom-Json) } catch {}

    if ($summary -and $summary.out_dir) {
      $indexRoot = [string]$summary.out_dir

      # Build aggregate.json for high-SNR team prompts
      $aggTool = Join-Path $repoRoot "tools\\summarize_chat_index.py"
      if (Test-Path $aggTool) {
        Save-Text (Join-Path $runDir "40_chat_aggregate_path.txt") { python $aggTool --run-dir $indexRoot }
      }

      $receipt = @{
        receipt_version = 1
        type = "chat_ingest"
        generated_at = (Get-Date).ToString("o")
        truth_label = "MEASURED"
        input_dir = $ChatInputDir
        out_dir = $indexRoot
        run_id = $summary.run_id
        counts = @{
          zip_files = $summary.zip_files
          conversations = $summary.conversations
          messages_index = $summary.messages_index
          messages_full = $summary.messages_full
        }
        artifacts = @()
      }

      foreach ($f in @("summary.json", "sources.json", "topics.json", "graph.json", "conversations.jsonl", "messages_index.jsonl")) {
        $p = Join-Path $indexRoot $f
        if (Test-Path $p) {
          $receipt.artifacts += @{
            path = $p
            sha256 = Sha256File $p
          }
        }
      }
      $aggPath = Join-Path $indexRoot "aggregate.json"
      if (Test-Path $aggPath) {
        $receipt.artifacts += @{
          path = $aggPath
          sha256 = Sha256File $aggPath
        }
      }

      $receiptPath = Join-Path $runDir "41_chat_ingest_receipt.json"
      ($receipt | ConvertTo-Json -Depth 6) | Out-File -FilePath $receiptPath -Encoding utf8
    }
  }

  # [Agent: OPS/FORENSICS] Windows crash logs ingestion (safe: copy small + index)
  if ($IngestCrashReports) {
    $crashTool = Join-Path $repoRoot "tools\\ingest_crash_reports.py"
    if (-not (Test-Path $crashTool)) { throw "Missing tool: $crashTool" }
    $crashOut = Join-Path $runDir "45_windows_crash_ingest_summary.json"
    Save-Text $crashOut { python $crashTool }
    $crashJson = Get-Content -Raw $crashOut

    $crash = $null
    try { $crash = ($crashJson | Out-String | ConvertFrom-Json) } catch {}
    if ($crash -and $crash.out_dir) {
      $crashReceipt = @{
        receipt_version = 1
        type = "windows_crash_ingest"
        generated_at = (Get-Date).ToString("o")
        truth_label = "MEASURED"
        out_dir = [string]$crash.out_dir
        files_indexed = $crash.files_indexed
        members_extracted = $crash.members_extracted
        artifacts = @()
      }
      foreach ($f in @("summary.json", "index.json")) {
        $p = Join-Path ([string]$crash.out_dir) $f
        if (Test-Path $p) {
          $crashReceipt.artifacts += @{
            path = $p
            sha256 = Sha256File $p
          }
        }
      }
      ($crashReceipt | ConvertTo-Json -Depth 6) | Out-File -FilePath (Join-Path $runDir "46_windows_crash_ingest_receipt.json") -Encoding utf8
    }
  }

  # [Agent: LLM TEAM] Run role agents on aggregate + codebase context
  if ($RunLLMTeam) {
    $aggFromReceipt = $null
    $ingestReceiptPath = Join-Path $runDir "41_chat_ingest_receipt.json"
    if (Test-Path $ingestReceiptPath) {
      try {
        $ing = Get-Content -Raw $ingestReceiptPath | ConvertFrom-Json
        $aggFromReceipt = ($ing.artifacts | Where-Object { $_.path -match 'aggregate\\.json$' } | Select-Object -First 1).path
      } catch {}
    }

    # Fallback: aggregate path hint file
    if (-not $aggFromReceipt) {
      $aggHint = Join-Path $runDir "40_chat_aggregate_path.txt"
      if (Test-Path $aggHint) {
        $cand = (Get-Content -Raw $aggHint).Trim()
        if ($cand) { $aggFromReceipt = $cand }
      }
    }

    $aggExists = $false
    if ($aggFromReceipt) {
      try { $aggExists = Test-Path $aggFromReceipt } catch { $aggExists = $false }
    }

    @(
      "ingestReceiptPath=$ingestReceiptPath"
      "aggregatePath=$aggFromReceipt"
      "aggregateExists=$aggExists"
    ) | Out-File -FilePath (Join-Path $runDir "49_llm_team_preflight.txt") -Encoding utf8

    if (-not $aggFromReceipt -or -not $aggExists) {
      throw "RunLLMTeam requires chat ingest (aggregate.json). Re-run without -SkipChatIngest."
    }

    $teamOut = Join-Path $runDir "50_llm_team_out_dir.txt"
    $teamRunner = Join-Path $repoRoot "ace-framework\\team-runner.js"
    if (-not (Test-Path $teamRunner)) { throw "Missing: $teamRunner" }

    $contextArgs = @()
    if (Test-Path $invTxt) { $contextArgs += @("--context", $invTxt) }
    $crashSummary = Join-Path $runDir "45_windows_crash_ingest_summary.json"
    if (Test-Path $crashSummary) { $contextArgs += @("--context", $crashSummary) }

    Save-Text $teamOut { node $teamRunner --aggregate $aggFromReceipt @contextArgs }
    $outDirFromNode = Get-Content -Raw $teamOut

    $teamPath = $null
    try {
      $teamPath = (($outDirFromNode -split "`r?`n") | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
    } catch {}
    if ($teamPath -and (Test-Path $teamPath)) {
      $teamReceipt = @{
        receipt_version = 1
        type = "llm_team_run"
        generated_at = (Get-Date).ToString("o")
        truth_label = "MEASURED"
        out_dir = $teamPath
        aggregate = $aggFromReceipt
        context_files = @($invTxt, $crashSummary) | Where-Object { $_ -and (Test-Path $_) }
      }
      ($teamReceipt | ConvertTo-Json -Depth 6) | Out-File -FilePath (Join-Path $runDir "51_llm_team_receipt.json") -Encoding utf8
    }
  }

  # Run manifest (single place to point at)
  $manifest = @{
    run_type = "activate_team"
    generated_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    run_dir = $runDir
    truth_label = "MEASURED"
    files = Get-ChildItem $runDir -File | ForEach-Object {
      @{
        name = $_.Name
        bytes = $_.Length
        sha256 = Sha256File $_.FullName
      }
    }
  }
  ($manifest | ConvertTo-Json -Depth 6) | Out-File -FilePath (Join-Path $runDir "ZZ_RUN_MANIFEST.json") -Encoding utf8

  Write-Host "Activation run complete: $runDir" -ForegroundColor Green
} finally {
  Pop-Location
}
