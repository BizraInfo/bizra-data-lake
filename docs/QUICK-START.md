# 🚀 DATA LAKE QUICK START GUIDE

**Goal:** Transform 1.37TB of scattered BIZRA data into organized Single Source of Truth  
**Time:** 30 minutes initial setup, then automated

---

## ⚡ IMMEDIATE ACTIONS

### **STEP 1: Test the Processor** (2 minutes)

```powershell
cd C:\BIZRA-DATA-LAKE
.\DataLakeProcessor.ps1 -ProcessOnce
```

**Expected:** Should show "No files found in INTAKE" (good! system working)

---

### **STEP 2: Ingest Cloud Drives** (DRY RUN FIRST)

#### **Option A: Test First (Recommended)**
```powershell
.\CloudIngestion.ps1 -DryRun
```

This shows what WOULD be copied without actually copying anything.

#### **Option B: Ingest OneDrive**
```powershell
.\CloudIngestion.ps1 -Source OneDrive
```

#### **Option C: Ingest Google Drive**
```powershell
.\CloudIngestion.ps1 -Source GoogleDrive
```

#### **Option D: Ingest Everything**
```powershell
.\CloudIngestion.ps1 -Source Both
```

---

### **STEP 3: Process Ingested Files**

After cloud ingestion completes:

```powershell
.\DataLakeProcessor.ps1 -ProcessOnce
```

Watch the magic:
- ✅ Deduplication happens automatically
- ✅ Files sorted by type
- ✅ Metadata created
- ✅ Originals backed up
- ✅ Duplicates quarantined

---

### **STEP 4: Start Continuous Monitoring**

```powershell
.\DataLakeProcessor.ps1 -Watch
```

Now ANY file dropped into `00_INTAKE/` gets auto-processed instantly.

---

## 📊 MONITORING PROGRESS

### **Check Processing Log**
```powershell
Get-Content .\processing.log -Tail 50
```

### **See What's Been Processed**
```powershell
Get-ChildItem .\02_PROCESSED -Recurse -File | Group-Object Directory | Select Name, Count | Sort Count -Desc
```

### **Check for Duplicates**
```powershell
$dups = Get-ChildItem .\99_QUARANTINE -File -Recurse
Write-Host "Duplicates caught: $($dups.Count) files"
Write-Host "Space saved: $([math]::Round(($dups | Measure-Object Length -Sum).Sum / 1MB, 2)) MB"
```

### **View Type Distribution**
```powershell
Get-ChildItem .\02_PROCESSED\* -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -File -Recurse).Count
    [PSCustomObject]@{
        Type = $_.Name
        Files = $count
    }
} | Sort-Object Files -Descending | Format-Table
```

---

## 🎯 RECOMMENDED WORKFLOW

### **For First-Time Setup:**

1. **Dry run to estimate** (understand scope)
   ```powershell
   .\CloudIngestion.ps1 -DryRun
   ```

2. **Ingest clouds** (may take hours for 100GB+)
   ```powershell
   .\CloudIngestion.ps1 -Source Both
   ```

3. **Process everything** (dedup + organize)
   ```powershell
   .\DataLakeProcessor.ps1 -ProcessOnce
   ```

4. **Review results**
   ```powershell
   Get-Content .\processing.log | Select-String "COMPLETE"
   ```

5. **Start continuous mode**
   ```powershell
   .\DataLakeProcessor.ps1 -Watch
   ```

---

### **For Ongoing Operations:**

Just leave the processor running in watch mode:
```powershell
.\DataLakeProcessor.ps1 -Watch
```

Then drag/drop ANY file into `00_INTAKE/` folder and it's auto-processed.

---

## 🚨 TROUBLESHOOTING

### **"Access Denied" on Cloud Files**

Some cloud files might be "online-only". Force download first:
```powershell
# For OneDrive
attrib -U /S "C:\Users\YourName\OneDrive\*"
```

### **Large Files Taking Forever**

Skip files > 1GB during ingestion:
```powershell
.\CloudIngestion.ps1 -SkipLargeFiles
```

### **Want to Customize File Size Limit**

Example: Only copy files under 500MB:
```powershell
.\CloudIngestion.ps1 -MaxFileSizeMB 500
```

---

## 📈 EXPECTED RESULTS

### **After Processing 100GB:**

```
02_PROCESSED/
├── text/           ~40GB  (largest - logs, chats, docs)
├── code/           ~20GB  (all source code)
├── documents/      ~15GB  (PDFs, Office files)
├── data/           ~10GB  (JSON, databases)
├── images/         ~8GB   (screenshots, diagrams)
├── models/         ~5GB   (AI models)
├── archives/       ~2GB   (compressed files)
```

### **Deduplication Savings:**

Typical: 10-30% reduction (10-30GB saved)  
Conservative estimate: 15GB of duplicates caught

---

## 🔗 NEXT PHASE: INTELLIGENCE

Once data is organized, we can:

1. **Generate Vector Embeddings** (03_INDEXED/)
   - Semantic search across all text
   - Find related concepts across files

2. **Connect to SAPE Knowledge Kernels**
   - Real data for evidence gathering
   - No more mock responses

3. **Train on Real History**
   - Extract patterns from dev conversations
   - Build consciousness from actual work

4. **Create Knowledge Graph**
   - Map relationships between concepts
   - Navigate 15,000 hours of insights

---

## 💎 GOLD CORPUS CREATION

Manually curate best files into `04_GOLD/`:

```powershell
# Example: Promote critical architecture docs
Copy-Item ".\02_PROCESSED\text\BIZRA_CORE_ARCHITECTURE.md" ".\04_GOLD\"
```

Gold corpus used for:
- Training data
- Quick reference
- Documentation generation
- Pattern extraction

---

**Ready to begin?** Run STEP 1 above! 🚀
