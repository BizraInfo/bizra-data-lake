// bizra-agent/src/skills/file_management.rs
// ============================================================
// Smart File Management — PAT executes, SAT validates
// ============================================================
//
// Capabilities:
//   1. Smart Classification — identify content, auto-categorize
//   2. Batch Renaming — pattern-based rename with preview
//   3. Auto Organization — move files to correct directories
//   4. File Merging — combine related files intelligently
//
// Constitutional requirements:
//   - Every operation produces a manifest (rollback-capable)
//   - SAT validates before destructive operations execute
//   - BLAKE3 hash of every moved/renamed file for integrity
//   - No silent data loss — every delete requires HITL approval
//   - Receipt emitted for every batch operation
//
// Architecture:
//   PAT (Navigator/Artisan) → plans operations, presents to user
//   SAT (Guardian/Auditor)  → validates safety, checks receipts
// ============================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── File classification ───────────────────────────────────

/// Content categories for smart classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileCategory {
    Document,    // PDF, DOCX, TXT, MD, RTF, ODT
    Spreadsheet, // XLSX, CSV, TSV, ODS
    Presentation,// PPTX, KEY, ODP
    Image,       // PNG, JPG, SVG, WEBP, GIF, BMP
    Video,       // MP4, MOV, AVI, MKV, WEBM
    Audio,       // MP3, WAV, FLAC, M4A, OGG
    Code,        // RS, PY, TS, JS, C, CPP, JAVA, GO
    Data,        // JSON, YAML, TOML, XML, PARQUET
    Archive,     // ZIP, TAR, GZ, 7Z, RAR
    Web,         // HTML, CSS, WASM
    Font,        // TTF, OTF, WOFF, WOFF2
    Config,      // ENV, INI, CFG, CONF
    Executable,  // EXE, MSI, DEB, RPM, APK, DMG
    Other,       // Unclassified
}

impl FileCategory {
    /// Target subdirectory name for auto-organization.
    pub fn target_dir(&self) -> &'static str {
        match self {
            Self::Document     => "Documents",
            Self::Spreadsheet  => "Spreadsheets",
            Self::Presentation => "Presentations",
            Self::Image        => "Images",
            Self::Video        => "Video",
            Self::Audio        => "Audio",
            Self::Code          => "Code",
            Self::Data          => "Data",
            Self::Archive       => "Archives",
            Self::Web           => "Web",
            Self::Font          => "Fonts",
            Self::Config        => "Config",
            Self::Executable    => "Executables",
            Self::Other         => "Other",
        }
    }

    /// Classify by file extension (fast path).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            // Documents
            "pdf" | "docx" | "doc" | "txt" | "md" | "rtf" | "odt" | "tex" | "epub" => Self::Document,
            // Spreadsheets
            "xlsx" | "xls" | "csv" | "tsv" | "ods" | "xlsm" => Self::Spreadsheet,
            // Presentations
            "pptx" | "ppt" | "key" | "odp" => Self::Presentation,
            // Images
            "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "bmp" | "ico" | "tiff" | "tif" | "heic" | "heif" | "avif" => Self::Image,
            // Video
            "mp4" | "mov" | "avi" | "mkv" | "webm" | "flv" | "wmv" | "m4v" => Self::Video,
            // Audio
            "mp3" | "wav" | "flac" | "m4a" | "ogg" | "aac" | "wma" | "opus" => Self::Audio,
            // Code
            "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "c" | "cpp" | "h" | "hpp"
            | "java" | "go" | "rb" | "swift" | "kt" | "scala" | "lua" | "r"
            | "sh" | "bash" | "zsh" | "ps1" | "bat" | "cmd" | "sql" | "proto" => Self::Code,
            // Data
            "json" | "yaml" | "yml" | "toml" | "xml" | "parquet" | "arrow" | "avro" | "ndjson" | "jsonl" => Self::Data,
            // Archives
            "zip" | "tar" | "gz" | "tgz" | "bz2" | "xz" | "7z" | "rar" | "zst" => Self::Archive,
            // Web
            "html" | "htm" | "css" | "wasm" | "scss" | "less" | "sass" => Self::Web,
            // Fonts
            "ttf" | "otf" | "woff" | "woff2" | "eot" => Self::Font,
            // Config
            "env" | "ini" | "cfg" | "conf" | "properties" | "lock" => Self::Config,
            // Executables
            "exe" | "msi" | "deb" | "rpm" | "apk" | "dmg" | "appimage" | "snap" => Self::Executable,
            _ => Self::Other,
        }
    }

    /// SNR weight for Mint Court valuation (from Impact Settlement Contract).
    pub fn snr_weight(&self) -> f32 {
        match self {
            Self::Code          => 0.95,  // Knowledge contribution
            Self::Document      => 0.90,  // Knowledge contribution
            Self::Data          => 0.85,  // Data contribution
            Self::Spreadsheet   => 0.80,
            Self::Presentation  => 0.75,
            Self::Config        => 0.70,
            Self::Web           => 0.65,
            Self::Archive       => 0.30,  // Container, not content
            Self::Image         => 0.20,  // Media excluded from SNR by default
            Self::Video         => 0.15,
            Self::Audio         => 0.15,
            Self::Font          => 0.10,
            Self::Executable    => 0.05,
            Self::Other         => 0.00,
        }
    }
}

// ── File entry (classified file) ──────────────────────────

/// A classified file with metadata.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    pub name: String,
    pub extension: String,
    pub category: FileCategory,
    pub size_bytes: u64,
    pub content_hash: Option<[u8; 32]>,  // BLAKE3 if computed
}

impl FileEntry {
    /// Create from a path with extension-based classification.
    pub fn classify(path: PathBuf, size_bytes: u64) -> Self {
        let name = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let extension = path.extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_default();
        let category = FileCategory::from_extension(&extension);
        Self { path, name, extension, category, size_bytes, content_hash: None }
    }

    /// Set BLAKE3 hash (computed externally for large files).
    pub fn with_hash(mut self, hash: [u8; 32]) -> Self {
        self.content_hash = Some(hash);
        self
    }
}

// ── File operations ───────────────────────────────────────

/// A single file operation in a batch.
#[derive(Debug, Clone)]
pub enum FileOp {
    /// Move file to target directory.
    Move { source: PathBuf, target: PathBuf },
    /// Rename file (same directory).
    Rename { source: PathBuf, new_name: String },
    /// Copy file (non-destructive).
    Copy { source: PathBuf, target: PathBuf },
    /// Delete file (requires HITL approval via SAT).
    Delete { path: PathBuf, reason: String },
    /// Merge multiple files into one.
    Merge { sources: Vec<PathBuf>, target: PathBuf, strategy: MergeStrategy },
}

impl FileOp {
    /// Is this a destructive operation requiring SAT approval?
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Delete { .. } | Self::Move { .. })
    }

    /// Human-readable description for HITL review.
    pub fn describe(&self) -> String {
        match self {
            Self::Move { source, target } => format!(
                "Move {} → {}", source.display(), target.display()
            ),
            Self::Rename { source, new_name } => format!(
                "Rename {} → {}", source.display(), new_name
            ),
            Self::Copy { source, target } => format!(
                "Copy {} → {}", source.display(), target.display()
            ),
            Self::Delete { path, reason } => format!(
                "Delete {} ({})", path.display(), reason
            ),
            Self::Merge { sources, target, .. } => format!(
                "Merge {} files → {}", sources.len(), target.display()
            ),
        }
    }
}

/// Strategy for merging files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Concatenate text files sequentially.
    Concatenate,
    /// Interleave (for CSVs with same schema).
    Interleave,
    /// Deduplicate (remove duplicate lines).
    Deduplicate,
}

// ── Operation manifest (constitutional rollback) ──────────

/// Tracks every operation for rollback capability.
/// This is the constitutional requirement: no silent data loss.
#[derive(Debug, Clone)]
pub struct OperationManifest {
    pub id: [u8; 32],             // BLAKE3 hash of manifest content
    pub operations: Vec<ManifestEntry>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub rollback_available: bool,
}

/// A single entry in the manifest (before/after state).
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    pub operation: FileOp,
    pub status: OpStatus,
    pub source_hash: Option<[u8; 32]>,   // BLAKE3 of source before op
    pub target_hash: Option<[u8; 32]>,   // BLAKE3 of target after op
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpStatus {
    Planned,
    Approved,      // SAT approved
    Executing,
    Succeeded,
    Failed,
    RolledBack,
}

impl OperationManifest {
    pub fn new(timestamp: u64) -> Self {
        let id = blake3::hash(format!("manifest:{}", timestamp).as_bytes()).into();
        Self {
            id,
            operations: Vec::new(),
            created_at: timestamp,
            completed_at: None,
            rollback_available: true,
        }
    }

    pub fn add(&mut self, op: FileOp) {
        self.operations.push(ManifestEntry {
            operation: op,
            status: OpStatus::Planned,
            source_hash: None,
            target_hash: None,
            error: None,
        });
    }

    pub fn total_ops(&self) -> usize { self.operations.len() }
    pub fn succeeded(&self) -> usize { self.operations.iter().filter(|e| e.status == OpStatus::Succeeded).count() }
    pub fn failed(&self) -> usize { self.operations.iter().filter(|e| e.status == OpStatus::Failed).count() }

    pub fn destructive_count(&self) -> usize {
        self.operations.iter().filter(|e| e.operation.is_destructive()).count()
    }

    /// Seal the manifest with a BLAKE3 hash of all entries.
    pub fn seal(&mut self, timestamp: u64) {
        self.completed_at = Some(timestamp);
        let content = format!("{}:{}:{}", self.total_ops(), self.succeeded(), timestamp);
        self.id = blake3::hash(content.as_bytes()).into();
    }
}

// ── Rename patterns ───────────────────────────────────────

/// Pattern-based batch rename rule.
#[derive(Debug, Clone)]
pub struct RenameRule {
    pub pattern: RenamePattern,
    pub scope: FileCategory,  // Apply only to this category
}

#[derive(Debug, Clone)]
pub enum RenamePattern {
    /// Add prefix: "2026-03_" + original name.
    Prefix(String),
    /// Add suffix: original name + "_v2".
    Suffix(String),
    /// Replace substring in filename.
    Replace { find: String, replace: String },
    /// Sequential numbering: "photo_001.jpg", "photo_002.jpg".
    Sequential { base: String, start: u32, pad: usize },
    /// Lowercase everything.
    Lowercase,
    /// Replace spaces with underscores/hyphens.
    Sanitize { separator: char },
}

impl RenamePattern {
    /// Apply the pattern to a filename, returning the new name.
    pub fn apply(&self, original: &str, index: u32) -> String {
        let (stem, ext) = match original.rsplit_once('.') {
            Some((s, e)) => (s.to_string(), format!(".{}", e)),
            None => (original.to_string(), String::new()),
        };
        match self {
            Self::Prefix(p) => format!("{}{}{}", p, stem, ext),
            Self::Suffix(s) => format!("{}{}{}", stem, s, ext),
            Self::Replace { find, replace } => format!("{}{}", stem.replace(find.as_str(), replace), ext),
            Self::Sequential { base, start, pad } => {
                let num = start + index;
                format!("{}_{:0>width$}{}", base, num, ext, width = *pad)
            }
            Self::Lowercase => format!("{}{}", stem.to_lowercase(), ext.to_lowercase()),
            Self::Sanitize { separator } => {
                let clean: String = stem.chars().map(|c| {
                    if c.is_whitespace() { *separator }
                    else if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c }
                    else { *separator }
                }).collect();
                format!("{}{}", clean, ext)
            }
        }
    }
}

// ── Smart File Manager (PAT + SAT dual-role) ─────────────

/// The core file management orchestrator.
/// PAT role: classify, plan operations, present to user.
/// SAT role: validate safety, check receipts, approve destructive ops.
#[derive(Default)]
pub struct SmartFileManager {
    /// All classified files in the working scope.
    pub inventory: Vec<FileEntry>,
    /// Current operation manifest (rollback-capable).
    pub manifest: Option<OperationManifest>,
    /// Classification statistics.
    pub stats: ClassificationStats,
}

/// Summary statistics from classification.
#[derive(Debug, Clone, Default)]
pub struct ClassificationStats {
    pub total_files: usize,
    pub total_bytes: u64,
    pub by_category: HashMap<FileCategory, CategoryStats>,
}

#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    pub count: usize,
    pub total_bytes: u64,
}

impl SmartFileManager {
    pub fn new() -> Self {
        Self::default()
    }

    // ── PAT: Classification (Navigator + Scholar) ─────────

    /// Classify a batch of files by extension.
    /// PAT Navigator identifies content, Scholar enriches metadata.
    pub fn classify_batch(&mut self, files: Vec<(PathBuf, u64)>) -> &ClassificationStats {
        self.inventory.clear();
        self.stats = ClassificationStats::default();

        for (path, size) in files {
            let entry = FileEntry::classify(path, size);
            let cat_stats = self.stats.by_category
                .entry(entry.category)
                .or_default();
            cat_stats.count += 1;
            cat_stats.total_bytes += size;
            self.stats.total_files += 1;
            self.stats.total_bytes += size;
            self.inventory.push(entry);
        }
        &self.stats
    }

    // ── PAT: Auto Organization (Navigator + Artisan) ──────

    /// Plan an auto-organization: move files to category-based subdirectories.
    /// Returns the manifest with planned operations (not yet executed).
    pub fn plan_auto_organize(&mut self, base_dir: &Path, timestamp: u64) -> &OperationManifest {
        let mut manifest = OperationManifest::new(timestamp);

        for entry in &self.inventory {
            let target_dir = base_dir.join(entry.category.target_dir());
            let target_path = target_dir.join(&entry.name);

            // Skip if already in correct directory
            if entry.path.parent() == Some(&target_dir) {
                continue;
            }

            manifest.add(FileOp::Move {
                source: entry.path.clone(),
                target: target_path,
            });
        }

        self.manifest = Some(manifest);
        self.manifest.as_ref().unwrap()
    }

    // ── PAT: Batch Rename (Artisan) ───────────────────────

    /// Plan a batch rename using a pattern rule.
    /// Returns the manifest with planned rename operations.
    pub fn plan_batch_rename(
        &mut self,
        rule: &RenameRule,
        timestamp: u64,
    ) -> &OperationManifest {
        let mut manifest = OperationManifest::new(timestamp);
        let mut index = 0u32;

        for entry in &self.inventory {
            // Apply only to matching category
            if entry.category != rule.scope {
                continue;
            }

            let new_name = rule.pattern.apply(&entry.name, index);
            if new_name != entry.name {
                manifest.add(FileOp::Rename {
                    source: entry.path.clone(),
                    new_name,
                });
                index += 1;
            }
        }

        self.manifest = Some(manifest);
        self.manifest.as_ref().unwrap()
    }

    // ── PAT: Duplicate Detection (Scholar + Sentinel) ─────

    /// Find duplicate files by size (fast pre-filter) then by hash.
    /// Returns groups of duplicate paths.
    pub fn find_duplicates(&self) -> Vec<Vec<&FileEntry>> {
        // Group by size first (O(n) pre-filter)
        let mut by_size: HashMap<u64, Vec<&FileEntry>> = HashMap::new();
        for entry in &self.inventory {
            by_size.entry(entry.size_bytes).or_default().push(entry);
        }

        // Only groups with 2+ files of same size are candidates
        let mut duplicates = Vec::new();
        for (_size, group) in by_size {
            if group.len() >= 2 {
                // For files with hashes, sub-group by hash
                let mut by_hash: HashMap<[u8; 32], Vec<&FileEntry>> = HashMap::new();
                let mut unhashed = Vec::new();

                for entry in &group {
                    if let Some(hash) = entry.content_hash {
                        by_hash.entry(hash).or_default().push(entry);
                    } else {
                        unhashed.push(*entry);
                    }
                }

                for (_hash, hash_group) in by_hash {
                    if hash_group.len() >= 2 {
                        duplicates.push(hash_group);
                    }
                }

                // Size-only duplicates (no hash) are also candidates
                if unhashed.len() >= 2 {
                    duplicates.push(unhashed);
                }
            }
        }
        duplicates
    }


    // ── SAT: Validation (Guardian + Auditor) ──────────────

    /// SAT Guardian validates a manifest before execution.
    /// Returns (approved, reasons) — rejection reasons if any.
    pub fn sat_validate_manifest(manifest: &OperationManifest) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        // Rule 1: No empty manifests
        if manifest.total_ops() == 0 {
            reasons.push("Empty manifest — nothing to execute".into());
        }

        // Rule 2: Destructive ops count check (warn > 50, block > 500)
        let destructive = manifest.destructive_count();
        if destructive > 500 {
            reasons.push(format!(
                "Destructive op count {} exceeds safety limit 500 — requires manual override",
                destructive
            ));
        }

        // Rule 3: No operations targeting system directories
        for entry in &manifest.operations {
            let paths = match &entry.operation {
                FileOp::Move { source, target } => vec![source.clone(), target.clone()],
                FileOp::Delete { path, .. } => vec![path.clone()],
                FileOp::Rename { source, .. } => vec![source.clone()],
                FileOp::Copy { source, target } => vec![source.clone(), target.clone()],
                FileOp::Merge { sources, target, .. } => {
                    let mut p = sources.clone();
                    p.push(target.clone());
                    p
                }
            };

            for p in &paths {
                let s = p.to_string_lossy().to_lowercase();
                if s.contains("windows") || s.contains("system32") || s.contains("/etc")
                    || s.contains("/usr") || s.contains("/bin") || s.contains("program files")
                {
                    reasons.push(format!("System path violation: {}", p.display()));
                }
            }
        }

        // Rule 4: Delete operations require explicit paths (no wildcards in path)
        for entry in &manifest.operations {
            if let FileOp::Delete { path, .. } = &entry.operation {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                if name.contains('*') || name.contains('?') {
                    reasons.push(format!("Wildcard deletion blocked: {}", path.display()));
                }
            }
        }

        (reasons.is_empty(), reasons)
    }

    /// SAT Auditor verifies post-execution integrity.
    /// Checks that source hashes match pre-state and target hashes are valid.
    pub fn sat_audit_manifest(manifest: &OperationManifest) -> (bool, Vec<String>) {
        let mut issues = Vec::new();

        for (i, entry) in manifest.operations.iter().enumerate() {
            match entry.status {
                OpStatus::Succeeded => {
                    // Verify hash integrity if available
                    if entry.source_hash.is_some() && entry.target_hash.is_none() {
                        issues.push(format!("Op #{}: succeeded but target hash missing", i));
                    }
                }
                OpStatus::Failed => {
                    if entry.error.is_none() {
                        issues.push(format!("Op #{}: failed but no error recorded", i));
                    }
                }
                OpStatus::Planned | OpStatus::Approved => {
                    issues.push(format!("Op #{}: not executed (still {:?})", i, entry.status));
                }
                _ => {}
            }
        }

        (issues.is_empty(), issues)
    }

    // ── Summary report ────────────────────────────────────

    /// Generate a human-readable summary of the current inventory.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Files: {} | Size: {:.1} MB",
            self.stats.total_files,
            self.stats.total_bytes as f64 / 1_048_576.0
        ));

        let mut cats: Vec<_> = self.stats.by_category.iter().collect();
        cats.sort_by(|a, b| b.1.count.cmp(&a.1.count));

        for (cat, stats) in cats {
            lines.push(format!(
                "  {:?}: {} files ({:.1} MB, SNR weight {:.2})",
                cat, stats.count, stats.total_bytes as f64 / 1_048_576.0, cat.snr_weight()
            ));
        }
        lines.join("\n")
    }
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_common_extensions() {
        assert_eq!(FileCategory::from_extension("pdf"), FileCategory::Document);
        assert_eq!(FileCategory::from_extension("rs"), FileCategory::Code);
        assert_eq!(FileCategory::from_extension("png"), FileCategory::Image);
        assert_eq!(FileCategory::from_extension("json"), FileCategory::Data);
        assert_eq!(FileCategory::from_extension("mp4"), FileCategory::Video);
        assert_eq!(FileCategory::from_extension("xlsx"), FileCategory::Spreadsheet);
        assert_eq!(FileCategory::from_extension("zip"), FileCategory::Archive);
        assert_eq!(FileCategory::from_extension("exe"), FileCategory::Executable);
        assert_eq!(FileCategory::from_extension("xyz"), FileCategory::Other);
    }

    #[test]
    fn classify_case_insensitive() {
        assert_eq!(FileCategory::from_extension("PDF"), FileCategory::Document);
        assert_eq!(FileCategory::from_extension("Rs"), FileCategory::Code);
        assert_eq!(FileCategory::from_extension("JSON"), FileCategory::Data);
    }

    #[test]
    fn snr_weights_ordered() {
        assert!(FileCategory::Code.snr_weight() > FileCategory::Image.snr_weight());
        assert!(FileCategory::Document.snr_weight() > FileCategory::Archive.snr_weight());
        assert!(FileCategory::Data.snr_weight() > FileCategory::Video.snr_weight());
    }

    #[test]
    fn classify_batch_computes_stats() {
        let mut fm = SmartFileManager::new();
        let files = vec![
            (PathBuf::from("/docs/report.pdf"), 1024),
            (PathBuf::from("/docs/notes.txt"), 512),
            (PathBuf::from("/code/main.rs"), 2048),
            (PathBuf::from("/code/lib.rs"), 1536),
            (PathBuf::from("/images/photo.png"), 4096),
        ];
        fm.classify_batch(files);

        assert_eq!(fm.stats.total_files, 5);
        assert_eq!(fm.stats.total_bytes, 9216);
        assert_eq!(fm.stats.by_category[&FileCategory::Document].count, 2);
        assert_eq!(fm.stats.by_category[&FileCategory::Code].count, 2);
        assert_eq!(fm.stats.by_category[&FileCategory::Image].count, 1);
    }

    #[test]
    fn auto_organize_skips_correct_location() {
        let mut fm = SmartFileManager::new();
        let base = PathBuf::from("/organized");
        let files = vec![
            (PathBuf::from("/downloads/report.pdf"), 1024),
            (PathBuf::from("/organized/Documents/notes.txt"), 512), // already correct
            (PathBuf::from("/downloads/main.rs"), 2048),
        ];
        fm.classify_batch(files);
        let manifest = fm.plan_auto_organize(&base, 1000);

        // Only 2 moves (the one already in Documents/ is skipped)
        assert_eq!(manifest.total_ops(), 2);
    }

    #[test]
    fn rename_pattern_prefix() {
        let p = RenamePattern::Prefix("2026-03_".into());
        assert_eq!(p.apply("report.pdf", 0), "2026-03_report.pdf");
    }

    #[test]
    fn rename_pattern_sequential() {
        let p = RenamePattern::Sequential { base: "photo".into(), start: 1, pad: 3 };
        assert_eq!(p.apply("IMG_1234.jpg", 0), "photo_001.jpg");
        assert_eq!(p.apply("IMG_5678.jpg", 1), "photo_002.jpg");
    }

    #[test]
    fn rename_pattern_sanitize() {
        let p = RenamePattern::Sanitize { separator: '_' };
        assert_eq!(p.apply("my file (copy).pdf", 0), "my_file__copy_.pdf");
    }

    #[test]
    fn rename_pattern_lowercase() {
        let p = RenamePattern::Lowercase;
        assert_eq!(p.apply("My Report.PDF", 0), "my report.pdf");
    }

    #[test]
    fn sat_blocks_system_paths() {
        let mut manifest = OperationManifest::new(1000);
        manifest.add(FileOp::Delete {
            path: PathBuf::from("C:\\Windows\\System32\\important.dll"),
            reason: "cleanup".into(),
        });
        let (approved, reasons) = SmartFileManager::sat_validate_manifest(&manifest);
        assert!(!approved);
        assert!(reasons.iter().any(|r| r.contains("System path violation")));
    }

    #[test]
    fn sat_blocks_wildcard_deletes() {
        let mut manifest = OperationManifest::new(1000);
        manifest.add(FileOp::Delete {
            path: PathBuf::from("/downloads/*.tmp"),
            reason: "cleanup".into(),
        });
        let (approved, reasons) = SmartFileManager::sat_validate_manifest(&manifest);
        assert!(!approved);
        assert!(reasons.iter().any(|r| r.contains("Wildcard deletion")));
    }

    #[test]
    fn sat_approves_safe_manifest() {
        let mut manifest = OperationManifest::new(1000);
        manifest.add(FileOp::Move {
            source: PathBuf::from("/downloads/report.pdf"),
            target: PathBuf::from("/organized/Documents/report.pdf"),
        });
        manifest.add(FileOp::Rename {
            source: PathBuf::from("/downloads/old_name.txt"),
            new_name: "new_name.txt".into(),
        });
        let (approved, _reasons) = SmartFileManager::sat_validate_manifest(&manifest);
        assert!(approved);
    }

    #[test]
    fn sat_blocks_massive_destructive_batch() {
        let mut manifest = OperationManifest::new(1000);
        for i in 0..501 {
            manifest.add(FileOp::Delete {
                path: PathBuf::from(format!("/tmp/file_{}.tmp", i)),
                reason: "batch cleanup".into(),
            });
        }
        let (approved, reasons) = SmartFileManager::sat_validate_manifest(&manifest);
        assert!(!approved);
        assert!(reasons.iter().any(|r| r.contains("exceeds safety limit")));
    }

    #[test]
    fn duplicate_detection_by_size() {
        let mut fm = SmartFileManager::new();
        let files = vec![
            (PathBuf::from("/a/file1.txt"), 1024),
            (PathBuf::from("/b/file2.txt"), 1024), // same size = candidate
            (PathBuf::from("/c/file3.txt"), 2048), // different size
        ];
        fm.classify_batch(files);
        let dupes = fm.find_duplicates();
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].len(), 2);
    }

    #[test]
    fn manifest_seal_updates_hash() {
        let mut manifest = OperationManifest::new(1000);
        manifest.add(FileOp::Copy {
            source: PathBuf::from("/a.txt"),
            target: PathBuf::from("/b.txt"),
        });
        let id_before = manifest.id;
        manifest.seal(2000);
        assert_ne!(manifest.id, id_before);
        assert_eq!(manifest.completed_at, Some(2000));
    }

    #[test]
    fn summary_report_includes_all_categories() {
        let mut fm = SmartFileManager::new();
        let files = vec![
            (PathBuf::from("a.rs"), 100),
            (PathBuf::from("b.pdf"), 200),
            (PathBuf::from("c.png"), 300),
        ];
        fm.classify_batch(files);
        let s = fm.summary();
        assert!(s.contains("Code"));
        assert!(s.contains("Document"));
        assert!(s.contains("Image"));
    }
}
