//! BIZRA Manifest History Cache — §Cycle-7 G3 Commit-2
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/manifest_history_cache.rs
//! Authority: cycle-7/niyyah.md §"Writer authority decision (HYBRID)" +
//!            §G3 "Persistent Local Memory"
//! Cycle position: 7, Phase 3
//!
//! Third of six dema_cache surfaces. Persists decoded ManifestArtifact
//! records so a freshly booted operator UI can list "what the node has
//! manifested" without a chain replay.
//!
//! Niyyah § on writer authority: derived and rebuildable from chain; if
//! the cache diverges, delete and rebuild.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;
use crate::manifest_artifact::ManifestArtifact;

pub const CACHE_SCHEMA_VERSION: &str = "v1";
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const MANIFEST_HISTORY_FILENAME: &str = "manifest_history.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestHistorySnapshot {
    pub chain_head: Blake3Hash,
    pub manifests: Vec<ManifestSummary>,
}

/// Owned, round-trippable projection of a ManifestArtifact suitable for
/// on-disk persistence. Mirrors the live struct but without the trait
/// machinery, so reads do not need the full payload store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSummary {
    pub manifest_id: Blake3Hash,
    pub window_start: u64,
    pub window_end: u64,
    pub receipt_refs: Vec<Blake3Hash>,
    pub integrity_hash: Blake3Hash,
    pub receipt_count: u32,
    pub chain_head_at_generation: Blake3Hash,
}

impl From<&ManifestArtifact> for ManifestSummary {
    fn from(m: &ManifestArtifact) -> Self {
        ManifestSummary {
            manifest_id: m.manifest_id,
            window_start: m.window_start,
            window_end: m.window_end,
            receipt_refs: m.receipt_refs.clone(),
            integrity_hash: m.integrity_hash,
            receipt_count: m.receipt_count,
            chain_head_at_generation: m.chain_head_at_generation,
        }
    }
}

#[derive(Debug)]
pub enum ManifestHistoryCacheError {
    DirCreate { path: PathBuf, msg: String },
    TempWrite { path: PathBuf, msg: String },
    Rename { from: PathBuf, to: PathBuf, msg: String },
    ReadFailed { path: PathBuf, msg: String },
    ParseFailed { path: PathBuf, msg: String },
    Malformed { path: PathBuf, reason: &'static str },
    SchemaMismatch { path: PathBuf, got: String, want: String },
    HexDecode { field: &'static str, reason: String },
    Serialize(String),
}

impl std::fmt::Display for ManifestHistoryCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirCreate { path, msg } => {
                write!(f, "create dema_cache dir {}: {}", path.display(), msg)
            }
            Self::TempWrite { path, msg } => {
                write!(f, "write cache temp {}: {}", path.display(), msg)
            }
            Self::Rename { from, to, msg } => {
                write!(f, "rename {} -> {}: {}", from.display(), to.display(), msg)
            }
            Self::ReadFailed { path, msg } => {
                write!(f, "read cache {}: {}", path.display(), msg)
            }
            Self::ParseFailed { path, msg } => {
                write!(f, "parse cache {}: {}", path.display(), msg)
            }
            Self::Malformed { path, reason } => {
                write!(f, "cache {} malformed: {}", path.display(), reason)
            }
            Self::SchemaMismatch { path, got, want } => {
                write!(f, "cache {} schema {}, expected {}", path.display(), got, want)
            }
            Self::HexDecode { field, reason } => {
                write!(f, "hex decode {}: {}", field, reason)
            }
            Self::Serialize(s) => write!(f, "serialize manifest history: {}", s),
        }
    }
}

impl std::error::Error for ManifestHistoryCacheError {}

#[derive(Debug, Clone)]
pub struct ManifestHistoryCache {
    cache_dir: PathBuf,
}

impl ManifestHistoryCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        ManifestHistoryCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        ManifestHistoryCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn history_path(&self) -> PathBuf {
        self.cache_dir.join(MANIFEST_HISTORY_FILENAME)
    }

    pub fn write(
        &self,
        snapshot: &ManifestHistorySnapshot,
    ) -> Result<(), ManifestHistoryCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| {
            ManifestHistoryCacheError::DirCreate {
                path: self.cache_dir.clone(),
                msg: e.to_string(),
            }
        })?;

        let mut manifests_json = Vec::with_capacity(snapshot.manifests.len());
        for m in &snapshot.manifests {
            let refs: Vec<String> = m.receipt_refs.iter().map(hex_encode).collect();
            manifests_json.push(json!({
                "manifest_id": hex_encode(&m.manifest_id),
                "window_start": m.window_start,
                "window_end": m.window_end,
                "receipt_refs": refs,
                "integrity_hash": hex_encode(&m.integrity_hash),
                "receipt_count": m.receipt_count,
                "chain_head_at_generation": hex_encode(&m.chain_head_at_generation),
            }));
        }

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "chain_head": hex_encode(&snapshot.chain_head),
            "manifests": manifests_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| ManifestHistoryCacheError::Serialize(e.to_string()))?;

        let final_path = self.history_path();
        let tmp_path = self.cache_dir.join(format!(
            "{}.tmp.{}",
            MANIFEST_HISTORY_FILENAME,
            std::process::id()
        ));

        {
            let mut f = fs::File::create(&tmp_path).map_err(|e| {
                ManifestHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
            f.write_all(&bytes).map_err(|e| {
                ManifestHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
            f.sync_all().map_err(|e| {
                ManifestHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| {
            ManifestHistoryCacheError::Rename {
                from: tmp_path,
                to: final_path,
                msg: e.to_string(),
            }
        })?;

        Ok(())
    }

    pub fn read(&self) -> Result<Option<ManifestHistorySnapshot>, ManifestHistoryCacheError> {
        let path = self.history_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| ManifestHistoryCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            ManifestHistoryCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            }
        })?;
        let obj = v.as_object().ok_or(ManifestHistoryCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj
            .get("schema_version")
            .and_then(|x| x.as_str())
            .ok_or(ManifestHistoryCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            })?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(ManifestHistoryCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let chain_head = field_hex(obj, &path, "chain_head")?;
        let arr = obj.get("manifests").and_then(|x| x.as_array()).ok_or(
            ManifestHistoryCacheError::Malformed {
                path: path.clone(),
                reason: "missing manifests array",
            },
        )?;

        let mut manifests = Vec::with_capacity(arr.len());
        for m in arr {
            let mobj = m.as_object().ok_or(ManifestHistoryCacheError::Malformed {
                path: path.clone(),
                reason: "manifest is not an object",
            })?;
            let manifest_id = field_hex(mobj, &path, "manifest_id")?;
            let window_start = mobj.get("window_start").and_then(|x| x.as_u64()).ok_or(
                ManifestHistoryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing window_start",
                },
            )?;
            let window_end = mobj.get("window_end").and_then(|x| x.as_u64()).ok_or(
                ManifestHistoryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing window_end",
                },
            )?;
            let refs_raw =
                mobj.get("receipt_refs")
                    .and_then(|x| x.as_array())
                    .ok_or(ManifestHistoryCacheError::Malformed {
                        path: path.clone(),
                        reason: "missing receipt_refs",
                    })?;
            let mut receipt_refs = Vec::with_capacity(refs_raw.len());
            for r in refs_raw {
                let s = r.as_str().ok_or(ManifestHistoryCacheError::Malformed {
                    path: path.clone(),
                    reason: "receipt_ref is not a string",
                })?;
                receipt_refs.push(hex_decode(s).map_err(|reason| {
                    ManifestHistoryCacheError::HexDecode {
                        field: "receipt_refs[]",
                        reason,
                    }
                })?);
            }
            let integrity_hash = field_hex(mobj, &path, "integrity_hash")?;
            let receipt_count = mobj.get("receipt_count").and_then(|x| x.as_u64()).ok_or(
                ManifestHistoryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing receipt_count",
                },
            )? as u32;
            let chain_head_at_generation = field_hex(mobj, &path, "chain_head_at_generation")?;
            manifests.push(ManifestSummary {
                manifest_id,
                window_start,
                window_end,
                receipt_refs,
                integrity_hash,
                receipt_count,
                chain_head_at_generation,
            });
        }

        Ok(Some(ManifestHistorySnapshot {
            chain_head,
            manifests,
        }))
    }

    pub fn delete(&self) -> Result<(), ManifestHistoryCacheError> {
        let path = self.history_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| ManifestHistoryCacheError::ReadFailed {
                path,
                msg: e.to_string(),
            })?;
        }
        Ok(())
    }
}

fn field_hex(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
    name: &'static str,
) -> Result<Blake3Hash, ManifestHistoryCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(ManifestHistoryCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing hex field",
        })?;
    hex_decode(s).map_err(|reason| ManifestHistoryCacheError::HexDecode { field: name, reason })
}

fn hex_encode(bytes: &Blake3Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_decode(s: &str) -> Result<Blake3Hash, String> {
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", s.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_nibble(s.as_bytes()[i * 2])?;
        let lo = hex_nibble(s.as_bytes()[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("non-hex byte 0x{:02x}", b)),
    }
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_summary(seed: u8) -> ManifestSummary {
        ManifestSummary {
            manifest_id: [seed; 32],
            window_start: 1_700_000_000_000 + seed as u64,
            window_end: 1_700_000_000_500 + seed as u64,
            receipt_refs: vec![[seed; 32], [seed.wrapping_add(1); 32]],
            integrity_hash: [seed.wrapping_mul(3); 32],
            receipt_count: 2,
            chain_head_at_generation: [seed.wrapping_mul(7); 32],
        }
    }

    fn sample_snapshot() -> ManifestHistorySnapshot {
        ManifestHistorySnapshot {
            chain_head: [0xAB; 32],
            manifests: vec![sample_summary(1), sample_summary(2), sample_summary(3)],
        }
    }

    #[test]
    fn write_then_read_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().expect("present");
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn write_creates_dir_if_missing() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        assert!(!cache.cache_dir().exists());
        cache.write(&sample_snapshot()).unwrap();
        assert!(cache.history_path().exists());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(!name.contains(".tmp."), "leftover temp file: {}", name);
        }
    }

    #[test]
    fn empty_manifests_round_trip() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        let empty = ManifestHistorySnapshot {
            chain_head: [0u8; 32],
            manifests: vec![],
        };
        cache.write(&empty).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, empty);
    }

    #[test]
    fn read_rejects_wrong_schema() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.history_path(),
            br#"{"schema_version":"v999","chain_head":"00","manifests":[]}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, ManifestHistoryCacheError::SchemaMismatch { .. }));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.history_path(), b"not json").unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, ManifestHistoryCacheError::ParseFailed { .. }));
    }

    #[test]
    fn read_rejects_non_hex_manifest_id() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        let zero = "0000000000000000000000000000000000000000000000000000000000000000";
        let body = format!(
            r#"{{"schema_version":"v1","chain_head":"{z}","manifests":[
            {{"manifest_id":"XYZ","window_start":0,"window_end":0,
              "receipt_refs":[],"integrity_hash":"{z}","receipt_count":0,
              "chain_head_at_generation":"{z}"}}
            ]}}"#,
            z = zero
        );
        fs::write(cache.history_path(), body.as_bytes()).unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, ManifestHistoryCacheError::HexDecode { .. }));
    }

    #[test]
    fn delete_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ManifestHistoryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        cache.delete().unwrap();
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival_simulation() {
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path().to_path_buf();
        let snap = sample_snapshot();
        {
            let cache = ManifestHistoryCache::at_sovereign_root(&root);
            cache.write(&snap).unwrap();
        }
        let cache2 = ManifestHistoryCache::at_sovereign_root(&root);
        let loaded = cache2.read().unwrap().unwrap();
        assert_eq!(loaded, snap);
    }

    #[test]
    fn summary_from_manifest_artifact_preserves_all_fields() {
        let m = ManifestArtifact::from_window(
            100,
            200,
            vec![[0x11; 32], [0x22; 32], [0x33; 32]],
            [0xAA; 32],
        );
        let s = ManifestSummary::from(&m);
        assert_eq!(s.manifest_id, m.manifest_id);
        assert_eq!(s.window_start, m.window_start);
        assert_eq!(s.window_end, m.window_end);
        assert_eq!(s.receipt_refs, m.receipt_refs);
        assert_eq!(s.integrity_hash, m.integrity_hash);
        assert_eq!(s.receipt_count, m.receipt_count);
        assert_eq!(s.chain_head_at_generation, m.chain_head_at_generation);
    }
}
