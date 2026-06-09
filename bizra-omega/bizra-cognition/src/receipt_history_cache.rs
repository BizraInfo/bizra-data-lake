//! BIZRA Receipt History Cache — §Cycle-7 G3 Commit-1
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/receipt_history_cache.rs
//! Authority: cycle-7/niyyah.md §"Writer authority decision (HYBRID)" +
//!            §G3 "Persistent Local Memory"
//! Cycle position: 7, Phase 3
//!
//! Second of six dema_cache surfaces. Mirrors the proven atomic-write,
//! schema-versioned-read pattern of principal_cache.rs.
//!
//! Niyyah § on writer authority:
//!   "Rust MAY write new local-only, non-chain surfaces … receipt history …
//!    These Rust-written surfaces are derived and rebuildable, never
//!    authoritative. If any Rust-written cache diverges from chain truth,
//!    rebuild from chain and mark the cache stale — never outrank chain."
//!
//! Cache semantics:
//!   - Atomic write: temp-then-rename on the same filesystem.
//!   - Schema-versioned JSON; cache content is a thin reflection of
//!     ReceiptChain.records() plus head and last_timestamp_ns.
//!   - Authoritative source is the in-memory ReceiptChain (and its payload
//!     store). This cache only exists so a freshly booted runtime can show
//!     the operator a receipt history list without a full chain replay.
//!   - Unknown ReceiptKind discriminants are rejected on read (fails
//!     closed); callers must delete-and-rebuild to recover.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;
use crate::receipts::{Receipt, ReceiptKind};

/// Schema version for the on-disk JSON payload.
pub const CACHE_SCHEMA_VERSION: &str = "v1";

/// Standard relative paths under sovereign_state/.
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const RECEIPT_HISTORY_FILENAME: &str = "receipt_history.json";

/// Snapshot of a ReceiptChain that round-trips through the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptHistorySnapshot {
    pub head: Blake3Hash,
    pub last_timestamp_ns: Option<u64>,
    pub records: Vec<Receipt>,
}

#[derive(Debug)]
pub enum ReceiptHistoryCacheError {
    DirCreate {
        path: PathBuf,
        msg: String,
    },
    TempWrite {
        path: PathBuf,
        msg: String,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
        msg: String,
    },
    ReadFailed {
        path: PathBuf,
        msg: String,
    },
    ParseFailed {
        path: PathBuf,
        msg: String,
    },
    Malformed {
        path: PathBuf,
        reason: &'static str,
    },
    SchemaMismatch {
        path: PathBuf,
        got: String,
        want: String,
    },
    UnknownKind {
        path: PathBuf,
        byte: u8,
    },
    HexDecode {
        field: &'static str,
        reason: String,
    },
    Serialize(String),
}

impl std::fmt::Display for ReceiptHistoryCacheError {
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
            Self::SchemaMismatch { path, got, want } => write!(
                f,
                "cache {} schema {}, expected {}",
                path.display(),
                got,
                want
            ),
            Self::UnknownKind { path, byte } => write!(
                f,
                "cache {} unknown ReceiptKind byte 0x{:02x}",
                path.display(),
                byte
            ),
            Self::HexDecode { field, reason } => {
                write!(f, "hex decode {}: {}", field, reason)
            }
            Self::Serialize(s) => write!(f, "serialize receipt history: {}", s),
        }
    }
}

impl std::error::Error for ReceiptHistoryCacheError {}

/// Receipt history cache rooted at a sovereign_state/ directory.
#[derive(Debug, Clone)]
pub struct ReceiptHistoryCache {
    cache_dir: PathBuf,
}

impl ReceiptHistoryCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        ReceiptHistoryCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        ReceiptHistoryCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn history_path(&self) -> PathBuf {
        self.cache_dir.join(RECEIPT_HISTORY_FILENAME)
    }

    /// Atomically write a receipt-history snapshot. Temp-then-rename on the
    /// same filesystem ensures a partial write cannot leave the file
    /// observable in a half-state.
    pub fn write(&self, snapshot: &ReceiptHistorySnapshot) -> Result<(), ReceiptHistoryCacheError> {
        Self::write_snapshot_file(&self.history_path(), snapshot, CACHE_SCHEMA_VERSION)
    }

    /// Write a snapshot to an arbitrary path with an explicit schema marker.
    /// Used by the authoritative receipt chain store (Cycle-6 Arc 3).
    pub fn write_snapshot_file(
        path: &Path,
        snapshot: &ReceiptHistorySnapshot,
        schema_version: &str,
    ) -> Result<(), ReceiptHistoryCacheError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| ReceiptHistoryCacheError::DirCreate {
                path: parent.to_path_buf(),
                msg: e.to_string(),
            })?;
        }

        let mut records_json = Vec::with_capacity(snapshot.records.len());
        for r in &snapshot.records {
            records_json.push(json!({
                "kind": r.kind as u8,
                "hash": hex_encode(&r.hash),
                "prev": hex_encode(&r.prev),
            }));
        }

        let payload = json!({
            "schema_version": schema_version,
            "head": hex_encode(&snapshot.head),
            "last_timestamp_ns": snapshot.last_timestamp_ns,
            "records": records_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| ReceiptHistoryCacheError::Serialize(e.to_string()))?;

        let parent = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("snapshot.json");
        let tmp_path = parent.join(format!("{}.tmp.{}", file_name, std::process::id()));

        {
            let mut f =
                fs::File::create(&tmp_path).map_err(|e| ReceiptHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.write_all(&bytes)
                .map_err(|e| ReceiptHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.sync_all()
                .map_err(|e| ReceiptHistoryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
        }

        fs::rename(&tmp_path, path).map_err(|e| ReceiptHistoryCacheError::Rename {
            from: tmp_path,
            to: path.to_path_buf(),
            msg: e.to_string(),
        })?;

        Ok(())
    }

    /// Read the receipt-history snapshot from disk if present. Returns
    /// `Ok(None)` when the cache file is absent. Fails closed on any
    /// schema, malformed-content, unknown-kind, or hex-decode error.
    pub fn read(&self) -> Result<Option<ReceiptHistorySnapshot>, ReceiptHistoryCacheError> {
        Self::read_snapshot_file(&self.history_path(), CACHE_SCHEMA_VERSION)
    }

    /// Read a snapshot from an arbitrary path with an explicit schema marker.
    pub fn read_snapshot_file(
        path: &Path,
        expected_schema: &str,
    ) -> Result<Option<ReceiptHistorySnapshot>, ReceiptHistoryCacheError> {
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path).map_err(|e| ReceiptHistoryCacheError::ReadFailed {
            path: path.to_path_buf(),
            msg: e.to_string(),
        })?;
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|e| ReceiptHistoryCacheError::ParseFailed {
                path: path.to_path_buf(),
                msg: e.to_string(),
            })?;
        let obj = v.as_object().ok_or(ReceiptHistoryCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "root is not an object",
        })?;
        let schema = obj.get("schema_version").and_then(|x| x.as_str()).ok_or(
            ReceiptHistoryCacheError::Malformed {
                path: path.to_path_buf(),
                reason: "missing schema_version",
            },
        )?;
        if schema != expected_schema {
            return Err(ReceiptHistoryCacheError::SchemaMismatch {
                path: path.to_path_buf(),
                got: schema.into(),
                want: expected_schema.into(),
            });
        }

        let head = field_hex(obj, path, "head")?;
        let last_timestamp_ns = match obj.get("last_timestamp_ns") {
            None => None,
            Some(Value::Null) => None,
            Some(x) => Some(x.as_u64().ok_or(ReceiptHistoryCacheError::Malformed {
                path: path.to_path_buf(),
                reason: "last_timestamp_ns not u64",
            })?),
        };

        let records_raw = obj.get("records").and_then(|x| x.as_array()).ok_or(
            ReceiptHistoryCacheError::Malformed {
                path: path.to_path_buf(),
                reason: "missing records array",
            },
        )?;
        let mut records = Vec::with_capacity(records_raw.len());
        for r in records_raw {
            let robj = r.as_object().ok_or(ReceiptHistoryCacheError::Malformed {
                path: path.to_path_buf(),
                reason: "record is not an object",
            })?;
            let kind_byte = robj.get("kind").and_then(|x| x.as_u64()).ok_or(
                ReceiptHistoryCacheError::Malformed {
                    path: path.to_path_buf(),
                    reason: "record.kind missing or not u8",
                },
            )?;
            if kind_byte > 0xFF {
                return Err(ReceiptHistoryCacheError::UnknownKind {
                    path: path.to_path_buf(),
                    byte: 0xFF,
                });
            }
            let kind = ReceiptKind::from_byte(kind_byte as u8).ok_or(
                ReceiptHistoryCacheError::UnknownKind {
                    path: path.to_path_buf(),
                    byte: kind_byte as u8,
                },
            )?;
            let hash = field_hex(robj, path, "hash")?;
            let prev = field_hex(robj, path, "prev")?;
            records.push(Receipt { kind, hash, prev });
        }

        Ok(Some(ReceiptHistorySnapshot {
            head,
            last_timestamp_ns,
            records,
        }))
    }

    /// Delete the cache file if it exists. Idempotent.
    pub fn delete(&self) -> Result<(), ReceiptHistoryCacheError> {
        let path = self.history_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| ReceiptHistoryCacheError::ReadFailed {
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
) -> Result<Blake3Hash, ReceiptHistoryCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(ReceiptHistoryCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing hex field",
        })?;
    hex_decode(s).map_err(|reason| ReceiptHistoryCacheError::HexDecode {
        field: name,
        reason,
    })
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

    fn sample_snapshot() -> ReceiptHistorySnapshot {
        ReceiptHistorySnapshot {
            head: [0xAB; 32],
            last_timestamp_ns: Some(1_700_000_000_000_000_000),
            records: vec![
                Receipt {
                    kind: ReceiptKind::Genesis,
                    hash: [0x01; 32],
                    prev: [0x00; 32],
                },
                Receipt {
                    kind: ReceiptKind::CognitionBoot,
                    hash: [0x02; 32],
                    prev: [0x01; 32],
                },
                Receipt {
                    kind: ReceiptKind::NodeLifecycle,
                    hash: [0x03; 32],
                    prev: [0x02; 32],
                },
                Receipt {
                    kind: ReceiptKind::Manifest,
                    hash: [0x04; 32],
                    prev: [0x03; 32],
                },
                Receipt {
                    kind: ReceiptKind::PrincipalActivation,
                    hash: [0xAB; 32],
                    prev: [0x04; 32],
                },
            ],
        }
    }

    #[test]
    fn write_then_read_round_trips_snapshot() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().expect("snapshot on disk");
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_file_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn write_creates_dema_cache_dir_if_missing() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        assert!(!cache.cache_dir().exists());
        cache.write(&sample_snapshot()).unwrap();
        assert!(cache.cache_dir().exists());
        assert!(cache.history_path().exists());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file_on_success() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(
                !name.contains(".tmp."),
                "unexpected leftover temp file: {}",
                name
            );
        }
    }

    #[test]
    fn write_overwrites_existing_cache() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();

        let mut snap2 = sample_snapshot();
        snap2.records.push(Receipt {
            kind: ReceiptKind::ReasoningSession,
            hash: [0x05; 32],
            prev: [0xAB; 32],
        });
        snap2.head = [0x05; 32];
        cache.write(&snap2).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.records.len(), 6);
        assert_eq!(loaded.head, [0x05; 32]);
    }

    #[test]
    fn empty_snapshot_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        let empty = ReceiptHistorySnapshot {
            head: [0u8; 32],
            last_timestamp_ns: None,
            records: vec![],
        };
        cache.write(&empty).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, empty);
    }

    #[test]
    fn null_last_timestamp_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        let snap = ReceiptHistorySnapshot {
            head: [0x11; 32],
            last_timestamp_ns: None,
            records: vec![Receipt {
                kind: ReceiptKind::CognitionBoot,
                hash: [0x11; 32],
                prev: [0x00; 32],
            }],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.last_timestamp_ns, None);
    }

    #[test]
    fn read_rejects_wrong_schema_version() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.history_path(),
            br#"{"schema_version":"v9999","head":"00","last_timestamp_ns":null,"records":[]}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(
            err,
            ReceiptHistoryCacheError::SchemaMismatch { .. }
        ));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.history_path(), b"not json").unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, ReceiptHistoryCacheError::ParseFailed { .. }));
    }

    #[test]
    fn read_rejects_unknown_receipt_kind() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        let zero = "0000000000000000000000000000000000000000000000000000000000000000";
        let body = format!(
            r#"{{"schema_version":"v1","head":"{z}","last_timestamp_ns":null,
            "records":[{{"kind":171,"hash":"{z}","prev":"{z}"}}]}}"#,
            z = zero
        );
        fs::write(cache.history_path(), body.as_bytes()).unwrap();
        let err = cache.read().unwrap_err();
        assert!(
            matches!(err, ReceiptHistoryCacheError::UnknownKind { byte: 171, .. }),
            "got {:?}",
            err
        );
    }

    #[test]
    fn read_rejects_non_hex_hash() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.history_path(),
            br#"{"schema_version":"v1","head":"XYZ","last_timestamp_ns":null,"records":[]}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, ReceiptHistoryCacheError::HexDecode { .. }));
    }

    #[test]
    fn delete_removes_file_and_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        assert!(cache.history_path().exists());
        cache.delete().unwrap();
        assert!(!cache.history_path().exists());
        cache.delete().unwrap(); // idempotent
    }

    #[test]
    fn restart_survival_simulation_reloads_identical_snapshot() {
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path().to_path_buf();
        let snap = sample_snapshot();

        {
            let cache = ReceiptHistoryCache::at_sovereign_root(&root);
            cache.write(&snap).unwrap();
        }

        {
            let cache2 = ReceiptHistoryCache::at_sovereign_root(&root);
            let loaded = cache2.read().unwrap().expect("snapshot survives restart");
            assert_eq!(loaded, snap);
        }
    }
}
