//! BIZRA State Snapshots Cache — §Cycle-7 G3 Commit-4
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/state_snapshots_cache.rs
//! Authority: cycle-7/niyyah.md §"Writer authority decision (HYBRID)" +
//!            §G3 "Persistent Local Memory"
//! Cycle position: 7, Phase 3
//!
//! Fifth of six dema_cache surfaces. Persists the FourStateModel
//! (current + ideal + gap) attached to each mission attempt so a
//! freshly booted operator UI can display "here is where NODE0 is,
//! here is where it aims to be, here is the gap."
//!
//! Niyyah §"Writer authority decision (HYBRID)": derived from the
//! missions registry; chain stays truth; cache is a read fast-path.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;

pub const CACHE_SCHEMA_VERSION: &str = "v1";
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const STATE_SNAPSHOTS_FILENAME: &str = "state_snapshots.json";

#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshotsSnapshot {
    pub chain_head: Blake3Hash,
    pub entries: Vec<StateSnapshotEntry>,
}

/// One row of the state-snapshots cache. Binds a mission_id to the
/// FourStateModel observed at submission time: current + ideal +
/// computed gap.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshotEntry {
    pub mission_id: Blake3Hash,
    pub timestamp_ns: u64,
    pub rejected: bool,
    pub current: StateSnapshotView,
    pub ideal: StateSnapshotView,
    pub gap: f64,
}

/// Owned projection of a single StateSnapshot (current or ideal).
#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshotView {
    pub hash: Blake3Hash,
    pub summary: String,
    pub metric: f64,
}

#[derive(Debug)]
pub enum StateSnapshotsCacheError {
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

impl std::fmt::Display for StateSnapshotsCacheError {
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
            Self::Serialize(s) => write!(f, "serialize state snapshots: {}", s),
        }
    }
}

impl std::error::Error for StateSnapshotsCacheError {}

#[derive(Debug, Clone)]
pub struct StateSnapshotsCache {
    cache_dir: PathBuf,
}

impl StateSnapshotsCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        StateSnapshotsCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        StateSnapshotsCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn snapshots_path(&self) -> PathBuf {
        self.cache_dir.join(STATE_SNAPSHOTS_FILENAME)
    }

    pub fn write(
        &self,
        snapshot: &StateSnapshotsSnapshot,
    ) -> Result<(), StateSnapshotsCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| {
            StateSnapshotsCacheError::DirCreate {
                path: self.cache_dir.clone(),
                msg: e.to_string(),
            }
        })?;

        let mut entries_json = Vec::with_capacity(snapshot.entries.len());
        for e in &snapshot.entries {
            entries_json.push(json!({
                "mission_id": hex_encode(&e.mission_id),
                "timestamp_ns": e.timestamp_ns,
                "rejected": e.rejected,
                "current": view_to_json(&e.current),
                "ideal": view_to_json(&e.ideal),
                "gap": e.gap,
            }));
        }

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "chain_head": hex_encode(&snapshot.chain_head),
            "entries": entries_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| StateSnapshotsCacheError::Serialize(e.to_string()))?;

        let final_path = self.snapshots_path();
        let tmp_path = self.cache_dir.join(format!(
            "{}.tmp.{}",
            STATE_SNAPSHOTS_FILENAME,
            std::process::id()
        ));

        {
            let mut f = fs::File::create(&tmp_path).map_err(|e| {
                StateSnapshotsCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
            f.write_all(&bytes).map_err(|e| StateSnapshotsCacheError::TempWrite {
                path: tmp_path.clone(),
                msg: e.to_string(),
            })?;
            f.sync_all().map_err(|e| StateSnapshotsCacheError::TempWrite {
                path: tmp_path.clone(),
                msg: e.to_string(),
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| {
            StateSnapshotsCacheError::Rename {
                from: tmp_path,
                to: final_path,
                msg: e.to_string(),
            }
        })?;

        Ok(())
    }

    pub fn read(&self) -> Result<Option<StateSnapshotsSnapshot>, StateSnapshotsCacheError> {
        let path = self.snapshots_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| StateSnapshotsCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            StateSnapshotsCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            }
        })?;
        let obj = v.as_object().ok_or(StateSnapshotsCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj
            .get("schema_version")
            .and_then(|x| x.as_str())
            .ok_or(StateSnapshotsCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            })?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(StateSnapshotsCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let chain_head = field_hex(obj, &path, "chain_head")?;
        let arr = obj
            .get("entries")
            .and_then(|x| x.as_array())
            .ok_or(StateSnapshotsCacheError::Malformed {
                path: path.clone(),
                reason: "missing entries array",
            })?;

        let mut entries = Vec::with_capacity(arr.len());
        for e in arr {
            let eobj = e.as_object().ok_or(StateSnapshotsCacheError::Malformed {
                path: path.clone(),
                reason: "entry is not an object",
            })?;
            let mission_id = field_hex(eobj, &path, "mission_id")?;
            let timestamp_ns = eobj.get("timestamp_ns").and_then(|x| x.as_u64()).ok_or(
                StateSnapshotsCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing timestamp_ns",
                },
            )?;
            let rejected = eobj.get("rejected").and_then(|x| x.as_bool()).ok_or(
                StateSnapshotsCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing rejected",
                },
            )?;
            let current = view_from_json(
                eobj.get("current")
                    .ok_or(StateSnapshotsCacheError::Malformed {
                        path: path.clone(),
                        reason: "missing current",
                    })?,
                &path,
                "current",
            )?;
            let ideal = view_from_json(
                eobj.get("ideal")
                    .ok_or(StateSnapshotsCacheError::Malformed {
                        path: path.clone(),
                        reason: "missing ideal",
                    })?,
                &path,
                "ideal",
            )?;
            let gap = eobj.get("gap").and_then(|x| x.as_f64()).ok_or(
                StateSnapshotsCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing gap",
                },
            )?;
            entries.push(StateSnapshotEntry {
                mission_id,
                timestamp_ns,
                rejected,
                current,
                ideal,
                gap,
            });
        }
        Ok(Some(StateSnapshotsSnapshot { chain_head, entries }))
    }

    pub fn delete(&self) -> Result<(), StateSnapshotsCacheError> {
        let path = self.snapshots_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| StateSnapshotsCacheError::ReadFailed {
                path,
                msg: e.to_string(),
            })?;
        }
        Ok(())
    }
}

fn view_to_json(v: &StateSnapshotView) -> Value {
    json!({
        "hash": hex_encode(&v.hash),
        "summary": v.summary,
        "metric": v.metric,
    })
}

fn view_from_json(
    v: &Value,
    path: &Path,
    _context: &'static str,
) -> Result<StateSnapshotView, StateSnapshotsCacheError> {
    let obj = v.as_object().ok_or(StateSnapshotsCacheError::Malformed {
        path: path.to_path_buf(),
        reason: "state view is not an object",
    })?;
    let hash = field_hex(obj, path, "hash")?;
    let summary = obj
        .get("summary")
        .and_then(|x| x.as_str())
        .ok_or(StateSnapshotsCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing summary",
        })?
        .to_string();
    let metric = obj.get("metric").and_then(|x| x.as_f64()).ok_or(
        StateSnapshotsCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing metric",
        },
    )?;
    Ok(StateSnapshotView { hash, summary, metric })
}

fn field_hex(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
    name: &'static str,
) -> Result<Blake3Hash, StateSnapshotsCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(StateSnapshotsCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing hex field",
        })?;
    hex_decode(s).map_err(|reason| StateSnapshotsCacheError::HexDecode { field: name, reason })
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

    fn sample_entry(seed: u8, rejected: bool) -> StateSnapshotEntry {
        StateSnapshotEntry {
            mission_id: [seed; 32],
            timestamp_ns: 1_700_000_000_000 + seed as u64,
            rejected,
            current: StateSnapshotView {
                hash: [seed.wrapping_add(1); 32],
                summary: "Current: not yet canonical".into(),
                metric: 0.2,
            },
            ideal: StateSnapshotView {
                hash: [seed.wrapping_add(2); 32],
                summary: "Ideal: canonical + receipted".into(),
                metric: 1.0,
            },
            gap: 0.8,
        }
    }

    fn sample_snapshot() -> StateSnapshotsSnapshot {
        StateSnapshotsSnapshot {
            chain_head: [0xAB; 32],
            entries: vec![sample_entry(1, false), sample_entry(2, true), sample_entry(3, false)],
        }
    }

    #[test]
    fn round_trip_preserves_entries() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(!name.contains(".tmp."), "leftover temp file: {}", name);
        }
    }

    #[test]
    fn empty_entries_round_trip() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        let empty = StateSnapshotsSnapshot {
            chain_head: [0u8; 32],
            entries: vec![],
        };
        cache.write(&empty).unwrap();
        assert_eq!(cache.read().unwrap().unwrap(), empty);
    }

    #[test]
    fn gap_f64_round_trips_without_loss() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        let mut e = sample_entry(7, false);
        e.gap = 0.123_456_789_012_345;
        e.current.metric = 0.987_654_321_098;
        let snap = StateSnapshotsSnapshot {
            chain_head: [0xBB; 32],
            entries: vec![e.clone()],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.entries[0].gap, e.gap);
        assert_eq!(loaded.entries[0].current.metric, e.current.metric);
    }

    #[test]
    fn read_rejects_wrong_schema() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.snapshots_path(),
            br#"{"schema_version":"v999","chain_head":"00","entries":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            StateSnapshotsCacheError::SchemaMismatch { .. }
        ));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.snapshots_path(), b"not json").unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            StateSnapshotsCacheError::ParseFailed { .. }
        ));
    }

    #[test]
    fn read_rejects_bad_current_hash() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        let z = "0000000000000000000000000000000000000000000000000000000000000000";
        let body = format!(
            r#"{{"schema_version":"v1","chain_head":"{z}","entries":[
            {{"mission_id":"{z}","timestamp_ns":0,"rejected":false,
              "current":{{"hash":"XYZ","summary":"c","metric":0.0}},
              "ideal":{{"hash":"{z}","summary":"i","metric":1.0}},
              "gap":1.0}}
            ]}}"#,
            z = z
        );
        fs::write(cache.snapshots_path(), body.as_bytes()).unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            StateSnapshotsCacheError::HexDecode { .. }
        ));
    }

    #[test]
    fn delete_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = StateSnapshotsCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        cache.delete().unwrap();
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival() {
        let td = tempfile::TempDir::new().unwrap();
        let snap = sample_snapshot();
        {
            let cache = StateSnapshotsCache::at_sovereign_root(td.path());
            cache.write(&snap).unwrap();
        }
        let cache2 = StateSnapshotsCache::at_sovereign_root(td.path());
        assert_eq!(cache2.read().unwrap().unwrap(), snap);
    }
}
