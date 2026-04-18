//! BIZRA Mission Log Cache — §Cycle-7 G3 Commit-3
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/mission_log_cache.rs
//! Authority: cycle-7/niyyah.md §"Writer authority decision (HYBRID)" +
//!            §G3 "Persistent Local Memory"
//! Cycle position: 7, Phase 3
//!
//! Fourth of six dema_cache surfaces. Persists the stream of
//! MissionRecord outcomes — both permitted and rejected — with
//! structured remediation text attached to rejections.
//!
//! Mission log vs receipt_history vs manifest_history:
//!   - receipt_history: thin chain records (kind+hash+prev) only
//!   - manifest_history: permit-path manifests only
//!   - mission_log: every operator attempt, permit AND reject,
//!     with human-readable intent + outcome + remediation
//!
//! Niyyah § on writer authority: derived and rebuildable; chain
//! stays truth.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;

pub const CACHE_SCHEMA_VERSION: &str = "v1";
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const MISSION_LOG_FILENAME: &str = "mission_log.json";

#[derive(Debug, Clone, PartialEq)]
pub struct MissionLogSnapshot {
    pub chain_head: Blake3Hash,
    pub entries: Vec<MissionLogEntry>,
}

/// One row of the operator-visible mission log. Mirrors the fields of
/// MissionRuntimeRecord that matter to a Dema consumer — intent,
/// outcome, and either a receipt (permit) or a remediation text
/// (reject). Derived from the runtime's missions registry.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionLogEntry {
    pub mission_id: Blake3Hash,
    pub intent_text: String,
    pub timestamp_ns: u64,
    pub rejected: bool,
    pub stage_byte: u8,
    pub receipt_id: Option<Blake3Hash>,
    pub chain_head_after: Blake3Hash,
    pub quality_score: f64,
    pub remediation: Option<String>,
}

#[derive(Debug)]
pub enum MissionLogCacheError {
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

impl std::fmt::Display for MissionLogCacheError {
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
            Self::Serialize(s) => write!(f, "serialize mission log: {}", s),
        }
    }
}

impl std::error::Error for MissionLogCacheError {}

#[derive(Debug, Clone)]
pub struct MissionLogCache {
    cache_dir: PathBuf,
}

impl MissionLogCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        MissionLogCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        MissionLogCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn log_path(&self) -> PathBuf {
        self.cache_dir.join(MISSION_LOG_FILENAME)
    }

    pub fn write(&self, snapshot: &MissionLogSnapshot) -> Result<(), MissionLogCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| MissionLogCacheError::DirCreate {
            path: self.cache_dir.clone(),
            msg: e.to_string(),
        })?;

        let mut entries_json = Vec::with_capacity(snapshot.entries.len());
        for e in &snapshot.entries {
            entries_json.push(json!({
                "mission_id": hex_encode(&e.mission_id),
                "intent_text": e.intent_text,
                "timestamp_ns": e.timestamp_ns,
                "rejected": e.rejected,
                "stage_byte": e.stage_byte,
                "receipt_id": e.receipt_id.as_ref().map(hex_encode),
                "chain_head_after": hex_encode(&e.chain_head_after),
                "quality_score": e.quality_score,
                "remediation": e.remediation,
            }));
        }

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "chain_head": hex_encode(&snapshot.chain_head),
            "entries": entries_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| MissionLogCacheError::Serialize(e.to_string()))?;

        let final_path = self.log_path();
        let tmp_path = self
            .cache_dir
            .join(format!("{}.tmp.{}", MISSION_LOG_FILENAME, std::process::id()));

        {
            let mut f = fs::File::create(&tmp_path).map_err(|e| {
                MissionLogCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
            f.write_all(&bytes).map_err(|e| MissionLogCacheError::TempWrite {
                path: tmp_path.clone(),
                msg: e.to_string(),
            })?;
            f.sync_all().map_err(|e| MissionLogCacheError::TempWrite {
                path: tmp_path.clone(),
                msg: e.to_string(),
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| MissionLogCacheError::Rename {
            from: tmp_path,
            to: final_path,
            msg: e.to_string(),
        })?;

        Ok(())
    }

    pub fn read(&self) -> Result<Option<MissionLogSnapshot>, MissionLogCacheError> {
        let path = self.log_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| MissionLogCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            MissionLogCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            }
        })?;
        let obj = v.as_object().ok_or(MissionLogCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj
            .get("schema_version")
            .and_then(|x| x.as_str())
            .ok_or(MissionLogCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            })?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(MissionLogCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let chain_head = field_hex(obj, &path, "chain_head")?;
        let arr = obj
            .get("entries")
            .and_then(|x| x.as_array())
            .ok_or(MissionLogCacheError::Malformed {
                path: path.clone(),
                reason: "missing entries array",
            })?;

        let mut entries = Vec::with_capacity(arr.len());
        for e in arr {
            let eobj = e.as_object().ok_or(MissionLogCacheError::Malformed {
                path: path.clone(),
                reason: "entry is not an object",
            })?;
            let mission_id = field_hex(eobj, &path, "mission_id")?;
            let intent_text = eobj
                .get("intent_text")
                .and_then(|x| x.as_str())
                .ok_or(MissionLogCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing intent_text",
                })?
                .to_string();
            let timestamp_ns = eobj.get("timestamp_ns").and_then(|x| x.as_u64()).ok_or(
                MissionLogCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing timestamp_ns",
                },
            )?;
            let rejected = eobj.get("rejected").and_then(|x| x.as_bool()).ok_or(
                MissionLogCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing rejected",
                },
            )?;
            let stage_byte = eobj.get("stage_byte").and_then(|x| x.as_u64()).ok_or(
                MissionLogCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing stage_byte",
                },
            )? as u8;
            let receipt_id = match eobj.get("receipt_id") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(hex_decode(s).map_err(|reason| {
                    MissionLogCacheError::HexDecode {
                        field: "receipt_id",
                        reason,
                    }
                })?),
                _ => {
                    return Err(MissionLogCacheError::Malformed {
                        path,
                        reason: "receipt_id neither string nor null",
                    })
                }
            };
            let chain_head_after = field_hex(eobj, &path, "chain_head_after")?;
            let quality_score = eobj.get("quality_score").and_then(|x| x.as_f64()).ok_or(
                MissionLogCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing quality_score",
                },
            )?;
            let remediation = match eobj.get("remediation") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                _ => {
                    return Err(MissionLogCacheError::Malformed {
                        path,
                        reason: "remediation neither string nor null",
                    })
                }
            };
            entries.push(MissionLogEntry {
                mission_id,
                intent_text,
                timestamp_ns,
                rejected,
                stage_byte,
                receipt_id,
                chain_head_after,
                quality_score,
                remediation,
            });
        }

        Ok(Some(MissionLogSnapshot { chain_head, entries }))
    }

    pub fn delete(&self) -> Result<(), MissionLogCacheError> {
        let path = self.log_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| MissionLogCacheError::ReadFailed {
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
) -> Result<Blake3Hash, MissionLogCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(MissionLogCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing hex field",
        })?;
    hex_decode(s).map_err(|reason| MissionLogCacheError::HexDecode { field: name, reason })
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

    fn permit_entry(seed: u8) -> MissionLogEntry {
        MissionLogEntry {
            mission_id: [seed; 32],
            intent_text: format!("mission #{}", seed),
            timestamp_ns: 1_700_000_000_000 + seed as u64,
            rejected: false,
            stage_byte: 0x08, // Replayability
            receipt_id: Some([seed.wrapping_add(1); 32]),
            chain_head_after: [seed.wrapping_add(2); 32],
            quality_score: 0.98,
            remediation: None,
        }
    }

    fn reject_entry(seed: u8) -> MissionLogEntry {
        MissionLogEntry {
            mission_id: [seed; 32],
            intent_text: format!("mission #{} (reject)", seed),
            timestamp_ns: 1_700_000_000_000 + seed as u64,
            rejected: true,
            stage_byte: 0x04, // Admissibility
            receipt_id: None,
            chain_head_after: [seed; 32],
            quality_score: 0.40,
            remediation: Some("IHSAN_FLOOR not met; raise quality_score >= 0.95".into()),
        }
    }

    fn sample_snapshot() -> MissionLogSnapshot {
        MissionLogSnapshot {
            chain_head: [0xAB; 32],
            entries: vec![permit_entry(1), reject_entry(2), permit_entry(3)],
        }
    }

    #[test]
    fn round_trip_preserves_mix_of_permit_and_reject_entries() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(!name.contains(".tmp."), "leftover temp file: {}", name);
        }
    }

    #[test]
    fn empty_entries_round_trip() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        let empty = MissionLogSnapshot {
            chain_head: [0u8; 32],
            entries: vec![],
        };
        cache.write(&empty).unwrap();
        assert_eq!(cache.read().unwrap().unwrap(), empty);
    }

    #[test]
    fn null_receipt_id_for_reject_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        let snap = MissionLogSnapshot {
            chain_head: [0x11; 32],
            entries: vec![reject_entry(9)],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.entries[0].receipt_id, None);
        assert!(loaded.entries[0].remediation.is_some());
    }

    #[test]
    fn read_rejects_wrong_schema() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.log_path(),
            br#"{"schema_version":"v999","chain_head":"00","entries":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            MissionLogCacheError::SchemaMismatch { .. }
        ));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.log_path(), b"{ not json").unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            MissionLogCacheError::ParseFailed { .. }
        ));
    }

    #[test]
    fn read_rejects_non_hex_mission_id() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        let z = "0000000000000000000000000000000000000000000000000000000000000000";
        let body = format!(
            r#"{{"schema_version":"v1","chain_head":"{z}","entries":[
            {{"mission_id":"XYZ","intent_text":"t","timestamp_ns":0,"rejected":false,
              "stage_byte":8,"receipt_id":null,"chain_head_after":"{z}",
              "quality_score":0.98,"remediation":null}}
            ]}}"#,
            z = z
        );
        fs::write(cache.log_path(), body.as_bytes()).unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            MissionLogCacheError::HexDecode { .. }
        ));
    }

    #[test]
    fn delete_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        cache.delete().unwrap();
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival() {
        let td = tempfile::TempDir::new().unwrap();
        let snap = sample_snapshot();
        {
            let cache = MissionLogCache::at_sovereign_root(td.path());
            cache.write(&snap).unwrap();
        }
        let cache2 = MissionLogCache::at_sovereign_root(td.path());
        assert_eq!(cache2.read().unwrap().unwrap(), snap);
    }

    #[test]
    fn f64_quality_score_round_trips_without_loss() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = MissionLogCache::at_sovereign_root(td.path());
        let mut e = permit_entry(5);
        e.quality_score = 0.987_654_321_098_765;
        let snap = MissionLogSnapshot {
            chain_head: [0xCC; 32],
            entries: vec![e.clone()],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.entries[0].quality_score, e.quality_score);
    }
}
