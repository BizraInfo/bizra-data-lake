//! BIZRA Proof-of-Impact Ledger — §Cycle-7 G6
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/poi_ledger.rs
//! Authority: cycle-7/niyyah.md §G6 "Local-only PoI ledger" +
//!            manifest §"Writer authority decision (HYBRID)"
//! Cycle position: 7, Phase 6
//!
//! Local-only, derived, rebuildable ledger of operator impact.
//!
//! One PoiEntry is produced per permitted lawful state-transition
//! receipt (PrincipalActivation 0x61 + MissionExecuted 0x70). The
//! entry carries:
//!   - receipt_id (source of truth — chain record)
//!   - quality_score (from the admissibility claim)
//!   - gate_min_score (weakest gate verdict — "chain is as strong
//!     as its weakest link")
//!   - entry_count (work volume, 0 for activation, >=0 for organize)
//!   - impact_score = quality_score * gate_min_score + ln(1+entry_count) * 0.01
//!   - principal_id (contributor, from Originator::Operator.session_id)
//!
//! Niyyah compliance:
//!   - Local-only: no federation, no external claims.
//!   - No interest: the "score" is operator discipline + work volume,
//!     not a monetary yield. ZANN_ZERO / RIBA_ZERO respected.
//!   - Rebuildable: if the cache diverges from chain, the runtime
//!     rebuilds from chain and marks the cache stale. Chain stays truth.
//!   - Honest valuation: impact_score is bounded by the admissibility
//!     verdict's weakest gate — dishonest missions cannot inflate
//!     their own score.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;

pub const CACHE_SCHEMA_VERSION: &str = "v1";
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const POI_LEDGER_FILENAME: &str = "poi_ledger.json";

/// ReceiptKind byte that this entry derives from. Kept as a raw u8
/// so the ledger is forward-compatible with new ReceiptKind variants
/// the enforcement law decides to promote into the ledger.
pub type ReceiptKindByte = u8;

/// A single row of the Proof-of-Impact ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct PoiEntry {
    /// Source chain record (MissionExecuted or PrincipalActivation).
    pub receipt_id: Blake3Hash,
    /// ReceiptKind byte. 0x61 = PrincipalActivation, 0x70 = MissionExecuted.
    pub receipt_kind_byte: ReceiptKindByte,
    /// Admissibility claim quality_score (0.0..1.0).
    pub quality_score: f64,
    /// Weakest gate verdict score in the admissibility chain (0.0..1.0).
    pub gate_min_score: f64,
    /// Work volume: listing entry count for organize, 0 for activation.
    pub entry_count: u32,
    /// Impact score computed by `compute_impact_score`.
    pub impact_score: f64,
    /// Mission timestamp (ns).
    pub timestamp_ns: u64,
    /// Contributor principal_id (Originator::Operator.session_id).
    /// None when the mission was submitted pre-activation.
    pub principal_id: Option<Blake3Hash>,
}

/// Compute the v1 lawful impact score.
///
/// Formula:  impact = quality_score * gate_min_score + ln(1+entry_count) * 0.01
///
/// Rationale:
///   - `quality_score * gate_min_score` bounds the score by admissibility
///     truth — dishonest missions cannot inflate their score past the
///     weakest gate that permitted them.
///   - `ln(1+entry_count) * 0.01` is a small positive nudge for work
///     volume so operators are credited for larger organizes without
///     swamping the quality signal (ln keeps it sublinear).
///   - Cap at 1.0 so the score stays interpretable as a fraction.
///
/// Operator policy may refine this formula in a future cycle. For G6
/// it is intentionally simple and documented.
pub fn compute_impact_score(quality_score: f64, gate_min_score: f64, entry_count: u32) -> f64 {
    let base = quality_score.clamp(0.0, 1.0) * gate_min_score.clamp(0.0, 1.0);
    let volume_bonus = ((entry_count as f64) + 1.0).ln() * 0.01;
    (base + volume_bonus).clamp(0.0, 1.0)
}

/// Snapshot of the entire ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct PoiLedgerSnapshot {
    pub chain_head: Blake3Hash,
    pub entries: Vec<PoiEntry>,
}

// ════════════════════════════════════════════════════════════════════
// Cache — atomic write + schema-versioned read
// ════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum PoiLedgerCacheError {
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
    HexDecode {
        field: &'static str,
        reason: String,
    },
    Serialize(String),
}

impl std::fmt::Display for PoiLedgerCacheError {
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
                write!(
                    f,
                    "cache {} schema {}, expected {}",
                    path.display(),
                    got,
                    want
                )
            }
            Self::HexDecode { field, reason } => {
                write!(f, "hex decode {}: {}", field, reason)
            }
            Self::Serialize(s) => write!(f, "serialize poi ledger: {}", s),
        }
    }
}

impl std::error::Error for PoiLedgerCacheError {}

#[derive(Debug, Clone)]
pub struct PoiLedgerCache {
    cache_dir: PathBuf,
}

impl PoiLedgerCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        PoiLedgerCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        PoiLedgerCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn ledger_path(&self) -> PathBuf {
        self.cache_dir.join(POI_LEDGER_FILENAME)
    }

    pub fn write(&self, snapshot: &PoiLedgerSnapshot) -> Result<(), PoiLedgerCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| PoiLedgerCacheError::DirCreate {
            path: self.cache_dir.clone(),
            msg: e.to_string(),
        })?;

        let mut entries_json = Vec::with_capacity(snapshot.entries.len());
        for e in &snapshot.entries {
            entries_json.push(json!({
                "receipt_id": hex_encode(&e.receipt_id),
                "receipt_kind_byte": e.receipt_kind_byte,
                "quality_score": e.quality_score,
                "gate_min_score": e.gate_min_score,
                "entry_count": e.entry_count,
                "impact_score": e.impact_score,
                "timestamp_ns": e.timestamp_ns,
                "principal_id": e.principal_id.as_ref().map(hex_encode),
            }));
        }

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "chain_head": hex_encode(&snapshot.chain_head),
            "entries": entries_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| PoiLedgerCacheError::Serialize(e.to_string()))?;

        let final_path = self.ledger_path();
        let tmp_path = self.cache_dir.join(format!(
            "{}.tmp.{}",
            POI_LEDGER_FILENAME,
            std::process::id()
        ));

        {
            let mut f =
                fs::File::create(&tmp_path).map_err(|e| PoiLedgerCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.write_all(&bytes)
                .map_err(|e| PoiLedgerCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.sync_all().map_err(|e| PoiLedgerCacheError::TempWrite {
                path: tmp_path.clone(),
                msg: e.to_string(),
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| PoiLedgerCacheError::Rename {
            from: tmp_path,
            to: final_path,
            msg: e.to_string(),
        })?;
        Ok(())
    }

    pub fn read(&self) -> Result<Option<PoiLedgerSnapshot>, PoiLedgerCacheError> {
        let path = self.ledger_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| PoiLedgerCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|e| PoiLedgerCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            })?;
        let obj = v.as_object().ok_or(PoiLedgerCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj.get("schema_version").and_then(|x| x.as_str()).ok_or(
            PoiLedgerCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            },
        )?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(PoiLedgerCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let chain_head = field_hex(obj, &path, "chain_head")?;
        let arr = obj.get("entries").and_then(|x| x.as_array()).ok_or(
            PoiLedgerCacheError::Malformed {
                path: path.clone(),
                reason: "missing entries array",
            },
        )?;
        let mut entries = Vec::with_capacity(arr.len());
        for e in arr {
            let eobj = e.as_object().ok_or(PoiLedgerCacheError::Malformed {
                path: path.clone(),
                reason: "entry is not an object",
            })?;
            let receipt_id = field_hex(eobj, &path, "receipt_id")?;
            let receipt_kind_byte = eobj
                .get("receipt_kind_byte")
                .and_then(|x| x.as_u64())
                .ok_or(PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing receipt_kind_byte",
                })? as u8;
            let quality_score = eobj.get("quality_score").and_then(|x| x.as_f64()).ok_or(
                PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing quality_score",
                },
            )?;
            let gate_min_score = eobj.get("gate_min_score").and_then(|x| x.as_f64()).ok_or(
                PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing gate_min_score",
                },
            )?;
            let entry_count = eobj.get("entry_count").and_then(|x| x.as_u64()).ok_or(
                PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing entry_count",
                },
            )? as u32;
            let impact_score = eobj.get("impact_score").and_then(|x| x.as_f64()).ok_or(
                PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing impact_score",
                },
            )?;
            let timestamp_ns = eobj.get("timestamp_ns").and_then(|x| x.as_u64()).ok_or(
                PoiLedgerCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing timestamp_ns",
                },
            )?;
            let principal_id = match eobj.get("principal_id") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => {
                    Some(
                        hex_decode(s).map_err(|reason| PoiLedgerCacheError::HexDecode {
                            field: "principal_id",
                            reason,
                        })?,
                    )
                }
                _ => {
                    return Err(PoiLedgerCacheError::Malformed {
                        path,
                        reason: "principal_id neither string nor null",
                    })
                }
            };
            entries.push(PoiEntry {
                receipt_id,
                receipt_kind_byte,
                quality_score,
                gate_min_score,
                entry_count,
                impact_score,
                timestamp_ns,
                principal_id,
            });
        }
        Ok(Some(PoiLedgerSnapshot {
            chain_head,
            entries,
        }))
    }

    pub fn delete(&self) -> Result<(), PoiLedgerCacheError> {
        let path = self.ledger_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| PoiLedgerCacheError::ReadFailed {
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
) -> Result<Blake3Hash, PoiLedgerCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(PoiLedgerCacheError::Malformed {
            path: path.to_path_buf(),
            reason: "missing hex field",
        })?;
    hex_decode(s).map_err(|reason| PoiLedgerCacheError::HexDecode {
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

    fn sample_entry(seed: u8, kind: u8) -> PoiEntry {
        let qs = 0.95 + (seed as f64) * 0.001;
        let gm = 0.97;
        let ec = seed as u32;
        PoiEntry {
            receipt_id: [seed; 32],
            receipt_kind_byte: kind,
            quality_score: qs,
            gate_min_score: gm,
            entry_count: ec,
            impact_score: compute_impact_score(qs, gm, ec),
            timestamp_ns: 1_700_000_000_000 + seed as u64,
            principal_id: Some([seed.wrapping_add(1); 32]),
        }
    }

    fn sample_snapshot() -> PoiLedgerSnapshot {
        PoiLedgerSnapshot {
            chain_head: [0xAB; 32],
            entries: vec![
                sample_entry(1, 0x61),
                sample_entry(2, 0x70),
                sample_entry(3, 0x70),
            ],
        }
    }

    // ─── scoring ─────────────────────────────────────────────────

    #[test]
    fn impact_score_bounded_by_weakest_gate() {
        // weakest gate = 0 -> base collapses, only volume_bonus remains
        let s = compute_impact_score(1.0, 0.0, 100);
        assert!(s < 0.1, "zero gate must crush impact to tiny volume-only");
    }

    #[test]
    fn impact_score_clamps_to_unit_interval() {
        // pathological inputs should clamp
        let s1 = compute_impact_score(5.0, 5.0, 1_000_000);
        assert!(s1 <= 1.0);
        let s2 = compute_impact_score(-1.0, 0.98, 10);
        assert!(s2 >= 0.0);
    }

    #[test]
    fn zero_entry_count_gives_no_volume_bonus() {
        // ln(1+0) = 0
        let s = compute_impact_score(0.98, 0.97, 0);
        assert!((s - 0.98 * 0.97).abs() < 1e-9);
    }

    #[test]
    fn volume_bonus_is_sublinear() {
        let low = compute_impact_score(0.95, 0.95, 10);
        let high = compute_impact_score(0.95, 0.95, 10_000);
        // 1000x work -> small impact delta
        assert!((high - low) < 0.1);
    }

    // ─── cache round-trip ────────────────────────────────────────

    #[test]
    fn round_trip_preserves_entries() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(!name.contains(".tmp."), "leftover temp file: {}", name);
        }
    }

    #[test]
    fn empty_entries_round_trip() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        let empty = PoiLedgerSnapshot {
            chain_head: [0u8; 32],
            entries: vec![],
        };
        cache.write(&empty).unwrap();
        assert_eq!(cache.read().unwrap().unwrap(), empty);
    }

    #[test]
    fn null_principal_id_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        let mut e = sample_entry(5, 0x70);
        e.principal_id = None;
        let snap = PoiLedgerSnapshot {
            chain_head: [0xCC; 32],
            entries: vec![e.clone()],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.entries[0].principal_id, None);
    }

    #[test]
    fn read_rejects_wrong_schema() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.ledger_path(),
            br#"{"schema_version":"v999","chain_head":"00","entries":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            PoiLedgerCacheError::SchemaMismatch { .. }
        ));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.ledger_path(), b"not json").unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            PoiLedgerCacheError::ParseFailed { .. }
        ));
    }

    #[test]
    fn read_rejects_non_hex_receipt_id() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        let z = "0000000000000000000000000000000000000000000000000000000000000000";
        let body = format!(
            r#"{{"schema_version":"v1","chain_head":"{z}","entries":[
            {{"receipt_id":"XYZ","receipt_kind_byte":112,"quality_score":0.98,
              "gate_min_score":0.97,"entry_count":3,"impact_score":0.95,
              "timestamp_ns":0,"principal_id":null}}
            ]}}"#,
            z = z
        );
        fs::write(cache.ledger_path(), body.as_bytes()).unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            PoiLedgerCacheError::HexDecode { .. }
        ));
    }

    #[test]
    fn delete_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        cache.delete().unwrap();
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival() {
        let td = tempfile::TempDir::new().unwrap();
        let snap = sample_snapshot();
        {
            let cache = PoiLedgerCache::at_sovereign_root(td.path());
            cache.write(&snap).unwrap();
        }
        let cache2 = PoiLedgerCache::at_sovereign_root(td.path());
        assert_eq!(cache2.read().unwrap().unwrap(), snap);
    }

    #[test]
    fn impact_scores_f64_precision_preserved() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PoiLedgerCache::at_sovereign_root(td.path());
        let mut e = sample_entry(7, 0x70);
        e.impact_score = 0.987_654_321_098_7;
        let snap = PoiLedgerSnapshot {
            chain_head: [0xDD; 32],
            entries: vec![e.clone()],
        };
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.entries[0].impact_score, e.impact_score);
    }
}
