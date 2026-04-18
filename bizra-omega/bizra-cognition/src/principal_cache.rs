//! BIZRA Principal Profile Cache — §Cycle-7 G2 Commit-3
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/principal_cache.rs
//! Authority: cycle-7/niyyah.md §"Writer authority decision (HYBRID)" +
//!            §"Storage location"
//! Cycle position: 7, Phase 2
//!
//! Disk persistence for the local PrincipalProfile under
//! sovereign_state/dema_cache/principal.json.
//!
//! Niyyah § on writer authority:
//!   "Rust MAY write new local-only, non-chain surfaces … principal
//!    profile … These Rust-written surfaces are derived and rebuildable,
//!    never authoritative. If any Rust-written cache diverges from chain
//!    truth, rebuild from chain and mark the cache stale — never outrank
//!    chain."
//!
//! Cache semantics:
//!   - Atomic write: temp-then-rename on the same filesystem, so a partial
//!     write cannot leave principal.json observable in a half-state.
//!   - Read fails closed on malformed content or hash mismatch vs. the
//!     profile_hash recorded in the PrincipalActivationReceipt (checked at
//!     the runtime-rebuild layer, not here — this module only does the
//!     serialization round-trip).
//!   - Schema-versioned JSON so future evolution can detect + reject
//!     incompatible on-disk state.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::canonical_hasher::Blake3Hash;
use crate::principal_activation::PrincipalProfile;

/// Schema version for the on-disk JSON payload. Bumped if field set changes.
pub const CACHE_SCHEMA_VERSION: &str = "v1";

/// Standard relative path under sovereign_state/ where the cache lives.
///
/// Niyyah: "sovereign_state/dema_cache/ — explicitly named so it is clear
/// these are Rust-authored, non-chain, derived surfaces."
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const PRINCIPAL_CACHE_FILENAME: &str = "principal.json";

#[derive(Debug)]
pub enum PrincipalCacheError {
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

impl std::fmt::Display for PrincipalCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirCreate { path, msg } => {
                write!(f, "create dema_cache dir {}: {}", path.display(), msg)
            }
            Self::TempWrite { path, msg } => {
                write!(f, "write cache temp {}: {}", path.display(), msg)
            }
            Self::Rename { from, to, msg } => write!(
                f,
                "rename {} -> {}: {}",
                from.display(),
                to.display(),
                msg
            ),
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
            Self::HexDecode { field, reason } => {
                write!(f, "hex decode {}: {}", field, reason)
            }
            Self::Serialize(s) => write!(f, "serialize principal profile: {}", s),
        }
    }
}

impl std::error::Error for PrincipalCacheError {}

/// Principal profile cache rooted at a sovereign_state/ directory.
///
/// Does not load or verify the chain — that discipline lives at the
/// runtime layer where PrincipalActivationReceipt.profile_hash can be
/// compared against the loaded profile's profile_hash().
#[derive(Debug, Clone)]
pub struct PrincipalProfileCache {
    cache_dir: PathBuf,
}

impl PrincipalProfileCache {
    /// Create a cache rooted at `<sovereign_state_root>/dema_cache/`.
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        PrincipalProfileCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    /// Direct constructor when the caller already has the dema_cache dir.
    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        PrincipalProfileCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn principal_path(&self) -> PathBuf {
        self.cache_dir.join(PRINCIPAL_CACHE_FILENAME)
    }

    /// Atomically write a PrincipalProfile to principal.json.
    ///
    /// Temp-then-rename on the same filesystem. A partial write cannot
    /// leave principal.json observable in a half-state.
    pub fn write(&self, profile: &PrincipalProfile) -> Result<(), PrincipalCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| PrincipalCacheError::DirCreate {
            path: self.cache_dir.clone(),
            msg: e.to_string(),
        })?;

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "principal_id": hex_encode(&profile.principal_id),
            "name": profile.name,
            "node_id": profile.node_id,
            "declared_role": profile.declared_role,
            "activation_receipt_id": hex_encode(&profile.activation_receipt_id),
            "activation_ns": profile.activation_ns,
            "profile_hash": hex_encode(&profile.profile_hash()),
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| PrincipalCacheError::Serialize(e.to_string()))?;

        let final_path = self.principal_path();
        let tmp_path = self.cache_dir.join(format!(
            "{}.tmp.{}",
            PRINCIPAL_CACHE_FILENAME,
            std::process::id()
        ));

        {
            let mut f = fs::File::create(&tmp_path).map_err(|e| {
                PrincipalCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                }
            })?;
            f.write_all(&bytes)
                .map_err(|e| PrincipalCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.sync_all()
                .map_err(|e| PrincipalCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| PrincipalCacheError::Rename {
            from: tmp_path,
            to: final_path,
            msg: e.to_string(),
        })?;

        Ok(())
    }

    /// Read the PrincipalProfile from disk if present. Returns Ok(None)
    /// when the cache file is absent. Fails closed on schema, malformed
    /// content, or hex-decode errors.
    pub fn read(&self) -> Result<Option<PrincipalProfile>, PrincipalCacheError> {
        let path = self.principal_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| PrincipalCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|e| PrincipalCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            })?;
        let obj = v.as_object().ok_or(PrincipalCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj
            .get("schema_version")
            .and_then(|x| x.as_str())
            .ok_or(PrincipalCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            })?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(PrincipalCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let principal_id = field_hex(obj, &path, "principal_id")?;
        let name = field_str(obj, &path, "name")?;
        let node_id = field_str(obj, &path, "node_id")?;
        let declared_role = field_str(obj, &path, "declared_role")?;
        let activation_receipt_id = field_hex(obj, &path, "activation_receipt_id")?;
        let activation_ns =
            obj.get("activation_ns")
                .and_then(|x| x.as_u64())
                .ok_or(PrincipalCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing activation_ns",
                })?;

        let profile = PrincipalProfile {
            principal_id,
            name,
            node_id,
            declared_role,
            activation_receipt_id,
            activation_ns,
        };
        Ok(Some(profile))
    }

    /// Delete the cache file if it exists. Idempotent.
    pub fn delete(&self) -> Result<(), PrincipalCacheError> {
        let path = self.principal_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| PrincipalCacheError::ReadFailed {
                path,
                msg: e.to_string(),
            })?;
        }
        Ok(())
    }
}

fn field_str(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
    name: &'static str,
) -> Result<String, PrincipalCacheError> {
    obj.get(name)
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .ok_or(PrincipalCacheError::Malformed {
            path: path.to_path_buf(),
            reason: leak_static("missing field", name),
        })
}

fn field_hex(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
    name: &'static str,
) -> Result<Blake3Hash, PrincipalCacheError> {
    let s = obj
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or(PrincipalCacheError::Malformed {
            path: path.to_path_buf(),
            reason: leak_static("missing hex field", name),
        })?;
    hex_decode(s).map_err(|reason| PrincipalCacheError::HexDecode { field: name, reason })
}

// `&'static str` reason slot is enumerated; we can't synthesize a
// per-field string for missing fields without leaking. The two callers
// above use a fixed reason pair. Keep them static.
fn leak_static(prefix: &'static str, _name: &'static str) -> &'static str {
    // Accept a small loss of field-name specificity to keep the error
    // type Copy-friendly. Callers reading the Debug repr can recover
    // the context from the surrounding path + reason prefix.
    prefix
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
    use crate::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};

    const TEST_PUBKEY_HEX: &str =
        "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

    fn test_profile() -> PrincipalProfile {
        let a = NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-13T23:54:59Z");
        let e = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            1_000,
        )
        .unwrap();
        PrincipalProfile::new(&e, [0xAB; 32], 2_000)
    }

    #[test]
    fn write_then_read_round_trips_profile_fields() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        let profile = test_profile();
        cache.write(&profile).unwrap();
        let loaded = cache.read().unwrap().expect("profile present on disk");
        assert_eq!(loaded.principal_id, profile.principal_id);
        assert_eq!(loaded.name, profile.name);
        assert_eq!(loaded.node_id, profile.node_id);
        assert_eq!(loaded.declared_role, profile.declared_role);
        assert_eq!(loaded.activation_receipt_id, profile.activation_receipt_id);
        assert_eq!(loaded.activation_ns, profile.activation_ns);
        assert_eq!(
            loaded.profile_hash(),
            profile.profile_hash(),
            "profile_hash must round-trip exactly"
        );
    }

    #[test]
    fn read_absent_file_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        let result = cache.read().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn write_creates_dema_cache_dir_if_missing() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        assert!(!cache.cache_dir().exists());
        cache.write(&test_profile()).unwrap();
        assert!(cache.cache_dir().exists());
        assert!(cache.principal_path().exists());
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file_on_success() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        cache.write(&test_profile()).unwrap();
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
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        cache.write(&test_profile()).unwrap();

        // Write a different profile (different activation receipt id) and
        // confirm the on-disk content changes.
        let a = NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "now");
        let e = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        let p2 = PrincipalProfile::new(&e, [0xCD; 32], 9_999);
        cache.write(&p2).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded.activation_receipt_id, [0xCD; 32]);
        assert_eq!(loaded.activation_ns, 9_999);
    }

    #[test]
    fn read_rejects_wrong_schema_version() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.principal_path(),
            br#"{"schema_version":"v9000","principal_id":"aa","name":"x","node_id":"y","declared_role":"z","activation_receipt_id":"bb","activation_ns":0}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, PrincipalCacheError::SchemaMismatch { .. }));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.principal_path(), b"not json").unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, PrincipalCacheError::ParseFailed { .. }));
    }

    #[test]
    fn read_rejects_missing_field() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.principal_path(),
            br#"{"schema_version":"v1","principal_id":"0000000000000000000000000000000000000000000000000000000000000000"}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, PrincipalCacheError::Malformed { .. }));
    }

    #[test]
    fn read_rejects_non_hex_principal_id() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.principal_path(),
            br#"{"schema_version":"v1","principal_id":"XXXX","name":"x","node_id":"y","declared_role":"z","activation_receipt_id":"0000000000000000000000000000000000000000000000000000000000000000","activation_ns":0}"#,
        )
        .unwrap();
        let err = cache.read().unwrap_err();
        assert!(matches!(err, PrincipalCacheError::HexDecode { .. }));
    }

    #[test]
    fn delete_removes_file_and_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = PrincipalProfileCache::at_sovereign_root(td.path());
        cache.write(&test_profile()).unwrap();
        assert!(cache.principal_path().exists());
        cache.delete().unwrap();
        assert!(!cache.principal_path().exists());
        // Second delete must succeed silently.
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival_simulation_reloads_identical_profile() {
        // 1. write profile, 2. drop cache handle, 3. new cache at same
        // root, 4. read → must match original byte-for-byte.
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path().to_path_buf();

        {
            let cache = PrincipalProfileCache::at_sovereign_root(&root);
            cache.write(&test_profile()).unwrap();
        } // drop — simulate process exit

        {
            let cache2 = PrincipalProfileCache::at_sovereign_root(&root);
            let loaded = cache2.read().unwrap().expect("profile survives restart");
            assert_eq!(loaded.profile_hash(), test_profile().profile_hash());
        }
    }
}
