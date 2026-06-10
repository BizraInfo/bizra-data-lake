//! BIZRA Receipt Chain Store — Cycle-6 Arc 3 authoritative persistence
//!
//! When `BIZRA_RECEIPT_STORE_PATH` is set (explicit path or the operator token
//! `default`), the gateway bootstraps with:
//!   - sled-backed payload storage under `<root>/payloads/`
//!   - an authoritative chain snapshot at `<root>/chain_snapshot.json`
//!
//! When unset, the gateway keeps the in-memory payload store (ephemeral).
//! Persistence is opt-in: the env var must be present; `default` expands to a
//! canonical operator path without enabling persistence implicitly.
//!
//! This is distinct from `BIZRA_DEMA_CACHE_ROOT` / `receipt_history.json`,
//! which remains derived and does not rehydrate `ReceiptChain` on boot.

use std::path::{Path, PathBuf};

use crate::receipt_history_cache::{
    ReceiptHistoryCache, ReceiptHistoryCacheError, ReceiptHistorySnapshot,
};
use crate::receipts::{Blake3Hash, ChainError, PayloadStore, ReceiptChain, StoreError};

#[cfg(feature = "sled-store")]
use crate::receipt_freeze_v1::SledPayloadStore;

/// Environment variable selecting the authoritative receipt store root.
pub const ENV_RECEIPT_STORE_PATH: &str = "BIZRA_RECEIPT_STORE_PATH";
pub const ENV_SOVEREIGN_STATE_PATH: &str = "BIZRA_SOVEREIGN_STATE_PATH";
pub const ENV_DATA_LAKE_ROOT: &str = "BIZRA_DATA_LAKE_ROOT";
pub const ENV_DEMA_HOME: &str = "DEMA_HOME";
/// Explicit opt-in token: `BIZRA_RECEIPT_STORE_PATH=default`.
pub const OPERATOR_DEFAULT_STORE_TOKEN: &str = "default";
pub const OPERATOR_STORE_DIRNAME: &str = "authoritative_receipt_store";
pub const SOVEREIGN_STATE_DIRNAME: &str = "sovereign_state";

/// How [`ReceiptChainStore::resolve_root_from_env`] resolved the store root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptStorePathMode {
    Explicit,
    OperatorDefault,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedReceiptStorePath {
    pub root: PathBuf,
    pub mode: ReceiptStorePathMode,
}

/// Authoritative chain metadata filename under the store root.
pub const CHAIN_SNAPSHOT_FILENAME: &str = "chain_snapshot.json";

/// Schema marker written into `chain_snapshot.json`.
pub const CHAIN_STORE_SCHEMA_VERSION: &str = "bizra.receipt_chain_store.v1";

/// Sled database directory for hash-addressed payloads.
pub const PAYLOADS_SUBDIR: &str = "payloads";

#[derive(Debug)]
pub enum ReceiptChainStoreError {
    Bootstrap(ChainError),
    Snapshot(ReceiptHistoryCacheError),
    FeatureDisabled,
}

impl std::fmt::Display for ReceiptChainStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bootstrap(e) => write!(f, "receipt chain bootstrap: {:?}", e),
            Self::Snapshot(e) => write!(f, "receipt chain snapshot: {}", e),
            Self::FeatureDisabled => {
                write!(
                    f,
                    "sled-store feature not enabled for authoritative persistence"
                )
            }
        }
    }
}

impl std::error::Error for ReceiptChainStoreError {}

impl From<ChainError> for ReceiptChainStoreError {
    fn from(e: ChainError) -> Self {
        Self::Bootstrap(e)
    }
}

impl From<ReceiptHistoryCacheError> for ReceiptChainStoreError {
    fn from(e: ReceiptHistoryCacheError) -> Self {
        Self::Snapshot(e)
    }
}

/// Authoritative local receipt store rooted at `BIZRA_RECEIPT_STORE_PATH`.
#[derive(Debug, Clone)]
pub struct ReceiptChainStore {
    root: PathBuf,
}

impl ReceiptChainStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn payloads_dir(&self) -> PathBuf {
        self.root.join(PAYLOADS_SUBDIR)
    }

    pub fn chain_snapshot_path(&self) -> PathBuf {
        self.root.join(CHAIN_SNAPSHOT_FILENAME)
    }

    /// Canonical operator store path when persistence is explicitly enabled with
    /// `BIZRA_RECEIPT_STORE_PATH=default`. Resolution order:
    /// 1. `$BIZRA_SOVEREIGN_STATE_PATH/authoritative_receipt_store`
    /// 2. `$BIZRA_DATA_LAKE_ROOT/sovereign_state/authoritative_receipt_store`
    /// 3. `$DEMA_HOME/authoritative_receipt_store`
    /// 4. `$HOME/.dema/authoritative_receipt_store`
    /// 5. `./sovereign_state/authoritative_receipt_store` (process cwd)
    pub fn default_operator_store_path() -> PathBuf {
        if let Ok(sovereign) = std::env::var(ENV_SOVEREIGN_STATE_PATH) {
            let trimmed = sovereign.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join(OPERATOR_STORE_DIRNAME);
            }
        }
        if let Ok(root) = std::env::var(ENV_DATA_LAKE_ROOT) {
            let trimmed = root.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed)
                    .join(SOVEREIGN_STATE_DIRNAME)
                    .join(OPERATOR_STORE_DIRNAME);
            }
        }
        if let Ok(home) = std::env::var(ENV_DEMA_HOME) {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join(OPERATOR_STORE_DIRNAME);
            }
        }
        if let Some(home) = dirs_home_via_env() {
            return home.join(".dema").join(OPERATOR_STORE_DIRNAME);
        }
        PathBuf::from(SOVEREIGN_STATE_DIRNAME).join(OPERATOR_STORE_DIRNAME)
    }

    pub fn resolve_root_from_env() -> Option<ResolvedReceiptStorePath> {
        let raw = std::env::var(ENV_RECEIPT_STORE_PATH).ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        if trimmed.eq_ignore_ascii_case(OPERATOR_DEFAULT_STORE_TOKEN) {
            Some(ResolvedReceiptStorePath {
                root: Self::default_operator_store_path(),
                mode: ReceiptStorePathMode::OperatorDefault,
            })
        } else {
            Some(ResolvedReceiptStorePath {
                root: PathBuf::from(trimmed),
                mode: ReceiptStorePathMode::Explicit,
            })
        }
    }

    pub fn root_from_env() -> Option<PathBuf> {
        Self::resolve_root_from_env().map(|resolved| resolved.root)
    }

    pub fn open_payload_store(&self) -> Result<Box<dyn PayloadStore>, StoreError> {
        #[cfg(feature = "sled-store")]
        {
            std::fs::create_dir_all(self.payloads_dir()).map_err(|e| {
                StoreError::IoError(format!(
                    "create payloads dir {}: {}",
                    self.payloads_dir().display(),
                    e
                ))
            })?;
            let path = self.payloads_dir().to_string_lossy().into_owned();
            Ok(Box::new(SledPayloadStore::open(&path)?))
        }
        #[cfg(not(feature = "sled-store"))]
        {
            let _ = self;
            Err(StoreError::IoError("sled-store feature not enabled".into()))
        }
    }

    pub fn read_snapshot(
        &self,
    ) -> Result<Option<ReceiptHistorySnapshot>, ReceiptHistoryCacheError> {
        ReceiptHistoryCache::read_snapshot_file(
            &self.chain_snapshot_path(),
            CHAIN_STORE_SCHEMA_VERSION,
        )
    }

    pub fn write_snapshot(
        &self,
        snapshot: &ReceiptHistorySnapshot,
    ) -> Result<(), ReceiptHistoryCacheError> {
        std::fs::create_dir_all(&self.root).map_err(|e| ReceiptHistoryCacheError::DirCreate {
            path: self.root.clone(),
            msg: e.to_string(),
        })?;
        ReceiptHistoryCache::write_snapshot_file(
            &self.chain_snapshot_path(),
            snapshot,
            CHAIN_STORE_SCHEMA_VERSION,
        )
    }

    pub fn bootstrap_chain(
        &self,
        genesis: Blake3Hash,
    ) -> Result<ReceiptChain, ReceiptChainStoreError> {
        let store = self.open_payload_store().map_err(ChainError::from)?;
        if let Some(snapshot) = self.read_snapshot()? {
            Ok(ReceiptChain::restore_from_snapshot(
                genesis, snapshot, store,
            )?)
        } else {
            Ok(ReceiptChain::new(genesis, store))
        }
    }
}

fn dirs_home_via_env() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod env_resolution_tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Process-global env is shared across parallel tests; serialize mutations.
    fn env_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn resolve_root_from_env_honors_explicit_path() {
        let _guard = env_test_lock();
        let td = tempfile::TempDir::new().unwrap();
        let explicit = td.path().join("custom_store");
        std::env::set_var(ENV_RECEIPT_STORE_PATH, explicit.to_string_lossy().as_ref());
        let resolved = ReceiptChainStore::resolve_root_from_env().unwrap();
        assert_eq!(resolved.mode, ReceiptStorePathMode::Explicit);
        assert_eq!(resolved.root, explicit);
        std::env::remove_var(ENV_RECEIPT_STORE_PATH);
    }

    #[test]
    fn resolve_root_from_env_default_token_uses_data_lake_root() {
        let _guard = env_test_lock();
        let td = tempfile::TempDir::new().unwrap();
        std::env::remove_var(ENV_SOVEREIGN_STATE_PATH);
        std::env::remove_var(ENV_DEMA_HOME);
        std::env::set_var(ENV_DATA_LAKE_ROOT, td.path().to_string_lossy().as_ref());
        std::env::set_var(ENV_RECEIPT_STORE_PATH, OPERATOR_DEFAULT_STORE_TOKEN);
        let resolved = ReceiptChainStore::resolve_root_from_env().unwrap();
        assert_eq!(resolved.mode, ReceiptStorePathMode::OperatorDefault);
        assert_eq!(
            resolved.root,
            td.path()
                .join(SOVEREIGN_STATE_DIRNAME)
                .join(OPERATOR_STORE_DIRNAME)
        );
        std::env::remove_var(ENV_RECEIPT_STORE_PATH);
        std::env::remove_var(ENV_DATA_LAKE_ROOT);
    }

    #[test]
    fn resolve_root_from_env_unset_is_none() {
        let _guard = env_test_lock();
        std::env::remove_var(ENV_RECEIPT_STORE_PATH);
        assert!(ReceiptChainStore::resolve_root_from_env().is_none());
    }
}

#[cfg(all(test, feature = "sled-store"))]
mod tests {
    use super::*;
    use crate::receipts::{InMemoryPayloadStore, ReceiptKind, ReceiptPayload};

    struct DummyPayload {
        kind: ReceiptKind,
        data: Vec<u8>,
    }

    impl ReceiptPayload for DummyPayload {
        fn kind(&self) -> ReceiptKind {
            self.kind
        }
        fn canonical_bytes(&self) -> Vec<u8> {
            self.data.clone()
        }
        fn hash(&self) -> Blake3Hash {
            let mut h = [0u8; 32];
            for (i, b) in self.data.iter().take(32).enumerate() {
                h[i] = *b;
            }
            h
        }
    }

    #[test]
    fn authoritative_store_survives_reload() {
        let td = tempfile::TempDir::new().unwrap();
        let store = ReceiptChainStore::new(td.path().to_path_buf());
        let genesis = [0u8; 32];

        let mut chain = store.bootstrap_chain(genesis).unwrap();
        chain
            .append_with_payload(DummyPayload {
                kind: ReceiptKind::CognitionBoot,
                data: vec![9, 8, 7, 6, 5],
            })
            .unwrap();
        store
            .write_snapshot(&ReceiptHistorySnapshot {
                head: chain.head(),
                last_timestamp_ns: chain.latest_timestamp(),
                records: chain.records().copied().collect(),
            })
            .unwrap();

        let expected_len = chain.len();
        let expected_head = chain.head();
        drop(chain);

        let reloaded = store.bootstrap_chain(genesis).unwrap();
        assert_eq!(reloaded.len(), expected_len);
        assert_eq!(reloaded.head(), expected_head);
        assert!(reloaded
            .fetch_payload_bytes(&reloaded.head())
            .unwrap()
            .is_some());
    }

    #[test]
    fn in_memory_store_does_not_use_authoritative_path() {
        let genesis = [0u8; 32];
        let mut chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        chain
            .append_with_payload(DummyPayload {
                kind: ReceiptKind::ReasoningSession,
                data: vec![1, 2, 3],
            })
            .unwrap();
        assert_eq!(chain.len(), 1);
    }
}
