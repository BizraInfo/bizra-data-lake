//! BIZRA Resource Registry — §Cycle-7 G4 Commit-1
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/resource_registry.rs
//! Authority: cycle-7/niyyah.md §G4 "Local resource registry + URP view"
//! Cycle position: 7, Phase 4
//!
//! Typed layer over the dema_cache resource_registry.json surface seeded
//! in G3. G4 owns: register / list / allowlist-query. G5 consumes the
//! allowlist from `dema organize <allowlisted>`.
//!
//! Niyyah §"Writer authority (HYBRID)": local-only, non-chain, derived
//! and rebuildable. Cache stays non-authoritative; no chain receipt is
//! emitted for a register call.
//!
//! Schema compatibility: the cache JSON format (schema v1) keeps `kind`
//! as a string. This module provides a typed enum over that string with
//! a Custom(String) escape hatch — so new variants can be added without
//! breaking cache files written by an older build.

use std::fmt;

use crate::resource_registry_cache::{ResourceEntry, ResourceRegistryCacheError};

// ════════════════════════════════════════════════════════════════════
// ResourceKind — typed enum with free-form escape hatch
// ════════════════════════════════════════════════════════════════════

/// Kinds of local resources Dema can register + allowlist.
///
/// Well-known variants have stable canonical string forms. Unknown
/// strings loaded from disk become `Custom(s)` so older/newer builds
/// do not reject each other's caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// A local filesystem path (absolute). Primary G5 use case.
    FilesystemPath,
    /// A network endpoint (host:port, URL). For future mission kinds.
    NetworkEndpoint,
    /// A process handle (pid or semantic name).
    ProcessHandle,
    /// A credential reference — NOT the credential itself. Only a
    /// handle that resolves through sovereign_state/identity/ later.
    Credential,
    /// Escape hatch for unknown strings read from disk and for
    /// operator-defined kinds G4-seed does not enumerate.
    Custom(String),
}

impl ResourceKind {
    /// Canonical lowercase string form used on disk and over the wire.
    pub fn as_str(&self) -> &str {
        match self {
            Self::FilesystemPath => "filesystem",
            Self::NetworkEndpoint => "network",
            Self::ProcessHandle => "process",
            Self::Credential => "credential",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parse from the canonical string. Unknown strings fall into
    /// `Custom` — never return an error. Niyyah §"derived + rebuildable":
    /// future builds should not reject older caches.
    pub fn from_str(s: &str) -> Self {
        match s {
            "filesystem" => Self::FilesystemPath,
            "network" => Self::NetworkEndpoint,
            "process" => Self::ProcessHandle,
            "credential" => Self::Credential,
            other => Self::Custom(other.to_string()),
        }
    }

    /// `true` when this kind is one of the G4-seeded well-known variants.
    /// Used by URP projection to decide whether to surface under its
    /// canonical heading or under a single "custom" bucket.
    pub fn is_well_known(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ════════════════════════════════════════════════════════════════════
// TypedResource — in-memory projection of a ResourceEntry
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedResource {
    pub kind: ResourceKind,
    pub id: String,
    pub summary: String,
    pub allowlisted: bool,
}

impl TypedResource {
    /// Validate + construct. Fails on empty id.
    pub fn new(
        kind: ResourceKind,
        id: String,
        summary: String,
        allowlisted: bool,
    ) -> Result<Self, ResourceRegistryError> {
        if id.trim().is_empty() {
            return Err(ResourceRegistryError::EmptyId);
        }
        Ok(TypedResource {
            kind,
            id,
            summary,
            allowlisted,
        })
    }

    pub fn to_cache_entry(&self) -> ResourceEntry {
        ResourceEntry {
            id: self.id.clone(),
            kind: self.kind.as_str().to_string(),
            summary: self.summary.clone(),
            allowlisted: self.allowlisted,
        }
    }
}

impl From<&ResourceEntry> for TypedResource {
    fn from(e: &ResourceEntry) -> Self {
        TypedResource {
            kind: ResourceKind::from_str(&e.kind),
            id: e.id.clone(),
            summary: e.summary.clone(),
            allowlisted: e.allowlisted,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum ResourceRegistryError {
    EmptyId,
    NoCacheAttached,
    Cache(ResourceRegistryCacheError),
}

impl fmt::Display for ResourceRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => write!(f, "resource id must be non-empty"),
            Self::NoCacheAttached => write!(
                f,
                "resource_registry cache is not attached; call attach_dema_cache first"
            ),
            Self::Cache(e) => write!(f, "resource_registry cache: {}", e),
        }
    }
}

impl std::error::Error for ResourceRegistryError {}

impl From<ResourceRegistryCacheError> for ResourceRegistryError {
    fn from(e: ResourceRegistryCacheError) -> Self {
        Self::Cache(e)
    }
}

// ════════════════════════════════════════════════════════════════════
// Registration outcome
// ════════════════════════════════════════════════════════════════════

/// Result of a `register_resource` call. Lets the operator know whether
/// their registration was novel, changed the allowlist flag on an
/// existing entry, or was a no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterOutcome {
    /// New (kind, id) pair — added.
    Created,
    /// (kind, id) already present; allowlist flag or summary changed.
    Updated,
    /// Exact match already present — write elided.
    Idempotent,
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_known_kinds_round_trip_through_string() {
        for k in [
            ResourceKind::FilesystemPath,
            ResourceKind::NetworkEndpoint,
            ResourceKind::ProcessHandle,
            ResourceKind::Credential,
        ] {
            let s = k.as_str().to_string();
            let parsed = ResourceKind::from_str(&s);
            assert_eq!(parsed, k);
            assert!(k.is_well_known());
        }
    }

    #[test]
    fn unknown_string_becomes_custom_without_error() {
        let k = ResourceKind::from_str("quantum-entanglement-channel");
        assert_eq!(
            k,
            ResourceKind::Custom("quantum-entanglement-channel".into())
        );
        assert!(!k.is_well_known());
        assert_eq!(k.as_str(), "quantum-entanglement-channel");
    }

    #[test]
    fn canonical_strings_are_stable_lowercase() {
        assert_eq!(ResourceKind::FilesystemPath.as_str(), "filesystem");
        assert_eq!(ResourceKind::NetworkEndpoint.as_str(), "network");
        assert_eq!(ResourceKind::ProcessHandle.as_str(), "process");
        assert_eq!(ResourceKind::Credential.as_str(), "credential");
    }

    #[test]
    fn typed_resource_rejects_empty_id() {
        let err = TypedResource::new(
            ResourceKind::FilesystemPath,
            "".into(),
            "empty".into(),
            true,
        )
        .unwrap_err();
        assert!(matches!(err, ResourceRegistryError::EmptyId));
    }

    #[test]
    fn typed_resource_rejects_whitespace_only_id() {
        let err = TypedResource::new(
            ResourceKind::FilesystemPath,
            "   \t  ".into(),
            "blanks".into(),
            true,
        )
        .unwrap_err();
        assert!(matches!(err, ResourceRegistryError::EmptyId));
    }

    #[test]
    fn typed_resource_round_trips_through_cache_entry() {
        let t = TypedResource::new(
            ResourceKind::FilesystemPath,
            "/home/mumo/docs".into(),
            "mumo's docs".into(),
            true,
        )
        .unwrap();
        let e = t.to_cache_entry();
        assert_eq!(e.kind, "filesystem");
        assert_eq!(e.id, "/home/mumo/docs");
        assert!(e.allowlisted);

        let back = TypedResource::from(&e);
        assert_eq!(back, t);
    }

    #[test]
    fn custom_kind_survives_cache_round_trip() {
        let t = TypedResource::new(
            ResourceKind::Custom("telescope-mount".into()),
            "scope-01".into(),
            "rooftop".into(),
            false,
        )
        .unwrap();
        let e = t.to_cache_entry();
        let back = TypedResource::from(&e);
        assert_eq!(back.kind, ResourceKind::Custom("telescope-mount".into()));
    }
}
