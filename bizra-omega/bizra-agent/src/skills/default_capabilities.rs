// bizra-agent/src/skills/default_capabilities.rs
// ============================================================
// Default Node Capabilities — constitutional defaults
// ============================================================
//
// Every BIZRA node ships with two pre-installed capabilities:
//   1. Local Filesystem Management (sovereign file operations)
//   2. Browser Management via MCP (sovereign web interaction)
//
// These are NOT optional plugins. They are constitutional
// defaults — the minimum capability set that makes a node
// useful from minute one. A node without file management
// cannot organize its own data. A node without browser
// management cannot interact with the web on behalf of its
// human. Both are required for sovereignty.
//
// PAT agents trained on both:
//   Navigator  — routes file/browser intents
//   Scholar    — deep search, content extraction
//   Artisan    — file organization, form filling
//   Sentinel   — security scanning, URL validation
//
// SAT agents validate both:
//   Guardian   — blocks system paths, dangerous URLs, sensitive data
//   Auditor    — verifies manifest integrity, receipt chains
// ============================================================

use super::{browser_management::UrlValidator, file_management::SmartFileManager};

/// The two constitutional default capabilities every node ships with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultCapability {
    /// Local filesystem: classify, organize, rename, merge, dedup.
    FileManagement,
    /// Browser via MCP: navigate, read, fill, click, screenshot.
    BrowserManagement,
}

impl DefaultCapability {
    /// All default capabilities (constitutional minimum).
    pub fn all() -> &'static [Self] {
        &[Self::FileManagement, Self::BrowserManagement]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FileManagement => "file_management",
            Self::BrowserManagement => "browser_management",
        }
    }

    /// MCP server identifier for this capability.
    pub fn mcp_server_id(&self) -> &'static str {
        match self {
            Self::FileManagement => "bizra-fs",
            Self::BrowserManagement => "bizra-browser",
        }
    }

    /// Which PAT agents are primary for this capability?
    pub fn pat_agents(&self) -> &'static [&'static str] {
        match self {
            Self::FileManagement => &["Navigator", "Artisan", "Scholar", "Sentinel"],
            Self::BrowserManagement => &["Navigator", "Scholar", "Artisan", "Sentinel"],
        }
    }

    /// Which SAT agents validate this capability?
    pub fn sat_agents(&self) -> &'static [&'static str] {
        match self {
            Self::FileManagement => &["Guardian", "Auditor"],
            Self::BrowserManagement => &["Guardian", "Sentinel"],
        }
    }
}

/// Default capability configuration for a new node.
/// This is what ships in the installer — no user config needed.
pub struct NodeCapabilityConfig {
    pub file_manager: SmartFileManager,
    pub url_validator: UrlValidator,
    pub capabilities: Vec<DefaultCapability>,
}

impl NodeCapabilityConfig {
    /// Create the constitutional default configuration.
    /// Every new node starts with this — no setup required.
    pub fn genesis() -> Self {
        Self {
            file_manager: SmartFileManager::new(),
            url_validator: UrlValidator::with_defaults(),
            capabilities: DefaultCapability::all().to_vec(),
        }
    }

    /// Check if a capability is installed.
    pub fn has_capability(&self, cap: DefaultCapability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// List all installed capabilities.
    pub fn installed(&self) -> &[DefaultCapability] {
        &self.capabilities
    }

    /// Human-readable capability summary for node status.
    pub fn summary(&self) -> String {
        let caps: Vec<&str> = self.capabilities.iter().map(|c| c.as_str()).collect();
        format!(
            "Installed capabilities: {} [{}]",
            caps.len(),
            caps.join(", ")
        )
    }
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_has_both_capabilities() {
        let config = NodeCapabilityConfig::genesis();
        assert!(config.has_capability(DefaultCapability::FileManagement));
        assert!(config.has_capability(DefaultCapability::BrowserManagement));
        assert_eq!(config.installed().len(), 2);
    }

    #[test]
    fn all_capabilities_have_mcp_ids() {
        for cap in DefaultCapability::all() {
            assert!(!cap.mcp_server_id().is_empty());
            assert!(!cap.as_str().is_empty());
        }
    }

    #[test]
    fn all_capabilities_have_pat_agents() {
        for cap in DefaultCapability::all() {
            assert!(!cap.pat_agents().is_empty());
            assert!(cap.pat_agents().contains(&"Navigator")); // Navigator routes all skills
        }
    }

    #[test]
    fn all_capabilities_have_sat_validators() {
        for cap in DefaultCapability::all() {
            assert!(!cap.sat_agents().is_empty());
            assert!(cap.sat_agents().contains(&"Guardian")); // Guardian validates all skills
        }
    }

    #[test]
    fn genesis_summary_readable() {
        let config = NodeCapabilityConfig::genesis();
        let s = config.summary();
        assert!(s.contains("file_management"));
        assert!(s.contains("browser_management"));
        assert!(s.contains("2"));
    }
}
