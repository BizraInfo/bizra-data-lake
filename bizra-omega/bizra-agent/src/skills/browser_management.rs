// bizra-agent/src/skills/browser_management.rs
// ============================================================
// Browser Management via MCP — PAT executes, SAT validates
// ============================================================
//
// Capabilities:
//   1. Page Navigation — URL navigation, history traversal
//   2. Page Reading — extract text, accessibility tree, forms
//   3. Form Interaction — fill forms, click buttons, inputs
//   4. Tab Management — open, close, switch tabs
//   5. Screenshot Capture — visual state for verification
//
// Constitutional requirements:
//   - Every navigation produces a receipt (URL + timestamp)
//   - SAT validates URLs against blocklist before navigation
//   - No credential entry without HITL approval
//   - No downloads from untrusted sources
//   - Cookie consent auto-declined (privacy-first)
//   - Receipt emitted for every page interaction
//
// Architecture:
//   PAT (Navigator/Scholar) → plans browser actions, reads pages
//   SAT (Guardian/Sentinel) → validates URLs, blocks harmful sites
// ============================================================

use std::collections::HashSet;

// ── Browser actions ────────────────────────────────────────

/// A browser action that PAT can request.
#[derive(Debug, Clone)]
pub enum BrowserAction {
    /// Navigate to a URL.
    Navigate { url: String },
    /// Go back in history.
    GoBack,
    /// Go forward in history.
    GoForward,
    /// Read the current page content (accessibility tree).
    ReadPage,
    /// Extract raw text from the current page.
    ExtractText,
    /// Fill a form field by element reference.
    FillField { element_ref: String, value: String },
    /// Click an element by reference.
    Click { element_ref: String },
    /// Open a new tab.
    NewTab,
    /// Close the current tab.
    CloseTab { tab_id: String },
    /// Switch to a tab.
    SwitchTab { tab_id: String },
    /// Take a screenshot for visual verification.
    Screenshot,
    /// Find elements matching a description.
    FindElements { description: String },
    /// Execute JavaScript (requires SAT approval).
    ExecuteJs { code: String },
}

impl BrowserAction {
    /// Does this action require SAT approval?
    pub fn requires_sat_approval(&self) -> bool {
        matches!(self, 
            Self::ExecuteJs { .. } |
            Self::FillField { .. } |
            Self::Click { .. }
        )
    }

    /// Does this action involve sensitive data entry?
    pub fn is_sensitive(&self) -> bool {
        if let Self::FillField { element_ref, .. } = self {
            let lower = element_ref.to_lowercase();
            lower.contains("password") || lower.contains("credit")
                || lower.contains("ssn") || lower.contains("secret")
                || lower.contains("token") || lower.contains("key")
        } else {
            false
        }
    }

    /// Human-readable description for HITL review.
    pub fn describe(&self) -> String {
        match self {
            Self::Navigate { url } => format!("Navigate to {}", url),
            Self::GoBack => "Go back in history".into(),
            Self::GoForward => "Go forward in history".into(),
            Self::ReadPage => "Read current page content".into(),
            Self::ExtractText => "Extract page text".into(),
            Self::FillField { element_ref, .. } => format!("Fill field: {}", element_ref),
            Self::Click { element_ref } => format!("Click: {}", element_ref),
            Self::NewTab => "Open new tab".into(),
            Self::CloseTab { tab_id } => format!("Close tab: {}", tab_id),
            Self::SwitchTab { tab_id } => format!("Switch to tab: {}", tab_id),
            Self::Screenshot => "Take screenshot".into(),
            Self::FindElements { description } => format!("Find: {}", description),
            Self::ExecuteJs { .. } => "Execute JavaScript (requires approval)".into(),
        }
    }
}

// ── Browser action plan ───────────────────────────────────

/// A planned sequence of browser actions with SAT validation.
#[derive(Debug, Clone)]
pub struct BrowserPlan {
    pub actions: Vec<BrowserActionEntry>,
    pub created_at: u64,
    pub sat_approved: bool,
}

#[derive(Debug, Clone)]
pub struct BrowserActionEntry {
    pub action: BrowserAction,
    pub status: BrowserOpStatus,
    pub result: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowserOpStatus {
    Planned,
    SatApproved,
    Executing,
    Succeeded,
    Failed,
    Blocked,  // SAT rejected
}

impl BrowserPlan {
    pub fn new(timestamp: u64) -> Self {
        Self { actions: Vec::new(), created_at: timestamp, sat_approved: false }
    }

    pub fn add(&mut self, action: BrowserAction) {
        self.actions.push(BrowserActionEntry {
            action,
            status: BrowserOpStatus::Planned,
            result: None,
            error: None,
        });
    }

    pub fn total_actions(&self) -> usize { self.actions.len() }

    pub fn needs_hitl(&self) -> bool {
        self.actions.iter().any(|a| a.action.is_sensitive())
    }

    pub fn needs_sat(&self) -> bool {
        self.actions.iter().any(|a| a.action.requires_sat_approval())
    }
}

// ── SAT: URL validation (Guardian + Sentinel) ─────────────

/// SAT Guardian validates URLs before navigation.
pub struct UrlValidator {
    /// Blocked domains (malware, phishing, adult content).
    blocked_domains: HashSet<String>,
    /// Allowed domains (user-configured trusted sites).
    allowed_domains: HashSet<String>,
    /// Whether to enforce allowlist-only mode.
    allowlist_only: bool,
}

impl UrlValidator {
    /// Create with sensible defaults — blocks known harmful patterns.
    pub fn with_defaults() -> Self {
        let mut blocked = HashSet::new();
        // Constitutional blocklist — sites that violate Ihsan principles
        for domain in &[
            "malware.com", "phishing.example",
            // Placeholder — real deployment loads from constitutional config
        ] {
            blocked.insert(domain.to_string());
        }
        Self {
            blocked_domains: blocked,
            allowed_domains: HashSet::new(),
            allowlist_only: false,
        }
    }

    /// Add a domain to the blocklist.
    pub fn block_domain(&mut self, domain: &str) {
        self.blocked_domains.insert(domain.to_lowercase());
    }

    /// Add a domain to the allowlist.
    pub fn allow_domain(&mut self, domain: &str) {
        self.allowed_domains.insert(domain.to_lowercase());
    }

    /// Validate a URL. Returns (allowed, reason).
    pub fn validate_url(&self, url: &str) -> (bool, String) {
        // Extract domain from URL
        let domain = extract_domain(url);

        // Check blocklist first
        if self.blocked_domains.contains(&domain) {
            return (false, format!("Domain {} is blocked by constitutional policy", domain));
        }

        // Check allowlist if in strict mode
        if self.allowlist_only && !self.allowed_domains.contains(&domain) {
            return (false, format!("Domain {} not in allowlist (strict mode)", domain));
        }

        // Block suspicious URL patterns
        if url.contains("javascript:") || url.contains("data:text/html") {
            return (false, "Suspicious URL scheme blocked".into());
        }

        // Block credential harvesting patterns
        if url.contains("login") && url.contains("redirect") {
            return (false, "Potential credential redirect detected".into());
        }

        (true, "URL approved".into())
    }

    /// Validate an entire browser plan.
    pub fn validate_plan(&self, plan: &BrowserPlan) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        for entry in &plan.actions {
            if let BrowserAction::Navigate { url } = &entry.action {
                let (ok, reason) = self.validate_url(url);
                if !ok {
                    reasons.push(reason);
                }
            }
            // Block JS execution without SAT approval
            if let BrowserAction::ExecuteJs { .. } = &entry.action {
                reasons.push("JavaScript execution requires explicit SAT approval".into());
            }
        }

        (reasons.is_empty(), reasons)
    }
}

/// Extract domain from a URL string (simple parser).
fn extract_domain(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://").or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    without_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_lowercase()
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigate_does_not_need_sat() {
        let action = BrowserAction::Navigate { url: "https://bizra.ai".into() };
        assert!(!action.requires_sat_approval());
    }

    #[test]
    fn execute_js_requires_sat() {
        let action = BrowserAction::ExecuteJs { code: "alert(1)".into() };
        assert!(action.requires_sat_approval());
    }

    #[test]
    fn fill_field_requires_sat() {
        let action = BrowserAction::FillField {
            element_ref: "search_input".into(),
            value: "BIZRA".into(),
        };
        assert!(action.requires_sat_approval());
    }

    #[test]
    fn password_field_is_sensitive() {
        let action = BrowserAction::FillField {
            element_ref: "password_input".into(),
            value: "secret123".into(),
        };
        assert!(action.is_sensitive());
    }

    #[test]
    fn search_field_not_sensitive() {
        let action = BrowserAction::FillField {
            element_ref: "search_box".into(),
            value: "hello".into(),
        };
        assert!(!action.is_sensitive());
    }

    #[test]
    fn url_validator_blocks_malware() {
        let mut v = UrlValidator::with_defaults();
        v.block_domain("evil.com");
        let (ok, _) = v.validate_url("https://evil.com/steal");
        assert!(!ok);
    }

    #[test]
    fn url_validator_allows_clean_url() {
        let v = UrlValidator::with_defaults();
        let (ok, _) = v.validate_url("https://docs.rust-lang.org/book/");
        assert!(ok);
    }

    #[test]
    fn url_validator_blocks_javascript_scheme() {
        let v = UrlValidator::with_defaults();
        let (ok, _) = v.validate_url("javascript:alert(1)");
        assert!(!ok);
    }

    #[test]
    fn extract_domain_works() {
        assert_eq!(extract_domain("https://bizra.ai/docs"), "bizra.ai");
        assert_eq!(extract_domain("http://localhost:3000/api"), "localhost");
        assert_eq!(extract_domain("https://sub.domain.com:443/path"), "sub.domain.com");
    }

    #[test]
    fn plan_tracks_hitl_need() {
        let mut plan = BrowserPlan::new(1000);
        plan.add(BrowserAction::Navigate { url: "https://bizra.ai".into() });
        assert!(!plan.needs_hitl());

        plan.add(BrowserAction::FillField {
            element_ref: "password_field".into(),
            value: "secret".into(),
        });
        assert!(plan.needs_hitl());
    }

    #[test]
    fn plan_tracks_sat_need() {
        let mut plan = BrowserPlan::new(1000);
        plan.add(BrowserAction::ReadPage);
        assert!(!plan.needs_sat());

        plan.add(BrowserAction::Click { element_ref: "submit_btn".into() });
        assert!(plan.needs_sat());
    }

    #[test]
    fn validate_plan_catches_js_and_bad_urls() {
        let mut v = UrlValidator::with_defaults();
        v.block_domain("phishing.example");

        let mut plan = BrowserPlan::new(1000);
        plan.add(BrowserAction::Navigate { url: "https://phishing.example/login".into() });
        plan.add(BrowserAction::ExecuteJs { code: "document.cookie".into() });

        let (ok, reasons) = v.validate_plan(&plan);
        assert!(!ok);
        assert_eq!(reasons.len(), 2); // blocked domain + JS execution
    }

    #[test]
    fn allowlist_strict_mode() {
        let mut v = UrlValidator::with_defaults();
        v.allowlist_only = true;
        v.allow_domain("bizra.ai");

        let (ok, _) = v.validate_url("https://bizra.ai/docs");
        assert!(ok);

        let (ok, _) = v.validate_url("https://random-site.com");
        assert!(!ok);
    }
}
