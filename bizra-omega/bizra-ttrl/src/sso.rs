//! # SSO — Spectral Sphere Optimizer Constraint (Paper 3)
//!
//! SSO guarantees that TTRL weight updates keep the model **on the spectral
//! manifold**.  Without this constraint, on-device GRPO updates could drift
//! the model's spectral norm, causing:
//! - Activation magnitude explosion (NaN/Inf outputs)
//! - Iḥsān gate regression (compiled reflexes become stale)
//! - Latency spikes
//!
//! SSO enforces:  `|spectral_norm_post - spectral_norm_pre| < EPSILON`
//!
//! ## What this module provides
//! - `SpectralSphereConstraint` — configuration + violation check
//! - `SpectralNorm` — newtype wrapper for a layer's operator norm
//! - `SsoCheckResult` — pass/fail with diagnostic detail
//!
//! ## Note on GPU integration
//! This module contains the **constraint logic** only.  Actual spectral-norm
//! computation is performed by the model runtime (Python torch or Rust
//! inference crate).  The Omni-Kernel calls `sso.check(pre, post)` after
//! each TTRL update.
//!
//! Standing on Giants:
//! - Yoshida & Miyato (2018): Spectral Normalisation for GANs
//! - SSO paper (2025): Sphere-constrained on-device RL updates

use serde::{Deserialize, Serialize};

/// Maximum allowed drift in spectral norm across a single TTRL update.
/// Config key: `sso_epsilon` in `config/proactive_config.yaml`.
/// Default matches the paper's stability bound.
pub const SSO_DEFAULT_EPSILON: f64 = 1e-3;

/// Newtype for spectral norm (operator 2-norm of a weight matrix).
/// Always non-negative.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SpectralNorm(pub f64);

impl SpectralNorm {
    pub fn new(v: f64) -> Self {
        assert!(v >= 0.0, "Spectral norm must be non-negative");
        Self(v)
    }

    pub fn drift_from(self, other: SpectralNorm) -> f64 {
        (self.0 - other.0).abs()
    }
}

/// Result of an SSO constraint check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsoCheckResult {
    pub passed: bool,
    pub pre_norm: SpectralNorm,
    pub post_norm: SpectralNorm,
    pub drift: f64,
    pub epsilon: f64,
}

impl SsoCheckResult {
    pub fn passed(&self) -> bool {
        self.passed
    }
}

/// The SSO constraint configuration.  Pass to `OmniKernel` at construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralSphereConstraint {
    /// Maximum allowed spectral-norm drift per update.
    pub epsilon: f64,
}

impl Default for SpectralSphereConstraint {
    fn default() -> Self {
        Self {
            epsilon: SSO_DEFAULT_EPSILON,
        }
    }
}

impl SpectralSphereConstraint {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Check whether the spectral norm stayed within the sphere after a
    /// TTRL update.  `pre` is measured before the update; `post` after.
    pub fn check(&self, pre: SpectralNorm, post: SpectralNorm) -> SsoCheckResult {
        let drift = post.drift_from(pre);
        SsoCheckResult {
            passed: drift < self.epsilon,
            pre_norm: pre,
            post_norm: post,
            drift,
            epsilon: self.epsilon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sso_passes_within_epsilon() {
        let sso = SpectralSphereConstraint::default();
        let pre = SpectralNorm::new(1.000_00);
        let post = SpectralNorm::new(1.000_50); // drift = 5e-4 < 1e-3
        assert!(sso.check(pre, post).passed());
    }

    #[test]
    fn test_sso_fails_outside_epsilon() {
        let sso = SpectralSphereConstraint::default();
        let pre = SpectralNorm::new(1.0);
        let post = SpectralNorm::new(1.002); // drift = 2e-3 > 1e-3
        assert!(!sso.check(pre, post).passed());
    }

    #[test]
    fn test_sso_drift_exact() {
        let pre = SpectralNorm::new(2.0);
        let post = SpectralNorm::new(1.5);
        assert!((post.drift_from(pre) - 0.5).abs() < 1e-12);
    }
}
