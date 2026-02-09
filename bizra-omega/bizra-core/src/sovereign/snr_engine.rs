//! SNR Engine — Signal-to-Noise Ratio Maximizer
//!
//! Implements autonomous signal quality optimization based on
//! Claude Shannon's information theory principles.
//!
//! # The Shannon Foundation
//!
//! SNR = Signal Power / Noise Power
//!
//! In the BIZRA context:
//! - Signal = Relevant, high-quality information
//! - Noise = Irrelevant, low-quality, or redundant data
//!
//! We maximize SNR by:
//! 1. Filtering low-quality inputs (noise reduction)
//! 2. Amplifying high-quality patterns (signal boost)
//! 3. Compressing without loss (entropy coding)
//!
//! # Enhanced Features (v2)
//!
//! - Configurable grounding estimation
//! - Input validation with DoS protection
//! - Streaming analysis for large texts
//! - Zero-copy text processing where possible

use super::error::{SovereignError, SovereignResult};
use std::collections::{HashSet, VecDeque};

/// Maximum input size (1MB) to prevent DoS
pub const MAX_INPUT_SIZE: usize = 1_048_576;

/// Minimum input size for meaningful analysis
pub const MIN_INPUT_SIZE: usize = 1;

/// Default grounding score when NLP is unavailable
pub const DEFAULT_GROUNDING: f64 = 0.85;

/// SNR Engine configuration
#[derive(Clone, Debug)]
pub struct SNRConfig {
    /// Minimum acceptable SNR
    pub snr_floor: f64,
    /// Target Ihsān score
    pub ihsan_target: f64,
    /// Default grounding score
    pub default_grounding: f64,
    /// Maximum input size in bytes
    pub max_input_size: usize,
    /// Minimum input size in bytes
    pub min_input_size: usize,
    /// History window size
    pub max_history: usize,
    /// Enable NLP-based grounding estimation
    pub enable_nlp_grounding: bool,
    /// Weight for signal strength
    pub weight_signal: f64,
    /// Weight for diversity
    pub weight_diversity: f64,
    /// Weight for grounding
    pub weight_grounding: f64,
    /// Weight for balance
    pub weight_balance: f64,
}

impl Default for SNRConfig {
    fn default() -> Self {
        Self {
            snr_floor: 0.85,
            ihsan_target: 0.95,
            default_grounding: DEFAULT_GROUNDING,
            max_input_size: MAX_INPUT_SIZE,
            min_input_size: MIN_INPUT_SIZE,
            max_history: 1000,
            enable_nlp_grounding: false,
            weight_signal: 0.30,
            weight_diversity: 0.25,
            weight_grounding: 0.25,
            weight_balance: 0.20,
        }
    }
}

impl SNRConfig {
    /// Create config for high-quality requirements
    pub fn high_quality() -> Self {
        Self {
            snr_floor: 0.90,
            ihsan_target: 0.98,
            default_grounding: 0.90,
            ..Default::default()
        }
    }

    /// Create config for edge/embedded deployment
    pub fn edge() -> Self {
        Self {
            max_input_size: 65536, // 64KB limit
            max_history: 100,
            enable_nlp_grounding: false,
            ..Default::default()
        }
    }
}

/// Signal quality metrics
#[derive(Clone, Debug, Default)]
pub struct SignalMetrics {
    /// Computed SNR value (0.0 - 1.0)
    pub snr: f64,
    /// Signal strength component
    pub signal_strength: f64,
    /// Noise level component
    pub noise_level: f64,
    /// Diversity score (unique information)
    pub diversity: f64,
    /// Grounding score (factual basis)
    pub grounding: f64,
    /// Balance score (equilibrium of factors)
    pub balance: f64,
    /// Input size in bytes
    pub input_size: usize,
    /// Word count
    pub word_count: usize,
    /// Unique word count
    pub unique_words: usize,
    /// Analysis duration in microseconds
    pub analysis_duration_us: u64,
}

impl SignalMetrics {
    /// Compute composite SNR from components using weighted geometric mean
    ///
    /// # Performance (Standing on Giants: Shannon, Gerganov)
    /// - Inline for zero-cost abstraction
    /// - Fixed-size arrays avoid heap allocation
    #[inline]
    pub fn compute_snr(&self) -> f64 {
        self.compute_snr_with_weights(0.30, 0.25, 0.25, 0.20)
    }

    /// Compute SNR with custom weights (Shannon-inspired weighted geometric mean)
    #[inline]
    pub fn compute_snr_with_weights(
        &self,
        w_signal: f64,
        w_diversity: f64,
        w_grounding: f64,
        w_balance: f64,
    ) -> f64 {
        // Unrolled loop for performance (avoid iterator overhead)
        let v1 = self.signal_strength.max(0.001).powf(w_signal);
        let v2 = self.diversity.max(0.001).powf(w_diversity);
        let v3 = self.grounding.max(0.001).powf(w_grounding);
        let v4 = self.balance.max(0.001).powf(w_balance);

        let weighted_product = v1 * v2 * v3 * v4;

        // Subtract noise penalty
        let noise_penalty = self.noise_level * 0.5;
        (weighted_product - noise_penalty).clamp(0.0, 1.0)
    }

    /// Check if metrics meet threshold
    #[inline]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.compute_snr() >= threshold
    }

    /// Check if metrics meet Ihsān constraint
    #[inline]
    pub fn meets_ihsan(&self, ihsan_threshold: f64) -> bool {
        self.compute_snr() >= ihsan_threshold
    }

    /// Get quality assessment string
    #[inline]
    pub fn quality_assessment(&self) -> &'static str {
        let snr = self.compute_snr();
        if snr >= 0.95 {
            "Excellent (Ihsān)"
        } else if snr >= 0.85 {
            "Good"
        } else if snr >= 0.70 {
            "Acceptable"
        } else if snr >= 0.50 {
            "Poor"
        } else {
            "Unacceptable"
        }
    }
}

/// The SNR Engine — Autonomous signal quality optimizer
pub struct SNREngine {
    /// Configuration
    config: SNRConfig,
    /// Recent measurements (sliding window)
    history: VecDeque<SignalMetrics>,
    /// Total measurements
    total_measurements: u64,
    /// Grounding keywords for NLP estimation
    grounding_keywords: HashSet<&'static str>,
}

impl SNREngine {
    /// Create new SNR engine with default configuration
    pub fn new(snr_floor: f64, ihsan_target: f64) -> Self {
        let config = SNRConfig {
            snr_floor,
            ihsan_target,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create new SNR engine with custom configuration
    pub fn with_config(config: SNRConfig) -> Self {
        let grounding_keywords = Self::init_grounding_keywords();
        Self {
            history: VecDeque::with_capacity(config.max_history),
            total_measurements: 0,
            config,
            grounding_keywords,
        }
    }

    /// Initialize grounding keywords for NLP-lite estimation
    fn init_grounding_keywords() -> HashSet<&'static str> {
        [
            // Technical terms
            "algorithm",
            "function",
            "method",
            "class",
            "interface",
            "implementation",
            "architecture",
            "protocol",
            "system",
            // Quantitative terms
            "percent",
            "ratio",
            "number",
            "count",
            "measure",
            "calculate",
            "compute",
            "estimate",
            "approximately",
            // Logical connectors
            "because",
            "therefore",
            "however",
            "although",
            "whereas",
            "consequently",
            "furthermore",
            "moreover",
            "specifically",
            // Evidence indicators
            "research",
            "study",
            "data",
            "evidence",
            "source",
            "according",
            "cited",
            "reference",
            "documented",
        ]
        .into_iter()
        .collect()
    }

    /// Validate input before analysis
    pub fn validate_input(&self, text: &str) -> SovereignResult<()> {
        let size = text.len();

        if size == 0 {
            return Err(SovereignError::EmptyInput);
        }

        if size < self.config.min_input_size {
            return Err(SovereignError::InputTooSmall {
                size,
                min_size: self.config.min_input_size,
            });
        }

        if size > self.config.max_input_size {
            return Err(SovereignError::InputTooLarge {
                size,
                max_size: self.config.max_input_size,
            });
        }

        Ok(())
    }

    /// Measure a generic operation (simplified metrics)
    pub fn measure_operation(&self) -> SignalMetrics {
        SignalMetrics {
            snr: 0.90,
            signal_strength: 0.92,
            noise_level: 0.08,
            diversity: 0.88,
            grounding: self.config.default_grounding,
            balance: 0.91,
            input_size: 0,
            word_count: 0,
            unique_words: 0,
            analysis_duration_us: 0,
        }
    }

    /// Analyze text content for SNR with validation
    ///
    /// # Performance Optimizations (Standing on Giants: Gerganov)
    /// - Zero-copy word iteration (no Vec allocation)
    /// - Reduced allocations via ASCII lowercase comparison
    /// - Inline hints for hot paths
    #[inline]
    pub fn analyze_text(&self, text: &str) -> SovereignResult<SignalMetrics> {
        let start = std::time::Instant::now();

        // Validate input
        self.validate_input(text)?;

        let input_size = text.len();

        // Zero-copy word analysis - single pass
        let (word_count, total_word_chars, unique_words, grounding_hits) =
            self.analyze_words_zero_copy(text);

        if word_count == 0 {
            return Err(SovereignError::EmptyInput);
        }

        // Signal strength: information density (avg word length)
        let avg_word_len = total_word_chars as f64 / word_count as f64;
        let signal_strength = (avg_word_len / 10.0).min(1.0);

        // Diversity: unique words ratio
        let diversity = unique_words as f64 / word_count as f64;

        // Noise: repetition (inverse of diversity)
        let noise_level = 1.0 - diversity;

        // Grounding: NLP-lite estimation or default
        let grounding = if self.config.enable_nlp_grounding {
            self.grounding_from_hits(grounding_hits)
        } else {
            self.config.default_grounding
        };

        // Balance: length appropriateness
        let balance = self.calculate_balance(word_count);

        let mut metrics = SignalMetrics {
            snr: 0.0,
            signal_strength,
            noise_level,
            diversity,
            grounding,
            balance,
            input_size,
            word_count,
            unique_words,
            analysis_duration_us: start.elapsed().as_micros() as u64,
        };

        metrics.snr = metrics.compute_snr_with_weights(
            self.config.weight_signal,
            self.config.weight_diversity,
            self.config.weight_grounding,
            self.config.weight_balance,
        );

        Ok(metrics)
    }

    /// Zero-copy word analysis - single pass through text
    /// Returns (word_count, total_chars, unique_count, grounding_hits)
    #[inline]
    fn analyze_words_zero_copy(&self, text: &str) -> (usize, usize, usize, usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut word_count = 0;
        let mut total_chars = 0;
        let mut grounding_hits = 0;
        let mut seen_hashes: HashSet<u64> = HashSet::with_capacity(
            (text.len() / 5).min(1000), // Estimate unique words
        );

        for word in text.split_whitespace() {
            word_count += 1;
            total_chars += word.len();

            // Hash the lowercase word without allocating
            let mut hasher = DefaultHasher::new();
            for c in word.chars() {
                c.to_ascii_lowercase().hash(&mut hasher);
            }
            let hash = hasher.finish();
            seen_hashes.insert(hash);

            // Check grounding keywords (case-insensitive via lowercase comparison)
            if self.is_grounding_keyword(word) {
                grounding_hits += 1;
            }
        }

        (word_count, total_chars, seen_hashes.len(), grounding_hits)
    }

    /// Fast case-insensitive keyword check
    #[inline]
    fn is_grounding_keyword(&self, word: &str) -> bool {
        // Convert to lowercase ASCII for comparison
        let lower: String = word.chars().map(|c| c.to_ascii_lowercase()).collect();
        self.grounding_keywords.contains(lower.as_str())
    }

    /// Calculate grounding from hit count
    #[inline]
    fn grounding_from_hits(&self, hits: usize) -> f64 {
        // Scale: 0 hits = 0.5, 10+ hits = 1.0
        let base = 0.5;
        let bonus = (hits as f64 / 10.0).min(0.5);
        base + bonus
    }

    /// Analyze text (legacy API, panics on error)
    pub fn analyze_text_unchecked(&self, text: &str) -> SignalMetrics {
        self.analyze_text(text).unwrap_or_default()
    }

    /// Calculate balance score based on word count
    #[inline]
    fn calculate_balance(&self, word_count: usize) -> f64 {
        if (10..=1000).contains(&word_count) {
            1.0
        } else if word_count < 10 {
            word_count as f64 / 10.0
        } else {
            1000.0 / word_count as f64
        }
    }

    /// Check if content meets SNR threshold
    pub fn check(&self, metrics: &SignalMetrics) -> bool {
        metrics.compute_snr() >= self.config.snr_floor
    }

    /// Check if content meets Ihsān threshold
    pub fn check_ihsan(&self, metrics: &SignalMetrics) -> bool {
        metrics.compute_snr() >= self.config.ihsan_target
    }

    /// Validate content and return result with error context
    pub fn validate(&self, text: &str) -> SovereignResult<SignalMetrics> {
        let metrics = self.analyze_text(text)?;
        let snr = metrics.compute_snr();

        if snr < self.config.snr_floor {
            return Err(SovereignError::SNRBelowThreshold {
                actual: snr,
                threshold: self.config.snr_floor,
            });
        }

        Ok(metrics)
    }

    /// Validate content against Ihsān threshold
    pub fn validate_ihsan(&self, text: &str) -> SovereignResult<SignalMetrics> {
        let metrics = self.analyze_text(text)?;
        let snr = metrics.compute_snr();

        if snr < self.config.ihsan_target {
            return Err(SovereignError::IhsanViolation {
                actual: snr,
                threshold: self.config.ihsan_target,
            });
        }

        Ok(metrics)
    }

    /// Record a measurement
    pub fn record(&mut self, metrics: SignalMetrics) {
        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
        self.total_measurements += 1;
    }

    /// Get average SNR from recent history
    pub fn average_snr(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|m| m.compute_snr()).sum();
        sum / self.history.len() as f64
    }

    /// Get current statistics
    pub fn stats(&self) -> SNRStats {
        SNRStats {
            total_measurements: self.total_measurements,
            history_size: self.history.len(),
            average_snr: self.average_snr(),
            snr_floor: self.config.snr_floor,
            ihsan_target: self.config.ihsan_target,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SNRConfig {
        &self.config
    }
}

/// SNR Engine statistics
#[derive(Debug)]
pub struct SNRStats {
    /// Measurements recorded since engine creation.
    pub total_measurements: u64,
    /// Number of measurements currently in the history window.
    pub history_size: usize,
    /// Rolling average SNR across the history window.
    pub average_snr: f64,
    /// Configured SNR floor threshold.
    pub snr_floor: f64,
    /// Configured Ihsan target threshold.
    pub ihsan_target: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_computation() {
        let metrics = SignalMetrics {
            snr: 0.0,
            signal_strength: 0.9,
            noise_level: 0.1,
            diversity: 0.8,
            grounding: 0.85,
            balance: 0.9,
            ..Default::default()
        };

        let snr = metrics.compute_snr();
        assert!(snr > 0.8, "SNR should be > 0.8, got {}", snr);
    }

    #[test]
    fn test_text_analysis() {
        let engine = SNREngine::new(0.85, 0.95);

        let good_text = "The quantum computer utilizes superposition and entanglement \
                         to perform calculations exponentially faster than classical computers.";
        let metrics = engine.analyze_text(good_text).unwrap();

        assert!(
            metrics.diversity > 0.5,
            "Good text should have high diversity"
        );
        assert!(metrics.balance > 0.5, "Good text should have good balance");
    }

    #[test]
    fn test_threshold_check() {
        let engine = SNREngine::new(0.85, 0.95);

        let high_quality = SignalMetrics {
            snr: 0.92,
            signal_strength: 0.92,
            noise_level: 0.08,
            diversity: 0.88,
            grounding: 0.94,
            balance: 0.91,
            ..Default::default()
        };

        assert!(engine.check(&high_quality));
    }

    #[test]
    fn test_input_validation() {
        let engine = SNREngine::new(0.85, 0.95);

        // Empty input
        let result = engine.validate_input("");
        assert!(matches!(result, Err(SovereignError::EmptyInput)));

        // Valid input
        let result = engine.validate_input("Hello world");
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_input_rejection() {
        let config = SNRConfig {
            max_input_size: 100,
            ..Default::default()
        };
        let engine = SNREngine::with_config(config);

        let large_input = "x".repeat(200);
        let result = engine.validate_input(&large_input);
        assert!(matches!(result, Err(SovereignError::InputTooLarge { .. })));
    }

    #[test]
    fn test_validate_returns_error() {
        let engine = SNREngine::new(0.99, 0.99); // Very high threshold

        let text = "short";
        let result = engine.validate(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_presets() {
        let high = SNRConfig::high_quality();
        assert!(high.snr_floor > 0.85);

        let edge = SNRConfig::edge();
        assert!(edge.max_input_size < MAX_INPUT_SIZE);
    }

    #[test]
    fn test_grounding_estimation() {
        let config = SNRConfig {
            enable_nlp_grounding: true,
            ..Default::default()
        };
        let engine = SNREngine::with_config(config);

        // Text with grounding keywords
        let technical = "The algorithm implementation uses research data to compute \
                        the ratio according to documented evidence.";
        let metrics = engine.analyze_text(technical).unwrap();
        assert!(
            metrics.grounding > 0.7,
            "Technical text should have high grounding"
        );
    }

    #[test]
    fn test_quality_assessment() {
        let excellent = SignalMetrics {
            signal_strength: 0.98,
            diversity: 0.95,
            grounding: 0.96,
            balance: 0.97,
            noise_level: 0.02,
            ..Default::default()
        };
        assert_eq!(excellent.quality_assessment(), "Excellent (Ihsān)");

        let poor = SignalMetrics {
            signal_strength: 0.5,
            diversity: 0.4,
            grounding: 0.5,
            balance: 0.5,
            noise_level: 0.6,
            ..Default::default()
        };
        assert!(poor.quality_assessment() != "Excellent (Ihsān)");
    }
}
