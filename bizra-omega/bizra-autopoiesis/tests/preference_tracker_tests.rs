//! Comprehensive tests for PreferenceTracker — observe, reinforce, threshold, apply
//!
//! Phase 13: Test Sprint

use bizra_autopoiesis::preference_tracker::*;

// ---------------------------------------------------------------------------
// PreferenceType
// ---------------------------------------------------------------------------

#[test]
fn preference_type_equality() {
    assert_eq!(PreferenceType::Style, PreferenceType::Style);
    assert_eq!(PreferenceType::Verbosity, PreferenceType::Verbosity);
    assert_eq!(PreferenceType::CodeStyle, PreferenceType::CodeStyle);
    assert_eq!(PreferenceType::Language, PreferenceType::Language);
    assert_eq!(
        PreferenceType::Custom("x".into()),
        PreferenceType::Custom("x".into())
    );
    assert_ne!(
        PreferenceType::Custom("x".into()),
        PreferenceType::Custom("y".into())
    );
    assert_ne!(PreferenceType::Style, PreferenceType::Verbosity);
}

// ---------------------------------------------------------------------------
// Preference
// ---------------------------------------------------------------------------

#[test]
fn preference_new_defaults() {
    let p = Preference::new(PreferenceType::Style, "tone".into(), "formal".into());
    assert_eq!(p.pref_type, PreferenceType::Style);
    assert_eq!(p.key, "tone");
    assert_eq!(p.value, "formal");
    assert!((p.confidence - 0.5).abs() < f64::EPSILON);
    assert_eq!(p.observations, 1);
}

#[test]
fn preference_reinforce_same_value_increases_confidence() {
    let mut p = Preference::new(PreferenceType::Style, "tone".into(), "formal".into());
    let initial = p.confidence;
    p.reinforce(true);
    assert!(p.confidence > initial);
    assert_eq!(p.observations, 2);
}

#[test]
fn preference_reinforce_different_value_decreases_confidence() {
    let mut p = Preference::new(PreferenceType::Style, "tone".into(), "formal".into());
    let initial = p.confidence;
    p.reinforce(false);
    assert!(p.confidence < initial);
    assert_eq!(p.observations, 2);
}

#[test]
fn preference_reinforce_confidence_converges_upward() {
    let mut p = Preference::new(PreferenceType::Verbosity, "level".into(), "concise".into());
    for _ in 0..50 {
        p.reinforce(true);
    }
    // After many same-value reinforcements, confidence should be high
    assert!(p.confidence > 0.9);
}

#[test]
fn preference_reinforce_decays_with_different_values() {
    let mut p = Preference::new(PreferenceType::Verbosity, "level".into(), "verbose".into());
    p.confidence = 0.9;
    for _ in 0..20 {
        p.reinforce(false);
    }
    // After many different-value signals, confidence should drop
    assert!(p.confidence < 0.3);
}

// ---------------------------------------------------------------------------
// PreferenceTracker
// ---------------------------------------------------------------------------

#[test]
fn tracker_new_empty() {
    let tracker = PreferenceTracker::new();
    // No preferences observed yet → get returns None
    assert!(tracker.get(&PreferenceType::Style, "anything").is_none());
}

#[test]
fn tracker_default_same_as_new() {
    let tracker = PreferenceTracker::default();
    assert!(tracker.get(&PreferenceType::Style, "anything").is_none());
}

#[test]
fn tracker_observe_creates_preference() {
    let mut tracker = PreferenceTracker::new();
    tracker.observe(PreferenceType::Style, "tone", "formal");
    // Initial confidence is 0.5, below activation threshold of 0.7
    assert!(tracker.get(&PreferenceType::Style, "tone").is_none());
}

#[test]
fn tracker_observe_repeated_activates_above_threshold() {
    let mut tracker = PreferenceTracker::new();
    // Observe same value many times to raise confidence above 0.7
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "formal");
    }
    let val = tracker.get(&PreferenceType::Style, "tone");
    assert_eq!(val, Some("formal"));
}

#[test]
fn tracker_observe_different_value_changes_stored() {
    let mut tracker = PreferenceTracker::new();
    // First: observe "formal" many times
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "formal");
    }
    assert_eq!(tracker.get(&PreferenceType::Style, "tone"), Some("formal"));

    // Now switch to "casual" — confidence drops because same_value=false
    tracker.observe(PreferenceType::Style, "tone", "casual");
    // After one different observation, confidence drops via *0.9
    // If it drops below 0.7, get() returns None
    // Let's check: after 20 reinforcements, confidence ≈ 0.5 + accumulated
    // The reinforce(true) formula: conf += (1.0 - conf) * 0.1
    // Starting 0.5: after 20 trues → confidence ~ 0.89
    // Then reinforce(false): 0.89 * 0.9 = 0.801 → still above 0.7
    let val = tracker.get(&PreferenceType::Style, "tone");
    assert_eq!(val, Some("casual"));
}

#[test]
fn tracker_multiple_preference_types_independent() {
    let mut tracker = PreferenceTracker::new();
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "formal");
        tracker.observe(PreferenceType::Verbosity, "level", "concise");
    }
    assert_eq!(tracker.get(&PreferenceType::Style, "tone"), Some("formal"));
    assert_eq!(
        tracker.get(&PreferenceType::Verbosity, "level"),
        Some("concise")
    );
}

#[test]
fn tracker_custom_preference_type() {
    let mut tracker = PreferenceTracker::new();
    for _ in 0..20 {
        tracker.observe(PreferenceType::Custom("emoji".into()), "usage", "never");
    }
    let val = tracker.get(&PreferenceType::Custom("emoji".into()), "usage");
    assert_eq!(val, Some("never"));
}

#[test]
fn tracker_get_below_threshold_returns_none() {
    let mut tracker = PreferenceTracker::new();
    // Single observation → confidence 0.5, below 0.7 threshold
    tracker.observe(PreferenceType::Language, "primary", "rust");
    assert!(tracker.get(&PreferenceType::Language, "primary").is_none());
}

#[test]
fn tracker_get_wrong_key_returns_none() {
    let mut tracker = PreferenceTracker::new();
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "formal");
    }
    assert!(tracker.get(&PreferenceType::Style, "color").is_none());
}

#[test]
fn tracker_get_wrong_type_returns_none() {
    let mut tracker = PreferenceTracker::new();
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "formal");
    }
    assert!(tracker.get(&PreferenceType::Verbosity, "tone").is_none());
}

// ---------------------------------------------------------------------------
// apply_to_prompt
// ---------------------------------------------------------------------------

#[test]
fn apply_to_prompt_no_preferences_returns_original() {
    let tracker = PreferenceTracker::new();
    let result = tracker.apply_to_prompt("Hello world");
    assert_eq!(result, "Hello world");
}

#[test]
fn apply_to_prompt_with_style_appends_preferences() {
    let mut tracker = PreferenceTracker::new();
    // Build up style preference above threshold
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "response", "professional");
    }
    let result = tracker.apply_to_prompt("Explain Rust");
    assert!(result.contains("Explain Rust"));
    assert!(result.contains("Preferences:"));
    assert!(result.contains("professional"));
}

#[test]
fn apply_to_prompt_ignores_non_response_style() {
    let mut tracker = PreferenceTracker::new();
    // Style key is "tone" not "response" → apply_to_prompt checks "response" key
    for _ in 0..20 {
        tracker.observe(PreferenceType::Style, "tone", "casual");
    }
    let result = tracker.apply_to_prompt("Test prompt");
    assert_eq!(result, "Test prompt"); // no modification
}
