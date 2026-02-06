//! Preference Tracker — Learn without retraining

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreferenceType {
    Style,
    Verbosity,
    CodeStyle,
    Language,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preference {
    pub pref_type: PreferenceType,
    pub key: String,
    pub value: String,
    pub confidence: f64,
    pub observations: u64,
    pub updated_at: DateTime<Utc>,
}

impl Preference {
    pub fn new(pref_type: PreferenceType, key: String, value: String) -> Self {
        Self {
            pref_type,
            key,
            value,
            confidence: 0.5,
            observations: 1,
            updated_at: Utc::now(),
        }
    }

    pub fn reinforce(&mut self, same_value: bool) {
        self.observations += 1;
        self.updated_at = Utc::now();
        if same_value {
            self.confidence += (1.0 - self.confidence) * 0.1;
        } else {
            self.confidence *= 0.9;
        }
    }
}

pub struct PreferenceTracker {
    preferences: HashMap<(PreferenceType, String), Preference>,
    activation_threshold: f64,
}

impl PreferenceTracker {
    pub fn new() -> Self {
        Self {
            preferences: HashMap::new(),
            activation_threshold: 0.7,
        }
    }

    pub fn observe(&mut self, pref_type: PreferenceType, key: &str, value: &str) {
        let lookup = (pref_type.clone(), key.to_string());
        if let Some(existing) = self.preferences.get_mut(&lookup) {
            let same = existing.value == value;
            if !same {
                existing.value = value.to_string();
            }
            existing.reinforce(same);
        } else {
            self.preferences
                .insert(lookup, Preference::new(pref_type, key.into(), value.into()));
        }
    }

    pub fn get(&self, pref_type: &PreferenceType, key: &str) -> Option<&str> {
        self.preferences
            .get(&(pref_type.clone(), key.to_string()))
            .filter(|p| p.confidence >= self.activation_threshold)
            .map(|p| p.value.as_str())
    }

    pub fn apply_to_prompt(&self, prompt: &str) -> String {
        let mut additions = Vec::new();
        if let Some(style) = self.get(&PreferenceType::Style, "response") {
            additions.push(format!("Use {} tone.", style));
        }
        if additions.is_empty() {
            prompt.to_string()
        } else {
            format!("{}\n\nPreferences: {}", prompt, additions.join(" "))
        }
    }
}

impl Default for PreferenceTracker {
    fn default() -> Self {
        Self::new()
    }
}
