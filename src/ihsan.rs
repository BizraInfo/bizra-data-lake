use serde::Deserialize;
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Debug, Deserialize)]
struct IhsanConstitutionFile {
    id: Option<String>,
    units: IhsanUnits,
    threshold_policy: Option<IhsanThresholdPolicyFile>,
    dimensions: BTreeMap<String, IhsanDimensionSpec>,
    invariants: Option<IhsanInvariants>,
}

#[derive(Debug, Deserialize)]
struct IhsanUnits {
    score_range: [f64; 2],
    threshold: f64,
}

#[derive(Debug, Deserialize)]
struct IhsanThresholdPolicyFile {
    version: Option<u32>,
    combine: Option<String>,
    default_env: Option<String>,
    thresholds_by_env: Option<BTreeMap<String, f64>>,
    thresholds_by_artifact_class: Option<BTreeMap<String, f64>>,
    normalization: Option<IhsanThresholdNormalizationFile>,
}

#[derive(Debug, Deserialize)]
struct IhsanThresholdNormalizationFile {
    env_aliases: Option<BTreeMap<String, String>>,
    artifact_class_aliases: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct IhsanDimensionSpec {
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct IhsanInvariants {
    weights_sum: Option<f64>,
    required_dimensions: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct IhsanConstitution {
    id: String,
    threshold: f64,
    default_env: String,
    threshold_combine: ThresholdCombine,
    thresholds_by_env: BTreeMap<String, f64>,
    thresholds_by_artifact_class: BTreeMap<String, f64>,
    env_aliases: BTreeMap<String, String>,
    artifact_class_aliases: BTreeMap<String, String>,
    weights: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
enum ThresholdCombine {
    Max,
    Min,
}

impl ThresholdCombine {
    fn parse(raw: Option<&str>) -> anyhow::Result<Self> {
        match raw.unwrap_or("max").trim().to_ascii_lowercase().as_str() {
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            other => anyhow::bail!("unsupported threshold_policy.combine: {other}"),
        }
    }
}

impl IhsanConstitution {
    fn normalize_key(raw: &str) -> String {
        raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
    }

    fn normalize_alias_map(map: Option<BTreeMap<String, String>>) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        let Some(map) = map else {
            return out;
        };

        for (k, v) in map {
            let key = Self::normalize_key(&k);
            let val = Self::normalize_key(&v);
            if !key.is_empty() && !val.is_empty() {
                out.insert(key, val);
            }
        }
        out
    }

    fn normalize_threshold_map(
        map: Option<BTreeMap<String, f64>>,
        min: f64,
        max: f64,
    ) -> anyhow::Result<BTreeMap<String, f64>> {
        let mut out = BTreeMap::new();
        let Some(map) = map else {
            return Ok(out);
        };

        for (k, v) in map {
            let key = Self::normalize_key(&k);
            if key.is_empty() {
                continue;
            }
            if !v.is_finite() || v < min || v > max {
                anyhow::bail!("threshold out of range for '{key}': {v} (expected {min}..={max})");
            }
            out.insert(key, v);
        }
        Ok(out)
    }

    fn canonicalize(key: &str, aliases: &BTreeMap<String, String>) -> String {
        let norm = Self::normalize_key(key);
        aliases.get(&norm).cloned().unwrap_or(norm)
    }

    fn from_yaml_str(yaml: &str) -> anyhow::Result<Self> {
        let parsed: IhsanConstitutionFile = serde_yaml::from_str(yaml)?;

        let expected_sum = parsed
            .invariants
            .as_ref()
            .and_then(|i| i.weights_sum)
            .unwrap_or(1.0);

        let weights: BTreeMap<String, f64> = parsed
            .dimensions
            .iter()
            .map(|(k, v)| (k.clone(), v.weight))
            .collect();

        let sum: f64 = weights.values().sum();
        if (sum - expected_sum).abs() > 1e-9 {
            anyhow::bail!("ihsan constitution weights do not sum to {expected_sum} (got {sum})");
        }

        for (name, weight) in &weights {
            if !weight.is_finite() || *weight < 0.0 || *weight > 1.0 {
                anyhow::bail!("ihsan weight out of range for {name}: {weight}");
            }
        }

        let min = parsed.units.score_range[0];
        let max = parsed.units.score_range[1];
        if !(min <= parsed.units.threshold && parsed.units.threshold <= max) {
            anyhow::bail!(
                "ihsan threshold {} outside score_range [{}, {}]",
                parsed.units.threshold,
                min,
                max
            );
        }

        let required = parsed
            .invariants
            .as_ref()
            .and_then(|i| i.required_dimensions.clone())
            .unwrap_or_else(|| weights.keys().cloned().collect());

        for dim in required {
            if !weights.contains_key(&dim) {
                anyhow::bail!("ihsan constitution missing required dimension: {dim}");
            }
        }

        let policy = parsed.threshold_policy;
        let default_env = policy
            .as_ref()
            .and_then(|p| p.default_env.clone())
            .unwrap_or_else(|| "development".to_string());

        let threshold_combine =
            ThresholdCombine::parse(policy.as_ref().and_then(|p| p.combine.as_deref()))?;

        let thresholds_by_env = Self::normalize_threshold_map(
            policy.as_ref().and_then(|p| p.thresholds_by_env.clone()),
            min,
            max,
        )?;
        let thresholds_by_artifact_class = Self::normalize_threshold_map(
            policy
                .as_ref()
                .and_then(|p| p.thresholds_by_artifact_class.clone()),
            min,
            max,
        )?;

        let env_aliases = Self::normalize_alias_map(
            policy
                .as_ref()
                .and_then(|p| p.normalization.as_ref())
                .and_then(|n| n.env_aliases.clone()),
        );
        let artifact_class_aliases = Self::normalize_alias_map(
            policy
                .as_ref()
                .and_then(|p| p.normalization.as_ref())
                .and_then(|n| n.artifact_class_aliases.clone()),
        );

        if let Some(v) = policy.as_ref().and_then(|p| p.version) {
            if v != 1 {
                anyhow::bail!("unsupported threshold_policy.version: {v}");
            }
        }

        Ok(Self {
            id: parsed.id.unwrap_or_else(|| "ihsan".to_string()),
            threshold: parsed.units.threshold,
            default_env,
            threshold_combine,
            thresholds_by_env,
            thresholds_by_artifact_class,
            env_aliases,
            artifact_class_aliases,
            weights,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn default_env(&self) -> &str {
        &self.default_env
    }

    pub fn threshold_for(&self, env: &str, artifact_class: &str) -> f64 {
        let env_key = Self::canonicalize(env, &self.env_aliases);
        let artifact_key = Self::canonicalize(artifact_class, &self.artifact_class_aliases);

        let mut candidates: Vec<f64> = Vec::new();
        if let Some(t) = self.thresholds_by_env.get(&env_key) {
            candidates.push(*t);
        }
        if let Some(t) = self.thresholds_by_artifact_class.get(&artifact_key) {
            candidates.push(*t);
        }

        if candidates.is_empty() {
            return self.threshold;
        }

        match self.threshold_combine {
            ThresholdCombine::Max => candidates
                .into_iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b)),
            ThresholdCombine::Min => candidates.into_iter().fold(f64::INFINITY, |a, b| a.min(b)),
        }
    }

    pub fn weights(&self) -> &BTreeMap<String, f64> {
        &self.weights
    }

    pub fn score(&self, scores: &BTreeMap<String, f64>) -> anyhow::Result<f64> {
        let min = 0.0;
        let max = 1.0;

        for (dim, weight) in &self.weights {
            let value = scores.get(dim).copied().ok_or_else(|| {
                anyhow::anyhow!("ihsan score input missing required dimension: {dim}")
            })?;
            if !value.is_finite() || value < min || value > max {
                anyhow::bail!("ihsan score input out of range for {dim}: {value}");
            }
            if !weight.is_finite() || *weight < 0.0 {
                anyhow::bail!("ihsan constitution weight invalid for {dim}: {weight}");
            }
        }

        Ok(self
            .weights
            .iter()
            .map(|(dim, w)| w * scores.get(dim).copied().unwrap_or(0.0))
            .sum())
    }
}

pub fn constitution() -> &'static IhsanConstitution {
    static ONCE: OnceLock<IhsanConstitution> = OnceLock::new();
    ONCE.get_or_init(|| {
        IhsanConstitution::from_yaml_str(include_str!("../constitution/ihsan_v1.yaml"))
            .expect("constitution/ihsan_v1.yaml must be parseable and valid")
    })
}

pub fn score(scores: &BTreeMap<String, f64>) -> anyhow::Result<f64> {
    constitution().score(scores)
}

pub fn current_env() -> String {
    // Check BIZRA_IHSAN_ENV first (set by docker-compose)
    if let Ok(v) = std::env::var("BIZRA_IHSAN_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if let Ok(v) = std::env::var("BIZRA_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if let Ok(v) = std::env::var("NODE_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if std::env::var("CI").is_ok() {
        return "ci".to_string();
    }
    constitution().default_env().to_string()
}

pub fn should_enforce() -> bool {
    if let Ok(v) = std::env::var("BIZRA_IHSAN_ENFORCE") {
        let val = v.trim().to_ascii_lowercase();
        if matches!(val.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
    }

    let env = current_env();
    let canonical = IhsanConstitution::canonicalize(&env, &constitution().env_aliases);
    matches!(canonical.as_str(), "ci" | "production")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constitution_weights_sum_to_one() {
        let sum: f64 = constitution().weights().values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum was {sum}");
    }

    #[test]
    fn score_matches_manual_dot_product() {
        let mut scores = BTreeMap::new();
        scores.insert("correctness".to_string(), 1.0);
        scores.insert("safety".to_string(), 1.0);
        scores.insert("user_benefit".to_string(), 0.5);
        scores.insert("efficiency".to_string(), 0.25);
        scores.insert("auditability".to_string(), 0.75);
        scores.insert("anti_centralization".to_string(), 0.0);
        scores.insert("robustness".to_string(), 0.9);
        scores.insert("adl_fairness".to_string(), 0.8);

        let actual = score(&scores).unwrap();
        let expected = 0.22 * 1.0
            + 0.22 * 1.0
            + 0.14 * 0.5
            + 0.12 * 0.25
            + 0.12 * 0.75
            + 0.08 * 0.0
            + 0.06 * 0.9
            + 0.04 * 0.8;

        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn threshold_policy_is_applied() {
        let c = constitution();
        assert!((c.threshold_for("production", "code") - 0.95).abs() < 1e-9);
        assert!((c.threshold_for("dev", "docs") - 0.80).abs() < 1e-9);
        assert!((c.threshold_for("ci", "docs") - 0.90).abs() < 1e-9);
        assert!((c.threshold_for("ci", "receipt") - 0.95).abs() < 1e-9);
    }
}
