// bizra-agent/src/reflex_compiler.rs
// ============================================================
// GENESIS Reflex Compiler — System-2 traces -> System-1 rules
// ============================================================

use std::collections::{HashMap, VecDeque};

use crate::{
    hash_namespace::TriggerHash,
    reflex_cache::{ActionTemplate, ReflexRule},
};

#[derive(Debug, Clone, Copy)]
pub struct CompilerConfig {
    pub min_success_chains: usize,
    pub min_compile_ihsan: f32,
    pub min_compile_snr: f32,
    pub max_path_variance: f32,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            min_success_chains: 3,
            min_compile_ihsan: 0.95,
            min_compile_snr: 0.90,
            max_path_variance: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileReasonCode {
    LowIhsan,
    LowSnr,
    PathVarianceHigh,
    InsufficientSamples,
}

impl CompileReasonCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LowIhsan => "low_ihsan",
            Self::LowSnr => "low_snr",
            Self::PathVarianceHigh => "path_variance_high",
            Self::InsufficientSamples => "insufficient_samples",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileSample {
    pub route_signature: String,
    pub path_signature: String,
    pub response_confidence: f32,
    pub context_richness: f32,
    pub guardian_approved: bool,
    pub ihsan_at_decision: f32,
    pub timestamp: u64,
}

impl CompileSample {
    pub fn snr(&self) -> f32 {
        snr_score(
            self.response_confidence,
            self.context_richness,
            self.guardian_approved,
        )
    }
}

#[derive(Debug, Default)]
struct CandidateBucket {
    samples: VecDeque<CompileSample>,
}

pub struct ReflexCompiler {
    candidates: HashMap<TriggerHash, CandidateBucket>,
    max_samples_per_trigger: usize,
}

impl ReflexCompiler {
    pub fn new(max_samples_per_trigger: usize) -> Self {
        Self {
            candidates: HashMap::new(),
            max_samples_per_trigger: max_samples_per_trigger.max(4),
        }
    }

    pub fn record_success(&mut self, trigger: TriggerHash, sample: CompileSample) {
        let bucket = self.candidates.entry(trigger).or_default();
        bucket.samples.push_back(sample);
        while bucket.samples.len() > self.max_samples_per_trigger {
            bucket.samples.pop_front();
        }
    }

    pub fn sample_count(&self, trigger: &TriggerHash) -> usize {
        self.candidates
            .get(trigger)
            .map(|b| b.samples.len())
            .unwrap_or(0)
    }

    pub fn evaluate(
        &self,
        trigger: TriggerHash,
        config: CompilerConfig,
        policy_hash: [u8; 32],
    ) -> Result<ReflexRule, CompileReasonCode> {
        let Some(bucket) = self.candidates.get(&trigger) else {
            return Err(CompileReasonCode::InsufficientSamples);
        };

        if bucket.samples.len() < config.min_success_chains {
            return Err(CompileReasonCode::InsufficientSamples);
        }

        let recent: Vec<&CompileSample> = bucket
            .samples
            .iter()
            .rev()
            .take(config.min_success_chains)
            .collect();

        let n = recent.len().max(1) as f32;
        let avg_ihsan = recent.iter().map(|s| s.ihsan_at_decision).sum::<f32>() / n;
        if avg_ihsan < config.min_compile_ihsan {
            return Err(CompileReasonCode::LowIhsan);
        }

        let avg_snr = recent.iter().map(|s| s.snr()).sum::<f32>() / n;
        if avg_snr < config.min_compile_snr {
            return Err(CompileReasonCode::LowSnr);
        }

        let variance = path_variance(&recent);
        if variance > config.max_path_variance {
            return Err(CompileReasonCode::PathVarianceHigh);
        }

        let route_signature = most_common_route(&recent);
        let primary_agent = infer_primary_agent(&route_signature);
        let compiled_at = recent.iter().map(|s| s.timestamp).max().unwrap_or(0);

        Ok(ReflexRule {
            trigger_hash: trigger,
            action_template: ActionTemplate {
                route_signature,
                primary_agent,
            },
            compile_ihsan: avg_ihsan,
            compile_snr: avg_snr,
            compiled_at,
            use_count: 0,
            last_used_at: 0,
            last_validated_at: compiled_at,
            quarantined: false,
            quarantine_reason: None,
            policy_hash,
        })
    }
}

impl Default for ReflexCompiler {
    fn default() -> Self {
        Self::new(32)
    }
}

pub fn snr_score(response_confidence: f32, context_richness: f32, guardian_approved: bool) -> f32 {
    let guardian = if guardian_approved { 1.0 } else { 0.0 };
    (0.5 * response_confidence) + (0.3 * context_richness) + (0.2 * guardian)
}

fn most_common_route(samples: &[&CompileSample]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in samples {
        *counts.entry(s.route_signature.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(route, _)| route.to_string())
        .unwrap_or_else(|| "GenerateResponse".to_string())
}

fn infer_primary_agent(route_signature: &str) -> String {
    if let Some(idx) = route_signature.find("roles=") {
        let rest = &route_signature[idx + 6..];
        if let Some(first) = rest.split('>').next() {
            return first.trim().to_string();
        }
    }
    "Navigator".to_string()
}

fn path_variance(samples: &[&CompileSample]) -> f32 {
    if samples.len() <= 1 {
        return 0.0;
    }

    let mut distances = Vec::new();
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            distances.push(signature_distance(
                &samples[i].path_signature,
                &samples[j].path_signature,
            ));
        }
    }

    if distances.is_empty() {
        return 0.0;
    }
    distances.iter().sum::<f32>() / distances.len() as f32
}

fn signature_distance(a: &str, b: &str) -> f32 {
    if a == b {
        return 0.0;
    }

    let left: Vec<&str> = a.split('>').filter(|s| !s.is_empty()).collect();
    let right: Vec<&str> = b.split('>').filter(|s| !s.is_empty()).collect();

    if left.is_empty() && right.is_empty() {
        return 0.0;
    }

    let max_len = left.len().max(right.len()) as f32;
    let min_len = left.len().min(right.len());

    let mut mismatches = (left.len() as i32 - right.len() as i32).unsigned_abs() as f32;
    for i in 0..min_len {
        if left[i] != right[i] {
            mismatches += 1.0;
        }
    }

    (mismatches / max_len).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(route: &str, path: &str, ihsan: f32, conf: f32, rich: f32, ts: u64) -> CompileSample {
        CompileSample {
            route_signature: route.to_string(),
            path_signature: path.to_string(),
            response_confidence: conf,
            context_richness: rich,
            guardian_approved: true,
            ihsan_at_decision: ihsan,
            timestamp: ts,
        }
    }

    #[test]
    fn fails_with_insufficient_samples() {
        let mut c = ReflexCompiler::new(8);
        let trigger = TriggerHash([1u8; 32]);
        c.record_success(trigger, sample("A", "a>b", 0.99, 0.95, 0.90, 1));
        let err = c
            .evaluate(trigger, CompilerConfig::default(), [7u8; 32])
            .expect_err("must fail");
        assert_eq!(err, CompileReasonCode::InsufficientSamples);
    }

    #[test]
    fn blocks_high_path_variance() {
        let mut c = ReflexCompiler::new(8);
        let trigger = TriggerHash([2u8; 32]);
        c.record_success(trigger, sample("A", "a>b>c", 0.99, 0.98, 0.95, 1));
        c.record_success(trigger, sample("A", "x>y>z", 0.99, 0.98, 0.95, 2));
        c.record_success(trigger, sample("A", "m>n>o", 0.99, 0.98, 0.95, 3));

        let err = c
            .evaluate(trigger, CompilerConfig::default(), [9u8; 32])
            .expect_err("must fail on variance");
        assert_eq!(err, CompileReasonCode::PathVarianceHigh);
    }

    #[test]
    fn compiles_when_gates_pass() {
        let mut c = ReflexCompiler::new(8);
        let trigger = TriggerHash([3u8; 32]);
        c.record_success(
            trigger,
            sample(
                "Retrieve>Generate|roles=Scholar>Artisan",
                "Retrieve>Verify>Act",
                0.98,
                0.95,
                0.92,
                10,
            ),
        );
        c.record_success(
            trigger,
            sample(
                "Retrieve>Generate|roles=Scholar>Artisan",
                "Retrieve>Verify>Act",
                0.97,
                0.94,
                0.91,
                11,
            ),
        );
        c.record_success(
            trigger,
            sample(
                "Retrieve>Generate|roles=Scholar>Artisan",
                "Retrieve>Verify>Act",
                0.99,
                0.96,
                0.93,
                12,
            ),
        );

        let rule = c
            .evaluate(trigger, CompilerConfig::default(), [1u8; 32])
            .expect("should compile");
        assert_eq!(rule.trigger_hash, trigger);
        assert!(!rule.quarantined);
        assert_eq!(rule.action_template.primary_agent, "Scholar");
        assert!(rule.compile_ihsan >= 0.95);
        assert!(rule.compile_snr >= 0.90);
    }
}
