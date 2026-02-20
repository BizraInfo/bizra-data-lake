// bizra-agent/src/hash_namespace.rs
// ============================================================
// GENESIS Hash Namespace — typed, domain-separated BLAKE3 hashes
// ============================================================

use blake3::Hasher;

pub const TRIGGER_DOMAIN: &str = "genesis/trigger/v1";
pub const ACTION_DOMAIN: &str = "genesis/action/v1";
pub const ARTIFACT_DOMAIN: &str = "genesis/artifact/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriggerHash(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionHash(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactHash(pub [u8; 32]);

impl TriggerHash {
    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }
}

impl ActionHash {
    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }
}

impl ArtifactHash {
    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }
}

pub fn parse_hex_32(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

pub fn compute_trigger_hash(
    normalized_intent: &str,
    selected_traits: &[(String, String)],
    policy_hash: &[u8; 32],
) -> TriggerHash {
    let mut canonical_traits: Vec<(String, String)> = selected_traits
        .iter()
        .map(|(k, v)| {
            (
                k.trim().to_ascii_lowercase(),
                normalize_ws(v.trim()).to_ascii_lowercase(),
            )
        })
        .collect();
    canonical_traits.sort_by(|a, b| a.cmp(b));

    let mut payload = Vec::new();
    let intent = normalize_ws(normalized_intent.trim()).to_ascii_lowercase();
    write_len_prefixed(&mut payload, intent.as_bytes());
    write_u32(&mut payload, canonical_traits.len() as u32);
    for (k, v) in &canonical_traits {
        write_len_prefixed(&mut payload, k.as_bytes());
        write_len_prefixed(&mut payload, v.as_bytes());
    }
    payload.extend_from_slice(policy_hash);
    TriggerHash(domain_hash(TRIGGER_DOMAIN, &payload))
}

pub fn compute_action_hash(
    trigger_hash: &TriggerHash,
    chosen_route: &str,
    timestamp: u64,
) -> ActionHash {
    let mut payload = Vec::new();
    payload.extend_from_slice(&trigger_hash.0);
    write_len_prefixed(&mut payload, normalize_ws(chosen_route).as_bytes());
    write_u64(&mut payload, timestamp);
    ActionHash(domain_hash(ACTION_DOMAIN, &payload))
}

pub fn compute_artifact_hash(action_hash: &ActionHash, serialized_artifact: &str) -> ArtifactHash {
    let mut payload = Vec::new();
    payload.extend_from_slice(&action_hash.0);
    write_len_prefixed(&mut payload, normalize_ws(serialized_artifact).as_bytes());
    ArtifactHash(domain_hash(ARTIFACT_DOMAIN, &payload))
}

fn domain_hash(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(b":");
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

fn write_len_prefixed(buf: &mut Vec<u8>, data: &[u8]) {
    write_u32(buf, data.len() as u32);
    buf.extend_from_slice(data);
}

fn write_u32(buf: &mut Vec<u8>, n: u32) {
    buf.extend_from_slice(&n.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, n: u64) {
    buf.extend_from_slice(&n.to_le_bytes());
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_roundtrip() {
        let raw = [7u8; 32];
        let hex = hex_encode(&raw);
        assert_eq!(parse_hex_32(&hex), Some(raw));
    }

    #[test]
    fn trigger_hash_stable_with_sorted_traits() {
        let policy = [3u8; 32];
        let a = vec![
            ("role".to_string(), "founder".to_string()),
            ("domain".to_string(), "infra".to_string()),
        ];
        let b = vec![
            ("domain".to_string(), "infra".to_string()),
            ("role".to_string(), "founder".to_string()),
        ];
        let ha = compute_trigger_hash("Plan", &a, &policy);
        let hb = compute_trigger_hash("Plan", &b, &policy);
        assert_eq!(ha, hb);
    }

    #[test]
    fn trigger_hash_changes_on_policy_change() {
        let traits = vec![("role".to_string(), "founder".to_string())];
        let h1 = compute_trigger_hash("plan", &traits, &[1u8; 32]);
        let h2 = compute_trigger_hash("plan", &traits, &[2u8; 32]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn domain_separation_changes_output() {
        let trigger = TriggerHash([1u8; 32]);
        let action = compute_action_hash(&trigger, "route", 100);
        let artifact = compute_artifact_hash(&action, "payload");
        assert_ne!(action.to_hex(), artifact.to_hex());
    }
}
