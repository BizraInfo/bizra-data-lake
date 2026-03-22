// bizra-node/src/audit_hook.rs
// ============================================================
// PostDeliver Audit Hook — Append-Only JSONL Audit Trail
// ============================================================
// Writes receipt events to an append-only JSONL audit log.
// PostDeliver hooks cannot halt (by design — Gem 2), so
// audit logging never blocks event dispatch.
//
// Standing on: Lamport (happened-before, 1978), Shannon (1948)
// ============================================================

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

use bizra_hooks::types::{Event, HookResult};

/// Default path for the audit log file.
pub const AUDIT_LOG_PATH: &str = "data/audit/action_receipts.jsonl";

/// Max audit file size before rotation (50 MB).
pub const MAX_AUDIT_FILE_SIZE: u64 = 50_000_000;

/// Resolve the active audit log path.
///
/// Allows tests and controlled migration to override the path while
/// keeping the production default stable.
pub fn audit_log_path() -> String {
    std::env::var("BIZRA_AUDIT_LOG_PATH").unwrap_or_else(|_| AUDIT_LOG_PATH.to_string())
}

/// PostDeliver hook function for action.receipt events.
///
/// Extracts receipt metadata from the event payload and appends
/// a JSONL entry to the audit log. Cannot halt (by design).
pub fn audit_receipt_hook(event: &Event) -> (HookResult, Option<Event>) {
    let bytes = event.payload.as_bytes();
    if bytes.len() < 32 {
        return (HookResult::Continue, None);
    }

    let receipt_hash_hex = hex_encode(&bytes[..32]);
    let action_id = if bytes.len() > 33 && bytes[32] == 0 {
        std::str::from_utf8(&bytes[33..]).unwrap_or("unknown")
    } else {
        "unknown"
    };

    let entry = serde_json::json!({
        "ts": event.id.timestamp_nanos(),
        "receipt_hash": receipt_hash_hex,
        "action_id": action_id,
        "source": format!("{}", event.source),
        "ihsan": event.ihsan_score.as_f64(),
        "priority": format!("{:?}", event.priority),
        "topic": event.topic.as_str(),
    });

    let path = audit_log_path();
    if let Err(e) = append_audit_line(&path, &entry.to_string()) {
        eprintln!("[WARN] Audit log write failed: {e}");
    }

    (HookResult::Continue, None)
}

/// Append a single line to the audit log file (O_APPEND for atomicity).
///
/// Creates parent directories if they do not exist. Rotates the file
/// when it exceeds `MAX_AUDIT_FILE_SIZE`.
pub fn append_audit_line(path: &str, line: &str) -> std::io::Result<()> {
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(file, "{line}")?;

    // Check rotation threshold.
    if let Ok(meta) = fs::metadata(&path) {
        if meta.len() > MAX_AUDIT_FILE_SIZE {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let archive = format!("{}.{}", path.display(), ts);
            let _ = fs::rename(&path, archive);
        }
    }
    Ok(())
}

/// Encode a byte slice as lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use bizra_hooks::types::*;

    use super::*;

    fn make_receipt_event(receipt_hash: [u8; 32], action_id: &str) -> Event {
        let mut payload_bytes = Vec::with_capacity(32 + 1 + action_id.len());
        payload_bytes.extend_from_slice(&receipt_hash);
        payload_bytes.push(0x00);
        payload_bytes.extend_from_slice(action_id.as_bytes());

        Event {
            id: EventId::new(1_000_000_000, 0),
            source: ComponentId::from_name("action_executor", "1.0.0"),
            topic: Topic::new("action.receipt"),
            priority: Priority::High,
            payload: Payload::new(&payload_bytes),
            ihsan_score: IhsanScore::from_raw(9900),
        }
    }

    #[test]
    fn audit_hook_returns_continue() {
        let event = make_receipt_event([0xABu8; 32], "act_00000001");
        let (result, transformed) = audit_receipt_hook(&event);
        assert_eq!(result, HookResult::Continue);
        assert!(transformed.is_none());
    }

    #[test]
    fn audit_hook_skips_short_payload() {
        let event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::from_name("test", "1.0.0"),
            topic: Topic::new("action.receipt"),
            priority: Priority::High,
            payload: Payload::new(&[0u8; 10]),
            ihsan_score: IhsanScore::MAX,
        };
        let (result, _) = audit_receipt_hook(&event);
        assert_eq!(result, HookResult::Continue);
    }

    #[test]
    fn hex_encode_correctness() {
        assert_eq!(hex_encode(&[0x00, 0xFF, 0xAB]), "00ffab");
        assert_eq!(hex_encode(&[]), "");
    }

    #[test]
    fn append_audit_line_creates_file() {
        let dir = std::env::temp_dir().join("bizra_audit_test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("test_audit.jsonl");
        let path_str = path.to_str().unwrap();

        append_audit_line(path_str, r#"{"ts":1,"test":true}"#).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains(r#""test":true"#));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
