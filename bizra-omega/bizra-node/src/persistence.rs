// bizra-node/src/persistence.rs
// ============================================================
// Persistence — Knowledge as text files you own
// ============================================================
//
// The persistence layer is deliberately simple:
//   - State directory: ~/.bizra/node-{hash}/
//   - Seed files: plain text, one TEACH command per line
//   - Save: dump accumulated knowledge as TEACH commands
//   - Load: replay TEACH commands through the node
//
// No database. No binary format. No vendor lock-in.
// Your knowledge is a text file. Back it up. Guard it. Own it.
// ============================================================

use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use bizra_agent::action_types::ActionReceipt;
use bizra_agent::hash_namespace::{parse_hex_32, TriggerHash};
use bizra_agent::reflex_cache::{ActionTemplate, QuarantineReason, ReflexRule};

use crate::node::Node;

// ============================================================
// STATE DIRECTORY
// ============================================================

/// Determine the default state directory for a given user hash.
///
/// Returns `~/.bizra/node-{hash}` (platform-appropriate home dir).
pub fn state_dir(user_hash: u32) -> PathBuf {
    let home = home_dir();
    home.join(".bizra").join(format!("node-{}", user_hash))
}

/// Get the user home directory.
fn home_dir() -> PathBuf {
    // Try HOME env var first (works on Linux, macOS, WSL)
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home);
    }
    // Fallback: USERPROFILE (Windows)
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home);
    }
    // Last resort
    PathBuf::from(".")
}

// ============================================================
// LOAD SEED — replay commands into a node
// ============================================================

/// Load a seed file into the node by replaying its commands.
///
/// Each line in the seed file is a protocol command (typically TEACH).
/// Lines starting with '#' are comments. Empty lines are skipped.
///
/// Returns `(loaded_count, error_count)`.
pub fn load_seed(node: &mut Node, path: &Path) -> io::Result<(usize, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut loaded = 0usize;
    let mut errors = 0usize;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let response = node.execute(trimmed);
        if response.starts_with("ERR\t") {
            errors += 1;
        } else {
            loaded += 1;
        }
    }

    Ok((loaded, errors))
}

// ============================================================
// SAVE STATE — dump knowledge as TEACH commands
// ============================================================

/// Save the node's accumulated knowledge to a seed file.
///
/// Writes TEACH commands that can be replayed to restore knowledge.
/// Returns the number of items saved.
pub fn save_state(node: &Node, path: &Path) -> io::Result<usize> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(writer, "# bizra-node knowledge seed")?;
    writeln!(writer, "# user: {}", node.user_hash())?;
    writeln!(writer, "# saved: {}", timestamp_now())?;
    writeln!(writer)?;

    let kinds = [
        ("fact", bizra_memory::AtomKind::Fact),
        ("preference", bizra_memory::AtomKind::Preference),
        ("pattern", bizra_memory::AtomKind::Pattern),
        ("relationship", bizra_memory::AtomKind::Relationship),
        ("goal", bizra_memory::AtomKind::Goal),
        ("expertise", bizra_memory::AtomKind::Expertise),
        ("context", bizra_memory::AtomKind::Context),
        ("principle", bizra_memory::AtomKind::Principle),
        ("temporal", bizra_memory::AtomKind::Temporal),
        ("negation", bizra_memory::AtomKind::Negation),
    ];

    let health = node.runtime().health();
    writeln!(writer, "# fragments: {}", health.fragments_stored)?;
    writeln!(writer, "# insights: {}", health.insights_stored)?;
    writeln!(writer, "# knows_me: {:.4}", health.knows_me_score)?;
    writeln!(writer)?;

    // Export atoms as TEACH commands via immutable pipeline access.
    let store = node.runtime().pipeline().store();
    let mut saved = 0usize;

    for (kind_name, kind_enum) in &kinds {
        for atom in store.atoms_by_kind(*kind_enum) {
            if let Some(content) = store.atom_content(atom) {
                let confidence = atom.header.confidence.base;
                let ihsan = (confidence * 10000.0) as u32;
                let ts = atom.header.confidence.last_reinforced;
                // Escape tabs and newlines in content for protocol safety
                let escaped = content
                    .replace('\\', "\\\\")
                    .replace('\t', "\\t")
                    .replace('\n', "\\n");
                writeln!(
                    writer,
                    "TEACH\t{}\t{}\t{}\t{}",
                    kind_name, escaped, ihsan, ts
                )?;
                saved += 1;
            }
        }
    }

    Ok(saved)
}

/// Load compiled reflex rules from `reflex.cache`.
///
/// Returns `(loaded_count, quarantined_count)`.
pub fn load_reflex_cache(node: &mut Node, path: &Path) -> io::Result<(usize, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut loaded = 0usize;
    let mut quarantined = 0usize;
    let mut rules = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !trimmed.starts_with("RULE\t") {
            continue;
        }

        if let Some(rule) = parse_reflex_rule_line(trimmed) {
            if rule.quarantined {
                quarantined += 1;
            }
            loaded += 1;
            rules.push(rule);
        }
    }

    node.runtime_mut().import_reflex_rules(rules);
    Ok((loaded, quarantined))
}

/// Save compiled reflex rules to `reflex.cache`.
pub fn save_reflex_cache(node: &Node, path: &Path) -> io::Result<usize> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# bizra-node reflex cache")?;
    writeln!(writer, "# user: {}", node.user_hash())?;
    writeln!(writer, "# saved: {}", timestamp_now())?;
    writeln!(writer)?;

    let rules = node.runtime().export_reflex_rules();
    let mut saved = 0usize;
    for rule in rules {
        let line = serialize_reflex_rule_line(&rule);
        writeln!(writer, "{}", line)?;
        saved += 1;
    }
    Ok(saved)
}

/// Load hash-chained action receipts from `actions.log`.
///
/// Returns `(loaded_count, rejected_count)`.
pub fn load_action_log(node: &mut Node, path: &Path) -> io::Result<(usize, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut loaded = Vec::new();
    let mut rejected = 0usize;
    let mut expected_prev = [0u8; 32];

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(receipt) = ActionReceipt::from_jsonl(trimmed) else {
            rejected += 1;
            continue;
        };
        if !receipt.verify_chain(&expected_prev) {
            rejected += 1;
            continue;
        }
        expected_prev = receipt.receipt_hash;
        loaded.push(receipt);
    }

    let loaded_count = loaded.len();
    node.action_executor_mut().import_receipts(loaded);
    Ok((loaded_count, rejected))
}

/// Save action receipt history to `actions.log` as compact JSONL.
pub fn save_action_log(node: &Node, path: &Path) -> io::Result<usize> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut saved = 0usize;
    for receipt in node.action_executor().receipts() {
        writeln!(writer, "{}", receipt.to_jsonl())?;
        saved += 1;
    }
    Ok(saved)
}

/// Simple timestamp (seconds since epoch, best-effort).
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn serialize_reflex_rule_line(rule: &ReflexRule) -> String {
    let q_reason = rule
        .quarantine_reason
        .map(|q| q.as_str().to_string())
        .unwrap_or_else(|| "none".to_string());
    let policy_hex: String = rule
        .policy_hash
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    format!(
        "RULE\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        rule.trigger_hash.to_hex(),
        sanitize(&rule.action_template.route_signature),
        sanitize(&rule.action_template.primary_agent),
        rule.compile_ihsan,
        rule.compile_snr,
        rule.compiled_at,
        rule.use_count,
        rule.last_used_at,
        rule.last_validated_at,
        rule.quarantined,
        q_reason,
        policy_hex
    )
}

fn parse_reflex_rule_line(line: &str) -> Option<ReflexRule> {
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() < 13 || parts[0] != "RULE" {
        return None;
    }

    let trigger_raw = parse_hex_32(parts[1])?;
    let policy_raw = parse_hex_32(parts[12])?;
    let compile_ihsan = parts[4].parse::<f32>().ok()?;
    let compile_snr = parts[5].parse::<f32>().ok()?;
    let compiled_at = parts[6].parse::<u64>().ok()?;
    let use_count = parts[7].parse::<u64>().ok()?;
    let last_used_at = parts[8].parse::<u64>().ok()?;
    let last_validated_at = parts[9].parse::<u64>().ok()?;
    let quarantined = parts[10].parse::<bool>().ok()?;
    let quarantine_reason = if parts[11] == "none" {
        None
    } else {
        QuarantineReason::parse(parts[11])
    };

    Some(ReflexRule {
        trigger_hash: TriggerHash(trigger_raw),
        action_template: ActionTemplate {
            route_signature: parts[2].to_string(),
            primary_agent: parts[3].to_string(),
        },
        compile_ihsan,
        compile_snr,
        compiled_at,
        use_count,
        last_used_at,
        last_validated_at,
        quarantined,
        quarantine_reason,
        policy_hash: policy_raw,
    })
}

fn sanitize(s: &str) -> String {
    s.replace(['\t', '\n'], " ")
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeConfig;
    use bizra_agent::action_types::{ActionChannel, ActionKind, ActionReceipt};
    use bizra_agent::reflex_cache::{ActionTemplate, ReflexRule};

    #[test]
    fn state_dir_format() {
        let dir = state_dir(42);
        let dir_str = dir.to_string_lossy();
        assert!(dir_str.contains(".bizra"));
        assert!(dir_str.contains("node-42"));
    }

    #[test]
    fn load_seed_from_empty_file() {
        let dir = std::env::temp_dir().join("bizra-test-load-empty");
        let _ = std::fs::create_dir_all(&dir);
        let seed_path = dir.join("test.seed");

        std::fs::write(&seed_path, "# empty seed\n\n").unwrap();

        let mut node = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });

        let (loaded, errors) = load_seed(&mut node, &seed_path).unwrap();
        assert_eq!(loaded, 0);
        assert_eq!(errors, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_seed_with_commands() {
        let dir = std::env::temp_dir().join("bizra-test-load-cmds");
        let _ = std::fs::create_dir_all(&dir);
        let seed_path = dir.join("test.seed");

        let seed_content = "# seed file\n\
            TEACH\tfact\ttest fact\t9000\t1000\n\
            TEACH\tpreference\ttest pref\t8500\t2000\n\
            # comment line\n\
            \n\
            TEACH\tboguskind\tinvalid\t9000\t3000\n";

        std::fs::write(&seed_path, seed_content).unwrap();

        let mut node = Node::new(NodeConfig {
            show_banner: false,
            ..Default::default()
        });

        let (loaded, errors) = load_seed(&mut node, &seed_path).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(errors, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_state_creates_file() {
        let dir = std::env::temp_dir().join("bizra-test-save");
        let _ = std::fs::create_dir_all(&dir);
        let seed_path = dir.join("save.seed");

        let node = Node::new(NodeConfig {
            show_banner: false,
            ..Default::default()
        });

        let count = save_state(&node, &seed_path).unwrap();
        assert_eq!(count, 0); // fresh node has no atoms to export

        // File should exist and contain header
        let content = std::fs::read_to_string(&seed_path).unwrap();
        assert!(content.contains("bizra-node knowledge seed"));
        assert!(content.contains("user: 1"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn teach_kind_roundtrip_preserves_fidelity() {
        let dir = std::env::temp_dir().join("bizra-test-teach-roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let seed_path = dir.join("roundtrip.seed");

        // Phase 1: TEACH diverse kinds into a node
        let mut node = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });

        let teach_lines = [
            "TEACH\tfact\tFounder and CEO of BIZRA\t9900\t1",
            "TEACH\tpreference\tPrefers Rust over Python\t9500\t2",
            "TEACH\tprinciple\tIhsan excellence standard\t9800\t3",
            "TEACH\tnegation\tNever send data off-device\t9950\t4",
            "TEACH\tgoal\tComplete NODE0 v3 GENESIS\t9700\t5",
            "TEACH\texpertise\tDistributed systems\t9200\t6",
            "TEACH\tpattern\tDeep focused sessions\t9300\t7",
            "TEACH\trelationship\tAlpha-100 co-builders\t9600\t8",
            "TEACH\ttemporal\tPreparing investor materials\t9100\t9",
            "TEACH\tcontext\tGMT+4 Dubai timezone\t9400\t10",
        ];

        for line in &teach_lines {
            let resp = node.execute(line);
            assert!(
                resp.starts_with("OK"),
                "TEACH should succeed: {} -> {}",
                line,
                resp
            );
        }

        // Phase 2: Save state
        let saved = save_state(&node, &seed_path).unwrap();
        assert_eq!(saved, 10, "All 10 atoms should be saved");

        // Phase 3: Read the saved file and verify kind preservation
        let content = std::fs::read_to_string(&seed_path).unwrap();

        // Every kind that was taught should appear in the export
        assert!(content.contains("TEACH\tfact\t"), "fact kind missing");
        assert!(
            content.contains("TEACH\tpreference\t"),
            "preference kind missing"
        );
        assert!(
            content.contains("TEACH\tprinciple\t"),
            "principle kind missing"
        );
        assert!(
            content.contains("TEACH\tnegation\t"),
            "negation kind missing"
        );
        assert!(content.contains("TEACH\tgoal\t"), "goal kind missing");
        assert!(
            content.contains("TEACH\texpertise\t"),
            "expertise kind missing"
        );
        assert!(content.contains("TEACH\tpattern\t"), "pattern kind missing");
        assert!(
            content.contains("TEACH\trelationship\t"),
            "relationship kind missing"
        );
        assert!(
            content.contains("TEACH\ttemporal\t"),
            "temporal kind missing"
        );
        assert!(content.contains("TEACH\tcontext\t"), "context kind missing");

        // Phase 4: Load into a fresh node and verify
        let mut restored = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });
        let (loaded, errors) = load_seed(&mut restored, &seed_path).unwrap();
        assert_eq!(loaded, 10, "All 10 atoms should reload");
        assert_eq!(errors, 0, "No reload errors");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_load_reflex_cache_roundtrip() {
        let dir = std::env::temp_dir().join("bizra-test-reflex-cache");
        let _ = std::fs::create_dir_all(&dir);
        let cache_path = dir.join("reflex.cache");

        let mut node = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });

        let rule = ReflexRule {
            trigger_hash: TriggerHash([1u8; 32]),
            action_template: ActionTemplate {
                route_signature: "tasks=RetrieveContext>GenerateResponse|roles=Scholar>Artisan"
                    .to_string(),
                primary_agent: "Scholar".to_string(),
            },
            compile_ihsan: 0.97,
            compile_snr: 0.93,
            compiled_at: 100,
            use_count: 2,
            last_used_at: 120,
            last_validated_at: 130,
            quarantined: false,
            quarantine_reason: None,
            policy_hash: [0u8; 32],
        };
        node.runtime_mut().import_reflex_rules(vec![rule]);

        let saved = save_reflex_cache(&node, &cache_path).unwrap();
        assert_eq!(saved, 1);

        let mut restored = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });
        let (loaded, _quarantined) = load_reflex_cache(&mut restored, &cache_path).unwrap();
        assert_eq!(loaded, 1);

        let stats = restored.runtime().reflex_stats();
        assert_eq!(stats.size, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_load_action_log_roundtrip() {
        let dir = std::env::temp_dir().join("bizra-test-action-log");
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("actions.log");

        let mut node = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });

        let mut r1 = ActionReceipt {
            action_id: "act_1".to_string(),
            plan_id: "pln_1".to_string(),
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            timestamp: 100,
            result: "ok".to_string(),
            guardian_verdict: true,
            permit_hash: [1u8; 32],
            policy_hash: [2u8; 32],
            receipt_hash: [0u8; 32],
            prev_receipt_hash: [0u8; 32],
        };
        r1.seal();
        let mut r2 = ActionReceipt {
            action_id: "act_2".to_string(),
            plan_id: "pln_1".to_string(),
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::TypeText,
            timestamp: 101,
            result: "ok".to_string(),
            guardian_verdict: true,
            permit_hash: [1u8; 32],
            policy_hash: [2u8; 32],
            receipt_hash: [0u8; 32],
            prev_receipt_hash: r1.receipt_hash,
        };
        r2.seal();
        node.action_executor_mut()
            .import_receipts(vec![r1.clone(), r2.clone()]);

        let saved = save_action_log(&node, &log_path).unwrap();
        assert_eq!(saved, 2);

        let mut restored = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });
        let (loaded, rejected) = load_action_log(&mut restored, &log_path).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(rejected, 0);
        assert_eq!(restored.action_executor().receipts().len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn action_log_tamper_is_rejected_by_hash_chain() {
        let dir = std::env::temp_dir().join("bizra-test-action-log-tamper");
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("actions.log");

        let mut node = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });

        let mut r1 = ActionReceipt {
            action_id: "act_1".to_string(),
            plan_id: "pln_1".to_string(),
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            timestamp: 100,
            result: "ok".to_string(),
            guardian_verdict: true,
            permit_hash: [1u8; 32],
            policy_hash: [2u8; 32],
            receipt_hash: [0u8; 32],
            prev_receipt_hash: [0u8; 32],
        };
        r1.seal();
        let mut r2 = ActionReceipt {
            action_id: "act_2".to_string(),
            plan_id: "pln_1".to_string(),
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::TypeText,
            timestamp: 101,
            result: "ok".to_string(),
            guardian_verdict: true,
            permit_hash: [1u8; 32],
            policy_hash: [2u8; 32],
            receipt_hash: [0u8; 32],
            prev_receipt_hash: r1.receipt_hash,
        };
        r2.seal();
        node.action_executor_mut().import_receipts(vec![r1, r2]);
        let saved = save_action_log(&node, &log_path).unwrap();
        assert_eq!(saved, 2);

        // Tamper with second row without recomputing receipt hash.
        let original = std::fs::read_to_string(&log_path).unwrap();
        let mut lines = original.lines().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        lines[1] = lines[1].replace("\"result\":\"ok\"", "\"result\":\"tampered\"");
        let tampered = format!("{}\n{}\n", lines[0], lines[1]);
        std::fs::write(&log_path, tampered).unwrap();

        let mut restored = Node::new(NodeConfig {
            show_banner: false,
            auto_start_session: false,
            ..Default::default()
        });
        let (loaded, rejected) = load_action_log(&mut restored, &log_path).unwrap();
        assert_eq!(loaded, 1);
        assert_eq!(rejected, 1);
        assert_eq!(restored.action_executor().receipts().len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
