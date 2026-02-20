// bizra-node/src/handler.rs
// ============================================================
// Command Handler — dispatches parsed commands to node state
// ============================================================
//
// The handler is the bridge between the wire protocol (protocol.rs)
// and the living node state (node.rs). Each command maps to one or
// more operations on the AgentRuntime, MemoryPipeline, or IhsanScore.
//
// Pure function: (Command, &mut NodeInternals) -> Response
// No I/O. No side effects beyond mutating the node state passed in.
// ============================================================

use crate::protocol::{Command, ErrorCode, Response, NODE_NAME, NODE_VERSION, PROTOCOL_VERSION};
use bizra_agent::runtime::{AgentRuntime, RuntimeState};
use bizra_agent::types::{Message, MessageId};
use bizra_hooks::IhsanScore;
use bizra_memory::types::FragmentKind;

// ============================================================
// NODE INTERNALS — the mutable state the handler operates on
// ============================================================

/// The internal state of the node, exposed to the handler.
/// This is a borrowed view — the Node struct owns the real data.
pub struct NodeInternals<'a> {
    pub runtime: &'a mut AgentRuntime,
    pub ihsan: &'a mut IhsanScore,
    pub session_counter: &'a mut u64,
    pub message_counter: &'a mut u64,
    pub ihsan_floor: u16,
    pub user_hash: u32,
    pub stopped: &'a mut bool,
}

// ============================================================
// DISPATCH — Command -> Response
// ============================================================

/// Handle a parsed command, mutating node internals and returning
/// a wire-ready Response.
pub fn handle(cmd: Command, state: &mut NodeInternals<'_>) -> Response {
    match cmd {
        Command::Ping => handle_ping(),
        Command::Version => handle_version(),
        Command::Shutdown => handle_shutdown(state),
        Command::Health => handle_health(state),
        Command::Profile => handle_profile(state),
        Command::KnowsMe => handle_knows_me(state),
        Command::Explain { action_hash } => handle_explain(state, &action_hash),
        Command::ReflexStats => handle_reflex_stats(state),
        Command::ReflexInvalidate { trigger_hash } => {
            handle_reflex_invalidate(state, &trigger_hash)
        }
        Command::Ihsan { score } => handle_ihsan(state, score),
        Command::Receive { content, timestamp } => handle_receive(state, &content, timestamp),
        Command::Teach {
            kind,
            content,
            confidence,
            timestamp,
        } => handle_teach(state, &kind, &content, confidence, timestamp),
        Command::Synthesize { timestamp } => handle_synthesize(state, timestamp),
        Command::Query { key } => handle_query(state, &key),
        Command::StartSession { timestamp } => handle_start_session(state, timestamp),
        Command::EndSession { timestamp } => handle_end_session(state, timestamp),
    }
}

// ============================================================
// INDIVIDUAL COMMAND HANDLERS
// ============================================================

fn handle_ping() -> Response {
    Response::ok_single("pong", "true")
}

fn handle_version() -> Response {
    Response::ok(vec![
        ("node", NODE_NAME.to_string()),
        ("version", NODE_VERSION.to_string()),
        ("protocol", PROTOCOL_VERSION.to_string()),
    ])
}

fn handle_shutdown(state: &mut NodeInternals<'_>) -> Response {
    // End any active session gracefully
    let ts = 0; // shutdown timestamp
    state.runtime.end_conversation(ts);
    state.runtime.shutdown(ts);
    *state.stopped = true;
    Response::ok_single("shutdown", "true")
}

fn handle_health(state: &mut NodeInternals<'_>) -> Response {
    let health = state.runtime.health();
    let state_str = match health.state {
        RuntimeState::Ready => "Ready",
        RuntimeState::Degraded => "Degraded",
        RuntimeState::Processing => "Processing",
        RuntimeState::Stopped => "Stopped",
        RuntimeState::Uninitialized => "Uninitialized",
    };
    Response::ok(vec![
        ("state", state_str.to_string()),
        ("ihsan", format!("{}", state.ihsan.raw())),
        ("agents_registered", format!("{}", health.agents_registered)),
        ("agents_active", format!("{}", health.agents_active)),
        (
            "messages_processed",
            format!("{}", health.messages_processed),
        ),
        ("pipeline_fragments", format!("{}", health.fragments_stored)),
        ("pipeline_insights", format!("{}", health.insights_stored)),
        ("knows_me", format!("{:.4}", health.knows_me_score)),
        ("active_session", format!("{}", health.active_session)),
        (
            "total_conversations",
            format!("{}", health.total_conversations),
        ),
        ("total_vetoes", format!("{}", health.total_vetoes)),
        ("reflex_mode", health.reflex_mode.as_str().to_string()),
        ("reflex_rules", format!("{}", health.reflex_rules)),
        ("reflex_hits", format!("{}", health.reflex_hits)),
        ("reflex_misses", format!("{}", health.reflex_misses)),
        (
            "decision_artifacts",
            format!("{}", health.decision_artifacts),
        ),
    ])
}

fn handle_profile(state: &mut NodeInternals<'_>) -> Response {
    let profile = state.runtime.memory().profile();
    Response::ok(vec![
        ("completeness", format!("{:.4}", profile.completeness())),
        ("total_atoms", format!("{}", profile.total_atoms)),
        ("total_insights", format!("{}", profile.total_insights)),
        ("active_atoms", format!("{}", profile.active_atoms)),
        ("sections", format!("{}", profile.section_count())),
        ("confidence", format!("{:.4}", profile.confidence)),
    ])
}

fn handle_knows_me(state: &mut NodeInternals<'_>) -> Response {
    let score = state.runtime.knows_me_score();
    let summary = state.runtime.memory().knowledge_summary();
    Response::ok(vec![
        ("score", format!("{:.4}", score)),
        ("fragments", format!("{}", summary.total_fragments)),
        ("atoms", format!("{}", summary.total_atoms)),
        ("insights", format!("{}", summary.total_insights)),
    ])
}

fn handle_ihsan(state: &mut NodeInternals<'_>, score: u16) -> Response {
    let ihsan = IhsanScore::from_raw(score);
    *state.ihsan = ihsan;
    state.runtime.update_ihsan(ihsan);
    Response::ok(vec![
        ("ihsan", format!("{}", score)),
        ("as_f64", format!("{:.4}", ihsan.as_f64())),
    ])
}

fn handle_receive(state: &mut NodeInternals<'_>, content: &str, timestamp: u64) -> Response {
    *state.message_counter += 1;
    let msg_seq = *state.message_counter as u32;

    let msg = Message::inbound(MessageId::new(msg_seq, 1), content, timestamp, *state.ihsan);
    let result = state.runtime.receive(msg, timestamp);

    Response::ok(vec![
        ("received", "true".to_string()),
        ("agents_consulted", format!("{}", result.agents_consulted)),
        (
            "fragments_extracted",
            format!("{}", result.fragments_extracted),
        ),
        ("guardian_approved", format!("{}", result.guardian_approved)),
        ("knows_me", format!("{:.4}", result.knows_me_score)),
        ("decision_mode", result.decision_mode.as_str().to_string()),
        ("action_hash", result.action_hash),
        ("reflex_hit", format!("{}", result.reflex_hit)),
    ])
}

fn handle_explain(state: &mut NodeInternals<'_>, action_hash: &str) -> Response {
    let Some(artifact) = state.runtime.explain_action(action_hash) else {
        return Response::ok(vec![
            ("found", "false".to_string()),
            ("action_hash", action_hash.to_string()),
        ]);
    };

    let rejected = artifact
        .rejected_alternatives
        .iter()
        .map(|alt| format!("{}:{}", alt.route, alt.reason))
        .collect::<Vec<_>>()
        .join("|");
    let micro_path = artifact.micro_path.join(">");

    Response::ok(vec![
        ("found", "true".to_string()),
        ("action_hash", artifact.action_hash.to_hex()),
        ("trigger_hash", artifact.trigger_hash.to_hex()),
        ("decision_mode", artifact.decision_mode.as_str().to_string()),
        ("mission_phase", artifact.mission_phase.as_str().to_string()),
        ("chosen_route", artifact.chosen_route),
        ("micro_path", micro_path),
        ("guardian_verdict", format!("{}", artifact.guardian_verdict)),
        ("ihsan", format!("{:.4}", artifact.ihsan_at_decision)),
        ("snr", format!("{:.4}", artifact.snr_at_decision)),
        ("timestamp", format!("{}", artifact.timestamp)),
        ("rejected_alternatives", rejected),
    ])
}

fn handle_reflex_stats(state: &mut NodeInternals<'_>) -> Response {
    let stats = state.runtime.reflex_stats();
    let mode = state.runtime.effective_reflex_mode().as_str().to_string();
    Response::ok(vec![
        ("mode", mode),
        ("size", format!("{}", stats.size)),
        ("hits", format!("{}", stats.hits)),
        ("misses", format!("{}", stats.misses)),
        ("compiled", format!("{}", stats.compiled)),
        ("quarantined", format!("{}", stats.quarantined)),
        ("invalidated", format!("{}", stats.invalidated)),
        ("revalidations", format!("{}", stats.revalidations)),
        (
            "revalidation_failures",
            format!("{}", stats.revalidation_failures),
        ),
    ])
}

fn handle_reflex_invalidate(state: &mut NodeInternals<'_>, trigger_hash: &str) -> Response {
    let invalidated = state.runtime.invalidate_reflex(trigger_hash);
    Response::ok(vec![
        ("invalidated", format!("{}", invalidated)),
        ("trigger_hash", trigger_hash.to_string()),
    ])
}

fn handle_teach(
    state: &mut NodeInternals<'_>,
    kind: &str,
    content: &str,
    confidence_raw: u16,
    timestamp: u64,
) -> Response {
    let frag_kind = match teach_kind_to_fragment(kind) {
        Some(k) => k,
        None => {
            return Response::err(ErrorCode::InvalidArg, &format!("unknown kind: {:?}", kind));
        }
    };

    let conf = bizra_memory::Confidence::new(confidence_raw as f32 / 10000.0, timestamp);
    let ok = state.runtime.teach(frag_kind, content, conf, timestamp);

    Response::ok(vec![
        ("taught", format!("{}", ok)),
        ("kind", kind.to_string()),
    ])
}

fn handle_synthesize(state: &mut NodeInternals<'_>, timestamp: u64) -> Response {
    let insights = state.runtime.synthesize(timestamp);
    Response::ok(vec![
        ("synthesized", "true".to_string()),
        ("insights_produced", format!("{}", insights)),
    ])
}

fn handle_query(state: &mut NodeInternals<'_>, key: &str) -> Response {
    // Simple key-based query mapping to atom kinds
    let kind = match key {
        "facts" => bizra_memory::AtomKind::Fact,
        "preferences" => bizra_memory::AtomKind::Preference,
        "goals" => bizra_memory::AtomKind::Goal,
        "patterns" => bizra_memory::AtomKind::Pattern,
        "expertise" => bizra_memory::AtomKind::Expertise,
        "principles" => bizra_memory::AtomKind::Principle,
        "negations" => bizra_memory::AtomKind::Negation,
        _ => {
            return Response::ok(vec![("key", key.to_string()), ("results", "0".to_string())]);
        }
    };

    let results = state.runtime.memory().query_facts(kind, 0);
    let count = results.len();
    Response::ok(vec![
        ("key", key.to_string()),
        ("results", format!("{}", count)),
    ])
}

fn handle_start_session(state: &mut NodeInternals<'_>, timestamp: u64) -> Response {
    let session_id = state.runtime.start_conversation(timestamp);
    *state.session_counter += 1;
    Response::ok(vec![
        ("session_started", "true".to_string()),
        ("session_id", format!("{}", session_id)),
    ])
}

fn handle_end_session(state: &mut NodeInternals<'_>, timestamp: u64) -> Response {
    let insights = state.runtime.end_conversation(timestamp);
    Response::ok(vec![
        ("session_ended", "true".to_string()),
        ("insights_produced", format!("{}", insights.unwrap_or(0))),
    ])
}

// ============================================================
// HELPERS
// ============================================================

/// Map a TEACH kind string to a MemoryPipeline FragmentKind.
///
/// TEACH puts knowledge directly into the memory pipeline, so we map
/// the human-friendly kind names to the fragment kinds the pipeline
/// understands. Most teach commands ingest as UserMessage since the
/// pipeline's rule-based extractor will pull typed atoms from content.
fn teach_kind_to_fragment(kind: &str) -> Option<FragmentKind> {
    match kind {
        "fact" => Some(FragmentKind::Observation),
        "preference" => Some(FragmentKind::UserMessage),
        "pattern" => Some(FragmentKind::Observation),
        "relationship" => Some(FragmentKind::Observation),
        "goal" => Some(FragmentKind::UserMessage),
        "expertise" => Some(FragmentKind::UserMessage),
        "context" => Some(FragmentKind::Observation),
        "principle" => Some(FragmentKind::UserMessage),
        "temporal" => Some(FragmentKind::Observation),
        "negation" => Some(FragmentKind::UserMessage),
        _ => None,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_internals() -> (AgentRuntime, IhsanScore, u64, u64, bool) {
        let runtime = AgentRuntime::for_user(1);
        let ihsan = IhsanScore::from_raw(9900);
        (runtime, ihsan, 0, 0, false)
    }

    fn with_internals<'a>(
        rt: &'a mut AgentRuntime,
        ihsan: &'a mut IhsanScore,
        sc: &'a mut u64,
        mc: &'a mut u64,
        stopped: &'a mut bool,
    ) -> NodeInternals<'a> {
        NodeInternals {
            runtime: rt,
            ihsan,
            session_counter: sc,
            message_counter: mc,
            ihsan_floor: 9500,
            user_hash: 1,
            stopped,
        }
    }

    #[test]
    fn handle_ping_returns_pong() {
        let (mut rt, mut ih, mut sc, mut mc, mut st) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st);
        let resp = handle(Command::Ping, &mut state);
        assert_eq!(resp.to_wire(), "OK\tpong=true");
    }

    #[test]
    fn handle_version_contains_name() {
        let (mut rt, mut ih, mut sc, mut mc, mut st) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st);
        let resp = handle(Command::Version, &mut state);
        let wire = resp.to_wire();
        assert!(wire.contains("bizra-node"));
        assert!(wire.contains("0.1.0"));
    }

    #[test]
    fn handle_shutdown_sets_stopped() {
        let (mut rt, mut ih, mut sc, mut mc, mut st) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st);
        let resp = handle(Command::Shutdown, &mut state);
        assert!(resp.to_wire().contains("shutdown=true"));
        assert!(st);
    }
}
