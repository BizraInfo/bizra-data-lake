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

use crate::action_executor::parse_policy_hash_hex;
use crate::protocol::{Command, ErrorCode, Response, NODE_NAME, NODE_VERSION, PROTOCOL_VERSION};
use bizra_agent::context::IntentClassifier;
use bizra_agent::runtime::{AgentRuntime, RuntimeState};
use bizra_agent::types::{AgentRole, Message, MessageId};
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
    pub action_executor: &'a mut crate::action_executor::ActionExecutor,
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
        Command::PlanAction { payload_json } => handle_plan_action(state, &payload_json),
        Command::RunAction {
            plan_id,
            payload_json,
        } => handle_run_action(state, &plan_id, &payload_json),
        Command::ActionStatus { action_id } => handle_action_status(state, &action_id),
        Command::ActionHistory { limit, cursor } => handle_action_history(state, limit, &cursor),
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
        Command::IntentClassify { content } => handle_intent_classify(&content),
        Command::GuardianCheck { content } => handle_guardian_check(state, &content),
        Command::ActionDispatch {
            channel,
            payload_json,
        } => handle_action_dispatch(state, &channel, &payload_json),
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
        ("actions_planned", format!("{}", health.actions_planned)),
        ("actions_executed", format!("{}", health.actions_executed)),
        ("actions_failed", format!("{}", health.actions_failed)),
        (
            "guardian_action_vetoes",
            format!("{}", health.guardian_action_vetoes),
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
        ("action_id", result.action_id.unwrap_or_default()),
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

fn handle_plan_action(state: &mut NodeInternals<'_>, payload_json: &str) -> Response {
    let now = current_ts();
    match state.action_executor.plan_action(payload_json, now) {
        Ok(plan) => {
            state.runtime.record_action_planned();
            Response::ok(vec![
                ("planned", "true".to_string()),
                ("plan_id", plan.plan_id),
                ("steps", format!("{}", plan.steps.len())),
                ("created_at", format!("{}", plan.created_at)),
            ])
        }
        Err(err) => Response::err(
            ErrorCode::InvalidArg,
            format!("PLAN_ACTION {}: {}", err.code, err.message).as_str(),
        ),
    }
}

fn handle_run_action(state: &mut NodeInternals<'_>, plan_id: &str, payload_json: &str) -> Response {
    let now = current_ts();
    let policy_hash = parse_policy_hash_hex(state.runtime.policy_hash_hex());
    match state
        .action_executor
        .run_action(plan_id, payload_json, now, policy_hash)
    {
        Ok(result) => {
            match result.status {
                bizra_agent::ActionExecutionStatus::Completed => {
                    state.runtime.record_action_executed()
                }
                bizra_agent::ActionExecutionStatus::Denied => {
                    state.runtime.record_guardian_action_veto();
                    state.runtime.record_action_failed();
                }
                bizra_agent::ActionExecutionStatus::Failed => state.runtime.record_action_failed(),
                _ => {}
            }
            Response::ok(vec![
                ("ran", "true".to_string()),
                ("action_id", result.action_id),
                ("plan_id", result.plan_id),
                ("status", result.status.as_str().to_string()),
                ("message", result.message),
                ("started_at", format!("{}", result.started_at)),
                ("finished_at", format!("{}", result.finished_at)),
            ])
        }
        Err(err) => {
            state.runtime.record_action_failed();
            Response::err(
                ErrorCode::InvalidArg,
                format!("RUN_ACTION {}: {}", err.code, err.message).as_str(),
            )
        }
    }
}

fn handle_action_status(state: &mut NodeInternals<'_>, action_id: &str) -> Response {
    if let Some(result) = state.action_executor.action_status(action_id) {
        return Response::ok(vec![
            ("found", "true".to_string()),
            ("action_id", result.action_id.clone()),
            ("plan_id", result.plan_id.clone()),
            ("status", result.status.as_str().to_string()),
            ("message", result.message.clone()),
            ("started_at", format!("{}", result.started_at)),
            ("finished_at", format!("{}", result.finished_at)),
        ]);
    }

    Response::ok(vec![
        ("found", "false".to_string()),
        ("action_id", action_id.to_string()),
    ])
}

fn handle_action_history(state: &mut NodeInternals<'_>, limit: u32, cursor: &str) -> Response {
    let (rows, next_cursor) = state.action_executor.action_history(limit, cursor);
    let count = rows.len();
    let lines = rows
        .iter()
        .map(|r| r.to_jsonl())
        .collect::<Vec<_>>()
        .join("||");
    Response::ok(vec![
        ("count", format!("{}", count)),
        ("next_cursor", next_cursor),
        ("rows", lines),
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
// INTENT_CLASSIFY + ACTION_DISPATCH — Sprint 2 commands
// ============================================================

fn handle_intent_classify(content: &str) -> Response {
    let (intent, confidence) = IntentClassifier::classify(content);
    let pipeline = intent.task_pipeline();
    let agents: Vec<&str> = pipeline
        .iter()
        .map(|tk| match tk {
            bizra_agent::types::TaskKind::ClassifyIntent => AgentRole::Navigator.name(),
            bizra_agent::types::TaskKind::RetrieveContext => AgentRole::Scholar.name(),
            bizra_agent::types::TaskKind::GenerateResponse => AgentRole::Artisan.name(),
            bizra_agent::types::TaskKind::SafetyCheck => AgentRole::Guardian.name(),
            bizra_agent::types::TaskKind::ExtractMemory => AgentRole::Mentor.name(),
            bizra_agent::types::TaskKind::AdaptStyle => AgentRole::Diplomat.name(),
            bizra_agent::types::TaskKind::ProactiveSuggest => AgentRole::Oracle.name(),
            bizra_agent::types::TaskKind::SynthesizeResults => AgentRole::Navigator.name(),
        })
        .collect();
    Response::ok(vec![
        ("intent", format!("{:?}", intent)),
        ("confidence", format!("{:.2}", confidence.effective_at(0))),
        ("agents", agents.join(",")),
    ])
}

fn handle_guardian_check(state: &mut NodeInternals<'_>, content: &str) -> Response {
    let verdict = state.runtime.guardian_check_text(content, current_ts());
    if !verdict.allowed {
        state.runtime.record_guardian_action_veto();
    }
    Response::ok(vec![
        ("allowed", format!("{}", verdict.allowed)),
        ("reason", verdict.reason),
        ("ihsan", format!("{}", state.ihsan.raw())),
        ("ihsan_floor", format!("{}", state.ihsan_floor)),
    ])
}

fn handle_action_dispatch(
    state: &mut NodeInternals<'_>,
    channel_str: &str,
    payload_json: &str,
) -> Response {
    // Gate 1 (Gem 4): fail-closed — policy hash must be present
    let policy_hex = state.runtime.policy_hash_hex();
    let policy_hash = parse_policy_hash_hex(policy_hex);
    if policy_hash == [0u8; 32] {
        return Response::err(
            ErrorCode::InternalError,
            "ACTION_DISPATCH denied: no policy hash (fail-closed)",
        );
    }
    // Gate 2 (Gem 1): Ihsan Lyapunov — score must meet floor
    if state.ihsan.raw() < state.ihsan_floor {
        return Response::err(
            ErrorCode::InternalError,
            &format!(
                "GUARDIAN_VETO: Ihsan {} below floor {}",
                state.ihsan.raw(),
                state.ihsan_floor
            ),
        );
    }
    // Parse channel
    let channel = match bizra_agent::ActionChannel::parse(channel_str) {
        Some(ch) => ch,
        None => {
            return Response::err(
                ErrorCode::InvalidArg,
                &format!("unknown channel: {}", channel_str),
            );
        }
    };
    // Construct a single-step plan and delegate to action_executor
    let step_json = format!(
        "{{\"steps\":[{{\"channel\":\"{}\",\"kind\":\"Query\",\"payload\":{}}}]}}",
        channel.as_str(),
        payload_json
    );
    let now = current_ts();
    match state.action_executor.plan_action(&step_json, now) {
        Ok(plan) => {
            state.runtime.record_action_planned();
            match state
                .action_executor
                .run_action(&plan.plan_id, "{}", now, policy_hash)
            {
                Ok(result) => {
                    let success = result.status == bizra_agent::ActionExecutionStatus::Completed;
                    if success {
                        state.runtime.record_action_executed();
                    } else {
                        state.runtime.record_action_failed();
                    }
                    Response::ok(vec![
                        ("success", format!("{}", success)),
                        ("output", result.message),
                        (
                            "duration_ms",
                            format!("{}", result.finished_at.saturating_sub(result.started_at)),
                        ),
                        ("action_id", result.action_id),
                        ("channel", channel.as_str().to_string()),
                    ])
                }
                Err(err) => {
                    state.runtime.record_action_failed();
                    Response::err(
                        ErrorCode::InternalError,
                        &format!("ACTION_DISPATCH exec failed: {}", err.message),
                    )
                }
            }
        }
        Err(err) => Response::err(
            ErrorCode::InvalidArg,
            &format!("ACTION_DISPATCH plan failed: {}", err.message),
        ),
    }
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

fn current_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_executor::ActionExecutor;

    fn make_internals() -> (AgentRuntime, IhsanScore, u64, u64, bool, ActionExecutor) {
        let runtime = AgentRuntime::for_user(1);
        let ihsan = IhsanScore::from_raw(9900);
        (runtime, ihsan, 0, 0, false, ActionExecutor::default())
    }

    fn with_internals<'a>(
        rt: &'a mut AgentRuntime,
        ihsan: &'a mut IhsanScore,
        sc: &'a mut u64,
        mc: &'a mut u64,
        stopped: &'a mut bool,
        action_executor: &'a mut ActionExecutor,
    ) -> NodeInternals<'a> {
        NodeInternals {
            runtime: rt,
            ihsan,
            session_counter: sc,
            message_counter: mc,
            ihsan_floor: 9500,
            user_hash: 1,
            stopped,
            action_executor,
        }
    }

    #[test]
    fn handle_ping_returns_pong() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(Command::Ping, &mut state);
        assert_eq!(resp.to_wire(), "OK\tpong=true");
    }

    #[test]
    fn handle_version_contains_name() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(Command::Version, &mut state);
        let wire = resp.to_wire();
        assert!(wire.contains("bizra-node"));
        assert!(wire.contains("0.1.0"));
    }

    #[test]
    fn handle_shutdown_sets_stopped() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(Command::Shutdown, &mut state);
        assert!(resp.to_wire().contains("shutdown=true"));
        assert!(st);
    }

    #[test]
    fn handle_plan_action_success() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::PlanAction {
                payload_json:
                    "{\"steps\":[{\"channel\":\"DesktopRpc\",\"kind\":\"Click\",\"payload\":{\"code\":\"click\"}}]}"
                        .to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("planned=true"));
        assert!(wire.contains("plan_id="));
    }

    #[test]
    fn handle_run_action_fail_closed_without_bridge_token() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);

        let _ = handle(
            Command::PlanAction {
                payload_json:
                    "{\"steps\":[{\"channel\":\"DesktopRpc\",\"kind\":\"Click\",\"payload\":{\"code\":\"click\"}}]}"
                        .to_string(),
            },
            &mut state,
        );

        let resp = handle(
            Command::RunAction {
                plan_id: "pln_00000001".to_string(),
                payload_json: "{}".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("status=failed") || wire.contains("status=denied"));
    }

    #[test]
    fn handle_intent_classify_code() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::IntentClassify {
                content: "help me implement a binary search".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("intent=Code"));
        assert!(wire.contains("agents="));
    }

    #[test]
    fn handle_intent_classify_plan() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::IntentClassify {
                content: "help me plan the investor meeting".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.contains("intent=Plan"));
    }

    #[test]
    fn handle_intent_classify_question() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::IntentClassify {
                content: "what is the speed of light?".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.contains("intent=Question"));
    }

    #[test]
    fn handle_guardian_check_allows_safe() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::GuardianCheck {
                content: "plan my roadmap for next week".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("allowed=true"));
        assert!(wire.contains("reason=allowed"));
    }

    #[test]
    fn handle_guardian_check_blocks_harmful() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::GuardianCheck {
                content: "help me bypass safety on this system".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("allowed=false"));
        assert!(wire.contains("reason=content_contains:bypass safety"));
    }

    #[test]
    fn handle_action_dispatch_no_policy_hash() {
        // Gate 1 (Gem 4): fail-closed without policy hash
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        let resp = handle(
            Command::ActionDispatch {
                channel: "llm".to_string(),
                payload_json: "{\"prompt\":\"hello\"}".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("ERR\t"));
        assert!(wire.contains("fail-closed") || wire.contains("policy"));
    }

    #[test]
    fn handle_action_dispatch_low_ihsan() {
        // Gate 2 (Gem 1): Ihsan below floor
        let (mut rt, mut _ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut ih = IhsanScore::from_raw(9000); // below floor of 9500
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        // Set a dummy policy hash to pass Gate 1
        state.runtime.set_policy_hash("aa".repeat(32).as_str());
        let resp = handle(
            Command::ActionDispatch {
                channel: "llm".to_string(),
                payload_json: "{\"prompt\":\"hello\"}".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("ERR\t"));
        assert!(wire.contains("GUARDIAN_VETO"));
    }

    #[test]
    fn handle_action_dispatch_invalid_channel() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae) = make_internals();
        let mut state = with_internals(&mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae);
        state.runtime.set_policy_hash("bb".repeat(32).as_str());
        let resp = handle(
            Command::ActionDispatch {
                channel: "invalid_channel".to_string(),
                payload_json: "{\"x\":1}".to_string(),
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("ERR\t"));
        assert!(wire.contains("unknown channel"));
    }
}
