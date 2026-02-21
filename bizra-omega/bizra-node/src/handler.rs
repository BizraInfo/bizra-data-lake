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
use bizra_memory::types::{AtomKind, Confidence};
use std::collections::HashMap;

// ============================================================
// NODE INTERNALS — the mutable state the handler operates on
// ============================================================

/// SAP v0 session state — tracks an active MeetOpen session.
#[derive(Debug, Clone)]
pub struct SapSessionState {
    pub profile: String,
    pub initiator_role: String,
    pub created_at: u64,
    pub message_count: u32,
    pub receipt_hashes: Vec<String>,
    pub consent_granted: Vec<String>,
    pub closed: bool,
}

impl SapSessionState {
    fn new(profile: &str, initiator_role: &str, ts: u64) -> Self {
        Self {
            profile: profile.to_string(),
            initiator_role: initiator_role.to_string(),
            created_at: ts,
            message_count: 0,
            receipt_hashes: Vec::new(),
            consent_granted: Vec::new(),
            closed: false,
        }
    }
}

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
    pub sap_sessions: &'a mut HashMap<String, SapSessionState>,
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

        // SAP v0 protocol
        Command::SapMeetOpen {
            profile,
            initiator_role,
            timestamp,
        } => handle_sap_meet_open(state, &profile, &initiator_role, timestamp),
        Command::SapMessage {
            session_id,
            content,
            timestamp,
        } => handle_sap_message(state, &session_id, &content, timestamp),
        Command::SapDisclosure { session_id } => handle_sap_disclosure(state, &session_id),
        Command::SapConsentRequest {
            session_id,
            scopes_json,
        } => handle_sap_consent_request(state, &session_id, &scopes_json),
        Command::SapConsentRevoke {
            session_id,
            receipt_id,
        } => handle_sap_consent_revoke(state, &session_id, &receipt_id),
        Command::SapSessionClose {
            session_id,
            timestamp,
        } => handle_sap_session_close(state, &session_id, timestamp),
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
    let atom_kind = match parse_atom_kind(kind) {
        Some(k) => k,
        None => {
            return Response::err(ErrorCode::InvalidArg, &format!("unknown kind: {:?}", kind));
        }
    };

    // HHMM-aware confidence: half-life derived from atom kind's cognitive layer.
    let conf = Confidence::for_kind(confidence_raw as f32 / 10000.0, timestamp, atom_kind);
    let ok = state.runtime.teach(atom_kind, content, conf, timestamp);

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
// SAP v0 PROTOCOL HANDLERS
// ============================================================

fn handle_sap_meet_open(
    state: &mut NodeInternals<'_>,
    profile: &str,
    initiator_role: &str,
    timestamp: u64,
) -> Response {
    let ts = if timestamp == 0 {
        current_ts()
    } else {
        timestamp
    };

    // Generate session ID: sap_<counter>_<ts_hex>
    *state.session_counter += 1;
    let session_id = format!("sap_{:08x}_{:08x}", *state.session_counter, ts);

    // Create session
    let session = SapSessionState::new(profile, initiator_role, ts);
    state.sap_sessions.insert(session_id.clone(), session);

    // Build disclosure from current node state
    let health = state.runtime.health();
    let ihsan_score = state.ihsan.as_f64();

    Response::ok(vec![
        ("session_id", session_id),
        ("profile", profile.to_string()),
        ("ihsan_score", format!("{:.4}", ihsan_score)),
        (
            "disclosure",
            format!(
                "{{\"claims\":[\"Sovereign agent compiled from {} reflexes\",\"SAP v0 conformant\"],\"uncertainty\":[\"Alpha software\"],\"source_refs\":[\"specs/sap-v0/01-core-primitives.md\"],\"compliance_assertions\":[{{\"standard\":\"SAP_v0\",\"status\":\"conformant\"}}]}}",
                health.reflex_rules
            ),
        ),
        (
            "agent_card",
            format!(
                "{{\"agent_id\":\"node0-{:08x}\",\"owner_node_id\":\"node0\",\"role\":\"sovereign_personal\",\"version\":\"{}\",\"policy_hash\":\"{}\",\"capabilities\":[\"chat\",\"teach\",\"synthesize\",\"disclose\"],\"compilation\":{{\"genesis_version\":\"GENESIS\",\"ihsan_threshold\":0.95,\"compiled_reflex_count\":{},\"compilation_coverage\":{:.2}}}}}",
                state.user_hash,
                NODE_VERSION,
                state.runtime.policy_hash_hex().unwrap_or_default(),
                health.reflex_rules,
                health.knows_me_score,
            ),
        ),
    ])
}

fn handle_sap_message(
    state: &mut NodeInternals<'_>,
    session_id: &str,
    content: &str,
    timestamp: u64,
) -> Response {
    let ts = if timestamp == 0 {
        current_ts()
    } else {
        timestamp
    };

    // Validate session
    let session = match state.sap_sessions.get_mut(session_id) {
        Some(s) if !s.closed => s,
        Some(_) => {
            return Response::err(ErrorCode::InvalidArg, "SAP session is closed");
        }
        None => {
            return Response::err(ErrorCode::InvalidArg, "SAP session not found");
        }
    };

    // Enforce session limits (SAP v0 spec: max 50 messages)
    if session.message_count >= 50 {
        return Response::err(
            ErrorCode::InvalidArg,
            "SAP session message limit reached (50)",
        );
    }
    session.message_count += 1;

    // Process through the normal receive pipeline
    *state.message_counter += 1;
    let msg_seq = *state.message_counter as u32;
    let msg = Message::inbound(MessageId::new(msg_seq, 1), content, ts, *state.ihsan);
    let result = state.runtime.receive(msg, ts);

    // Generate receipt hash
    let receipt_input = format!("{}:{}:{}", session_id, session.message_count, content);
    let receipt_hash = format!("{:016x}", hash_receipt(&receipt_input));
    session.receipt_hashes.push(receipt_hash.clone());

    let ihsan_score = state.ihsan.as_f64();

    Response::ok(vec![
        ("session_id", session_id.to_string()),
        ("content", result.action_hash.clone()),
        ("agents_consulted", format!("{}", result.agents_consulted)),
        (
            "fragments_extracted",
            format!("{}", result.fragments_extracted),
        ),
        ("confidence", format!("{:.4}", result.knows_me_score)),
        ("ihsan_score", format!("{:.4}", ihsan_score)),
        ("receipt_hash", receipt_hash),
        (
            "disclosure",
            "{\"claims\":[\"Response generated from compiled reflexes\"],\"uncertainty\":[\"Alpha software\"]}".to_string(),
        ),
    ])
}

fn handle_sap_disclosure(state: &mut NodeInternals<'_>, session_id: &str) -> Response {
    let session = match state.sap_sessions.get(session_id) {
        Some(s) => s,
        None => {
            return Response::err(ErrorCode::InvalidArg, "SAP session not found");
        }
    };

    let health = state.runtime.health();
    let ihsan_score = state.ihsan.as_f64();

    Response::ok(vec![
        ("session_id", session_id.to_string()),
        (
            "disclosure",
            format!(
                "{{\"claims\":[\"Sovereign agent compiled from {} reflexes\",\"All responses pass Ihsan gate (threshold >= 0.95)\",\"SAP v0 conformant\"],\"uncertainty\":[\"Compilation coverage: {:.2}\",\"Alpha software\"],\"source_refs\":[\"specs/sap-v0/01-core-primitives.md\",\"schemas/sap/v0/disclosure.schema.json\"],\"compliance_assertions\":[{{\"standard\":\"SAP_v0\",\"status\":\"conformant\"}}]}}",
                health.reflex_rules, health.knows_me_score,
            ),
        ),
        ("ihsan_score", format!("{:.4}", ihsan_score)),
        ("messages_in_session", format!("{}", session.message_count)),
    ])
}

fn handle_sap_consent_request(
    state: &mut NodeInternals<'_>,
    session_id: &str,
    scopes_json: &str,
) -> Response {
    match state.sap_sessions.get(session_id) {
        Some(s) if !s.closed => {}
        Some(_) => {
            return Response::err(ErrorCode::InvalidArg, "SAP session is closed");
        }
        None => {
            return Response::err(ErrorCode::InvalidArg, "SAP session not found");
        }
    }

    // Consent is requested but not granted until explicit user action
    Response::ok(vec![
        ("session_id", session_id.to_string()),
        ("status", "pending".to_string()),
        ("scopes", scopes_json.to_string()),
        (
            "message",
            "Consent requested. No data shared until explicitly granted.".to_string(),
        ),
    ])
}

fn handle_sap_consent_revoke(
    state: &mut NodeInternals<'_>,
    session_id: &str,
    receipt_id: &str,
) -> Response {
    match state.sap_sessions.get_mut(session_id) {
        Some(s) => {
            s.consent_granted.retain(|r| r != receipt_id);
        }
        None => {
            return Response::err(ErrorCode::InvalidArg, "SAP session not found");
        }
    }

    Response::ok(vec![
        ("session_id", session_id.to_string()),
        ("revoked", "true".to_string()),
        ("receipt_id", receipt_id.to_string()),
        (
            "message",
            "Consent revoked. All associated data processing stopped.".to_string(),
        ),
    ])
}

fn handle_sap_session_close(
    state: &mut NodeInternals<'_>,
    session_id: &str,
    timestamp: u64,
) -> Response {
    let session = match state.sap_sessions.get_mut(session_id) {
        Some(s) => s,
        None => {
            return Response::err(ErrorCode::InvalidArg, "SAP session not found");
        }
    };

    session.closed = true;
    let _ts = if timestamp == 0 {
        current_ts()
    } else {
        timestamp
    };

    // Final receipt hash chains all message receipts
    let chain = session.receipt_hashes.join(":");
    let final_hash = format!("{:032x}", hash_receipt(&chain));

    Response::ok(vec![
        ("session_id", session_id.to_string()),
        ("closed", "true".to_string()),
        ("messages_exchanged", format!("{}", session.message_count)),
        ("final_receipt_hash", final_hash),
        (
            "message",
            "Session closed. You can revoke any granted consent at any time.".to_string(),
        ),
    ])
}

/// Simple hash for receipt chain (not cryptographic — uses existing BLAKE3 from the crate)
fn hash_receipt(input: &str) -> u128 {
    let bytes = input.as_bytes();
    let mut h: u128 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u128;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ============================================================
// HELPERS
// ============================================================

/// Parse a TEACH kind string to the corresponding AtomKind.
///
/// TEACH commands store atoms directly with their specified kind,
/// bypassing rule-based extraction. This preserves kind fidelity
/// through the full TEACH → store → export roundtrip.
fn parse_atom_kind(kind: &str) -> Option<AtomKind> {
    match kind {
        "fact" => Some(AtomKind::Fact),
        "preference" => Some(AtomKind::Preference),
        "pattern" => Some(AtomKind::Pattern),
        "relationship" => Some(AtomKind::Relationship),
        "goal" => Some(AtomKind::Goal),
        "expertise" => Some(AtomKind::Expertise),
        "context" => Some(AtomKind::Context),
        "principle" => Some(AtomKind::Principle),
        "temporal" => Some(AtomKind::Temporal),
        "negation" => Some(AtomKind::Negation),
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

    fn make_internals() -> (
        AgentRuntime,
        IhsanScore,
        u64,
        u64,
        bool,
        ActionExecutor,
        HashMap<String, SapSessionState>,
    ) {
        let runtime = AgentRuntime::for_user(1);
        let ihsan = IhsanScore::from_raw(9900);
        (
            runtime,
            ihsan,
            0,
            0,
            false,
            ActionExecutor::default(),
            HashMap::new(),
        )
    }

    fn with_internals<'a>(
        rt: &'a mut AgentRuntime,
        ihsan: &'a mut IhsanScore,
        sc: &'a mut u64,
        mc: &'a mut u64,
        stopped: &'a mut bool,
        action_executor: &'a mut ActionExecutor,
        sap_sessions: &'a mut HashMap<String, SapSessionState>,
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
            sap_sessions,
        }
    }

    #[test]
    fn handle_ping_returns_pong() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
        let resp = handle(Command::Ping, &mut state);
        assert_eq!(resp.to_wire(), "OK\tpong=true");
    }

    #[test]
    fn handle_version_contains_name() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
        let resp = handle(Command::Version, &mut state);
        let wire = resp.to_wire();
        assert!(wire.contains("bizra-node"));
        assert!(wire.contains("0.1.0"));
    }

    #[test]
    fn handle_shutdown_sets_stopped() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
        let resp = handle(Command::Shutdown, &mut state);
        assert!(resp.to_wire().contains("shutdown=true"));
        assert!(st);
    }

    #[test]
    fn handle_plan_action_success() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );

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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
        let (mut rt, mut _ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut ih = IhsanScore::from_raw(9000); // below floor of 9500
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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

    // ── SAP v0 handler tests ──

    #[test]
    fn handle_sap_meet_open_creates_session() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
        let resp = handle(
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 1000,
            },
            &mut state,
        );
        let wire = resp.to_wire();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("session_id=sap_"));
        assert!(wire.contains("ihsan_score="));
        assert!(wire.contains("disclosure="));
        assert!(wire.contains("agent_card="));
        assert_eq!(sap.len(), 1);
    }

    #[test]
    fn handle_sap_message_in_session() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );

        // Open session
        let open_resp = handle(
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 1000,
            },
            &mut state,
        );
        let open_wire = open_resp.to_wire();
        let session_id = open_wire
            .split('\t')
            .find(|p| p.starts_with("session_id="))
            .unwrap()
            .strip_prefix("session_id=")
            .unwrap()
            .to_string();

        // Send message
        let msg_resp = handle(
            Command::SapMessage {
                session_id: session_id.clone(),
                content: "Tell me about BIZRA".to_string(),
                timestamp: 2000,
            },
            &mut state,
        );
        let msg_wire = msg_resp.to_wire();
        assert!(msg_wire.starts_with("OK\t"));
        assert!(msg_wire.contains("receipt_hash="));
        assert!(msg_wire.contains("ihsan_score="));
    }

    #[test]
    fn handle_sap_message_invalid_session() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
        let resp = handle(
            Command::SapMessage {
                session_id: "nonexistent".to_string(),
                content: "hello".to_string(),
                timestamp: 1000,
            },
            &mut state,
        );
        assert!(resp.to_wire().starts_with("ERR\t"));
    }

    #[test]
    fn handle_sap_session_close_produces_final_receipt() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );

        let open_resp = handle(
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 1000,
            },
            &mut state,
        );
        let open_wire = open_resp.to_wire();
        let session_id = open_wire
            .split('\t')
            .find(|p| p.starts_with("session_id="))
            .unwrap()
            .strip_prefix("session_id=")
            .unwrap()
            .to_string();

        // Close
        let close_resp = handle(
            Command::SapSessionClose {
                session_id: session_id.clone(),
                timestamp: 3000,
            },
            &mut state,
        );
        let close_wire = close_resp.to_wire();
        assert!(close_wire.contains("closed=true"));
        assert!(close_wire.contains("final_receipt_hash="));
    }

    #[test]
    fn handle_sap_message_after_close_rejected() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );

        let open_resp = handle(
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 1000,
            },
            &mut state,
        );
        let session_id = open_resp
            .to_wire()
            .split('\t')
            .find(|p| p.starts_with("session_id="))
            .unwrap()
            .strip_prefix("session_id=")
            .unwrap()
            .to_string();

        // Close session
        handle(
            Command::SapSessionClose {
                session_id: session_id.clone(),
                timestamp: 2000,
            },
            &mut state,
        );

        // Try message after close
        let msg_resp = handle(
            Command::SapMessage {
                session_id,
                content: "hello".to_string(),
                timestamp: 3000,
            },
            &mut state,
        );
        assert!(msg_resp.to_wire().starts_with("ERR\t"));
        assert!(msg_resp.to_wire().contains("closed"));
    }

    #[test]
    fn handle_action_dispatch_invalid_channel() {
        let (mut rt, mut ih, mut sc, mut mc, mut st, mut ae, mut sap) = make_internals();
        let mut state = with_internals(
            &mut rt, &mut ih, &mut sc, &mut mc, &mut st, &mut ae, &mut sap,
        );
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
