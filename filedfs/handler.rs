// bizra-node/src/handler.rs
// ============================================================
// Command Handler — the dispatcher
// ============================================================

use crate::protocol::{Command, Response, ErrorCode, PROTOCOL_VERSION, NODE_VERSION, NODE_NAME};
use bizra_agent::runtime::AgentRuntime;
use bizra_agent::types::{Message, MessageId};
use bizra_memory::types::{FragmentKind, Confidence};

// ============================================================
// CORE DISPATCH
// ============================================================

pub fn handle(cmd: Command, runtime: &mut AgentRuntime) -> Response {
    match cmd {
        Command::Receive { content, timestamp } => handle_receive(runtime, &content, timestamp),
        Command::Teach { kind, content, confidence, timestamp } => {
            handle_teach(runtime, &kind, &content, confidence, timestamp)
        }
        Command::Synthesize { timestamp } => handle_synthesize(runtime, timestamp),
        Command::Query { key } => handle_query(runtime, &key),
        Command::Profile => handle_profile(runtime),
        Command::KnowsMe => handle_knows_me(runtime),
        Command::Health => handle_health(runtime),
        Command::StartSession { timestamp } => handle_start_session(runtime, timestamp),
        Command::EndSession { timestamp } => handle_end_session(runtime, timestamp),
        Command::Ihsan { score } => handle_ihsan(runtime, score),
        Command::Shutdown => handle_shutdown(runtime),
        Command::Ping => handle_ping(),
        Command::Version => handle_version(),
    }
}

// ============================================================
// COMMAND HANDLERS
// ============================================================

fn handle_receive(runtime: &mut AgentRuntime, content: &str, timestamp: u64) -> Response {
    if content.is_empty() {
        return Response::err(ErrorCode::InvalidArg, "Content cannot be empty");
    }

    let seq = (timestamp & 0xFFFF) as u32;
    let msg_id = MessageId::new(1, seq);
    let ihsan = runtime.current_ihsan();
    let message = Message::inbound(msg_id, content, timestamp, ihsan);

    let result = runtime.receive(message, timestamp);

    let resp_content = result.response.content.as_str();
    let content_str = if resp_content.is_empty() { "[processed]" } else { resp_content };

    Response::ok()
        .field("content", content_str)
        .field("confidence", format!("{:.4}", result.response.confidence.as_f32()))
        .field("agents_consulted", result.agents_consulted)
        .field("fragments_extracted", result.fragments_extracted)
        .field("guardian_approved", result.guardian_approved)
        .field("knows_me", format!("{:.4}", result.knows_me_score))
        .field("session_messages", result.session_messages)
        .field("vetoed", result.response.vetoed)
}

fn handle_teach(runtime: &mut AgentRuntime, kind: &str, content: &str, confidence: u16, timestamp: u64) -> Response {
    if content.is_empty() {
        return Response::err(ErrorCode::InvalidArg, "Content cannot be empty");
    }

    let fragment_kind = match kind.to_lowercase().as_str() {
        "preference" => FragmentKind::Preference,
        "fact" => FragmentKind::Fact,
        "goal" => FragmentKind::Goal,
        "expertise" | "skill" => FragmentKind::Expertise,
        "style" => FragmentKind::Style,
        "pattern" => FragmentKind::Pattern,
        "emotion" => FragmentKind::Emotion,
        "relationship" => FragmentKind::Relationship,
        "temporal" => FragmentKind::Temporal,
        "domain" => FragmentKind::Domain,
        _ => {
            return Response::err(ErrorCode::InvalidArg,
                &format!("Unknown kind: '{}'. Use: preference, fact, goal, expertise, style, pattern, emotion, relationship, temporal, domain", kind));
        }
    };

    runtime.teach(fragment_kind, content, Confidence::new(confidence), timestamp);

    Response::ok()
        .field("taught", content)
        .field("kind", kind)
        .field("confidence", confidence)
}

fn handle_synthesize(runtime: &mut AgentRuntime, timestamp: u64) -> Response {
    let insights = runtime.synthesize(timestamp);
    Response::ok()
        .field("insights_generated", insights)
        .field("knows_me", format!("{:.4}", runtime.knows_me_score()))
}

fn handle_query(runtime: &mut AgentRuntime, key: &str) -> Response {
    if key.is_empty() {
        return Response::err(ErrorCode::InvalidArg, "Key cannot be empty");
    }
    match runtime.query_trait(key) {
        Some((value, confidence)) => {
            Response::ok()
                .field("key", key)
                .field("value", value)
                .field("confidence", confidence.raw())
        }
        None => {
            Response::ok()
                .field("key", key)
                .field("value", "")
                .field("found", false)
        }
    }
}

fn handle_profile(runtime: &mut AgentRuntime) -> Response {
    let traits = runtime.query_profile();
    let trait_count = traits.len();
    let mut resp = Response::ok().field("trait_count", trait_count);
    for (i, (key, value, confidence)) in traits.iter().enumerate() {
        resp = resp.field(&format!("trait_{}", i), &format!("{}={}@{}", key, value, confidence.raw()));
    }
    resp.field("knows_me", format!("{:.4}", runtime.knows_me_score()))
}

fn handle_knows_me(runtime: &mut AgentRuntime) -> Response {
    Response::ok().field("score", format!("{:.4}", runtime.knows_me_score()))
}

fn handle_health(runtime: &mut AgentRuntime) -> Response {
    let health = runtime.health();
    Response::ok()
        .field("state", format!("{:?}", health.state))
        .field("ihsan", health.current_ihsan.raw())
        .field("knows_me", format!("{:.4}", health.knows_me_score))
        .field("agents_registered", health.agents_registered)
        .field("agents_active", health.agents_active)
        .field("total_vetoes", health.total_vetoes)
        .field("messages_processed", health.messages_processed)
        .field("total_tasks", health.total_tasks)
        .field("pipeline_fragments", health.fragments_stored)
        .field("pipeline_insights", health.insights_stored)
        .field("pipeline_traits", health.profile_traits)
        .field("pipeline_synthesis_rounds", health.synthesis_rounds)
        .field("active_session", health.active_session)
        .field("total_conversations", health.total_conversations)
}

fn handle_start_session(runtime: &mut AgentRuntime, timestamp: u64) -> Response {
    let session_id = runtime.start_conversation(timestamp);
    Response::ok().field("session_id", session_id)
}

fn handle_end_session(runtime: &mut AgentRuntime, timestamp: u64) -> Response {
    match runtime.end_conversation(timestamp) {
        Some(insights) => Response::ok().field("ended", true).field("insights_generated", insights),
        None => Response::ok().field("ended", false).field("reason", "no active session"),
    }
}

fn handle_ihsan(runtime: &mut AgentRuntime, score: u16) -> Response {
    if score > 10000 {
        return Response::err(ErrorCode::InvalidArg, "Score must be 0-10000");
    }
    runtime.update_ihsan(bizra_hooks::IhsanScore::new(score));
    Response::ok()
        .field("ihsan", score)
        .field("state", format!("{:?}", runtime.state()))
}

fn handle_shutdown(runtime: &mut AgentRuntime) -> Response {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    runtime.shutdown(now);
    Response::ok().field("shutdown", true)
}

fn handle_ping() -> Response {
    Response::ok().field("pong", true)
}

fn handle_version() -> Response {
    Response::ok()
        .field("node", NODE_NAME)
        .field("version", NODE_VERSION)
        .field("protocol", PROTOCOL_VERSION)
        .field("hooks", "0.1.0")
        .field("memory", "0.1.0")
        .field("agent", "0.1.0")
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_agent::runtime::RuntimeConfig;

    fn make_runtime() -> AgentRuntime {
        let config = RuntimeConfig::default();
        let mut rt = AgentRuntime::new();
        rt.start_conversation(1000);
        rt
    }

    fn assert_ok(resp: &Response) -> &Vec<(String, String)> {
        match resp {
            Response::Ok(fields) => fields,
            Response::Err { code, message } => panic!("Expected OK, got ERR {}: {}", code, message),
        }
    }

    fn get_field<'a>(fields: &'a [(String, String)], key: &str) -> &'a str {
        fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("Field '{}' not found", key))
    }

    #[test]
    fn test_ping() {
        let mut rt = make_runtime();
        let resp = handle(Command::Ping, &mut rt);
        let fields = assert_ok(&resp);
        assert_eq!(get_field(fields, "pong"), "true");
    }

    #[test]
    fn test_version() {
        let mut rt = make_runtime();
        let resp = handle(Command::Version, &mut rt);
        let fields = assert_ok(&resp);
        assert_eq!(get_field(fields, "node"), "bizra-node");
    }

    #[test]
    fn test_health() {
        let mut rt = make_runtime();
        let resp = handle(Command::Health, &mut rt);
        let fields = assert_ok(&resp);
        assert!(!get_field(fields, "state").is_empty());
        assert!(!get_field(fields, "ihsan").is_empty());
    }

    #[test]
    fn test_receive() {
        let mut rt = make_runtime();
        let resp = handle(Command::Receive { content: "I prefer Rust".to_string(), timestamp: 2000 }, &mut rt);
        let fields = assert_ok(&resp);
        assert!(!get_field(fields, "content").is_empty());
    }

    #[test]
    fn test_receive_empty_errors() {
        let mut rt = make_runtime();
        match handle(Command::Receive { content: String::new(), timestamp: 2000 }, &mut rt) {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::InvalidArg),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_teach_and_query() {
        let mut rt = make_runtime();
        assert_ok(&handle(Command::Teach { kind: "preference".to_string(), content: "loves systems programming".to_string(), confidence: 9000, timestamp: 2000 }, &mut rt));
        assert_ok(&handle(Command::Query { key: "preference".to_string() }, &mut rt));
    }

    #[test]
    fn test_teach_invalid_kind() {
        let mut rt = make_runtime();
        match handle(Command::Teach { kind: "invalid".to_string(), content: "test".to_string(), confidence: 9000, timestamp: 2000 }, &mut rt) {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::InvalidArg),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_knows_me() {
        let mut rt = make_runtime();
        let resp = handle(Command::KnowsMe, &mut rt);
        let fields = assert_ok(&resp);
        let score: f32 = get_field(fields, "score").parse().unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_ihsan() {
        let mut rt = make_runtime();
        let resp = handle(Command::Ihsan { score: 9800 }, &mut rt);
        let fields = assert_ok(&resp);
        assert_eq!(get_field(fields, "ihsan"), "9800");
    }

    #[test]
    fn test_ihsan_invalid() {
        let mut rt = make_runtime();
        match handle(Command::Ihsan { score: 10001 }, &mut rt) {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::InvalidArg),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_session_lifecycle() {
        let mut rt = AgentRuntime::new();
        // Start session
        let resp = handle(Command::StartSession { timestamp: 1000 }, &mut rt);
        let fields = assert_ok(&resp);
        assert!(!get_field(fields, "session_id").is_empty());
        // Send a few messages so end_session has fragments
        for i in 0..6 {
            let _ = handle(Command::Teach {
                kind: "fact".to_string(),
                content: format!("fact {}", i),
                confidence: 8000,
                timestamp: 1100 + i,
            }, &mut rt);
        }
        // End session — now has enough fragments for synthesis
        let resp = handle(Command::EndSession { timestamp: 2000 }, &mut rt);
        assert_ok(&resp);
    }

    #[test]
    fn test_profile() {
        let mut rt = make_runtime();
        let resp = handle(Command::Profile, &mut rt);
        let fields = assert_ok(&resp);
        assert!(!get_field(fields, "trait_count").is_empty());
    }

    #[test]
    fn test_shutdown() {
        let mut rt = make_runtime();
        let resp = handle(Command::Shutdown, &mut rt);
        let fields = assert_ok(&resp);
        assert_eq!(get_field(fields, "shutdown"), "true");
    }

    #[test]
    fn test_full_lifecycle() {
        let mut rt = AgentRuntime::new();
        assert_ok(&handle(Command::Version, &mut rt));
        assert_ok(&handle(Command::StartSession { timestamp: 1000 }, &mut rt));
        assert_ok(&handle(Command::Receive { content: "Building distributed AI in Rust".to_string(), timestamp: 2000 }, &mut rt));
        assert_ok(&handle(Command::Teach { kind: "goal".to_string(), content: "democratize AI".to_string(), confidence: 9500, timestamp: 3000 }, &mut rt));
        assert_ok(&handle(Command::KnowsMe, &mut rt));
        let h = handle(Command::Health, &mut rt);
        let fields = assert_ok(&h);
        let msgs: u64 = get_field(fields, "messages_processed").parse().unwrap();
        assert!(msgs >= 1);
        assert_ok(&handle(Command::EndSession { timestamp: 4000 }, &mut rt));
        assert_ok(&handle(Command::Shutdown, &mut rt));
    }
}
