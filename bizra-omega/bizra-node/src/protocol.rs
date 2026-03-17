// bizra-node/src/protocol.rs
// ============================================================
// Wire Protocol — Tab-separated command/response format
// ============================================================
//
// Wire format:
//   Request:  VERB\targ1\targ2\n
//   Response: OK\tfield=value\tfield=value\n
//   Error:    ERR\tcode\tmessage\n
//
// Every command is a single line. Every response is a single line.
// No JSON. No XML. No protobuf. Just tabs and newlines.
// Any process that can read stdin can drive a sovereign node.
// ============================================================

/// Protocol version — bumped on breaking wire format changes.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Node software version.
pub const NODE_VERSION: &str = "0.1.0";

/// Node name identifier.
pub const NODE_NAME: &str = "bizra-node";

// ============================================================
// ERROR CODES
// ============================================================

/// Error codes returned in ERR responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Unknown or unrecognized command verb
    BadCommand,
    /// Command recognized but required argument missing
    MissingArg,
    /// Argument present but could not be parsed (e.g. non-numeric)
    ParseError,
    /// Argument value is invalid (e.g. empty content, unknown kind)
    InvalidArg,
    /// Internal processing error
    InternalError,
}

impl ErrorCode {
    /// Wire representation of this error code.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::BadCommand => "BAD_COMMAND",
            ErrorCode::MissingArg => "MISSING_ARG",
            ErrorCode::ParseError => "PARSE_ERROR",
            ErrorCode::InvalidArg => "INVALID_ARG",
            ErrorCode::InternalError => "INTERNAL_ERROR",
        }
    }
}

// ============================================================
// COMMAND — parsed from wire input
// ============================================================

/// A parsed protocol command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// RECEIVE <content> <timestamp>
    Receive { content: String, timestamp: u64 },
    /// TEACH <kind> <content> <confidence_raw> <timestamp>
    Teach {
        kind: String,
        content: String,
        confidence: u16,
        timestamp: u64,
    },
    /// SYNTHESIZE <timestamp>
    Synthesize { timestamp: u64 },
    /// QUERY <key>
    Query { key: String },
    /// PROFILE
    Profile,
    /// KNOWS_ME
    KnowsMe,
    /// HEALTH
    Health,
    /// EXPLAIN <action_hash_hex>
    Explain { action_hash: String },
    /// REFLEX_STATS
    ReflexStats,
    /// REFLEX_INVALIDATE <trigger_hash_hex>
    ReflexInvalidate { trigger_hash: String },
    /// PLAN_ACTION <json_payload>
    PlanAction { payload_json: String },
    /// RUN_ACTION <plan_id> <json_payload>
    RunAction {
        plan_id: String,
        payload_json: String,
    },
    /// ACTION_STATUS <action_id>
    ActionStatus { action_id: String },
    /// ACTION_HISTORY <limit> <cursor>
    ActionHistory { limit: u32, cursor: String },
    /// START_SESSION <timestamp>
    StartSession { timestamp: u64 },
    /// END_SESSION <timestamp>
    EndSession { timestamp: u64 },
    /// IHSAN <score_raw_u16>
    Ihsan { score: u16 },
    /// PING
    Ping,
    /// VERSION
    Version,
    /// SHUTDOWN
    Shutdown,
    /// INTENT_CLASSIFY <content>
    IntentClassify { content: String },
    /// GUARDIAN_CHECK <content>
    GuardianCheck { content: String },
    /// ACTION_DISPATCH <channel> <json_payload>
    ActionDispatch {
        channel: String,
        payload_json: String,
    },

    // ── SAP v0 Protocol ──────────────────────────────
    /// SAP_MEET_OPEN <profile> <initiator_role> <timestamp>
    SapMeetOpen {
        profile: String,
        initiator_role: String,
        timestamp: u64,
    },
    /// SAP_MESSAGE <session_id> <content> <timestamp>
    SapMessage {
        session_id: String,
        content: String,
        timestamp: u64,
    },
    /// SAP_DISCLOSURE <session_id>
    SapDisclosure { session_id: String },
    /// SAP_CONSENT_REQUEST <session_id> <scopes_json>
    SapConsentRequest {
        session_id: String,
        scopes_json: String,
    },
    /// SAP_CONSENT_REVOKE <session_id> <receipt_id>
    SapConsentRevoke {
        session_id: String,
        receipt_id: String,
    },
    /// SAP_SESSION_CLOSE <session_id> <timestamp>
    SapSessionClose { session_id: String, timestamp: u64 },
    /// RESOURCES — sovereign resource manifest (hardware + models)
    Resources,
    /// RESOURCES_REFRESH — re-discover all resources
    ResourcesRefresh,
    /// MISSION_RECEIVE <content> <timestamp> — governed mission lifecycle
    MissionReceive { content: String, timestamp: u64 },
    /// HEARTBEAT <timestamp_ms> — timer-driven pulse for self-sustaining operation
    /// Phase 86-B: Drives the 4-loop HHMM EventBus.
    Heartbeat { timestamp_ms: u64 },
}

// ============================================================
// RESPONSE — serializable to wire output
// ============================================================

/// A protocol response (OK or ERR).
#[derive(Debug, Clone)]
pub enum Response {
    /// Successful response with key=value fields.
    Ok(Vec<(String, String)>),
    /// Error response with code and human-readable message.
    Err(ErrorCode, String),
}

impl Response {
    /// Create an OK response with a single field.
    pub fn ok_single(key: &str, value: &str) -> Self {
        Response::Ok(vec![(key.to_string(), value.to_string())])
    }

    /// Create an OK response from a vec of (key, value) pairs.
    pub fn ok(fields: Vec<(&str, String)>) -> Self {
        Response::Ok(
            fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    /// Create an ERR response.
    pub fn err(code: ErrorCode, message: &str) -> Self {
        Response::Err(code, message.to_string())
    }

    /// Serialize to wire format (single line, no trailing newline).
    pub fn to_wire(&self) -> String {
        match self {
            Response::Ok(fields) => {
                let mut parts = vec!["OK".to_string()];
                for (k, v) in fields {
                    // Escape any tabs or newlines in values
                    let safe_v = v.replace(['\t', '\n'], " ");
                    parts.push(format!("{k}={safe_v}"));
                }
                parts.join("\t")
            }
            Response::Err(code, msg) => {
                let safe_msg = msg.replace(['\t', '\n'], " ");
                format!("ERR\t{}\t{}", code.as_str(), safe_msg)
            }
        }
    }

    /// Is this an error response?
    pub fn is_err(&self) -> bool {
        matches!(self, Response::Err(..))
    }
}

// ============================================================
// PARSER — line to Command
// ============================================================

/// Parse a single wire-format line into a Command.
///
/// Returns `Err((ErrorCode, message))` on failure.
pub fn parse_command(line: &str) -> Result<Command, (ErrorCode, String)> {
    let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
    let parts: Vec<&str> = trimmed.split('\t').collect();

    if parts.is_empty() || parts[0].is_empty() {
        return Err((ErrorCode::BadCommand, "empty command".to_string()));
    }

    let verb = parts[0];
    match verb {
        "PING" => Ok(Command::Ping),
        "VERSION" => Ok(Command::Version),
        "SHUTDOWN" => Ok(Command::Shutdown),
        "PROFILE" => Ok(Command::Profile),
        "KNOWS_ME" => Ok(Command::KnowsMe),
        "HEALTH" => Ok(Command::Health),
        "REFLEX_STATS" => Ok(Command::ReflexStats),
        "RESOURCES" => Ok(Command::Resources),
        "RESOURCES_REFRESH" => Ok(Command::ResourcesRefresh),

        "EXPLAIN" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "EXPLAIN requires <action_hash>".to_string(),
                ));
            }
            if parts[1].is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "EXPLAIN action_hash must not be empty".to_string(),
                ));
            }
            Ok(Command::Explain {
                action_hash: parts[1].to_string(),
            })
        }

        "REFLEX_INVALIDATE" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "REFLEX_INVALIDATE requires <trigger_hash>".to_string(),
                ));
            }
            if parts[1].is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "REFLEX_INVALIDATE trigger_hash must not be empty".to_string(),
                ));
            }
            Ok(Command::ReflexInvalidate {
                trigger_hash: parts[1].to_string(),
            })
        }

        "PLAN_ACTION" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "PLAN_ACTION requires <json_payload>".to_string(),
                ));
            }
            if parts[1].trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "PLAN_ACTION payload must not be empty".to_string(),
                ));
            }
            Ok(Command::PlanAction {
                payload_json: parts[1].to_string(),
            })
        }

        "RUN_ACTION" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "RUN_ACTION requires <plan_id> <json_payload>".to_string(),
                ));
            }
            if parts[1].trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "RUN_ACTION plan_id must not be empty".to_string(),
                ));
            }
            if parts[2].trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "RUN_ACTION payload must not be empty".to_string(),
                ));
            }
            Ok(Command::RunAction {
                plan_id: parts[1].to_string(),
                payload_json: parts[2].to_string(),
            })
        }

        "ACTION_STATUS" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "ACTION_STATUS requires <action_id>".to_string(),
                ));
            }
            if parts[1].trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "ACTION_STATUS action_id must not be empty".to_string(),
                ));
            }
            Ok(Command::ActionStatus {
                action_id: parts[1].to_string(),
            })
        }

        "ACTION_HISTORY" => {
            let limit = if parts.len() >= 2 && !parts[1].trim().is_empty() {
                parts[1].parse::<u32>().map_err(|_| {
                    (
                        ErrorCode::ParseError,
                        "invalid action history limit".to_string(),
                    )
                })?
            } else {
                20
            };
            let cursor = if parts.len() >= 3 {
                parts[2].to_string()
            } else {
                String::new()
            };
            Ok(Command::ActionHistory { limit, cursor })
        }

        "RECEIVE" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "RECEIVE requires <content> <timestamp>".to_string(),
                ));
            }
            let content = parts[1];
            if content.is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "RECEIVE content must not be empty".to_string(),
                ));
            }
            let ts = parse_u64(parts[2], "timestamp")?;
            Ok(Command::Receive {
                content: content.to_string(),
                timestamp: ts,
            })
        }

        "MISSION_RECEIVE" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "MISSION_RECEIVE requires <content> <timestamp>".to_string(),
                ));
            }
            let content = parts[1];
            if content.is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "MISSION_RECEIVE content must not be empty".to_string(),
                ));
            }
            let ts = parse_u64(parts[2], "timestamp")?;
            Ok(Command::MissionReceive {
                content: content.to_string(),
                timestamp: ts,
            })
        }

        "TEACH" => {
            if parts.len() < 5 {
                return Err((
                    ErrorCode::MissingArg,
                    "TEACH requires <kind> <content> <confidence> <timestamp>".to_string(),
                ));
            }
            let kind = parts[1].to_string();
            let content = parts[2].to_string();
            if content.is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "TEACH content must not be empty".to_string(),
                ));
            }
            let conf = parse_u16(parts[3], "confidence")?;
            let ts = parse_u64(parts[4], "timestamp")?;
            validate_teach_kind(&kind)?;
            Ok(Command::Teach {
                kind,
                content,
                confidence: conf,
                timestamp: ts,
            })
        }

        "SYNTHESIZE" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "SYNTHESIZE requires <timestamp>".to_string(),
                ));
            }
            let ts = parse_u64(parts[1], "timestamp")?;
            Ok(Command::Synthesize { timestamp: ts })
        }

        "QUERY" => {
            if parts.len() < 2 {
                return Err((ErrorCode::MissingArg, "QUERY requires <key>".to_string()));
            }
            Ok(Command::Query {
                key: parts[1].to_string(),
            })
        }

        "START_SESSION" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "START_SESSION requires <timestamp>".to_string(),
                ));
            }
            let ts = parse_u64(parts[1], "timestamp")?;
            Ok(Command::StartSession { timestamp: ts })
        }

        "END_SESSION" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "END_SESSION requires <timestamp>".to_string(),
                ));
            }
            let ts = parse_u64(parts[1], "timestamp")?;
            Ok(Command::EndSession { timestamp: ts })
        }

        "IHSAN" => {
            if parts.len() < 2 {
                return Err((ErrorCode::MissingArg, "IHSAN requires <score>".to_string()));
            }
            let score = parse_u16(parts[1], "score")?;
            Ok(Command::Ihsan { score })
        }

        "INTENT_CLASSIFY" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "INTENT_CLASSIFY requires <content>".to_string(),
                ));
            }
            let content = parts[1..].join("\t"); // rejoin in case content had tabs
            if content.trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "INTENT_CLASSIFY content must not be empty".to_string(),
                ));
            }
            Ok(Command::IntentClassify { content })
        }

        "GUARDIAN_CHECK" => {
            if parts.len() < 2 {
                return Err((
                    ErrorCode::MissingArg,
                    "GUARDIAN_CHECK requires <content>".to_string(),
                ));
            }
            let content = parts[1..].join("\t");
            if content.trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "GUARDIAN_CHECK content must not be empty".to_string(),
                ));
            }
            Ok(Command::GuardianCheck { content })
        }

        "ACTION_DISPATCH" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "ACTION_DISPATCH requires <channel> <json_payload>".to_string(),
                ));
            }
            let channel = parts[1].to_string();
            if channel.trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "ACTION_DISPATCH channel must not be empty".to_string(),
                ));
            }
            let payload_json = parts[2..].join("\t");
            if payload_json.trim().is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "ACTION_DISPATCH payload must not be empty".to_string(),
                ));
            }
            Ok(Command::ActionDispatch {
                channel,
                payload_json,
            })
        }

        // SAP v0 protocol commands
        "SAP_MEET_OPEN" => {
            let profile = if parts.len() >= 2 && !parts[1].is_empty() {
                parts[1].to_string()
            } else {
                "sap-ads-retail-v0".to_string()
            };
            let initiator_role = if parts.len() >= 3 && !parts[2].is_empty() {
                parts[2].to_string()
            } else {
                "visitor".to_string()
            };
            let ts = if parts.len() >= 4 {
                parse_u64(parts[3], "timestamp")?
            } else {
                0
            };
            Ok(Command::SapMeetOpen {
                profile,
                initiator_role,
                timestamp: ts,
            })
        }

        "SAP_MESSAGE" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "SAP_MESSAGE requires <session_id> <content> [timestamp]".to_string(),
                ));
            }
            if parts[1].is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "SAP_MESSAGE session_id must not be empty".to_string(),
                ));
            }
            if parts[2].is_empty() {
                return Err((
                    ErrorCode::InvalidArg,
                    "SAP_MESSAGE content must not be empty".to_string(),
                ));
            }
            let ts = if parts.len() >= 4 {
                parse_u64(parts[3], "timestamp")?
            } else {
                0
            };
            Ok(Command::SapMessage {
                session_id: parts[1].to_string(),
                content: parts[2].to_string(),
                timestamp: ts,
            })
        }

        "SAP_DISCLOSURE" => {
            if parts.len() < 2 || parts[1].is_empty() {
                return Err((
                    ErrorCode::MissingArg,
                    "SAP_DISCLOSURE requires <session_id>".to_string(),
                ));
            }
            Ok(Command::SapDisclosure {
                session_id: parts[1].to_string(),
            })
        }

        "SAP_CONSENT_REQUEST" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "SAP_CONSENT_REQUEST requires <session_id> <scopes_json>".to_string(),
                ));
            }
            Ok(Command::SapConsentRequest {
                session_id: parts[1].to_string(),
                scopes_json: parts[2].to_string(),
            })
        }

        "SAP_CONSENT_REVOKE" => {
            if parts.len() < 3 {
                return Err((
                    ErrorCode::MissingArg,
                    "SAP_CONSENT_REVOKE requires <session_id> <receipt_id>".to_string(),
                ));
            }
            Ok(Command::SapConsentRevoke {
                session_id: parts[1].to_string(),
                receipt_id: parts[2].to_string(),
            })
        }

        "SAP_SESSION_CLOSE" => {
            if parts.len() < 2 || parts[1].is_empty() {
                return Err((
                    ErrorCode::MissingArg,
                    "SAP_SESSION_CLOSE requires <session_id> [timestamp]".to_string(),
                ));
            }
            let ts = if parts.len() >= 3 {
                parse_u64(parts[2], "timestamp")?
            } else {
                0
            };
            Ok(Command::SapSessionClose {
                session_id: parts[1].to_string(),
                timestamp: ts,
            })
        }

        // ── Phase 86-B: Heartbeat ──────────────────────────
        "HEARTBEAT" => {
            let ts = if parts.len() >= 2 && !parts[1].trim().is_empty() {
                parse_u64(parts[1], "timestamp_ms")?
            } else {
                0 // caller may omit; node uses wall clock
            };
            Ok(Command::Heartbeat { timestamp_ms: ts })
        }

        _ => Err((ErrorCode::BadCommand, format!("unknown command: {verb}"))),
    }
}

// ============================================================
// HELPERS
// ============================================================

fn parse_u64(s: &str, field: &str) -> Result<u64, (ErrorCode, String)> {
    s.parse::<u64>()
        .map_err(|_| (ErrorCode::ParseError, format!("invalid {field}: {s:?}")))
}

fn parse_u16(s: &str, field: &str) -> Result<u16, (ErrorCode, String)> {
    s.parse::<u16>()
        .map_err(|_| (ErrorCode::ParseError, format!("invalid {field}: {s:?}")))
}

/// Validate a TEACH kind string.
/// Accepted: fact, preference, pattern, relationship, goal, expertise, context, principle, temporal, negation
fn validate_teach_kind(kind: &str) -> Result<(), (ErrorCode, String)> {
    match kind {
        "fact" | "preference" | "pattern" | "relationship" | "goal" | "expertise" | "context"
        | "principle" | "temporal" | "negation" => Ok(()),
        _ => Err((
            ErrorCode::InvalidArg,
            format!("unknown TEACH kind: {kind:?}"),
        )),
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ping() {
        assert_eq!(parse_command("PING").unwrap(), Command::Ping);
    }

    #[test]
    fn parse_version() {
        assert_eq!(parse_command("VERSION").unwrap(), Command::Version);
    }

    #[test]
    fn parse_shutdown() {
        assert_eq!(parse_command("SHUTDOWN").unwrap(), Command::Shutdown);
    }

    #[test]
    fn parse_receive() {
        let cmd = parse_command("RECEIVE\thello world\t1000").unwrap();
        assert_eq!(
            cmd,
            Command::Receive {
                content: "hello world".to_string(),
                timestamp: 1000,
            }
        );
    }

    #[test]
    fn parse_receive_missing_args() {
        let err = parse_command("RECEIVE").unwrap_err();
        assert_eq!(err.0, ErrorCode::MissingArg);
    }

    #[test]
    fn parse_receive_empty_content() {
        let err = parse_command("RECEIVE\t\t1000").unwrap_err();
        assert_eq!(err.0, ErrorCode::InvalidArg);
    }

    #[test]
    fn parse_teach() {
        let cmd = parse_command("TEACH\tfact\tI live in Dubai\t9500\t4000").unwrap();
        assert_eq!(
            cmd,
            Command::Teach {
                kind: "fact".to_string(),
                content: "I live in Dubai".to_string(),
                confidence: 9500,
                timestamp: 4000,
            }
        );
    }

    #[test]
    fn parse_teach_invalid_kind() {
        let err = parse_command("TEACH\tboguskind\ttest\t9000\t1000").unwrap_err();
        assert_eq!(err.0, ErrorCode::InvalidArg);
    }

    #[test]
    fn parse_ihsan() {
        let cmd = parse_command("IHSAN\t5000").unwrap();
        assert_eq!(cmd, Command::Ihsan { score: 5000 });
    }

    #[test]
    fn parse_ihsan_bad_number() {
        let err = parse_command("IHSAN\tnotanumber").unwrap_err();
        assert_eq!(err.0, ErrorCode::ParseError);
    }

    #[test]
    fn parse_explain() {
        let cmd = parse_command("EXPLAIN\tabcdef").unwrap();
        assert_eq!(
            cmd,
            Command::Explain {
                action_hash: "abcdef".to_string(),
            }
        );
    }

    #[test]
    fn parse_reflex_stats() {
        let cmd = parse_command("REFLEX_STATS").unwrap();
        assert_eq!(cmd, Command::ReflexStats);
    }

    #[test]
    fn parse_reflex_invalidate() {
        let cmd = parse_command("REFLEX_INVALIDATE\t0011").unwrap();
        assert_eq!(
            cmd,
            Command::ReflexInvalidate {
                trigger_hash: "0011".to_string(),
            }
        );
    }

    #[test]
    fn parse_plan_action() {
        let cmd = parse_command("PLAN_ACTION\t{\"kind\":\"Click\"}").unwrap();
        assert_eq!(
            cmd,
            Command::PlanAction {
                payload_json: "{\"kind\":\"Click\"}".to_string(),
            }
        );
    }

    #[test]
    fn parse_run_action() {
        let cmd = parse_command("RUN_ACTION\tpln_1\t{\"kind\":\"Click\"}").unwrap();
        assert_eq!(
            cmd,
            Command::RunAction {
                plan_id: "pln_1".to_string(),
                payload_json: "{\"kind\":\"Click\"}".to_string(),
            }
        );
    }

    #[test]
    fn parse_action_status() {
        let cmd = parse_command("ACTION_STATUS\tact_1").unwrap();
        assert_eq!(
            cmd,
            Command::ActionStatus {
                action_id: "act_1".to_string(),
            }
        );
    }

    #[test]
    fn parse_action_history_defaults() {
        let cmd = parse_command("ACTION_HISTORY").unwrap();
        assert_eq!(
            cmd,
            Command::ActionHistory {
                limit: 20,
                cursor: String::new(),
            }
        );
    }

    #[test]
    fn parse_intent_classify() {
        let cmd = parse_command("INTENT_CLASSIFY\thelp me plan the meeting").unwrap();
        assert_eq!(
            cmd,
            Command::IntentClassify {
                content: "help me plan the meeting".to_string(),
            }
        );
    }

    #[test]
    fn parse_intent_classify_missing_content() {
        let err = parse_command("INTENT_CLASSIFY").unwrap_err();
        assert_eq!(err.0, ErrorCode::MissingArg);
    }

    #[test]
    fn parse_intent_classify_empty_content() {
        let err = parse_command("INTENT_CLASSIFY\t").unwrap_err();
        assert_eq!(err.0, ErrorCode::InvalidArg);
    }

    #[test]
    fn parse_action_dispatch() {
        let cmd = parse_command("ACTION_DISPATCH\tllm\t{\"prompt\":\"hello\"}").unwrap();
        assert_eq!(
            cmd,
            Command::ActionDispatch {
                channel: "llm".to_string(),
                payload_json: "{\"prompt\":\"hello\"}".to_string(),
            }
        );
    }

    #[test]
    fn parse_action_dispatch_missing_args() {
        let err = parse_command("ACTION_DISPATCH\tllm").unwrap_err();
        assert_eq!(err.0, ErrorCode::MissingArg);
    }

    #[test]
    fn parse_action_dispatch_empty_channel() {
        let err = parse_command("ACTION_DISPATCH\t\t{\"a\":1}").unwrap_err();
        assert_eq!(err.0, ErrorCode::InvalidArg);
    }

    #[test]
    fn parse_guardian_check() {
        let cmd = parse_command("GUARDIAN_CHECK\tplease plan my project").unwrap();
        assert_eq!(
            cmd,
            Command::GuardianCheck {
                content: "please plan my project".to_string(),
            }
        );
    }

    #[test]
    fn parse_guardian_check_missing_content() {
        let err = parse_command("GUARDIAN_CHECK").unwrap_err();
        assert_eq!(err.0, ErrorCode::MissingArg);
    }

    #[test]
    fn parse_bogus_command() {
        let err = parse_command("BOGUS").unwrap_err();
        assert_eq!(err.0, ErrorCode::BadCommand);
    }

    // ── SAP v0 parser tests ──

    #[test]
    fn parse_sap_meet_open() {
        let cmd = parse_command("SAP_MEET_OPEN\tsap-ads-retail-v0\tvisitor\t5000").unwrap();
        assert_eq!(
            cmd,
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 5000,
            }
        );
    }

    #[test]
    fn parse_sap_meet_open_defaults() {
        let cmd = parse_command("SAP_MEET_OPEN").unwrap();
        assert_eq!(
            cmd,
            Command::SapMeetOpen {
                profile: "sap-ads-retail-v0".to_string(),
                initiator_role: "visitor".to_string(),
                timestamp: 0,
            }
        );
    }

    #[test]
    fn parse_sap_message() {
        let cmd = parse_command("SAP_MESSAGE\tsap_123\thello\t6000").unwrap();
        assert_eq!(
            cmd,
            Command::SapMessage {
                session_id: "sap_123".to_string(),
                content: "hello".to_string(),
                timestamp: 6000,
            }
        );
    }

    #[test]
    fn parse_sap_message_missing_content() {
        let err = parse_command("SAP_MESSAGE\tsap_123").unwrap_err();
        assert_eq!(err.0, ErrorCode::MissingArg);
    }

    #[test]
    fn parse_sap_disclosure() {
        let cmd = parse_command("SAP_DISCLOSURE\tsap_123").unwrap();
        assert_eq!(
            cmd,
            Command::SapDisclosure {
                session_id: "sap_123".to_string(),
            }
        );
    }

    #[test]
    fn parse_sap_session_close() {
        let cmd = parse_command("SAP_SESSION_CLOSE\tsap_123\t7000").unwrap();
        assert_eq!(
            cmd,
            Command::SapSessionClose {
                session_id: "sap_123".to_string(),
                timestamp: 7000,
            }
        );
    }

    // ── Phase 86-B: Heartbeat parser tests ─────────────────

    #[test]
    fn parse_heartbeat_with_timestamp() {
        let cmd = parse_command("HEARTBEAT\t5000").unwrap();
        assert_eq!(cmd, Command::Heartbeat { timestamp_ms: 5000 });
    }

    #[test]
    fn parse_heartbeat_no_timestamp() {
        let cmd = parse_command("HEARTBEAT").unwrap();
        assert_eq!(cmd, Command::Heartbeat { timestamp_ms: 0 });
    }

    #[test]
    fn parse_heartbeat_bad_timestamp() {
        let err = parse_command("HEARTBEAT\tnot_a_number").unwrap_err();
        assert_eq!(err.0, ErrorCode::ParseError);
    }

    #[test]
    fn response_ok_wire_format() {
        let r = Response::ok_single("pong", "true");
        assert_eq!(r.to_wire(), "OK\tpong=true");
    }

    #[test]
    fn response_err_wire_format() {
        let r = Response::err(ErrorCode::BadCommand, "unknown");
        let wire = r.to_wire();
        assert!(wire.starts_with("ERR\t"));
        assert!(wire.contains("BAD_COMMAND"));
    }

    #[test]
    fn response_ok_no_newlines() {
        let r = Response::ok(vec![
            ("a", "1".to_string()),
            ("b", "value\nwith\nnewlines".to_string()),
        ]);
        let wire = r.to_wire();
        assert!(!wire.contains('\n'));
    }
}
