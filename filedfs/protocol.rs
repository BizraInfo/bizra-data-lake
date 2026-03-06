// bizra-node/src/protocol.rs
// ============================================================
// Node Protocol — the wire format
// ============================================================
// Design principles (standing on giants):
//
//   Redis RESP   → simplicity. One line = one command.
//   LSP          → stdio transport. Spawn binary, pipe text.
//   Unix         → text streams. Composable. Debuggable.
//   NATS         → subject-based routing. Clean verbs.
//
// Format:
//   Request:  VERB\targ1\targ2\t...\n
//   Response: OK\tfield=value\tfield=value\n
//          or ERR\tcode\tmessage\n
//
// Every request gets exactly one response.
// Tab-delimited. Newline-framed. Zero parsing ambiguity.
// Any language can implement a client in 30 lines.
// ============================================================

use core::fmt;

// ============================================================
// PROTOCOL COMMANDS — what clients can ask
// ============================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Process a user message through the full agent pipeline
    /// RECEIVE\t<content>\t<timestamp>
    Receive {
        content: String,
        timestamp: u64,
    },

    /// Teach the node something directly
    /// TEACH\t<kind>\t<content>\t<confidence>\t<timestamp>
    Teach {
        kind: String,
        content: String,
        confidence: u16,
        timestamp: u64,
    },

    /// Force a synthesis round
    /// SYNTHESIZE\t<timestamp>
    Synthesize {
        timestamp: u64,
    },

    /// Query a specific trait
    /// QUERY\t<key>
    Query {
        key: String,
    },

    /// Get the full user profile
    /// PROFILE
    Profile,

    /// Get the "knows me" score
    /// KNOWS_ME
    KnowsMe,

    /// Get full system health
    /// HEALTH
    Health,

    /// Start a conversation session
    /// START_SESSION\t<timestamp>
    StartSession {
        timestamp: u64,
    },

    /// End current conversation session
    /// END_SESSION\t<timestamp>
    EndSession {
        timestamp: u64,
    },

    /// Update إحسان score
    /// IHSAN\t<score>
    Ihsan {
        score: u16,
    },

    /// Graceful shutdown
    /// SHUTDOWN
    Shutdown,

    /// Echo (health check / keepalive)
    /// PING
    Ping,

    /// Get node version info
    /// VERSION
    Version,
}

// ============================================================
// PROTOCOL RESPONSES — what the node answers
// ============================================================

#[derive(Debug, Clone)]
pub enum Response {
    /// Successful response with key-value fields
    Ok(Vec<(String, String)>),

    /// Error response
    Err {
        code: ErrorCode,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Unknown or malformed command
    BadCommand = 1,
    /// Missing required argument
    MissingArg = 2,
    /// Invalid argument value
    InvalidArg = 3,
    /// System degraded (إحسان below threshold)
    Degraded = 4,
    /// Runtime is stopped
    Stopped = 5,
    /// Internal error
    Internal = 6,
    /// Parse error in argument
    ParseError = 7,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::BadCommand => write!(f, "BAD_COMMAND"),
            ErrorCode::MissingArg => write!(f, "MISSING_ARG"),
            ErrorCode::InvalidArg => write!(f, "INVALID_ARG"),
            ErrorCode::Degraded => write!(f, "DEGRADED"),
            ErrorCode::Stopped => write!(f, "STOPPED"),
            ErrorCode::Internal => write!(f, "INTERNAL"),
            ErrorCode::ParseError => write!(f, "PARSE_ERROR"),
        }
    }
}

// ============================================================
// RESPONSE BUILDERS — ergonomic construction
// ============================================================

impl Response {
    pub fn ok() -> Self {
        Response::Ok(Vec::new())
    }

    pub fn ok_with(fields: Vec<(&str, String)>) -> Self {
        Response::Ok(
            fields.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        )
    }

    pub fn err(code: ErrorCode, message: &str) -> Self {
        Response::Err {
            code,
            message: message.to_string(),
        }
    }

    pub fn field(mut self, key: &str, value: impl fmt::Display) -> Self {
        if let Response::Ok(ref mut fields) = self {
            fields.push((key.to_string(), value.to_string()));
        }
        self
    }
}

// ============================================================
// SERIALIZATION — Response → wire format
// ============================================================

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Response::Ok(fields) => {
                write!(f, "OK")?;
                for (key, value) in fields {
                    // Escape tabs and newlines in values
                    let escaped = escape_value(value);
                    write!(f, "\t{}={}", key, escaped)?;
                }
                Ok(())
            }
            Response::Err { code, message } => {
                let escaped = escape_value(message);
                write!(f, "ERR\t{}\t{}", code, escaped)
            }
        }
    }
}

/// Escape tab and newline characters in protocol values
fn escape_value(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('\t', "\\t")
     .replace('\n', "\\n")
     .replace('\r', "\\r")
}

/// Unescape protocol values
fn unescape_value(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => result.push('\t'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

// ============================================================
// PARSING — wire format → Command
// ============================================================

/// Parse a single line into a Command
/// Returns None for empty lines, Err for malformed commands
pub fn parse_command(line: &str) -> Result<Option<Command>, Response> {
    let line = line.trim();

    // Empty lines are ignored (keepalive-friendly)
    if line.is_empty() {
        return Ok(None);
    }

    // Split on tabs
    let parts: Vec<&str> = line.split('\t').collect();
    let verb = parts[0].to_uppercase();

    match verb.as_str() {
        "RECEIVE" => {
            let content = parts.get(1)
                .ok_or_else(|| Response::err(ErrorCode::MissingArg, "RECEIVE requires content"))?;
            let timestamp = parse_u64(parts.get(2), "timestamp")?;
            Ok(Some(Command::Receive {
                content: unescape_value(content),
                timestamp,
            }))
        }

        "TEACH" => {
            let kind = parts.get(1)
                .ok_or_else(|| Response::err(ErrorCode::MissingArg, "TEACH requires kind"))?;
            let content = parts.get(2)
                .ok_or_else(|| Response::err(ErrorCode::MissingArg, "TEACH requires content"))?;
            let confidence = parse_u16(parts.get(3), "confidence")?;
            let timestamp = parse_u64(parts.get(4), "timestamp")?;
            Ok(Some(Command::Teach {
                kind: kind.to_string(),
                content: unescape_value(content),
                confidence,
                timestamp,
            }))
        }

        "SYNTHESIZE" => {
            let timestamp = parse_u64(parts.get(1), "timestamp")?;
            Ok(Some(Command::Synthesize { timestamp }))
        }

        "QUERY" => {
            let key = parts.get(1)
                .ok_or_else(|| Response::err(ErrorCode::MissingArg, "QUERY requires key"))?;
            Ok(Some(Command::Query { key: key.to_string() }))
        }

        "PROFILE" => Ok(Some(Command::Profile)),

        "KNOWS_ME" => Ok(Some(Command::KnowsMe)),

        "HEALTH" => Ok(Some(Command::Health)),

        "START_SESSION" => {
            let timestamp = parse_u64(parts.get(1), "timestamp")?;
            Ok(Some(Command::StartSession { timestamp }))
        }

        "END_SESSION" => {
            let timestamp = parse_u64(parts.get(1), "timestamp")?;
            Ok(Some(Command::EndSession { timestamp }))
        }

        "IHSAN" => {
            let score = parse_u16(parts.get(1), "score")?;
            Ok(Some(Command::Ihsan { score }))
        }

        "SHUTDOWN" => Ok(Some(Command::Shutdown)),

        "PING" => Ok(Some(Command::Ping)),

        "VERSION" => Ok(Some(Command::Version)),

        _ => Err(Response::err(
            ErrorCode::BadCommand,
            &format!("Unknown command: {}", verb),
        )),
    }
}

/// Parse a u64 from an optional string reference
fn parse_u64(s: Option<&&str>, name: &str) -> Result<u64, Response> {
    let s = s.ok_or_else(|| Response::err(
        ErrorCode::MissingArg,
        &format!("{} is required", name),
    ))?;
    s.parse::<u64>().map_err(|_| Response::err(
        ErrorCode::ParseError,
        &format!("Invalid {}: '{}'", name, s),
    ))
}

/// Parse a u16 from an optional string reference
fn parse_u16(s: Option<&&str>, name: &str) -> Result<u16, Response> {
    let s = s.ok_or_else(|| Response::err(
        ErrorCode::MissingArg,
        &format!("{} is required", name),
    ))?;
    s.parse::<u16>().map_err(|_| Response::err(
        ErrorCode::ParseError,
        &format!("Invalid {}: '{}'", name, s),
    ))
}

// ============================================================
// PROTOCOL CONSTANTS
// ============================================================

pub const PROTOCOL_VERSION: &str = "1.0";
pub const NODE_VERSION: &str = "0.1.0";
pub const NODE_NAME: &str = "bizra-node";

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_receive_command() {
        let cmd = parse_command("RECEIVE\tHello world\t1000").unwrap().unwrap();
        match cmd {
            Command::Receive { content, timestamp } => {
                assert_eq!(content, "Hello world");
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("Expected Receive"),
        }
    }

    #[test]
    fn parse_teach_command() {
        let cmd = parse_command("TEACH\tpreference\tUser likes Rust\t8500\t1000").unwrap().unwrap();
        match cmd {
            Command::Teach { kind, content, confidence, timestamp } => {
                assert_eq!(kind, "preference");
                assert_eq!(content, "User likes Rust");
                assert_eq!(confidence, 8500);
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("Expected Teach"),
        }
    }

    #[test]
    fn parse_simple_commands() {
        assert!(matches!(
            parse_command("HEALTH").unwrap().unwrap(),
            Command::Health
        ));
        assert!(matches!(
            parse_command("PING").unwrap().unwrap(),
            Command::Ping
        ));
        assert!(matches!(
            parse_command("SHUTDOWN").unwrap().unwrap(),
            Command::Shutdown
        ));
        assert!(matches!(
            parse_command("KNOWS_ME").unwrap().unwrap(),
            Command::KnowsMe
        ));
        assert!(matches!(
            parse_command("PROFILE").unwrap().unwrap(),
            Command::Profile
        ));
        assert!(matches!(
            parse_command("VERSION").unwrap().unwrap(),
            Command::Version
        ));
    }

    #[test]
    fn parse_case_insensitive() {
        assert!(parse_command("health").unwrap().is_some());
        assert!(parse_command("Health").unwrap().is_some());
        assert!(parse_command("HEALTH").unwrap().is_some());
    }

    #[test]
    fn parse_empty_line_returns_none() {
        assert!(parse_command("").unwrap().is_none());
        assert!(parse_command("  ").unwrap().is_none());
    }

    #[test]
    fn parse_unknown_command_errors() {
        let err = parse_command("INVALID_CMD").unwrap_err();
        match err {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::BadCommand),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn parse_missing_args_errors() {
        let err = parse_command("RECEIVE").unwrap_err();
        match err {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::MissingArg),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn parse_invalid_number_errors() {
        let err = parse_command("RECEIVE\tHello\tnotanumber").unwrap_err();
        match err {
            Response::Err { code, .. } => assert_eq!(code, ErrorCode::ParseError),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn response_ok_format() {
        let resp = Response::ok()
            .field("score", 0.67)
            .field("agents", 3);
        let wire = resp.to_string();
        assert_eq!(wire, "OK\tscore=0.67\tagents=3");
    }

    #[test]
    fn response_err_format() {
        let resp = Response::err(ErrorCode::Degraded, "ihsan below threshold");
        let wire = resp.to_string();
        assert_eq!(wire, "ERR\tDEGRADED\tihsan below threshold");
    }

    #[test]
    fn escape_roundtrip() {
        let original = "line1\nline2\ttab\there\\backslash";
        let escaped = escape_value(original);
        let restored = unescape_value(&escaped);
        assert_eq!(restored, original);
    }

    #[test]
    fn response_ok_with_builder() {
        let resp = Response::ok_with(vec![
            ("state", "Ready".to_string()),
            ("ihsan", "9900".to_string()),
        ]);
        let wire = resp.to_string();
        assert!(wire.starts_with("OK\t"));
        assert!(wire.contains("state=Ready"));
        assert!(wire.contains("ihsan=9900"));
    }

    #[test]
    fn parse_ihsan_command() {
        let cmd = parse_command("IHSAN\t9500").unwrap().unwrap();
        match cmd {
            Command::Ihsan { score } => assert_eq!(score, 9500),
            _ => panic!("Expected Ihsan"),
        }
    }

    #[test]
    fn parse_session_commands() {
        let start = parse_command("START_SESSION\t1000").unwrap().unwrap();
        assert!(matches!(start, Command::StartSession { timestamp: 1000 }));

        let end = parse_command("END_SESSION\t2000").unwrap().unwrap();
        assert!(matches!(end, Command::EndSession { timestamp: 2000 }));
    }

    #[test]
    fn parse_query_command() {
        let cmd = parse_command("QUERY\tpreference").unwrap().unwrap();
        match cmd {
            Command::Query { key } => assert_eq!(key, "preference"),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn escaped_content_in_receive() {
        // Content with escaped newlines
        let cmd = parse_command("RECEIVE\tLine1\\nLine2\\tTabbed\t1000").unwrap().unwrap();
        match cmd {
            Command::Receive { content, .. } => {
                assert_eq!(content, "Line1\nLine2\tTabbed");
            }
            _ => panic!("Expected Receive"),
        }
    }
}
