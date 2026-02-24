// bizra-agent/src/action_types.rs
// ============================================================
// Action Types — typed action contract for Action Bus
// ============================================================

use crate::hash_namespace::{compute_receipt_hash, parse_hex_32, ReceiptHash};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionChannel {
    DesktopRpc,
    ToolCall,
    LlmCall,
    FileOp,
    BrowserNav,
}

impl ActionChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DesktopRpc => "DesktopRpc",
            Self::ToolCall => "ToolCall",
            Self::LlmCall => "LlmCall",
            Self::FileOp => "FileOp",
            Self::BrowserNav => "BrowserNav",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "desktoprpc" | "desktop_rpc" | "desktop" => Some(Self::DesktopRpc),
            "toolcall" | "tool_call" | "tool" => Some(Self::ToolCall),
            "llmcall" | "llm_call" | "llm" => Some(Self::LlmCall),
            "fileop" | "file_op" | "file" => Some(Self::FileOp),
            "browsernav" | "browser_nav" | "browser" => Some(Self::BrowserNav),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Click,
    TypeText,
    InvokeSkill,
    ToolCall,
    WriteFile,
    Navigate,
    Query,
}

impl ActionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Click => "Click",
            Self::TypeText => "TypeText",
            Self::InvokeSkill => "InvokeSkill",
            Self::ToolCall => "ToolCall",
            Self::WriteFile => "WriteFile",
            Self::Navigate => "Navigate",
            Self::Query => "Query",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "click" => Some(Self::Click),
            "typetext" | "type_text" | "type" => Some(Self::TypeText),
            "invokeskill" | "invoke_skill" => Some(Self::InvokeSkill),
            "toolcall" | "tool_call" => Some(Self::ToolCall),
            "writefile" | "write_file" => Some(Self::WriteFile),
            "navigate" => Some(Self::Navigate),
            "query" => Some(Self::Query),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionExecutionStatus {
    Planned,
    Running,
    Completed,
    Failed,
    Denied,
}

impl ActionExecutionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Denied => "denied",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedStep {
    pub channel: ActionChannel,
    pub kind: ActionKind,
    pub payload: String,
}

impl PlannedStep {
    pub fn validate(&self) -> Result<(), ActionError> {
        if self.payload.trim().is_empty() {
            return Err(ActionError::new(
                "EMPTY_PAYLOAD",
                "Action payload must not be empty",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ActionPlan {
    pub plan_id: String,
    pub created_at: u64,
    pub steps: Vec<PlannedStep>,
}

impl ActionPlan {
    pub fn validate(&self) -> Result<(), ActionError> {
        if self.plan_id.trim().is_empty() {
            return Err(ActionError::new(
                "INVALID_PLAN",
                "plan_id must not be empty",
            ));
        }
        if self.steps.is_empty() {
            return Err(ActionError::new(
                "EMPTY_PLAN",
                "Action plan must contain at least one step",
            ));
        }
        for step in &self.steps {
            step.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ActionError {
    pub code: String,
    pub message: String,
}

impl ActionError {
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionResult {
    pub action_id: String,
    pub plan_id: String,
    pub status: ActionExecutionStatus,
    pub message: String,
    pub started_at: u64,
    pub finished_at: u64,
}

#[derive(Debug, Clone)]
pub struct ActionReceipt {
    pub action_id: String,
    pub plan_id: String,
    pub channel: ActionChannel,
    pub kind: ActionKind,
    pub timestamp: u64,
    pub result: String,
    pub guardian_verdict: bool,
    pub permit_hash: [u8; 32],
    pub policy_hash: [u8; 32],
    pub receipt_hash: [u8; 32],
    pub prev_receipt_hash: [u8; 32],
    /// SHA-256 hash of post-action screenshot, proving the action's
    /// effect on the desktop. Currently only proves attempt, not success;
    /// this field proves both.  `None` when no screenshot was captured.
    pub outcome_hash: Option<[u8; 32]>,
}

impl ActionReceipt {
    pub fn canonical_without_hash(&self) -> String {
        let outcome = match &self.outcome_hash {
            Some(h) => to_hex(*h),
            None => "none".to_string(),
        };
        format!(
            "v=2|id={}|plan={}|ch={}|kind={}|ts={}|guardian={}|permit={}|policy={}|result={}|prev={}|outcome={}",
            self.action_id,
            self.plan_id,
            self.channel.as_str(),
            self.kind.as_str(),
            self.timestamp,
            self.guardian_verdict,
            to_hex(self.permit_hash),
            to_hex(self.policy_hash),
            self.result,
            to_hex(self.prev_receipt_hash),
            outcome,
        )
    }

    pub fn seal(&mut self) {
        self.receipt_hash = compute_receipt_hash(
            self.canonical_without_hash().as_str(),
            &self.prev_receipt_hash,
        )
        .0;
    }

    pub fn verify_chain(&self, expected_prev: &[u8; 32]) -> bool {
        if &self.prev_receipt_hash != expected_prev {
            return false;
        }
        compute_receipt_hash(self.canonical_without_hash().as_str(), expected_prev).0
            == self.receipt_hash
    }

    pub fn to_jsonl(&self) -> String {
        let outcome_str = match &self.outcome_hash {
            Some(h) => format!(",\"outcome\":\"{}\"", to_hex(*h)),
            None => String::new(),
        };
        format!(
            "{{\"v\":2,\"id\":\"{}\",\"plan\":\"{}\",\"ch\":\"{}\",\"kind\":\"{}\",\"ts\":{},\"guardian\":{},\"permit\":\"{}\",\"policy\":\"{}\",\"result\":\"{}\",\"receipt\":\"{}\",\"prev\":\"{}\"{}}}",
            sanitize(self.action_id.as_str()),
            sanitize(self.plan_id.as_str()),
            self.channel.as_str(),
            self.kind.as_str(),
            self.timestamp,
            self.guardian_verdict,
            to_hex(self.permit_hash),
            to_hex(self.policy_hash),
            sanitize(self.result.as_str()),
            to_hex(self.receipt_hash),
            to_hex(self.prev_receipt_hash),
            outcome_str,
        )
    }

    pub fn from_jsonl(line: &str) -> Option<Self> {
        // Minimal parser for locked schema; fail-closed on shape mismatch.
        let raw = line.trim();
        if !raw.starts_with('{') || !raw.ends_with('}') {
            return None;
        }
        let get = |k: &str| -> Option<String> { extract_json_string(raw, k) };
        let action_id = get("id")?;
        let plan_id = get("plan")?;
        let ch = ActionChannel::parse(get("ch")?.as_str())?;
        let kind = ActionKind::parse(get("kind")?.as_str())?;
        let timestamp = extract_json_u64(raw, "ts")?;
        let guardian_verdict = extract_json_bool(raw, "guardian")?;
        let permit_hash = parse_hex_32(get("permit")?.as_str())?;
        let policy_hash = parse_hex_32(get("policy")?.as_str())?;
        let result = get("result")?;
        let receipt_hash = parse_hex_32(get("receipt")?.as_str())?;
        let prev_receipt_hash = parse_hex_32(get("prev")?.as_str())?;

        // outcome_hash is optional — absent in v1 receipts
        let outcome_hash = get("outcome").and_then(|h| parse_hex_32(h.as_str()));

        Some(Self {
            action_id,
            plan_id,
            channel: ch,
            kind,
            timestamp,
            result,
            guardian_verdict,
            permit_hash,
            policy_hash,
            receipt_hash,
            prev_receipt_hash,
            outcome_hash,
        })
    }

    pub fn receipt_hash_hex(&self) -> String {
        ReceiptHash(self.receipt_hash).to_hex()
    }
}

fn to_hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sanitize(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let mut out = String::new();
    let mut escaped = false;
    for ch in rest.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            return Some(out);
        }
        out.push(ch);
    }
    None
}

fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find([',', '}'])?;
    rest[..end].trim().parse::<u64>().ok()
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find([',', '}'])?;
    match rest[..end].trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_plan_validation() {
        let plan = ActionPlan {
            plan_id: "p1".to_string(),
            created_at: 1,
            steps: vec![PlannedStep {
                channel: ActionChannel::DesktopRpc,
                kind: ActionKind::Click,
                payload: "{\"target\":\"ok\"}".to_string(),
            }],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn receipt_roundtrip_and_chain_verify() {
        let mut r = ActionReceipt {
            action_id: "a1".to_string(),
            plan_id: "p1".to_string(),
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            timestamp: 123,
            result: "ok".to_string(),
            guardian_verdict: true,
            permit_hash: [1u8; 32],
            policy_hash: [2u8; 32],
            receipt_hash: [0u8; 32],
            prev_receipt_hash: [0u8; 32],
            outcome_hash: None,
        };
        r.seal();
        assert!(r.verify_chain(&[0u8; 32]));
        let line = r.to_jsonl();
        let p = ActionReceipt::from_jsonl(&line).expect("must parse");
        assert_eq!(p.action_id, "a1");
        assert_eq!(p.receipt_hash, r.receipt_hash);
    }
}
