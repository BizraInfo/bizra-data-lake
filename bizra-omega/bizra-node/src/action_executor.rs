// bizra-node/src/action_executor.rs
// ============================================================
// Action Executor — protocol-facing Action Bus adapter
// ============================================================

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use bizra_agent::action_bus::ActionBus;
use bizra_agent::action_types::{
    ActionChannel, ActionError, ActionExecutionStatus, ActionKind, ActionPlan, ActionReceipt,
    ActionResult, PlannedStep,
};
use bizra_agent::hash_namespace::parse_hex_32;
use bizra_agent::key_vault::KeyVault;
use bizra_agent::permit_guard::PermitUsage;
use bizra_telescript::{Authority, Capability, Permit, ResourceLimits};
use serde_json::{json, Value};

pub struct ActionExecutorConfig {
    pub bridge_host: String,
    pub bridge_port: u16,
    pub timeout_ms: u64,
}

impl Default for ActionExecutorConfig {
    fn default() -> Self {
        Self {
            bridge_host: "127.0.0.1".to_string(),
            bridge_port: 9742,
            timeout_ms: 3000,
        }
    }
}

pub struct ActionExecutor {
    config: ActionExecutorConfig,
    action_bus: ActionBus,
    permit: Permit,
    usage: PermitUsage,
    plan_seq: u64,
    action_seq: u64,
    plans: HashMap<String, ActionPlan>,
    actions: HashMap<String, ActionResult>,
    receipts: Vec<ActionReceipt>,
    prev_receipt_hash: [u8; 32],
    /// Optional KeyVault for secure secret resolution (Phase 4).
    key_vault: Option<KeyVault>,
    /// Whether to write audit log entries on each receipt.
    audit_log_enabled: bool,
}

impl ActionExecutor {
    pub fn new(config: ActionExecutorConfig) -> Self {
        let mut action_bus = ActionBus::default();
        // Explicitly lock executors for this milestone.
        action_bus.unregister_executor(ActionChannel::LlmCall);
        action_bus.unregister_executor(ActionChannel::FileOp);
        action_bus.unregister_executor(ActionChannel::BrowserNav);

        let permit = Permit::new(
            Authority::genesis(),
            vec![Capability::Compute, Capability::Network, Capability::Store],
            ResourceLimits::default(),
            600,
        );

        Self {
            config,
            action_bus,
            permit,
            usage: PermitUsage::default(),
            plan_seq: 0,
            action_seq: 0,
            plans: HashMap::new(),
            actions: HashMap::new(),
            receipts: Vec::new(),
            prev_receipt_hash: [0u8; 32],
            key_vault: None,
            audit_log_enabled: false,
        }
    }

    /// Create an executor with audit logging enabled.
    pub fn with_audit(mut self) -> Self {
        self.audit_log_enabled = true;
        self
    }

    /// Create an executor with a KeyVault for secret resolution.
    pub fn with_vault(mut self, vault: KeyVault) -> Self {
        self.key_vault = Some(vault);
        self
    }

    /// Whether audit logging is enabled.
    pub fn audit_log_enabled(&self) -> bool {
        self.audit_log_enabled
    }

    /// Set audit logging on/off at runtime.
    pub fn set_audit_log_enabled(&mut self, enabled: bool) {
        self.audit_log_enabled = enabled;
    }

    pub fn plan_action(&mut self, payload_json: &str, now: u64) -> Result<ActionPlan, ActionError> {
        let mut plan = self.parse_plan(payload_json, now)?;
        self.plan_seq += 1;
        plan.plan_id = format!("pln_{:08x}", self.plan_seq);
        self.action_bus
            .validate_plan(&plan, &self.permit, &self.usage, now)?;
        self.plans.insert(plan.plan_id.clone(), plan.clone());
        Ok(plan)
    }

    pub fn run_action(
        &mut self,
        plan_id: &str,
        payload_json: &str,
        now: u64,
        policy_hash: [u8; 32],
    ) -> Result<ActionResult, ActionError> {
        let plan = if plan_id.trim().is_empty() {
            self.parse_plan(payload_json, now)?
        } else if let Some(existing) = self.plans.get(plan_id) {
            existing.clone()
        } else {
            return Err(ActionError::new("PLAN_NOT_FOUND", "Plan not found"));
        };

        self.action_bus
            .validate_plan(&plan, &self.permit, &self.usage, now)?;

        self.action_seq += 1;
        let action_id = format!("act_{:08x}", self.action_seq);
        let started_at = now;
        let mut final_status = ActionExecutionStatus::Completed;
        let mut final_message = "ok".to_string();

        for step in &plan.steps {
            let guard_ok = self.guardian_allows(step);
            if !guard_ok {
                final_status = ActionExecutionStatus::Denied;
                final_message = "Guardian veto".to_string();
                self.append_receipt(
                    &action_id,
                    &plan.plan_id,
                    step,
                    now,
                    "denied:guardian_veto",
                    false,
                    policy_hash,
                );
                break;
            }

            let exec = self.execute_step(step, now);
            match exec {
                Ok(_) => {
                    self.append_receipt(
                        &action_id,
                        &plan.plan_id,
                        step,
                        now,
                        "ok",
                        true,
                        policy_hash,
                    );
                }
                Err(err) => {
                    final_status = ActionExecutionStatus::Failed;
                    final_message = err.message;
                    self.append_receipt(
                        &action_id,
                        &plan.plan_id,
                        step,
                        now,
                        format!("err:{}", err.code).as_str(),
                        true,
                        policy_hash,
                    );
                    break;
                }
            }
        }

        let result = ActionResult {
            action_id: action_id.clone(),
            plan_id: plan.plan_id.clone(),
            status: final_status,
            message: final_message,
            started_at,
            finished_at: now,
        };
        self.actions.insert(action_id, result.clone());
        Ok(result)
    }

    pub fn action_status(&self, action_id: &str) -> Option<&ActionResult> {
        self.actions.get(action_id)
    }

    pub fn action_history(&self, limit: u32, cursor: &str) -> (Vec<ActionReceipt>, String) {
        if self.receipts.is_empty() {
            return (Vec::new(), String::new());
        }

        let end = if cursor.trim().is_empty() {
            self.receipts.len()
        } else {
            cursor
                .parse::<usize>()
                .ok()
                .map(|v| v.min(self.receipts.len()))
                .unwrap_or(self.receipts.len())
        };
        let lim = limit.max(1) as usize;
        let start = end.saturating_sub(lim);
        let slice = self.receipts[start..end].to_vec();
        let next_cursor = start.to_string();
        (slice, next_cursor)
    }

    pub fn import_receipts(&mut self, receipts: Vec<ActionReceipt>) {
        // Derive sequence counters from restored receipts to avoid ID collisions.
        for r in &receipts {
            if let Some(n) = r
                .action_id
                .strip_prefix("act_")
                .and_then(|s| u64::from_str_radix(s, 16).ok())
            {
                self.action_seq = self.action_seq.max(n);
            }
            if let Some(n) = r
                .plan_id
                .strip_prefix("pln_")
                .and_then(|s| u64::from_str_radix(s, 16).ok())
            {
                self.plan_seq = self.plan_seq.max(n);
            }
        }
        self.prev_receipt_hash = receipts.last().map(|r| r.receipt_hash).unwrap_or([0u8; 32]);
        self.receipts = receipts;
    }

    pub fn receipts(&self) -> &[ActionReceipt] {
        &self.receipts
    }

    #[allow(clippy::too_many_arguments)]
    fn append_receipt(
        &mut self,
        action_id: &str,
        plan_id: &str,
        step: &PlannedStep,
        now: u64,
        result: &str,
        guardian_verdict: bool,
        policy_hash: [u8; 32],
    ) {
        let mut receipt = ActionReceipt {
            action_id: action_id.to_string(),
            plan_id: plan_id.to_string(),
            channel: step.channel,
            kind: step.kind,
            timestamp: now,
            result: result.to_string(),
            guardian_verdict,
            permit_hash: self.permit.permit_hash,
            policy_hash,
            receipt_hash: [0u8; 32],
            prev_receipt_hash: self.prev_receipt_hash,
        };
        receipt.seal();
        self.prev_receipt_hash = receipt.receipt_hash;
        if self.audit_log_enabled {
            self.write_audit_entry(&receipt);
        }
        self.receipts.push(receipt);
    }

    /// Write a single receipt to the JSONL audit log.
    fn write_audit_entry(&self, receipt: &ActionReceipt) {
        let entry = json!({
            "ts": receipt.timestamp,
            "receipt_hash": receipt.receipt_hash_hex(),
            "action_id": &receipt.action_id,
            "plan_id": &receipt.plan_id,
            "channel": receipt.channel.as_str(),
            "kind": receipt.kind.as_str(),
            "result": &receipt.result,
            "guardian_verdict": receipt.guardian_verdict,
        });
        if let Err(e) = crate::audit_hook::append_audit_line(
            crate::audit_hook::AUDIT_LOG_PATH,
            &entry.to_string(),
        ) {
            eprintln!("[WARN] Audit log write failed: {e}");
        }
    }

    fn parse_plan(&self, payload_json: &str, now: u64) -> Result<ActionPlan, ActionError> {
        let value: Value = serde_json::from_str(payload_json)
            .map_err(|_| ActionError::new("BAD_JSON", "Invalid JSON payload"))?;

        let steps = if let Some(raw_steps) = value.get("steps").and_then(|v| v.as_array()) {
            let mut out = Vec::new();
            for raw in raw_steps {
                out.push(self.parse_step(raw)?);
            }
            out
        } else {
            vec![self.parse_step(&value)?]
        };

        let plan = ActionPlan {
            plan_id: "pending".to_string(),
            created_at: now,
            steps,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn parse_step(&self, value: &Value) -> Result<PlannedStep, ActionError> {
        let channel = value
            .get("channel")
            .and_then(|v| v.as_str())
            .and_then(ActionChannel::parse)
            .ok_or_else(|| ActionError::new("INVALID_CHANNEL", "Missing or invalid channel"))?;
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .and_then(ActionKind::parse)
            .ok_or_else(|| ActionError::new("INVALID_KIND", "Missing or invalid kind"))?;

        let payload = value
            .get("payload")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()));
        let payload = if payload.is_string() {
            payload.as_str().unwrap_or_default().to_string()
        } else {
            payload.to_string()
        };

        Ok(PlannedStep {
            channel,
            kind,
            payload,
        })
    }

    fn guardian_allows(&self, step: &PlannedStep) -> bool {
        let payload = step.payload.to_ascii_lowercase();
        let blocked = ["rm -rf", "format c:", "bypass safety", "override guardian"];
        !blocked.iter().any(|x| payload.contains(x))
    }

    fn execute_step(&mut self, step: &PlannedStep, now: u64) -> Result<(), ActionError> {
        let _ = self
            .action_bus
            .dispatch_step(step, &self.permit, &mut self.usage, now)?;

        match (step.channel, step.kind) {
            (ActionChannel::DesktopRpc, ActionKind::InvokeSkill) => {
                let payload: Value = serde_json::from_str(step.payload.as_str())
                    .map_err(|_| ActionError::new("BAD_PAYLOAD", "Invalid skill payload JSON"))?;
                let skill = payload
                    .get("skill")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ActionError::new("BAD_PAYLOAD", "payload.skill is required"))?;
                let inputs = payload.get("inputs").cloned().unwrap_or_else(|| json!({}));
                let params = json!({ "skill": skill, "inputs": inputs });
                let _ = self.call_bridge("invoke_skill", params)?;
                Ok(())
            }
            (ActionChannel::DesktopRpc, _) => {
                let payload: Value = serde_json::from_str(step.payload.as_str())
                    .unwrap_or_else(|_| json!({ "code": step.payload }));
                let code = payload
                    .get("code")
                    .and_then(|v| v.as_str())
                    .or_else(|| payload.get("target").and_then(|v| v.as_str()))
                    .unwrap_or(step.payload.as_str());
                let intent = payload
                    .get("intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("execute");
                let target_app = payload.get("target_app").cloned().unwrap_or(Value::Null);
                let params = json!({
                    "code": code,
                    "intent": intent,
                    "target_app": target_app,
                });
                let _ = self.call_bridge("actuator_execute", params)?;
                Ok(())
            }
            (ActionChannel::ToolCall, _) => {
                let payload: Value = serde_json::from_str(step.payload.as_str())
                    .map_err(|_| ActionError::new("BAD_PAYLOAD", "Invalid tool payload JSON"))?;
                let skill = payload
                    .get("skill")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ActionError::new("BAD_PAYLOAD", "payload.skill is required"))?;
                let inputs = payload.get("inputs").cloned().unwrap_or_else(|| json!({}));
                let params = json!({ "skill": skill, "inputs": inputs });
                let _ = self.call_bridge("invoke_skill", params)?;
                Ok(())
            }
            _ => Err(ActionError::new(
                "UNAVAILABLE_EXECUTOR",
                "No safe executor available for requested channel/kind",
            )),
        }
    }

    fn call_bridge(&mut self, method: &str, params: Value) -> Result<Value, ActionError> {
        // Resolve bridge token via vault (preferred) or env var (fallback).
        let token = if let Some(vault) = &mut self.key_vault {
            vault
                .get("bridge_token")
                .map(|s| s.expose().to_string())
                .map_err(|e| ActionError::new("MISSING_BRIDGE_TOKEN", &format!("vault: {e}")))?
        } else {
            std::env::var("BIZRA_BRIDGE_TOKEN").map_err(|_| {
                ActionError::new("MISSING_BRIDGE_TOKEN", "BIZRA_BRIDGE_TOKEN not set")
            })?
        };
        let ts = chrono_like_now_ms();
        let nonce = format!("n{}", ts);
        let request = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": ts,
            "headers": {
                "X-BIZRA-TOKEN": token,
                "X-BIZRA-TS": ts,
                "X-BIZRA-NONCE": nonce,
            }
        });

        let addr = resolve_addr(self.config.bridge_host.as_str(), self.config.bridge_port)
            .ok_or_else(|| {
                ActionError::new("BRIDGE_UNREACHABLE", "Bridge address resolve failed")
            })?;
        let mut stream =
            TcpStream::connect_timeout(&addr, Duration::from_millis(self.config.timeout_ms))
                .map_err(|_| {
                    ActionError::new("BRIDGE_UNREACHABLE", "Desktop bridge unavailable")
                })?;
        let _ = stream.set_read_timeout(Some(Duration::from_millis(self.config.timeout_ms)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(self.config.timeout_ms)));

        let mut wire = request.to_string();
        wire.push('\n');
        stream
            .write_all(wire.as_bytes())
            .map_err(|_| ActionError::new("BRIDGE_WRITE_FAILED", "Failed to write to bridge"))?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|_| {
            ActionError::new("BRIDGE_READ_FAILED", "Failed to read bridge response")
        })?;
        if line.trim().is_empty() {
            return Err(ActionError::new(
                "BRIDGE_EMPTY_RESPONSE",
                "Bridge returned empty response",
            ));
        }

        let value: Value = serde_json::from_str(line.trim())
            .map_err(|_| ActionError::new("BRIDGE_BAD_JSON", "Bridge returned invalid JSON"))?;
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Bridge error");
            return Err(ActionError::new("BRIDGE_ERROR", msg));
        }
        let result = value.get("result").cloned().ok_or_else(|| {
            ActionError::new("BRIDGE_MISSING_RESULT", "Bridge response missing result")
        })?;
        // Check for application-level denial inside the result payload.
        if let Some(err) = result.get("error") {
            let msg = err
                .as_str()
                .or_else(|| err.get("message").and_then(|v| v.as_str()))
                .unwrap_or("Bridge execution denied");
            return Err(ActionError::new("BRIDGE_DENIED", msg));
        }
        Ok(result)
    }
}

impl Default for ActionExecutor {
    fn default() -> Self {
        Self::new(ActionExecutorConfig::default())
    }
}

impl ActionExecutor {
    /// Mutable reference to the KeyVault (if set).
    pub fn vault_mut(&mut self) -> Option<&mut KeyVault> {
        self.key_vault.as_mut()
    }
}

fn resolve_addr(host: &str, port: u16) -> Option<SocketAddr> {
    (host, port).to_socket_addrs().ok()?.next()
}

fn chrono_like_now_ms() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    now as i64
}

pub fn parse_policy_hash_hex(hex: Option<String>) -> [u8; 32] {
    hex.and_then(|h| parse_hex_32(h.as_str()))
        .unwrap_or([0u8; 32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_and_history_roundtrip() {
        let mut exec = ActionExecutor::default();
        let plan = exec
            .plan_action(
                r#"{"steps":[{"channel":"DesktopRpc","kind":"Click","payload":{"code":"click button"}}]}"#,
                100,
            )
            .expect("plan should parse");
        assert!(plan.plan_id.starts_with("pln_"));
    }

    #[test]
    fn policy_hash_parse_fallback() {
        let out = parse_policy_hash_hex(None);
        assert_eq!(out, [0u8; 32]);
    }
}
