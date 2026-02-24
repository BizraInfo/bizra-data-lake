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
use bizra_hooks::{
    ComponentId, Event, EventBus, HookError, HookFn, HookId, HookPhase, HookPipeline, IhsanScore,
    Payload, Priority, Topic,
};
use bizra_telescript::{Authority, Capability, Permit, ResourceLimits};
use serde_json::{json, Value};

pub struct ActionExecutorConfig {
    pub bridge_host: String,
    pub bridge_port: u16,
    pub timeout_ms: u64,
    /// When true, route actions through bizra-action's constitutional Dispatcher
    /// instead of the raw TCP bridge path. Default: false (legacy path).
    pub use_constitutional_dispatcher: bool,
}

impl Default for ActionExecutorConfig {
    fn default() -> Self {
        Self {
            bridge_host: "127.0.0.1".to_string(),
            bridge_port: 9742,
            timeout_ms: 3000,
            use_constitutional_dispatcher: false,
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
    event_bus: EventBus,
    post_deliver_pipeline: HookPipeline,
    executor_component: ComponentId,
    event_ihsan_score: IhsanScore,
    receipt_events_enabled: bool,
    /// Optional KeyVault for secure secret resolution (Phase 4).
    key_vault: Option<KeyVault>,
    /// Whether to write audit log entries on each receipt.
    audit_log_enabled: bool,
    /// Transitional guard: allow direct-audit fallback even when EventBus path is active.
    direct_audit_fallback_on_eventbus: bool,
    /// Optional audit path override (primarily for deterministic tests/migration checks).
    audit_log_path_override: Option<String>,
    /// Production constitutional Dispatcher (bizra-action).
    /// When present and config.use_constitutional_dispatcher is true,
    /// actions route through the 7-gate Guardian pipeline.
    constitutional_dispatcher: Option<bizra_action::Dispatcher>,
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
        let executor_component = ComponentId::from_name("action_executor", "1.0.0");

        let constitutional_dispatcher = if config.use_constitutional_dispatcher {
            Some(crate::action_bridge::create_default_dispatcher())
        } else {
            None
        };

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
            event_bus: EventBus::new(),
            post_deliver_pipeline: HookPipeline::new(),
            executor_component,
            event_ihsan_score: IhsanScore::from_raw(9900),
            receipt_events_enabled: false,
            key_vault: None,
            audit_log_enabled: false,
            direct_audit_fallback_on_eventbus: false,
            audit_log_path_override: None,
            constitutional_dispatcher,
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

    /// Register a PostDeliver hook for receipt events.
    pub fn register_post_deliver_hook(
        &mut self,
        name: &str,
        priority: u8,
        hook_fn: HookFn,
    ) -> Result<HookId, HookError> {
        let hook_id =
            self.post_deliver_pipeline
                .register(HookPhase::PostDeliver, name, priority, hook_fn)?;
        self.receipt_events_enabled = true;
        Ok(hook_id)
    }

    /// Set ihsan score carried in emitted action.receipt events.
    pub fn set_event_ihsan_score(&mut self, score: IhsanScore) {
        self.event_ihsan_score = score;
    }

    /// Enable/disable direct-audit fallback when EventBus path is active.
    pub fn set_direct_audit_fallback_on_eventbus(&mut self, enabled: bool) {
        self.direct_audit_fallback_on_eventbus = enabled;
    }

    /// Override direct-write audit path.
    pub fn set_audit_log_path_override(&mut self, path: Option<String>) {
        self.audit_log_path_override = path;
    }

    /// Number of emitted receipt events (for diagnostics/tests).
    pub fn receipt_events_emitted(&self) -> u64 {
        self.event_bus.total_emitted()
    }

    /// Number of registered PostDeliver hooks.
    pub fn post_deliver_hook_count(&self) -> usize {
        self.post_deliver_pipeline.total_hooks()
    }

    pub fn plan_action(&mut self, payload_json: &str, now: u64) -> Result<ActionPlan, ActionError> {
        let mut plan = self.parse_plan(payload_json, now)?;
        self.plan_seq += 1;
        plan.plan_id = format!("pln_{:08x}", self.plan_seq);
        // When constitutional dispatcher is active, skip legacy executor validation
        // (the Dispatcher has its own channel registry and Guardian gating).
        if self.constitutional_dispatcher.is_none() {
            self.action_bus
                .validate_plan(&plan, &self.permit, &self.usage, now)?;
        } else {
            plan.validate()?;
        }
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

        if self.constitutional_dispatcher.is_none() {
            self.action_bus
                .validate_plan(&plan, &self.permit, &self.usage, now)?;
        } else {
            plan.validate()?;
        }

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
            outcome_hash: None,
        };
        receipt.seal();
        self.prev_receipt_hash = receipt.receipt_hash;
        if self.receipt_events_enabled {
            self.emit_receipt_event(&receipt);
        }
        if self.audit_log_enabled
            && (!self.receipt_events_enabled || self.direct_audit_fallback_on_eventbus)
        {
            self.write_audit_entry(&receipt);
        }
        self.receipts.push(receipt);
    }

    fn emit_receipt_event(&mut self, receipt: &ActionReceipt) {
        let mut payload_bytes = Vec::with_capacity(32 + 1 + receipt.action_id.len());
        payload_bytes.extend_from_slice(&receipt.receipt_hash);
        payload_bytes.push(0x00);
        payload_bytes.extend_from_slice(receipt.action_id.as_bytes());

        let ts_nanos = receipt.timestamp.saturating_mul(1_000_000_000);
        let event = Event {
            id: self.event_bus.next_event_id(ts_nanos),
            source: self.executor_component,
            topic: Topic::new("action.receipt"),
            priority: Priority::High,
            payload: Payload::new(&payload_bytes),
            ihsan_score: self.event_ihsan_score,
        };
        let _ = self.event_bus.emit(event);
        self.post_deliver_pipeline.process_post_delivery(&event);
    }

    /// Write a single receipt to the JSONL audit log.
    fn write_audit_entry(&self, receipt: &ActionReceipt) {
        let outcome_hex: Value = match &receipt.outcome_hash {
            Some(h) => Value::String(h.iter().map(|b| format!("{b:02x}")).collect()),
            None => Value::Null,
        };
        let entry = json!({
            "ts": receipt.timestamp,
            "receipt_hash": receipt.receipt_hash_hex(),
            "action_id": &receipt.action_id,
            "plan_id": &receipt.plan_id,
            "channel": receipt.channel.as_str(),
            "kind": receipt.kind.as_str(),
            "result": &receipt.result,
            "guardian_verdict": receipt.guardian_verdict,
            "outcome_hash": outcome_hex,
        });
        let audit_path = self
            .audit_log_path_override
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(crate::audit_hook::audit_log_path);
        if let Err(e) = crate::audit_hook::append_audit_line(&audit_path, &entry.to_string()) {
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
        // ── Constitutional Dispatcher path (production) ──────
        if let Some(ref mut dispatcher) = self.constitutional_dispatcher {
            let action = crate::action_bridge::translate_step(step)?;
            let permit = bizra_action::Permit::user_default();
            let ihsan = bizra_action::IhsanScore::new(self.event_ihsan_score.as_f64());
            match dispatcher.dispatch(action, permit, ihsan, "action_executor") {
                Ok(_result) => return Ok(()),
                Err(e) => return Err(crate::action_bridge::dispatch_error_to_legacy(&e)),
            }
        }

        // ── Legacy bridge path (MVP) ─────────────────────────
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

    /// Whether the constitutional Dispatcher is active.
    pub fn uses_constitutional_dispatcher(&self) -> bool {
        self.constitutional_dispatcher.is_some()
    }

    /// Get constitutional Dispatcher health (if active).
    pub fn dispatcher_health(&self) -> Option<bizra_action::DispatcherHealth> {
        self.constitutional_dispatcher.as_ref().map(|d| d.health())
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
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    static RECEIPT_HOOK_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static CUSTOM_HOOK_AUDIT_PATH: OnceLock<String> = OnceLock::new();

    fn counting_post_deliver_hook(_event: &Event) -> (bizra_hooks::HookResult, Option<Event>) {
        RECEIPT_HOOK_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
        (bizra_hooks::HookResult::Continue, None)
    }

    fn file_post_deliver_hook(event: &Event) -> (bizra_hooks::HookResult, Option<Event>) {
        let path = match CUSTOM_HOOK_AUDIT_PATH.get() {
            Some(p) => p,
            None => return (bizra_hooks::HookResult::Continue, None),
        };
        let bytes = event.payload.as_bytes();
        if bytes.len() < 32 {
            return (bizra_hooks::HookResult::Continue, None);
        }
        let receipt_hash = bytes[..32]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let action_id = if bytes.len() > 33 && bytes[32] == 0 {
            std::str::from_utf8(&bytes[33..]).unwrap_or("unknown")
        } else {
            "unknown"
        };
        let entry = serde_json::json!({
            "ts": event.id.timestamp_nanos(),
            "receipt_hash": receipt_hash,
            "action_id": action_id,
            "topic": event.topic.as_str(),
        });
        let _ = crate::audit_hook::append_audit_line(path, &entry.to_string());
        (bizra_hooks::HookResult::Continue, None)
    }

    fn denied_step_payload() -> &'static str {
        r#"{"steps":[{"channel":"DesktopRpc","kind":"Click","payload":"rm -rf /"}]}"#
    }

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

    #[test]
    fn receipt_emits_post_deliver_event() {
        RECEIPT_HOOK_INVOCATIONS.store(0, Ordering::Relaxed);
        let mut exec = ActionExecutor::default();
        exec.register_post_deliver_hook("test.count", 0, counting_post_deliver_hook)
            .expect("hook registration");

        let result = exec
            .run_action("", denied_step_payload(), 100, [0u8; 32])
            .expect("run_action should produce denied result");
        assert_eq!(result.status, ActionExecutionStatus::Denied);
        assert_eq!(exec.receipt_events_emitted(), 1);
        assert_eq!(RECEIPT_HOOK_INVOCATIONS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn transitional_guard_prevents_duplicate_audit_writes() {
        let default_path = format!(
            "{}/bizra_audit_guard_action_executor.jsonl",
            std::env::temp_dir().display()
        );
        let hook_path = CUSTOM_HOOK_AUDIT_PATH
            .get_or_init(|| default_path.clone())
            .clone();
        let path = std::path::PathBuf::from(&hook_path);
        let _ = fs::remove_file(&path);

        let mut exec = ActionExecutor::default().with_audit();
        exec.set_audit_log_path_override(Some(hook_path.clone()));
        exec.register_post_deliver_hook("audit.receipt", 0, file_post_deliver_hook)
            .expect("hook registration");
        exec.set_direct_audit_fallback_on_eventbus(false);

        let result = exec
            .run_action("", denied_step_payload(), 200, [0u8; 32])
            .expect("run_action should produce denied result");
        assert_eq!(result.status, ActionExecutionStatus::Denied);

        let lines = fs::read_to_string(&path)
            .expect("audit file should exist")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        assert_eq!(lines, 1);

        let _ = fs::remove_file(&path);
    }
}
