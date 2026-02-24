// bizra-node/src/action_bridge.rs
// ============================================================
// Action Bridge — translates MVP types → production Dispatcher
// ============================================================
//
// The bridge maps bizra-agent's PlannedStep (5 channels, string payload)
// to bizra-action's BizraAction (8 channels, typed variants), enabling
// a non-breaking migration path from the MVP ActionBus to the production
// constitutional Dispatcher.
//
// Standing on:
//   bizra-agent  v0.1.0 → MVP action types
//   bizra-action v0.1.0 → production Dispatcher + Guardian
//
// Integration strategy:
//   PlannedStep → translate() → BizraAction → Dispatcher.dispatch()
//   ActionResult (bizra-action) → into_legacy_result() → ActionResult (bizra-agent)
// ============================================================

use bizra_action::{BizraAction, Channel, DispatchError, Dispatcher};
use bizra_agent::action_types::{
    ActionChannel, ActionError, ActionExecutionStatus, ActionKind, ActionResult as LegacyResult,
    PlannedStep,
};

// ── Channel Mapping ─────────────────────────────────────────

/// Map MVP ActionChannel → production Channel.
pub fn map_channel(ch: ActionChannel) -> Channel {
    match ch {
        ActionChannel::DesktopRpc => Channel::Ahk,
        ActionChannel::ToolCall => Channel::Mcp,
        ActionChannel::LlmCall => Channel::Llm,
        ActionChannel::FileOp => Channel::FileSystem,
        ActionChannel::BrowserNav => Channel::Browser,
    }
}

/// Map production Channel → MVP ActionChannel (best-effort).
pub fn map_channel_reverse(ch: Channel) -> ActionChannel {
    match ch {
        Channel::Ahk => ActionChannel::DesktopRpc,
        Channel::Llm => ActionChannel::LlmCall,
        Channel::Memory => ActionChannel::ToolCall, // No direct MVP equivalent
        Channel::Mcp => ActionChannel::ToolCall,
        Channel::FileSystem => ActionChannel::FileOp,
        Channel::Browser => ActionChannel::BrowserNav,
        Channel::Response => ActionChannel::ToolCall, // Best-effort
        Channel::Telescript => ActionChannel::ToolCall, // Best-effort
    }
}

// ── Step Translation ────────────────────────────────────────

/// Translate a PlannedStep into a BizraAction.
///
/// The payload string is parsed as JSON where possible. Falls back
/// to sensible defaults for minimal payloads.
pub fn translate_step(step: &PlannedStep) -> Result<BizraAction, ActionError> {
    match (step.channel, step.kind) {
        // ── Desktop ─────────────────────────────────────
        (ActionChannel::DesktopRpc, ActionKind::Click) => {
            let (window, element) = parse_desktop_payload(&step.payload);
            Ok(BizraAction::AhkClick {
                window,
                element_path: element,
            })
        }
        (ActionChannel::DesktopRpc, ActionKind::TypeText) => {
            let (window, text) = parse_desktop_payload(&step.payload);
            Ok(BizraAction::AhkType {
                window,
                element_path: String::new(),
                text,
            })
        }
        (ActionChannel::DesktopRpc, ActionKind::InvokeSkill) => {
            let (exe, args) = parse_launch_payload(&step.payload);
            Ok(BizraAction::AhkLaunch {
                executable: exe,
                args,
            })
        }
        (ActionChannel::DesktopRpc, ActionKind::Query) => Ok(BizraAction::AhkPerceive),

        // ── LLM ─────────────────────────────────────────
        (ActionChannel::LlmCall, _) => {
            let (model, prompt) = parse_llm_payload(&step.payload);
            Ok(BizraAction::LlmQuery {
                provider: "local".to_string(),
                model,
                system_prompt: String::new(),
                user_prompt: prompt,
                max_tokens: 2048,
                temperature: 0.7,
            })
        }

        // ── Tool / MCP ──────────────────────────────────
        (ActionChannel::ToolCall, _) => {
            let (server, tool, arguments) = parse_tool_payload(&step.payload);
            Ok(BizraAction::McpToolCall {
                server,
                tool_name: tool,
                arguments,
            })
        }

        // ── File ────────────────────────────────────────
        (ActionChannel::FileOp, ActionKind::WriteFile) => {
            let (path, content) = parse_file_payload(&step.payload);
            Ok(BizraAction::FileCreate {
                path,
                content: content.into_bytes(),
            })
        }
        (ActionChannel::FileOp, ActionKind::Query) => {
            let (path, _) = parse_file_payload(&step.payload);
            Ok(BizraAction::FileRead { path })
        }
        (ActionChannel::FileOp, _) => {
            let (path, _) = parse_file_payload(&step.payload);
            Ok(BizraAction::FileRead { path })
        }

        // ── Browser ─────────────────────────────────────
        (ActionChannel::BrowserNav, ActionKind::Navigate) => {
            let url = extract_string_field(&step.payload, "url")
                .unwrap_or_else(|| step.payload.trim().to_string());
            Ok(BizraAction::BrowserNavigate { url })
        }
        (ActionChannel::BrowserNav, _) => {
            let url = extract_string_field(&step.payload, "url")
                .unwrap_or_else(|| step.payload.trim().to_string());
            Ok(BizraAction::BrowserFetch {
                url,
                method: "GET".to_string(),
                headers: Vec::new(),
            })
        }

        // ── Catch-all ───────────────────────────────────
        _ => Err(ActionError::new(
            "UNTRANSLATABLE",
            &format!(
                "Cannot translate {}:{} to production action",
                step.channel.as_str(),
                step.kind.as_str()
            ),
        )),
    }
}

// ── Result Translation ──────────────────────────────────────

/// Convert a production ActionResult into a legacy ActionResult.
pub fn into_legacy_result(
    result: &bizra_action::ActionResult,
    plan_id: &str,
    started_at: u64,
    now: u64,
) -> LegacyResult {
    let status = if result.success {
        ActionExecutionStatus::Completed
    } else {
        ActionExecutionStatus::Failed
    };
    let message = match &result.payload {
        bizra_action::ActionPayload::Text(t) => t.clone(),
        bizra_action::ActionPayload::Error(e) => e.clone(),
        bizra_action::ActionPayload::Empty => "ok".to_string(),
        bizra_action::ActionPayload::Bytes(b) => format!("{} bytes", b.len()),
        bizra_action::ActionPayload::Structured { entries } => {
            format!("{} entries", entries.len())
        }
    };

    LegacyResult {
        action_id: format!("act_{:08x}", result.action_id.0),
        plan_id: plan_id.to_string(),
        status,
        message,
        started_at,
        finished_at: now,
    }
}

/// Convert a DispatchError into a legacy ActionError.
pub fn dispatch_error_to_legacy(err: &DispatchError) -> ActionError {
    match err {
        DispatchError::GuardianDenied { reason, .. } => ActionError::new("GUARDIAN_DENIED", reason),
        DispatchError::HitlRequired { summary, .. } => ActionError::new("HITL_REQUIRED", summary),
        DispatchError::ChannelNotRegistered { channel } => ActionError::new(
            "CHANNEL_NOT_REGISTERED",
            &format!("No handler for channel: {:?}", channel),
        ),
        DispatchError::ChannelUnavailable { channel, status } => ActionError::new(
            "CHANNEL_UNAVAILABLE",
            &format!("{:?} unavailable: {}", channel, status),
        ),
    }
}

// ── Dispatcher Factory ──────────────────────────────────────

/// Create a production Dispatcher with all stub channels registered.
pub fn create_default_dispatcher() -> Dispatcher {
    use bizra_action::channels::*;

    let mut d = Dispatcher::new();
    d.register_channel(Box::new(AhkChannel::new()));
    d.register_channel(Box::new(LlmChannel::new()));
    d.register_channel(Box::new(MemoryChannel::new()));
    d.register_channel(Box::new(McpChannel::new()));
    d.register_channel(Box::new(FileSystemChannel::new()));
    d.register_channel(Box::new(BrowserChannel::new()));
    d.register_channel(Box::new(ResponseChannel::new()));
    d.register_channel(Box::new(TelescriptChannel::new()));
    d
}

/// Create a strict Dispatcher for visiting agents.
pub fn create_strict_dispatcher() -> Dispatcher {
    use bizra_action::channels::*;

    let mut d = Dispatcher::strict();
    // Only register safe channels for visitors
    d.register_channel(Box::new(LlmChannel::new()));
    d.register_channel(Box::new(MemoryChannel::new()));
    d.register_channel(Box::new(ResponseChannel::new()));
    d
}

// ── Payload Parsers (minimal JSON extraction) ───────────────

fn extract_string_field(json: &str, key: &str) -> Option<String> {
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

fn parse_desktop_payload(payload: &str) -> (String, String) {
    let window = extract_string_field(payload, "target_app")
        .or_else(|| extract_string_field(payload, "window"))
        .unwrap_or_else(|| "foreground".to_string());
    let element = extract_string_field(payload, "target")
        .or_else(|| extract_string_field(payload, "code"))
        .or_else(|| extract_string_field(payload, "element"))
        .unwrap_or_else(|| payload.trim().to_string());
    (window, element)
}

fn parse_llm_payload(payload: &str) -> (String, String) {
    let model = extract_string_field(payload, "model").unwrap_or_else(|| "default".to_string());
    let prompt = extract_string_field(payload, "prompt")
        .or_else(|| extract_string_field(payload, "query"))
        .unwrap_or_else(|| payload.trim().to_string());
    (model, prompt)
}

fn parse_tool_payload(payload: &str) -> (String, String, String) {
    let server = extract_string_field(payload, "server").unwrap_or_else(|| "localhost".to_string());
    let tool = extract_string_field(payload, "skill")
        .or_else(|| extract_string_field(payload, "tool"))
        .unwrap_or_else(|| "unknown".to_string());
    let arguments = extract_string_field(payload, "inputs")
        .or_else(|| extract_string_field(payload, "arguments"))
        .unwrap_or_else(|| "{}".to_string());
    (server, tool, arguments)
}

fn parse_file_payload(payload: &str) -> (String, String) {
    let path = extract_string_field(payload, "path").unwrap_or_else(|| payload.trim().to_string());
    let content = extract_string_field(payload, "content").unwrap_or_default();
    (path, content)
}

fn parse_launch_payload(payload: &str) -> (String, Vec<String>) {
    let exe = extract_string_field(payload, "skill")
        .or_else(|| extract_string_field(payload, "executable"))
        .unwrap_or_else(|| payload.trim().to_string());
    // Args parsing: simplified — production would parse a JSON array
    let args = extract_string_field(payload, "args")
        .map(|a| vec![a])
        .unwrap_or_default();
    (exe, args)
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_action::{IhsanScore as ActionIhsan, Permit};
    use bizra_agent::action_types::{ActionChannel, ActionKind, PlannedStep};

    #[test]
    fn translate_desktop_click() {
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: r#"{"target":"OK","target_app":"Notepad"}"#.to_string(),
        };
        let action = translate_step(&step).expect("should translate");
        assert_eq!(action.channel(), Channel::Ahk);
        assert!(action.summary().contains("Notepad"));
    }

    #[test]
    fn translate_llm_query() {
        let step = PlannedStep {
            channel: ActionChannel::LlmCall,
            kind: ActionKind::Query,
            payload: r#"{"model":"qwen2","prompt":"hello"}"#.to_string(),
        };
        let action = translate_step(&step).expect("should translate");
        assert_eq!(action.channel(), Channel::Llm);
        assert!(action.summary().contains("qwen2"));
    }

    #[test]
    fn translate_tool_call() {
        let step = PlannedStep {
            channel: ActionChannel::ToolCall,
            kind: ActionKind::ToolCall,
            payload: r#"{"skill":"file_search","inputs":"{}"}"#.to_string(),
        };
        let action = translate_step(&step).expect("should translate");
        assert_eq!(action.channel(), Channel::Mcp);
    }

    #[test]
    fn translate_file_write() {
        let step = PlannedStep {
            channel: ActionChannel::FileOp,
            kind: ActionKind::WriteFile,
            payload: r#"{"path":"/tmp/test.txt","content":"hello"}"#.to_string(),
        };
        let action = translate_step(&step).expect("should translate");
        assert_eq!(action.channel(), Channel::FileSystem);
    }

    #[test]
    fn translate_browser_nav() {
        let step = PlannedStep {
            channel: ActionChannel::BrowserNav,
            kind: ActionKind::Navigate,
            payload: r#"{"url":"https://example.com"}"#.to_string(),
        };
        let action = translate_step(&step).expect("should translate");
        assert_eq!(action.channel(), Channel::Browser);
    }

    #[test]
    fn channel_mapping_roundtrip() {
        for ch in [
            ActionChannel::DesktopRpc,
            ActionChannel::ToolCall,
            ActionChannel::LlmCall,
            ActionChannel::FileOp,
            ActionChannel::BrowserNav,
        ] {
            let prod = map_channel(ch);
            let back = map_channel_reverse(prod);
            // DesktopRpc→Ahk→DesktopRpc, LlmCall→Llm→LlmCall, etc.
            assert_eq!(back, ch, "Channel roundtrip failed for {:?}", ch);
        }
    }

    #[test]
    fn dispatcher_factory_creates_all_channels() {
        let d = create_default_dispatcher();
        let status = d.channel_status();
        assert_eq!(status.len(), 8, "All 8 channels should be registered");
    }

    #[test]
    fn strict_dispatcher_limits_channels() {
        let d = create_strict_dispatcher();
        let status = d.channel_status();
        assert_eq!(status.len(), 3, "Strict dispatcher should have 3 channels");
    }

    #[test]
    fn full_bridge_dispatch_cycle() {
        let mut dispatcher = create_default_dispatcher();
        let step = PlannedStep {
            channel: ActionChannel::LlmCall,
            kind: ActionKind::Query,
            payload: r#"{"model":"test","prompt":"hello world"}"#.to_string(),
        };

        let action = translate_step(&step).expect("translation");
        let permit = Permit::user_default();
        let ihsan = ActionIhsan::new(0.97);

        let result = dispatcher
            .dispatch(action, permit, ihsan, "bridge_test")
            .expect("dispatch should succeed with stub channel");

        assert!(result.success);
        assert!(result.duration_ns > 0);

        let legacy = into_legacy_result(&result, "pln_test", 100, 200);
        assert_eq!(legacy.status, ActionExecutionStatus::Completed);
    }

    #[test]
    fn bridge_guardian_denies_low_ihsan() {
        let mut dispatcher = create_default_dispatcher();
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: r#"{"target":"ok","target_app":"Notepad"}"#.to_string(),
        };

        let action = translate_step(&step).expect("translation");
        let permit = Permit::user_default();
        // AHK is High risk → requires 0.98 Ihsan
        let ihsan = ActionIhsan::new(0.50);

        let err = dispatcher.dispatch(action, permit, ihsan, "bridge_test");
        assert!(err.is_err());

        let legacy_err = dispatch_error_to_legacy(&err.unwrap_err());
        assert_eq!(legacy_err.code, "GUARDIAN_DENIED");
    }
}
