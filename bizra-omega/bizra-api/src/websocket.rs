//! WebSocket Handler for Real-Time Updates

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use std::sync::Arc;

use crate::state::AppState;

/// WebSocket upgrade handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected");

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "welcome",
        "version": env!("CARGO_PKG_VERSION"),
        "node_id": state.identity.read().await.as_ref().map(|i| i.node_id().0.clone()),
    });

    if socket
        .send(Message::Text(welcome.to_string()))
        .await
        .is_err()
    {
        return;
    }

    // Main message loop
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse and handle message
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                    let msg_type = parsed.get("type").and_then(|t| t.as_str());

                    match msg_type {
                        Some("ping") => {
                            let pong = serde_json::json!({
                                "type": "pong",
                                "timestamp": chrono::Utc::now().to_rfc3339(),
                            });
                            if socket.send(Message::Text(pong.to_string())).await.is_err() {
                                break;
                            }
                        }
                        Some("subscribe") => {
                            let topic = parsed.get("topic").and_then(|t| t.as_str());
                            let ack = serde_json::json!({
                                "type": "subscribed",
                                "topic": topic,
                            });
                            if socket.send(Message::Text(ack.to_string())).await.is_err() {
                                break;
                            }
                        }
                        Some("status") => {
                            let status = serde_json::json!({
                                "type": "status",
                                "uptime_secs": state.uptime_secs(),
                                "requests": state.get_request_count(),
                            });
                            if socket
                                .send(Message::Text(status.to_string()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        _ => {
                            let error = serde_json::json!({
                                "type": "error",
                                "message": "Unknown message type",
                            });
                            let _ = socket.send(Message::Text(error.to_string())).await;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => {}
        }
    }

    tracing::info!("WebSocket client disconnected");
}

/// Event types for broadcasting
#[derive(Clone, Debug)]
pub enum ServerEvent {
    InferenceComplete {
        request_id: String,
        model: String,
        tokens: usize,
    },
    PatternElevated {
        pattern_id: String,
        votes: usize,
    },
    PeerJoined {
        node_id: String,
    },
    PeerLeft {
        node_id: String,
    },
}
