//! BIZRA Sovereign API Server
//!
//! Production-ready REST/WebSocket server for BIZRA sovereign operations.
//!
//! Run: cargo run -p bizra-api --release
//! Or:  ./target/release/bizra-api

use ed25519_dalek::SigningKey;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use bizra_api::{serve, AppState, ServerConfig};
use bizra_core::{Constitution, NodeIdentity};
use bizra_federation::{ConsensusEngine, GossipProtocol};
use bizra_inference::{
    backends::{ollama::OllamaBackend, BackendConfig},
    selector::ModelTier,
    InferenceGateway,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "bizra_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let port = args
        .iter()
        .position(|a| a == "--port" || a == "-p")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3001);

    let host = args
        .iter()
        .position(|a| a == "--host")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("127.0.0.1");

    ensure_auth_configuration()?;

    // Display banner
    print_banner();

    // Load identity secret (to allow creating multiple instances)
    let secret_bytes = load_or_create_identity_bytes()?;
    let identity = NodeIdentity::from_secret_bytes(&secret_bytes);
    tracing::info!(node_id = %identity.node_id().0, "Node identity loaded");

    // Initialize constitution
    let constitution = Constitution::default();
    tracing::info!(
        ihsan = constitution.ihsan.minimum,
        snr = constitution.snr_threshold,
        "Constitution initialized"
    );

    // Initialize inference gateway (create fresh identity for gateway)
    let gateway_identity = NodeIdentity::from_secret_bytes(&secret_bytes);
    let gateway = InferenceGateway::new(gateway_identity, constitution.clone());

    // Register Ollama backend if available
    if check_ollama_available().await {
        let model = detect_ollama_model()
            .await
            .unwrap_or_else(|| "qwen2.5:7b".into());
        let ollama_config = BackendConfig {
            name: "ollama-local".into(),
            model: model.clone(),
            context_length: 4096,
            gpu_layers: -1,
        };
        let ollama = Arc::new(OllamaBackend::new(ollama_config, None));
        gateway
            .register_backend(ModelTier::Local, ollama.clone())
            .await;
        gateway.register_backend(ModelTier::Edge, ollama).await;
        tracing::info!(model = %model, "Ollama backend registered");
    } else {
        tracing::warn!("Ollama not available - inference will return errors");
    }

    // Initialize gossip protocol with signing key for secure message authentication
    let gossip_addr: SocketAddr = format!("{host}:7946").parse()?;
    let gossip_signing_key = SigningKey::from_bytes(&secret_bytes);
    let gossip = GossipProtocol::new(identity.node_id().clone(), gossip_addr, gossip_signing_key);
    tracing::info!(addr = %gossip_addr, "Gossip protocol initialized with Ed25519 signing");

    // Initialize consensus engine (create fresh identity for consensus)
    let consensus_identity = NodeIdentity::from_secret_bytes(&secret_bytes);
    let consensus = ConsensusEngine::new(consensus_identity);
    tracing::info!("Consensus engine initialized");

    // Build application state
    let state = Arc::new(
        AppState::new(constitution)
            .with_identity(identity)
            .await
            .with_gateway(gateway)
            .await
            .with_gossip(gossip)
            .await
            .with_consensus(consensus)
            .await,
    );

    // Server configuration
    let config = ServerConfig {
        host: host.to_string(),
        port,
        enable_metrics: true,
        max_connections: 10000,
        request_timeout_ms: 30000,
    };

    // Print startup info
    println!("\n   ┌─────────────────────────────────────────────────────────────┐");
    println!(
        "   │  BIZRA Sovereign API Server v{}                        │",
        env!("CARGO_PKG_VERSION")
    );
    println!("   ├─────────────────────────────────────────────────────────────┤");
    println!("   │  Endpoints:                                                 │");
    println!("   │    GET  /api/v1/health          Health check                │");
    println!("   │    GET  /api/v1/status          System status               │");
    println!("   │    POST /api/v1/identity/*      Identity operations         │");
    println!("   │    POST /api/v1/pci/*           PCI protocol                │");
    println!("   │    POST /api/v1/inference/*     LLM inference               │");
    println!("   │    GET  /api/v1/federation/*    Federation status           │");
    println!("   │    GET  /api/v1/ws              WebSocket                   │");
    println!("   └─────────────────────────────────────────────────────────────┘");
    println!("\n   Listening on: http://{host}:{port}\n");

    // Start server
    serve(config, state).await?;

    Ok(())
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn ensure_auth_configuration() -> anyhow::Result<()> {
    let token_set = std::env::var("BIZRA_API_TOKEN")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .is_some();
    if token_set {
        return Ok(());
    }

    if parse_env_bool("BIZRA_API_ALLOW_INSECURE_DEV") {
        tracing::warn!(
            "BIZRA_API_TOKEN is unset; running in insecure dev mode because BIZRA_API_ALLOW_INSECURE_DEV=true"
        );
        return Ok(());
    }

    anyhow::bail!(
        "BIZRA_API_TOKEN is required for startup. Set BIZRA_API_TOKEN or explicitly set BIZRA_API_ALLOW_INSECURE_DEV=true for local development."
    );
}

fn print_banner() {
    println!(
        r#"
   ╔══════════════════════════════════════════════════════════════════════╗
   ║   ██████╗ ██╗███████╗██████╗  █████╗                                 ║
   ║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                                ║
   ║   ██████╔╝██║  ███╔╝ ██████╔╝███████║                                ║
   ║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                                ║
   ║   ██████╔╝██║███████╗██║  ██║██║  ██║                                ║
   ║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                                ║
   ║                                                                      ║
   ║   Sovereign API Gateway — Every human is a node, every node is a seed║
   ╚══════════════════════════════════════════════════════════════════════╝
    "#
    );
}

fn load_or_create_identity_bytes() -> anyhow::Result<[u8; 32]> {
    if let Ok(secret_hex) = std::env::var("NODE_SECRET") {
        let secret_array = parse_identity_hex(secret_hex.trim())?;
        tracing::info!("Node identity loaded from NODE_SECRET");
        return Ok(secret_array);
    }

    let identity_dir = identity_directory();
    load_or_create_identity_bytes_at(&identity_dir)
}

fn identity_directory() -> PathBuf {
    if let Ok(dir) = std::env::var("BIZRA_IDENTITY_DIR") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".bizra")
}

fn parse_identity_hex(hex_key: &str) -> anyhow::Result<[u8; 32]> {
    let secret = hex::decode(hex_key.trim())?;
    let secret_array: [u8; 32] = secret
        .try_into()
        .map_err(|_| anyhow::anyhow!("NODE_SECRET must decode to exactly 32 bytes"))?;
    Ok(secret_array)
}

fn load_or_create_identity_bytes_at(identity_dir: &Path) -> anyhow::Result<[u8; 32]> {
    let identity_file = identity_dir.join("identity.key");

    if identity_file.exists() {
        let hex_key = std::fs::read_to_string(&identity_file)?;
        let secret_array = parse_identity_hex(hex_key.trim())?;
        enforce_private_permissions(&identity_file)?;
        return Ok(secret_array);
    }

    std::fs::create_dir_all(identity_dir)?;
    let identity = NodeIdentity::generate();
    let secret_bytes = identity.secret_bytes();
    let hex_key = hex::encode(secret_bytes);

    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&identity_file)?;
        file.write_all(hex_key.as_bytes())?;
    }

    #[cfg(not(unix))]
    {
        std::fs::write(&identity_file, hex_key)?;
    }

    enforce_private_permissions(&identity_file)?;
    tracing::warn!(
        path = %identity_file.display(),
        "Auto-generated identity key; set NODE_SECRET for deterministic identity"
    );
    Ok(secret_bytes)
}

#[cfg(unix)]
fn enforce_private_permissions(identity_file: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::metadata(identity_file)?;
    let mut permissions = metadata.permissions();
    if permissions.mode() & 0o777 != 0o600 {
        permissions.set_mode(0o600);
        std::fs::set_permissions(identity_file, permissions)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn enforce_private_permissions(_identity_file: &Path) -> anyhow::Result<()> {
    Ok(())
}

async fn check_ollama_available() -> bool {
    let client = reqwest::Client::new();
    let base_url = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());

    match client.get(format!("{base_url}/api/tags")).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

async fn detect_ollama_model() -> Option<String> {
    let client = reqwest::Client::new();
    let base_url = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());

    #[derive(serde::Deserialize)]
    struct TagsResponse {
        models: Vec<ModelInfo>,
    }

    #[derive(serde::Deserialize)]
    struct ModelInfo {
        name: String,
    }

    let resp = client
        .get(format!("{base_url}/api/tags"))
        .send()
        .await
        .ok()?;

    let tags: TagsResponse = resp.json().await.ok()?;

    // Prefer Qwen, then Llama, then any available
    let preferred = ["qwen", "llama", "mistral", "phi"];

    for pref in preferred {
        if let Some(model) = tags
            .models
            .iter()
            .find(|m| m.name.to_lowercase().contains(pref))
        {
            return Some(model.name.clone());
        }
    }

    tags.models.first().map(|m| m.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_identity_hex_accepts_32_byte_secret() {
        let secret_hex = "11".repeat(32);
        let secret = parse_identity_hex(&secret_hex).expect("valid NODE_SECRET hex");
        assert_eq!(secret.len(), 32);
        assert!(secret.iter().all(|b| *b == 0x11));
    }

    #[test]
    fn node_secret_env_overrides_disk_identity() {
        let secret_hex = "22".repeat(32);
        std::env::set_var("NODE_SECRET", &secret_hex);

        let secret = load_or_create_identity_bytes().expect("env-backed identity");
        std::env::remove_var("NODE_SECRET");

        assert_eq!(secret.len(), 32);
        assert!(secret.iter().all(|b| *b == 0x22));
    }

    #[test]
    fn generated_identity_file_permissions_are_restricted() {
        let dir = tempdir().expect("tempdir");
        let identity = load_or_create_identity_bytes_at(dir.path()).expect("identity creation");
        assert_eq!(identity.len(), 32);

        let path = dir.path().join("identity.key");
        assert!(path.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }
}
