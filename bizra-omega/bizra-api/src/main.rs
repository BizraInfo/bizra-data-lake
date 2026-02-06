//! BIZRA Sovereign API Server
//!
//! Production-ready REST/WebSocket server for BIZRA sovereign operations.
//!
//! Run: cargo run -p bizra-api --release
//! Or:  ./target/release/bizra-api

use std::sync::Arc;
use std::net::SocketAddr;
use ed25519_dalek::SigningKey;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use bizra_core::{NodeIdentity, Constitution};
use bizra_inference::{
    InferenceGateway,
    backends::{BackendConfig, ollama::OllamaBackend},
    selector::ModelTier,
};
use bizra_federation::{GossipProtocol, ConsensusEngine};
use bizra_api::{AppState, ServerConfig, serve};

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
    let port = args.iter()
        .position(|a| a == "--port" || a == "-p")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3001);

    let host = args.iter()
        .position(|a| a == "--host")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("0.0.0.0");

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
        let model = detect_ollama_model().await.unwrap_or_else(|| "qwen2.5:7b".into());
        let ollama_config = BackendConfig {
            name: "ollama-local".into(),
            model: model.clone(),
            context_length: 4096,
            gpu_layers: -1,
        };
        let ollama = Arc::new(OllamaBackend::new(ollama_config, None));
        gateway.register_backend(ModelTier::Local, ollama.clone()).await;
        gateway.register_backend(ModelTier::Edge, ollama).await;
        tracing::info!(model = %model, "Ollama backend registered");
    } else {
        tracing::warn!("Ollama not available - inference will return errors");
    }

    // Initialize gossip protocol with signing key for secure message authentication
    let gossip_addr: SocketAddr = format!("{}:7946", host).parse()?;
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
            .with_identity(identity).await
            .with_gateway(gateway).await
            .with_gossip(gossip).await
            .with_consensus(consensus).await
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
    println!("   │  BIZRA Sovereign API Server v{}                        │", env!("CARGO_PKG_VERSION"));
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
    println!("\n   Listening on: http://{}:{}\n", host, port);

    // Start server
    serve(config, state).await?;

    Ok(())
}

fn print_banner() {
    println!(r#"
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
    "#);
}

fn load_or_create_identity_bytes() -> anyhow::Result<[u8; 32]> {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let identity_dir = home.join(".bizra");
    let identity_file = identity_dir.join("identity.key");

    if identity_file.exists() {
        let hex_key = std::fs::read_to_string(&identity_file)?;
        let secret = hex::decode(hex_key.trim())?;
        let secret_array: [u8; 32] = secret.try_into()
            .map_err(|_| anyhow::anyhow!("Invalid key length"))?;
        Ok(secret_array)
    } else {
        std::fs::create_dir_all(&identity_dir)?;
        let identity = NodeIdentity::generate();
        let secret_bytes = identity.secret_bytes();
        std::fs::write(&identity_file, hex::encode(&secret_bytes))?;
        tracing::info!(path = %identity_file.display(), "New identity created");
        Ok(secret_bytes)
    }
}

async fn check_ollama_available() -> bool {
    let client = reqwest::Client::new();
    let base_url = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".into());

    match client.get(format!("{}/api/tags", base_url)).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

async fn detect_ollama_model() -> Option<String> {
    let client = reqwest::Client::new();
    let base_url = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".into());

    #[derive(serde::Deserialize)]
    struct TagsResponse {
        models: Vec<ModelInfo>,
    }

    #[derive(serde::Deserialize)]
    struct ModelInfo {
        name: String,
    }

    let resp = client.get(format!("{}/api/tags", base_url))
        .send()
        .await
        .ok()?;

    let tags: TagsResponse = resp.json().await.ok()?;

    // Prefer Qwen, then Llama, then any available
    let preferred = ["qwen", "llama", "mistral", "phi"];

    for pref in preferred {
        if let Some(model) = tags.models.iter().find(|m| m.name.to_lowercase().contains(pref)) {
            return Some(model.name.clone());
        }
    }

    tags.models.first().map(|m| m.name.clone())
}
