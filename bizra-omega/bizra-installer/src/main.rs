//! BIZRA Universal CLI
//!
//! Complete command-line interface for BIZRA Sovereign Node operations.
//!
//! Giants: Torvalds (Unix philosophy), Pike (Go CLI patterns), Stallman (GNU)

mod hardware_detect;
mod model_cache;

use clap::{Args, Parser, Subcommand};
use hardware_detect::detect_hardware;
use model_cache::ModelSpec;
use std::fs;
use std::path::PathBuf;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_DATA_DIR: &str = ".bizra";
const DEFAULT_API_PORT: u16 = 3001;
const DEFAULT_GOSSIP_PORT: u16 = 7946;

#[derive(Parser)]
#[command(name = "bizra")]
#[command(author = "BIZRA Collective")]
#[command(version)]
#[command(about = "BIZRA Sovereign Node — Every human is a node, every node is a seed")]
#[command(long_about = r#"
BIZRA (بذرة) Sovereign Node CLI

A complete command-line interface for operating BIZRA sovereign infrastructure:
  • Identity management (Ed25519 + BLAKE3)
  • PCI Protocol (Proof-Carrying Inference)
  • Inference gateway (tiered model access)
  • Federation network (SWIM gossip + BFT consensus)

Examples:
  bizra init                    Initialize node with new identity
  bizra serve                   Start API server on port 3001
  bizra serve --port 8080       Start on custom port
  bizra join 192.168.1.100      Join federation via seed node
  bizra status                  Show node status
  bizra inference "Hello"       Run inference locally
"#)]
struct Cli {
    /// Data directory
    #[arg(short, long, default_value = DEFAULT_DATA_DIR, global = true)]
    data_dir: PathBuf,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new BIZRA node
    Init {
        /// Force reinitialize (overwrites existing)
        #[arg(short, long)]
        force: bool,
    },

    /// Start the API server
    Serve(ServeArgs),

    /// Join a federation network
    Join {
        /// Seed node address (host:port)
        seed: String,

        /// Local gossip port
        #[arg(short, long, default_value_t = DEFAULT_GOSSIP_PORT)]
        port: u16,
    },

    /// Show node status
    Status {
        /// Show detailed status
        #[arg(short, long)]
        detailed: bool,
    },

    /// Detect hardware capabilities
    Detect,

    /// Manage models
    Models {
        #[command(subcommand)]
        action: ModelCommands,
    },

    /// Run inference
    Inference(InferenceArgs),

    /// Federation operations
    Federation {
        #[command(subcommand)]
        action: FederationCommands,
    },

    /// Identity operations
    Identity {
        #[command(subcommand)]
        action: IdentityCommands,
    },

    /// PCI Protocol operations
    Pci {
        #[command(subcommand)]
        action: PCICommands,
    },

    /// Constitution and governance
    Constitution {
        /// Show current thresholds
        #[arg(long)]
        thresholds: bool,
    },
}

#[derive(Args)]
struct ServeArgs {
    /// API port
    #[arg(short, long, default_value_t = DEFAULT_API_PORT)]
    port: u16,

    /// Bind host
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Initialize identity on startup
    #[arg(long)]
    init_identity: bool,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,

    /// Daemonize (run in background)
    #[arg(short, long)]
    daemon: bool,
}

#[derive(Args)]
struct InferenceArgs {
    /// Prompt text
    prompt: String,

    /// System prompt
    #[arg(short, long)]
    system: Option<String>,

    /// Maximum tokens
    #[arg(short, long, default_value_t = 512)]
    max_tokens: usize,

    /// Model tier (edge, local, pool)
    #[arg(short, long, default_value = "auto")]
    tier: String,

    /// Temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

#[derive(Subcommand)]
enum ModelCommands {
    /// List available models
    List,
    /// Download a model
    Download {
        /// Model name
        name: String,
    },
    /// Show loaded models
    Loaded,
    /// Unload a model
    Unload {
        /// Model name
        name: String,
    },
}

#[derive(Subcommand)]
enum FederationCommands {
    /// Show federation status
    Status,
    /// List known peers
    Peers,
    /// Propose a pattern for elevation
    Propose {
        /// Pattern ID
        pattern_id: String,
    },
    /// Leave the federation
    Leave,
}

#[derive(Subcommand)]
enum IdentityCommands {
    /// Show current identity
    Show,
    /// Generate new identity
    Generate {
        /// Force overwrite existing
        #[arg(short, long)]
        force: bool,
    },
    /// Export identity (DANGEROUS)
    Export {
        /// Output file
        output: PathBuf,
    },
    /// Import identity
    Import {
        /// Input file
        input: PathBuf,
    },
    /// Sign a message
    Sign {
        /// Message to sign
        message: String,
    },
    /// Verify a signature
    Verify {
        /// Message
        message: String,
        /// Signature (hex)
        signature: String,
        /// Public key (hex)
        public_key: String,
    },
}

#[derive(Subcommand)]
enum PCICommands {
    /// Create a PCI envelope
    Create {
        /// Payload (JSON)
        payload: String,
        /// TTL in seconds
        #[arg(long, default_value_t = 3600)]
        ttl: u64,
    },
    /// Verify a PCI envelope
    Verify {
        /// Envelope (JSON file or stdin)
        envelope: Option<PathBuf>,
    },
    /// Check gates
    Gates {
        /// Content to check
        content: String,
        /// SNR score
        #[arg(long)]
        snr: Option<f64>,
        /// Ihsan score
        #[arg(long)]
        ihsan: Option<f64>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Expand home directory
    let data_dir = expand_home(&cli.data_dir);

    match cli.command {
        Commands::Init { force } => cmd_init(&data_dir, force).await?,
        Commands::Serve(args) => cmd_serve(&data_dir, args).await?,
        Commands::Join { seed, port } => cmd_join(&data_dir, &seed, port).await?,
        Commands::Status { detailed } => cmd_status(&data_dir, detailed).await?,
        Commands::Detect => cmd_detect().await?,
        Commands::Models { action } => cmd_models(action).await?,
        Commands::Inference(args) => cmd_inference(&data_dir, args).await?,
        Commands::Federation { action } => cmd_federation(&data_dir, action).await?,
        Commands::Identity { action } => cmd_identity(&data_dir, action).await?,
        Commands::Pci { action } => cmd_pci(&data_dir, action).await?,
        Commands::Constitution { thresholds } => cmd_constitution(thresholds).await?,
    }

    Ok(())
}

fn expand_home(path: &PathBuf) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str.starts_with("~/") || path_str == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.join(path_str.trim_start_matches("~/"));
        }
    }
    path.clone()
}

async fn cmd_init(data_dir: &PathBuf, force: bool) -> anyhow::Result<()> {
    println!(
        r#"
    ╔══════════════════════════════════════════════════════════════╗
    ║   ██████╗ ██╗███████╗██████╗  █████╗                         ║
    ║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                        ║
    ║   ██████╔╝██║  ███╔╝ ██████╔╝███████║                        ║
    ║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                        ║
    ║   ██████╔╝██║███████╗██║  ██║██║  ██║                        ║
    ║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                        ║
    ╚══════════════════════════════════════════════════════════════╝
    "#
    );

    println!("    Initializing BIZRA Sovereign Node v{}\n", VERSION);

    let identity_file = data_dir.join("identity.key");

    if identity_file.exists() && !force {
        println!(
            "    ⚠️  Identity already exists at {}",
            identity_file.display()
        );
        println!("    Use --force to reinitialize (WILL LOSE CURRENT IDENTITY)");
        return Ok(());
    }

    // Create data directory
    fs::create_dir_all(data_dir)?;
    fs::create_dir_all(data_dir.join("models"))?;
    fs::create_dir_all(data_dir.join("data"))?;
    fs::create_dir_all(data_dir.join("logs"))?;

    // Generate identity
    use bizra_core::NodeIdentity;
    let identity = NodeIdentity::generate();

    println!("    ✓ Node ID:     {}", identity.node_id().0);
    println!("    ✓ Public Key:  {}...", &identity.public_key_hex()[..16]);

    // Save identity (encrypted in production)
    let secret = identity.secret_bytes();
    fs::write(&identity_file, hex::encode(secret))?;

    // Create config
    let config = format!(
        r#"# BIZRA Node Configuration
version: "{}"
node_id: "{}"
data_dir: "{}"

api:
  host: "0.0.0.0"
  port: {}

federation:
  gossip_port: {}
  bootstrap_nodes: []

inference:
  default_tier: "auto"
  gpu_layers: -1

constitution:
  ihsan_threshold: 0.95
  snr_threshold: 0.85
"#,
        VERSION,
        identity.node_id().0,
        data_dir.display(),
        DEFAULT_API_PORT,
        DEFAULT_GOSSIP_PORT
    );

    fs::write(data_dir.join("config.yaml"), config)?;

    println!("\n    ✓ Data directory: {}", data_dir.display());
    println!("    ✓ Configuration saved");
    println!("\n    🌱 Node initialized. Run 'bizra serve' to start.");

    Ok(())
}

async fn cmd_serve(data_dir: &PathBuf, args: ServeArgs) -> anyhow::Result<()> {
    println!("🚀 Starting BIZRA API Server...\n");

    let identity_file = data_dir.join("identity.key");
    if !identity_file.exists() && !args.init_identity {
        println!("⚠️  No identity found. Run 'bizra init' first or use --init-identity");
        return Ok(());
    }

    println!("   Host: {}:{}", args.host, args.port);
    println!("   Data: {}", data_dir.display());

    if args.daemon {
        println!("   Mode: Daemon (background)");
        // In production: fork and detach
        println!("\n   [Daemon mode not yet implemented - running in foreground]");
    }

    // Launch the API server
    // In production: use bizra_api::serve()
    println!("\n   API Endpoints:");
    println!("     GET  /api/v1/health");
    println!("     POST /api/v1/identity/generate");
    println!("     POST /api/v1/pci/envelope/create");
    println!("     POST /api/v1/inference/generate");
    println!("     GET  /api/v1/federation/status");
    println!("\n   Press Ctrl+C to stop\n");

    // Keep running (in production: run actual server)
    tokio::signal::ctrl_c().await?;
    println!("\n   Shutting down...");

    Ok(())
}

async fn cmd_join(_data_dir: &PathBuf, seed: &str, port: u16) -> anyhow::Result<()> {
    println!("🔗 Joining federation...\n");
    println!("   Seed node: {}", seed);
    println!("   Local port: {}", port);

    // In production: initialize gossip protocol
    println!("\n   [Federation join not yet implemented]");
    println!(
        "   Would connect to {} and start gossip on port {}",
        seed, port
    );

    Ok(())
}

async fn cmd_status(data_dir: &PathBuf, detailed: bool) -> anyhow::Result<()> {
    let identity_file = data_dir.join("identity.key");

    println!("📊 BIZRA Node Status\n");

    if !data_dir.exists() {
        println!("   Status: Not initialized");
        println!("   Run 'bizra init' to initialize");
        return Ok(());
    }

    println!("   Data Dir:  {}", data_dir.display());

    if identity_file.exists() {
        // Load and show identity
        let hex_key = fs::read_to_string(&identity_file)?;
        let secret = hex::decode(hex_key.trim())?;
        let secret_array: [u8; 32] = secret
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid key"))?;

        use bizra_core::NodeIdentity;
        let identity = NodeIdentity::from_secret_bytes(&secret_array);

        println!("   Node ID:   {}", identity.node_id().0);
        println!("   Public:    {}...", &identity.public_key_hex()[..16]);
    } else {
        println!("   Identity:  Not generated");
    }

    // Check API status
    println!("   API:       Not running");
    println!("   Federation: Not connected");

    if detailed {
        println!("\n   Hardware:");
        let hw = detect_hardware();
        println!("     CPU: {} cores", hw.cpu_cores);
        println!("     RAM: {:.1} GB", hw.ram_gb);
        println!("     GPU: {}", hw.gpu_name.as_deref().unwrap_or("None"));
        println!("     Tier: {}", hw.recommended_tier());
    }

    Ok(())
}

async fn cmd_detect() -> anyhow::Result<()> {
    println!("🔍 Hardware Detection\n");

    let profile = detect_hardware();

    println!("   CPU Cores:  {}", profile.cpu_cores);
    println!("   RAM:        {:.1} GB", profile.ram_gb);
    println!(
        "   GPU:        {}",
        profile.gpu_name.as_deref().unwrap_or("None detected")
    );
    println!("   VRAM:       {:.1} GB", profile.vram_gb);

    println!("\n   Recommended Configuration:");
    println!("     Tier:  {}", profile.recommended_tier());
    println!("     Model: {}", profile.recommended_model());

    Ok(())
}

async fn cmd_models(action: ModelCommands) -> anyhow::Result<()> {
    match action {
        ModelCommands::List => {
            println!("📋 Available Models\n");
            println!(
                "   {:20} {:10} {:8} DESCRIPTION",
                "NAME", "TIER", "SIZE"
            );
            println!("   {}", "-".repeat(60));
            for m in ModelSpec::available() {
                println!(
                    "   {:20} {:10} {:6.1} GB {}",
                    m.name, m.tier, m.size_gb, m.desc
                );
            }
        }
        ModelCommands::Download { name } => {
            println!("⬇️  Downloading model: {}\n", name);
            println!("   [Download not yet implemented]");
        }
        ModelCommands::Loaded => {
            println!("📦 Loaded Models\n");
            println!("   [No models currently loaded]");
        }
        ModelCommands::Unload { name } => {
            println!("🗑️  Unloading model: {}", name);
        }
    }
    Ok(())
}

async fn cmd_inference(_data_dir: &PathBuf, args: InferenceArgs) -> anyhow::Result<()> {
    println!("🧠 Running Inference\n");
    println!("   Tier:        {}", args.tier);
    println!("   Max Tokens:  {}", args.max_tokens);
    println!("   Temperature: {}", args.temperature);
    println!(
        "   Prompt:      {}...",
        &args.prompt[..args.prompt.len().min(50)]
    );

    if let Some(sys) = &args.system {
        println!("   System:      {}...", &sys[..sys.len().min(30)]);
    }

    println!("\n   [Inference not yet connected to backend]");
    println!("   Start API server with 'bizra serve' for full inference");

    Ok(())
}

async fn cmd_federation(_data_dir: &PathBuf, action: FederationCommands) -> anyhow::Result<()> {
    match action {
        FederationCommands::Status => {
            println!("🌐 Federation Status\n");
            println!("   Connected: No");
            println!("   Peers:     0");
            println!("   Patterns:  0");
        }
        FederationCommands::Peers => {
            println!("👥 Federation Peers\n");
            println!("   [Not connected to federation]");
        }
        FederationCommands::Propose { pattern_id } => {
            println!("📤 Proposing pattern: {}", pattern_id);
            println!("   [Federation not connected]");
        }
        FederationCommands::Leave => {
            println!("👋 Leaving federation...");
            println!("   [Not connected]");
        }
    }
    Ok(())
}

async fn cmd_identity(data_dir: &PathBuf, action: IdentityCommands) -> anyhow::Result<()> {
    use bizra_core::NodeIdentity;

    match action {
        IdentityCommands::Show => {
            let identity_file = data_dir.join("identity.key");
            if !identity_file.exists() {
                println!("⚠️  No identity. Run 'bizra init' or 'bizra identity generate'");
                return Ok(());
            }

            let hex_key = fs::read_to_string(&identity_file)?;
            let secret = hex::decode(hex_key.trim())?;
            let secret_array: [u8; 32] = secret
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid key"))?;
            let identity = NodeIdentity::from_secret_bytes(&secret_array);

            println!("🔑 Current Identity\n");
            println!("   Node ID:    {}", identity.node_id().0);
            println!("   Public Key: {}", identity.public_key_hex());
        }
        IdentityCommands::Generate { force } => {
            let identity_file = data_dir.join("identity.key");
            if identity_file.exists() && !force {
                println!("⚠️  Identity exists. Use --force to overwrite");
                return Ok(());
            }

            fs::create_dir_all(data_dir)?;
            let identity = NodeIdentity::generate();

            fs::write(&identity_file, hex::encode(identity.secret_bytes()))?;

            println!("✓ Generated new identity");
            println!("  Node ID:    {}", identity.node_id().0);
            println!("  Public Key: {}", identity.public_key_hex());
        }
        IdentityCommands::Export { output } => {
            println!("⚠️  SECURITY WARNING: Exporting private key!\n");
            let identity_file = data_dir.join("identity.key");
            fs::copy(&identity_file, &output)?;
            println!("   Exported to: {}", output.display());
        }
        IdentityCommands::Import { input } => {
            let identity_file = data_dir.join("identity.key");
            fs::create_dir_all(data_dir)?;
            fs::copy(&input, &identity_file)?;
            println!("✓ Identity imported from {}", input.display());
        }
        IdentityCommands::Sign { message } => {
            let identity_file = data_dir.join("identity.key");
            let hex_key = fs::read_to_string(&identity_file)?;
            let secret = hex::decode(hex_key.trim())?;
            let secret_array: [u8; 32] = secret
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid key"))?;
            let identity = NodeIdentity::from_secret_bytes(&secret_array);

            let signature = identity.sign(message.as_bytes());
            println!("{}", signature);
        }
        IdentityCommands::Verify {
            message,
            signature,
            public_key,
        } => {
            let valid = NodeIdentity::verify_with_hex(message.as_bytes(), &signature, &public_key);
            if valid {
                println!("✓ Signature valid");
            } else {
                println!("✗ Signature INVALID");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

async fn cmd_pci(_data_dir: &PathBuf, action: PCICommands) -> anyhow::Result<()> {
    match action {
        PCICommands::Create { payload, ttl } => {
            println!("📦 Creating PCI Envelope\n");
            println!("   Payload: {}...", &payload[..payload.len().min(50)]);
            println!("   TTL:     {} seconds", ttl);
            println!("\n   [Requires initialized identity - use 'bizra init' first]");
        }
        PCICommands::Verify { envelope } => {
            println!("🔍 Verifying PCI Envelope\n");
            if let Some(path) = envelope {
                println!("   From file: {}", path.display());
            } else {
                println!("   From stdin...");
            }
            println!("\n   [Verification not yet implemented]");
        }
        PCICommands::Gates {
            content,
            snr,
            ihsan,
        } => {
            println!("🚧 Checking Gates\n");
            println!("   Content: {}...", &content[..content.len().min(50)]);
            if let Some(s) = snr {
                println!("   SNR:     {}", s);
            }
            if let Some(i) = ihsan {
                println!("   Ihsan:   {}", i);
            }
            println!("\n   [Gate check not yet implemented]");
        }
    }
    Ok(())
}

async fn cmd_constitution(thresholds: bool) -> anyhow::Result<()> {
    println!("📜 BIZRA Constitution\n");

    println!("   Version: 1.0");
    println!(
        "   Ihsan Threshold:  {} (Excellence)",
        bizra_core::IHSAN_THRESHOLD
    );
    println!(
        "   SNR Threshold:    {} (Signal Quality)",
        bizra_core::SNR_THRESHOLD
    );

    if thresholds {
        println!("\n   Rules:");
        println!("     R001: All outputs must meet Ihsan threshold (≥ 0.95)");
        println!("     R002: Signal-to-noise ratio must exceed threshold (≥ 0.85)");
        println!("     R003: All messages must be signed with Ed25519");
        println!("     R004: Full provenance chain required for derived content");
        println!("     R005: Byzantine consensus requires 2f+1 votes");
    }

    Ok(())
}
