//! BIZRA Resource Pool Node
//!
//! CLI for running a resource pool node.
//!
//! # Usage
//!
//! ```bash
//! # Start as Genesis node (Node0)
//! resourcepool-node genesis --name "Node0-MoMo"
//!
//! # Start Genesis using existing identity
//! resourcepool-node genesis --identity-file /mnt/c/BIZRA-DATA-LAKE/sovereign_state/node0_genesis.json
//!
//! # Join existing pool
//! resourcepool-node join --sponsor <sponsor_node_id> --pool <pool_address>
//!
//! # Contribute resources
//! resourcepool-node contribute --cpu 4000 --gpu 24.0 --memory 128G
//!
//! # Check Zakat status
//! resourcepool-node zakat status
//!
//! # List Harberger market
//! resourcepool-node market list
//! ```

use bizra_resourcepool::*;
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

const DEFAULT_IDENTITY_PATH: &str = "/mnt/c/BIZRA-DATA-LAKE/sovereign_state/node0_genesis.json";

#[derive(Debug, Deserialize)]
struct IdentityFile {
    identity: IdentityRecord,
}

#[derive(Debug, Deserialize)]
struct IdentityRecord {
    public_key: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .compact()
        .init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "genesis" => run_genesis(&args[2..]).await?,
        "status" => run_status().await?,
        "version" => print_version(),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
        }
    }

    Ok(())
}

async fn run_genesis(args: &[String]) -> anyhow::Result<()> {
    let mut name: Option<String> = None;
    let mut identity_file: Option<String> = None;
    let mut public_key_hex: Option<String> = None;
    let mut signing_key_hex: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                if i + 1 >= args.len() {
                    return Err(anyhow::anyhow!("--name requires a value"));
                }
                name = Some(args[i + 1].clone());
                i += 2;
            }
            "--identity-file" => {
                if i + 1 >= args.len() {
                    return Err(anyhow::anyhow!("--identity-file requires a path"));
                }
                identity_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--public-key" => {
                if i + 1 >= args.len() {
                    return Err(anyhow::anyhow!("--public-key requires a hex value"));
                }
                public_key_hex = Some(args[i + 1].clone());
                i += 2;
            }
            "--signing-key" => {
                if i + 1 >= args.len() {
                    return Err(anyhow::anyhow!("--signing-key requires a hex value"));
                }
                signing_key_hex = Some(args[i + 1].clone());
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    if identity_file.is_none() && Path::new(DEFAULT_IDENTITY_PATH).exists() {
        identity_file = Some(DEFAULT_IDENTITY_PATH.to_string());
    }

    let identity = if let Some(path) = &identity_file {
        Some(load_identity(Path::new(path))?)
    } else {
        None
    };

    if public_key_hex.is_none() {
        if let Some(id) = &identity {
            public_key_hex = Some(id.public_key.clone());
        }
    }

    if name.is_none() {
        if let Some(id) = &identity {
            if let Some(n) = &id.name {
                name = Some(n.clone());
            }
        }
    }

    let genesis_name = name.unwrap_or_else(|| "Node0-Genesis".to_string());

    info!("Initializing Genesis Resource Pool...");

    let (signing_key, verifying_key, node_id) = resolve_genesis_identity(
        signing_key_hex.as_deref(),
        public_key_hex.as_deref(),
    )?;

    if let Some(id) = &identity {
        if let Some(node_id_hint) = &id.node_id {
            if node_id_hint != &node_id {
                info!(
                    identity_node_id = %node_id_hint,
                    derived_node_id = %node_id,
                    "Identity node_id differs; using public key hex as node_id"
                );
            }
        }
    }

    info!(node_id = %node_id, "Genesis node ID prepared");

    // Create the Resource Pool
    let pool = ResourcePool::genesis(node_id.clone(), genesis_name.clone(), verifying_key).await?;

    let state = pool.state().await;
    let stats = pool.stats().await;

    println!("\n========================================");
    println!("  BIZRA RESOURCE POOL - GENESIS COMPLETE");
    println!("========================================\n");
    println!("Pool ID:        {}", state.pool_id);
    println!("Pool Name:      {}", state.name);
    println!("Genesis Node:   {}", state.genesis_node);
    println!("Genesis Time:   {}", state.genesis_at);
    println!("Initial Supply: {} tokens", state.total_supply);
    println!("\nPool Statistics:");
    println!("  Total Nodes:    {}", stats.total_nodes);
    println!("  Active Nodes:   {}", stats.active_nodes);
    println!("  Total Staked:   {} tokens", stats.total_staked);
    println!("  Avg Ihsan:      {}", stats.avg_ihsan);
    println!("\nGenesis Node Details:");
    println!("  Name:           {}", genesis_name);
    println!("  Node ID:        {}", node_id);
    println!("  Class:          {:?}", NodeClass::Genesis);

    if let Some(signing_key) = signing_key {
        println!("\n[IMPORTANT] Save your signing key securely!");
        println!("Signing Key (hex): {}", hex::encode(signing_key.to_bytes()));
    } else {
        println!("\n[WARNING] No signing key provided.");
        println!("Provide --signing-key <hex> to persist signing authority.");
    }

    println!("\n========================================\n");

    // Keep running (in production, would start network listener)
    info!("Genesis node running. Press Ctrl+C to exit.");

    // For demo, just show PAT creation
    let pat = pool.create_pat(&node_id).await?;
    info!(owner = %pat.owner_node, "Personal Agent Team (PAT) created with {} slots", PAT_SIZE);

    // Process initial Zakat cycle
    let zakat = pool.process_zakat().await?;
    info!(collected = zakat.total_collected, "Initial Zakat cycle processed");

    Ok(())
}

fn resolve_genesis_identity(
    signing_key_hex: Option<&str>,
    public_key_hex: Option<&str>,
) -> anyhow::Result<(Option<SigningKey>, VerifyingKey, String)> {
    if let Some(sk_hex) = signing_key_hex {
        let signing_key = parse_signing_key(sk_hex)?;
        let verifying_key = signing_key.verifying_key();
        let node_id = hex::encode(verifying_key.as_bytes());
        return Ok((Some(signing_key), verifying_key, node_id));
    }

    if let Some(pk_hex) = public_key_hex {
        let verifying_key = parse_verifying_key(pk_hex)?;
        let node_id = hex::encode(verifying_key.as_bytes());
        return Ok((None, verifying_key, node_id));
    }

    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    let node_id = hex::encode(verifying_key.as_bytes());
    Ok((Some(signing_key), verifying_key, node_id))
}

fn parse_signing_key(sk_hex: &str) -> anyhow::Result<SigningKey> {
    let sk_bytes = hex::decode(sk_hex)?;
    let sk_arr: [u8; 32] = sk_bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("Signing key must be 32 bytes"))?;
    Ok(SigningKey::from_bytes(&sk_arr))
}

fn parse_verifying_key(pk_hex: &str) -> anyhow::Result<VerifyingKey> {
    let pk_bytes = hex::decode(pk_hex)?;
    let pk_arr: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("Public key must be 32 bytes"))?;
    Ok(VerifyingKey::from_bytes(&pk_arr)? )
}

fn load_identity(path: &Path) -> anyhow::Result<IdentityRecord> {
    let data = fs::read_to_string(path)?;
    let parsed: IdentityFile = serde_json::from_str(&data)?;
    Ok(parsed.identity)
}

async fn run_status() -> anyhow::Result<()> {
    println!("Resource Pool Status");
    println!("--------------------");
    println!("Status: Not connected to a pool");
    println!("Use 'resourcepool-node genesis' to create a new pool");
    println!("Use 'resourcepool-node join' to join an existing pool");
    Ok(())
}

fn print_version() {
    println!("bizra-resourcepool v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("The Universal Fabric Where All Nodes Connect");
    println!();
    println!("Five Pillars:");
    println!("  1. Universal Financial System (Islamic finance)");
    println!("  2. Agent Marketplace (PAT/SAT)");
    println!("  3. Compute Commons (Proof-of-Resource)");
    println!("  4. MMORPG World Map (Telescript Places)");
    println!("  5. Web4 Infrastructure (Sovereign Internet)");
}

fn print_usage() {
    println!("BIZRA Resource Pool Node");
    println!();
    println!("Usage: resourcepool-node <command> [options]");
    println!();
    println!("Commands:");
    println!("  genesis           Create a new Resource Pool");
    println!("  join              Join an existing Resource Pool");
    println!("  contribute        Contribute resources to the pool");
    println!("  status            Show node and pool status");
    println!("  zakat             Manage Zakat obligations");
    println!("  market            Harberger tax marketplace");
    println!("  pat               Manage Personal Agent Team");
    println!("  sat               Manage Shared Agent Team");
    println!("  version           Show version information");
    println!("  help              Show this help message");
    println!();
    println!("Genesis Options:");
    println!("  --name <name>              Node display name");
    println!("  --identity-file <path>     Load identity.public_key and name from JSON");
    println!("  --public-key <hex>         Use existing Ed25519 public key (32 bytes)");
    println!("  --signing-key <hex>        Use existing Ed25519 signing key (32 bytes)");
    println!();
    println!("Examples:");
    println!("  resourcepool-node genesis --name \"Node0-MoMo\"");
    println!("  resourcepool-node genesis --identity-file /mnt/c/BIZRA-DATA-LAKE/sovereign_state/node0_genesis.json");
    println!("  resourcepool-node join --sponsor <node_id> --pool <address>");
    println!("  resourcepool-node contribute --cpu 4000 --gpu 24.0");
    println!("  resourcepool-node zakat status");
    println!("  resourcepool-node market list");
    println!();
    println!("Islamic Finance Principles:");
    println!("  - No Riba (interest): Only profit-sharing (Mudarabah)");
    println!("  - Zakat: 2.5% distribution when wealth > nisab");
    println!("  - Halal Filter: All services pass FATE gates");
    println!("  - Takaful: Mutual insurance for node failures");
}
