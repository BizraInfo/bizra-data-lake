//! BIZRA Hunter CLI — SNR-Maximized Vulnerability Discovery

use bizra_hunter::{
    ingestion::{FileSource, StaticSource},
    Hunter, HunterConfig,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bizra-hunter-snr",
    version,
    about = "SNR-Maximized Vulnerability Hunter"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the vulnerability scanner
    Scan {
        /// JSON-RPC endpoint URL
        #[arg(long)]
        rpc_url: Option<String>,

        /// Chain ID
        #[arg(long, default_value_t = 1)]
        chain_id: u64,

        /// Number of health-loop iterations (0 = infinite)
        #[arg(long, default_value_t = 100)]
        iterations: u32,

        /// Hex-encoded bytecodes to scan (with or without 0x prefix)
        #[arg(long)]
        bytecode: Vec<String>,

        /// File containing contracts (one per line: `addr bytecode` or just `bytecode`)
        #[arg(long)]
        file: Option<String>,
    },
    /// Print health check and stats
    Health,
    /// Decode EVM bytecode and show instruction summary + vulnerability analysis
    Decode {
        /// Hex-encoded bytecode (with or without 0x prefix)
        #[arg(long)]
        bytecode: String,
    },
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Scan {
            rpc_url,
            chain_id,
            iterations,
            bytecode,
            file,
        }) => {
            let config = HunterConfig {
                rpc_url,
                chain_id,
                ..HunterConfig::default()
            };
            let mut hunter: Hunter<65536> = Hunter::new(config);

            if !hunter.health_check() {
                eprintln!("Health check FAILED — aborting.");
                std::process::exit(1);
            }
            println!("Health check: OK");

            // Build a source from CLI args
            let has_input = !bytecode.is_empty() || file.is_some();

            if has_input {
                if let Some(path) = file {
                    // File-based scan
                    match FileSource::load(std::path::Path::new(&path), chain_id) {
                        Ok(mut source) => {
                            let findings = hunter.scan(&mut source);
                            print_findings(&findings);
                        }
                        Err(e) => {
                            eprintln!("Failed to open contract file: {e}");
                            std::process::exit(1);
                        }
                    }
                } else {
                    // Inline bytecode scan
                    let mut contracts = Vec::new();
                    for (i, bc_hex) in bytecode.iter().enumerate() {
                        let hex = bc_hex.strip_prefix("0x").unwrap_or(bc_hex);
                        match hex::decode(hex) {
                            Ok(bytes) => {
                                let mut addr = [0u8; 20];
                                addr[19] = (i + 1) as u8; // synthetic unique address
                                contracts.push((addr, bytes));
                            }
                            Err(e) => {
                                eprintln!("Invalid hex in bytecode #{}: {e}", i + 1);
                                std::process::exit(1);
                            }
                        }
                    }
                    let mut source = StaticSource::from_pairs(contracts);
                    let findings = hunter.scan(&mut source);
                    print_findings(&findings);
                }
            } else {
                // No input: legacy health-loop mode
                let stats = hunter.run_loop(iterations);
                println!(
                    "Loop complete. lane1={} filtered={} submitted={}",
                    stats.lane1_processed, stats.lane1_filtered, stats.lane2_submitted
                );
            }
        }
        Some(Commands::Health) => {
            let hunter: Hunter<65536> = Hunter::new(HunterConfig::default());
            let healthy = hunter.health_check();
            println!("Health: {}", if healthy { "OK" } else { "FAIL" });
        }
        Some(Commands::Decode { bytecode }) => {
            let hex_str = bytecode.strip_prefix("0x").unwrap_or(&bytecode);
            let bytes = match hex::decode(hex_str) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Invalid hex: {e}");
                    std::process::exit(1);
                }
            };

            // Decode instructions
            let instructions = bizra_hunter::EvmDecoder::decode(&bytes);
            println!("Instructions: {}", instructions.len());

            // Sequence detection
            let sequences = bizra_hunter::evm::detect_opcode_sequences(&instructions);
            if let Some((pattern, offset)) = sequences {
                println!("Sequence detected: {pattern:?} at offset {offset}");
            } else {
                println!("No known vulnerability sequences detected.");
            }

            // Full pipeline analysis (single-contract scan)
            let mut hunter: Hunter<1024> = Hunter::new(HunterConfig::default());
            if let Some(finding) = hunter.scan_one([0u8; 20], &bytes) {
                println!(
                    "Vulnerability: {:?} at offset {} (complexity: {:?}, bounty: ${}.{:02})",
                    finding.vuln_type,
                    finding.location,
                    finding.complexity,
                    finding.bounty_estimate / 100,
                    finding.bounty_estimate % 100
                );
            }
        }
        None => {
            // Default: health check
            let hunter: Hunter<65536> = Hunter::new(HunterConfig::default());
            let healthy = hunter.health_check();
            println!("Hunter health: {}", if healthy { "OK" } else { "NOT OK" });
        }
    }
}

fn print_findings(findings: &[bizra_hunter::Finding]) {
    if findings.is_empty() {
        println!("No findings. All contracts passed SNR filtering.");
        return;
    }
    println!("\n=== {} Finding(s) ===\n", findings.len());
    for (i, f) in findings.iter().enumerate() {
        println!(
            "[{}] {} — {:?} at offset {} (complexity: {:?})",
            i + 1,
            f.address_hex(),
            f.vuln_type,
            f.location,
            f.complexity
        );
        println!(
            "    Bounty: ${}.{:02} | Entropy avg: {:.3}",
            f.bounty_estimate / 100,
            f.bounty_estimate % 100,
            f.entropy.average()
        );
        if let Some(sub) = &f.submission {
            println!(
                "    Submission: {} — {}",
                if sub.accepted { "ACCEPTED" } else { "REJECTED" },
                sub.reason
            );
        }
        println!();
    }
}
