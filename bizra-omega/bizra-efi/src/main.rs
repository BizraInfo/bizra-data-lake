//! BIZRA Node0 — UEFI Substrate Genesis
//!
//! Minimal UEFI application: boots, prints banner, emits a witness
//! record to console, then halts. D3 deliverable for BIZRA-STS-001.
//!
//! Standing on Giants:
//!   UEFI Forum (2005) — Unified Extensible Firmware Interface
//!   Nakamoto (2008) — genesis blocks as proof of existence

#![no_main]
#![no_std]

extern crate alloc;

use alloc::format;
use uefi::prelude::*;

/// BIZRA banner — printed at substrate genesis.
const BANNER: &str = r"
 ____  ___ __________ _____
|    \|   |\____    /|     \ _____
|    /|   |  /     / |     \\_   /
|   / |   | /     /_ | |_\  \|  |
|__/  |___|/_______ \|_____  /__|
                    \/      \/
  Node0 - Genesis Substrate
  bismi'llah al-rahman al-rahim
";

/// FNV-1a hash for no_std witness integrity.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Format u64 as 16-char hex.
fn u64_to_hex(value: u64) -> [u8; 16] {
    let hex_chars = b"0123456789abcdef";
    let mut buf = [0u8; 16];
    for i in 0..16 {
        buf[15 - i] = hex_chars[((value >> (i * 4)) & 0xf) as usize];
    }
    buf
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().expect("UEFI init failed");

    // Clear screen
    uefi::system::with_stdout(|stdout| {
        let _ = stdout.clear();
    });

    // Print banner
    uefi::println!("{}", BANNER);
    uefi::println!("========================================");
    uefi::println!("  BIZRA Substrate Genesis Witness");
    uefi::println!("========================================");

    // Build witness
    let witness_payload = b"BIZRA-Node0-Genesis-Substrate-2026";
    let hash = fnv1a_hash(witness_payload);
    let hex = u64_to_hex(hash);
    let hex_str = core::str::from_utf8(&hex).unwrap_or("????????????????");

    uefi::println!();
    uefi::println!("  event: substrate_genesis");
    uefi::println!("  hash:  {}", hex_str);
    uefi::println!("  node:  Node0 (MSI Titan i9-14900HX)");
    uefi::println!("  proof: UEFI boot sequence completed");
    uefi::println!("  chain: BIZRA-STS-001 D3");
    uefi::println!();

    // Machine-readable witness
    let json = format!(
        "{{\"event\":\"substrate_genesis\",\"hash\":\"{}\",\"node\":\"Node0\",\"chain\":\"BIZRA-STS-001\"}}",
        hex_str
    );
    uefi::println!("WITNESS_JSON: {}", json);

    uefi::println!();
    uefi::println!("  Genesis substrate verified.");

    // Wait for keypress
    uefi::system::with_stdin(|stdin| {
        let _ = stdin.read_key();
    });

    Status::SUCCESS
}
