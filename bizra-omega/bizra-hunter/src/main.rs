//! BIZRA Hunter CLI

use bizra_hunter::{Hunter, HunterConfig};

fn main() {
    let config = HunterConfig::default();
    let mut hunter: Hunter<65536> = Hunter::new(config);

    // Basic health + warm loop
    let healthy = hunter.health_check();
    println!("Hunter health: {}", if healthy { "OK" } else { "NOT OK" });

    let stats = hunter.run_loop(10);
    println!("Loop complete. lane1_processed={}", stats.lane1_processed);
}
