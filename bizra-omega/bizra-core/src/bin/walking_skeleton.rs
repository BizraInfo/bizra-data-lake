//! Walking Skeleton Binary — Run the constitutional liveness proof and print the receipt.
//!
//! Usage: cargo run --bin walking_skeleton
//!
//! Exits 0 if the system is constitutionally alive, 1 if it fails.

fn main() {
    let start = std::time::Instant::now();

    match bizra_core::walking_skeleton::run_skeleton() {
        Ok(receipt) => {
            let json = serde_json::to_string_pretty(&receipt).expect("Failed to serialize receipt");
            println!("{json}");

            eprintln!(
                "WALKING SKELETON PASSED in {}us — system is constitutionally alive",
                start.elapsed().as_micros()
            );
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("WALKING SKELETON FAILED: {e}");
            std::process::exit(1);
        }
    }
}
