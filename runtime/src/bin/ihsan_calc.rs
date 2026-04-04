use std::{collections::BTreeMap, env};

fn main() -> anyhow::Result<()> {
    let input = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: ihsan_calc '{{\"correctness\":1.0,...}}'"))?;

    let scores: BTreeMap<String, f64> = serde_json::from_str(&input)?;
    let score = meta_alpha_dual_agentic::ihsan::score(&scores)?;

    println!("{score:.9}");
    Ok(())
}
