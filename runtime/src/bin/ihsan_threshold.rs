use std::env;

fn main() -> anyhow::Result<()> {
    let env_name = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: ihsan_threshold <env> <artifact_class>"))?;
    let artifact_class = env::args()
        .nth(2)
        .ok_or_else(|| anyhow::anyhow!("usage: ihsan_threshold <env> <artifact_class>"))?;

    let threshold =
        meta_alpha_dual_agentic::ihsan::constitution().threshold_for(&env_name, &artifact_class);

    println!("{threshold:.9}");
    Ok(())
}
