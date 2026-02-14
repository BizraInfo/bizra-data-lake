//! Model Cache

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub desc: String,
    pub size_gb: f64,
    pub url: String,
    pub tier: String,
}

impl ModelSpec {
    pub fn available() -> Vec<Self> {
        vec![
            Self {
                name: "qwen2.5-0.5b-q4".into(),
                desc: "Ultra lightweight".into(),
                size_gb: 0.4,
                url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
                tier: "EDGE".into(),
            },
            Self {
                name: "qwen2.5-1.5b-q4".into(),
                desc: "Lightweight".into(),
                size_gb: 1.1,
                url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
                tier: "EDGE".into(),
            },
            Self {
                name: "qwen2.5-7b-q4".into(),
                desc: "Balanced".into(),
                size_gb: 4.7,
                url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF".into(),
                tier: "LOCAL".into(),
            },
            Self {
                name: "llama3-8b-q4".into(),
                desc: "General purpose".into(),
                size_gb: 4.9,
                url: "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct".into(),
                tier: "LOCAL".into(),
            },
        ]
    }
}
