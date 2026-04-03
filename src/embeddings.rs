use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Arc;

pub struct EmbeddingEngine {
    model: Arc<TextEmbedding>,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::AllMiniLML6V2;
        options.show_download_progress = true;

        let model = TextEmbedding::try_new(options)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let documents = vec![text];
        let embeddings = self.model.embed(documents, None)?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding generated"))
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    pub fn precompute_concepts(&self, concepts: &[&str]) -> Result<Vec<(String, Vec<f32>)>> {
        let mut results = Vec::new();
        let embeddings = self.model.embed(concepts.to_vec(), None)?;

        for (concept, embedding) in concepts.iter().zip(embeddings) {
            results.push((concept.to_string(), embedding));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings() {
        let engine = EmbeddingEngine::new().expect("Failed to init embedding engine");

        let vec1 = engine.embed_text("hack the system").unwrap();
        let vec2 = engine.embed_text("malicious attack").unwrap();
        let vec3 = engine.embed_text("bake a cake").unwrap();

        let sim_threat = EmbeddingEngine::cosine_similarity(&vec1, &vec2);
        let sim_benign = EmbeddingEngine::cosine_similarity(&vec1, &vec3);

        println!("Similarity (hack vs attack): {}", sim_threat);
        println!("Similarity (hack vs cake): {}", sim_benign);

        assert!(sim_threat > sim_benign);
        assert!(sim_threat > 0.4);
    }
}
