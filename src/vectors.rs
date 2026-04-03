// src/vectors.rs - ChromaDB Vector Store Client
//
// BIZRA Semantic Search Layer
// ============================
// - Embedding storage for knowledge retrieval
// - Semantic similarity search
// - Integration with House of Wisdom (Neo4j)
// - Ollama embeddings generation

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// ChromaDB collection name for BIZRA knowledge
const COLLECTION_NAME: &str = "bizra_wisdom";

/// ChromaDB client for vector operations
#[derive(Clone)]
pub struct ChromaClient {
    client: Client,
    base_url: String,
    ollama_url: String,
    collection_id: Option<String>,
    available: bool,
}

/// Document to store in ChromaDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub id: String,
    pub content: String,
    pub metadata: VectorMetadata,
}

/// Metadata for vector documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub source: String,
    pub node_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neo4j_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Search result from ChromaDB
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: String,
    pub content: String,
    pub metadata: VectorMetadata,
    pub distance: f32,
    pub relevance_score: f64,
}

/// ChromaDB API structures
#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    name: String,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CollectionResponse {
    id: String,
    #[allow(dead_code)]
    name: String,
}

#[derive(Debug, Serialize)]
struct AddDocumentsRequest {
    ids: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    documents: Vec<String>,
    metadatas: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct QueryRequest {
    query_embeddings: Vec<Vec<f32>>,
    n_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    where_document: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct QueryResponse {
    ids: Vec<Vec<String>>,
    distances: Option<Vec<Vec<f32>>>,
    documents: Option<Vec<Vec<String>>>,
    metadatas: Option<Vec<Vec<serde_json::Value>>>,
}

/// Ollama embedding request
#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

impl ChromaClient {
    /// Create new ChromaDB client from environment
    #[instrument]
    pub async fn from_env() -> Result<Self> {
        let base_url =
            std::env::var("CHROMA_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
        let ollama_url =
            std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

        Self::connect(&base_url, &ollama_url).await
    }

    /// Connect to ChromaDB
    #[instrument(skip(base_url, ollama_url))]
    pub async fn connect(base_url: &str, ollama_url: &str) -> Result<Self> {
        info!(base_url = %base_url, "Connecting to ChromaDB");

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        let mut chroma = Self {
            client,
            base_url: base_url.to_string(),
            ollama_url: ollama_url.to_string(),
            collection_id: None,
            available: false,
        };

        // Test connection
        match chroma.health_check().await {
            Ok(true) => {
                info!("✅ ChromaDB connection established");
                chroma.available = true;

                // Ensure collection exists
                if let Err(e) = chroma.ensure_collection().await {
                    warn!("Failed to ensure collection: {}", e);
                }
            }
            _ => {
                warn!("⚠️ ChromaDB not available, running in fallback mode");
            }
        }

        Ok(chroma)
    }

    /// Create unavailable client (for graceful degradation)
    pub fn unavailable() -> Self {
        Self {
            client: Client::new(),
            base_url: String::new(),
            ollama_url: String::new(),
            collection_id: None,
            available: false,
        }
    }

    /// Check if ChromaDB is available
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Health check
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/v1/heartbeat", self.base_url);

        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => Ok(true),
            Ok(resp) => {
                warn!("ChromaDB health check failed: {}", resp.status());
                Ok(false)
            }
            Err(e) => {
                debug!("ChromaDB health check error: {}", e);
                Ok(false)
            }
        }
    }

    /// Ensure collection exists
    #[instrument(skip(self))]
    async fn ensure_collection(&mut self) -> Result<()> {
        let url = format!("{}/api/v1/collections", self.base_url);

        // Try to get existing collection
        let get_url = format!("{}/{}", url, COLLECTION_NAME);
        if let Ok(resp) = self.client.get(&get_url).send().await {
            if resp.status().is_success() {
                if let Ok(collection) = resp.json::<CollectionResponse>().await {
                    let cid = collection.id.clone();
                    self.collection_id = Some(collection.id);
                    info!(collection_id = %cid, "Using existing collection");
                    return Ok(());
                }
            }
        }

        // Create collection
        let request = CreateCollectionRequest {
            name: COLLECTION_NAME.to_string(),
            metadata: Some(serde_json::json!({
                "description": "BIZRA wisdom knowledge base",
                "embedding_model": "nomic-embed-text"
            })),
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to create collection")?;

        if resp.status().is_success() {
            let collection: CollectionResponse = resp.json().await?;
            self.collection_id = Some(collection.id.clone());
            info!(collection_id = %collection.id, "Created new collection");
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            warn!("Failed to create collection: {} - {}", status, text);
        }

        Ok(())
    }

    /// Generate embedding using Ollama
    #[instrument(skip(self, text))]
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.ollama_url);

        let request = OllamaEmbeddingRequest {
            model: "nomic-embed-text".to_string(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to generate embedding")?;

        if resp.status().is_success() {
            let result: OllamaEmbeddingResponse = resp.json().await?;
            debug!(dim = result.embedding.len(), "Generated embedding");
            Ok(result.embedding)
        } else {
            let status = resp.status();
            anyhow::bail!("Embedding generation failed: {}", status)
        }
    }

    /// Add document to collection
    #[instrument(skip(self, doc))]
    pub async fn add_document(&self, doc: VectorDocument) -> Result<()> {
        if !self.available {
            debug!("ChromaDB unavailable, document not stored");
            return Ok(());
        }

        let collection_id = self
            .collection_id
            .as_ref()
            .context("Collection not initialized")?;

        // Generate embedding
        let embedding = self.generate_embedding(&doc.content).await?;

        let url = format!("{}/api/v1/collections/{}/add", self.base_url, collection_id);

        let request = AddDocumentsRequest {
            ids: vec![doc.id.clone()],
            embeddings: vec![embedding],
            documents: vec![doc.content],
            metadatas: vec![serde_json::to_value(&doc.metadata)?],
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to add document")?;

        if resp.status().is_success() {
            info!(doc_id = %doc.id, "Document added to vector store");
            Ok(())
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Failed to add document: {} - {}", status, text)
        }
    }

    /// Add multiple documents
    #[instrument(skip(self, docs))]
    pub async fn add_documents(&self, docs: Vec<VectorDocument>) -> Result<usize> {
        if !self.available {
            debug!("ChromaDB unavailable, documents not stored");
            return Ok(0);
        }

        let collection_id = self
            .collection_id
            .as_ref()
            .context("Collection not initialized")?;

        // Generate embeddings for all documents
        let mut ids = Vec::with_capacity(docs.len());
        let mut embeddings = Vec::with_capacity(docs.len());
        let mut documents = Vec::with_capacity(docs.len());
        let mut metadatas = Vec::with_capacity(docs.len());

        for doc in &docs {
            let embedding = self.generate_embedding(&doc.content).await?;
            ids.push(doc.id.clone());
            embeddings.push(embedding);
            documents.push(doc.content.clone());
            metadatas.push(serde_json::to_value(&doc.metadata)?);
        }

        let url = format!("{}/api/v1/collections/{}/add", self.base_url, collection_id);

        let request = AddDocumentsRequest {
            ids,
            embeddings,
            documents,
            metadatas,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to add documents")?;

        if resp.status().is_success() {
            info!(count = docs.len(), "Documents added to vector store");
            Ok(docs.len())
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Failed to add documents: {} - {}", status, text)
        }
    }

    /// Semantic search
    #[instrument(skip(self))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<VectorSearchResult>> {
        if !self.available {
            debug!("ChromaDB unavailable, returning empty results");
            return Ok(vec![]);
        }

        let collection_id = self
            .collection_id
            .as_ref()
            .context("Collection not initialized")?;

        // Generate query embedding
        let query_embedding = self.generate_embedding(query).await?;

        let url = format!(
            "{}/api/v1/collections/{}/query",
            self.base_url, collection_id
        );

        let request = QueryRequest {
            query_embeddings: vec![query_embedding],
            n_results: limit,
            where_document: None,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to query collection")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Query failed: {} - {}", status, text);
        }

        let query_resp: QueryResponse = resp.json().await?;

        // Parse results
        let mut results = Vec::new();

        if let (Some(ids), Some(documents), Some(metadatas), Some(distances)) = (
            query_resp.ids.first(),
            query_resp.documents.as_ref().and_then(|d| d.first()),
            query_resp.metadatas.as_ref().and_then(|m| m.first()),
            query_resp.distances.as_ref().and_then(|d| d.first()),
        ) {
            for (i, id) in ids.iter().enumerate() {
                let content = documents.get(i).cloned().unwrap_or_default();
                let metadata: VectorMetadata = metadatas
                    .get(i)
                    .map(|m| {
                        serde_json::from_value(m.clone()).unwrap_or_else(|_| VectorMetadata {
                            source: "unknown".to_string(),
                            node_type: "unknown".to_string(),
                            neo4j_id: None,
                            timestamp: None,
                        })
                    })
                    .unwrap_or_else(|| VectorMetadata {
                        source: "unknown".to_string(),
                        node_type: "unknown".to_string(),
                        neo4j_id: None,
                        timestamp: None,
                    });
                let distance = distances.get(i).copied().unwrap_or(1.0);

                // Convert distance to relevance score (0-1, higher is better)
                let relevance_score = 1.0 / (1.0 + distance as f64);

                results.push(VectorSearchResult {
                    id: id.clone(),
                    content,
                    metadata,
                    distance,
                    relevance_score,
                });
            }
        }

        info!(query = %query, results = results.len(), "Semantic search completed");
        Ok(results)
    }

    /// Delete document by ID
    #[instrument(skip(self))]
    pub async fn delete_document(&self, doc_id: &str) -> Result<()> {
        if !self.available {
            return Ok(());
        }

        let collection_id = self
            .collection_id
            .as_ref()
            .context("Collection not initialized")?;

        let url = format!(
            "{}/api/v1/collections/{}/delete",
            self.base_url, collection_id
        );

        let body = serde_json::json!({
            "ids": [doc_id]
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to delete document")?;

        if resp.status().is_success() {
            info!(doc_id = %doc_id, "Document deleted from vector store");
            Ok(())
        } else {
            let status = resp.status();
            anyhow::bail!("Failed to delete document: {}", status)
        }
    }

    /// Get collection stats
    #[instrument(skip(self))]
    pub async fn collection_count(&self) -> Result<usize> {
        if !self.available {
            return Ok(0);
        }

        let collection_id = self
            .collection_id
            .as_ref()
            .context("Collection not initialized")?;

        let url = format!(
            "{}/api/v1/collections/{}/count",
            self.base_url, collection_id
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to get collection count")?;

        if resp.status().is_success() {
            let count: usize = resp.json().await.unwrap_or(0);
            Ok(count)
        } else {
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_metadata_serialization() {
        let metadata = VectorMetadata {
            source: "wisdom".to_string(),
            node_type: "concept".to_string(),
            neo4j_id: Some("node-123".to_string()),
            timestamp: None,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("wisdom"));
        assert!(json.contains("concept"));
        assert!(json.contains("node-123"));
        assert!(!json.contains("timestamp")); // skipped when None
    }

    #[test]
    fn test_vector_document_creation() {
        let doc = VectorDocument {
            id: "doc-001".to_string(),
            content: "BIZRA knowledge about Ihsān principles".to_string(),
            metadata: VectorMetadata {
                source: "constitution".to_string(),
                node_type: "principle".to_string(),
                neo4j_id: None,
                timestamp: Some("2025-01-20T00:00:00Z".to_string()),
            },
        };

        assert_eq!(doc.id, "doc-001");
        assert!(doc.content.contains("Ihsān"));
    }

    #[test]
    fn test_relevance_score_calculation() {
        // Distance 0 → relevance 1.0
        let relevance: f64 = 1.0 / (1.0 + 0.0);
        assert!((relevance - 1.0).abs() < 0.001);

        // Distance 1 → relevance 0.5
        let relevance: f64 = 1.0 / (1.0 + 1.0);
        assert!((relevance - 0.5).abs() < 0.001);

        // Distance 4 → relevance 0.2
        let relevance: f64 = 1.0 / (1.0 + 4.0);
        assert!((relevance - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_unavailable_client() {
        let client = ChromaClient::unavailable();
        assert!(!client.is_available());
    }
}
