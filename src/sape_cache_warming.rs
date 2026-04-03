// src/sape_cache_warming.rs - SAPE Cache Warming Implementation
// Standing on Shoulders of Giants Protocol: SAPE validation framework
// Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

use crate::errors::BridgeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

const CACHE_WARMUP_BATCH_SIZE: usize = 100;
const CACHE_TTL_SECONDS: u64 = 3600;
const INVARIANT_KEY: &str = "sape_invariant";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: String,
    pub invariant: u32,
    pub created_at: u64,
    pub expires_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupRequest {
    pub batch_id: String,
    pub entries: Vec<CacheEntryRequest>,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryRequest {
    pub key: String,
    pub value: String,
    pub invariant: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    pub batch_id: String,
    pub success: bool,
    pub entries_loaded: usize,
    pub entries_failed: usize,
    pub invariant_preserved: bool,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct SapEcacheWarmer {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<RwLock<CacheStats>>,
    invariant_table: Arc<RwLock<HashMap<String, u32>>>,
}

#[derive(Debug, Clone, Default)]
struct CacheStats {
    total_warmups: u64,
    successful_warmups: u64,
    failed_warmups: u64,
    invariant_violations: u64,
}

impl SapEcacheWarmer {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            invariant_table: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn warmup(&self, request: WarmupRequest) -> Result<WarmupResult, BridgeError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| BridgeError::Auth(format!("Time error: {}", e)))?
            .as_secs();

        let mut entries_loaded = 0;
        let mut entries_failed = 0;
        let mut invariant_preserved = true;

        {
            let invariant_table = self.invariant_table.read().await;
            for entry in &request.entries {
                if let Some(&stored_invariant) = invariant_table.get(&entry.key) {
                    if stored_invariant != entry.invariant {
                        invariant_preserved = false;
                        break;
                    }
                }
            }
        }

        if !invariant_preserved {
            let mut stats = self.stats.write().await;
            stats.failed_warmups += 1;
            stats.invariant_violations += 1;

            return Ok(WarmupResult {
                batch_id: request.batch_id,
                success: false,
                entries_loaded: 0,
                entries_failed: request.entries.len(),
                invariant_preserved: false,
                timestamp: now,
            });
        }

        let entries = request.entries.chunks(CACHE_WARMUP_BATCH_SIZE);
        
        for batch in entries {
            let mut cache = self.cache.write().await;
            
            for entry_request in batch {
                let cache_entry = CacheEntry {
                    key: entry_request.key.clone(),
                    value: entry_request.value.clone(),
                    invariant: entry_request.invariant,
                    created_at: now,
                    expires_at: now + CACHE_TTL_SECONDS,
                };
                
                cache.insert(entry_request.key.clone(), cache_entry);
                entries_loaded += 1;
            }
        }

        {
            let mut invariant_table = self.invariant_table.write().await;
            for entry in &request.entries {
                invariant_table.insert(entry.key.clone(), entry.invariant);
            }
        }

        let mut stats = self.stats.write().await;
        stats.total_warmups += 1;
        stats.successful_warmups += 1;

        Ok(WarmupResult {
            batch_id: request.batch_id,
            success: true,
            entries_loaded,
            entries_failed,
            invariant_preserved,
            timestamp: now,
        })
    }

    pub async fn get(&self, key: &str) -> Option<CacheEntry> {
        let cache = self.cache.read().await;
        let entry = cache.get(key)?.clone();
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        
        if entry.expires_at < now {
            return None;
        }
        
        Some(entry)
    }

    pub async fn invalidate(&self, key: &str) -> Result<(), BridgeError> {
        let mut cache = self.cache.write().await;
        cache.remove(key);
        
        let mut invariant_table = self.invariant_table.write().await;
        invariant_table.remove(key);
        
        Ok(())
    }

    pub async fn check_invariant(&self, key: &str) -> Option<u32> {
        let invariant_table = self.invariant_table.read().await;
        invariant_table.get(key).copied()
    }

    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    pub async fn prune_expired(&self) -> Result<usize, BridgeError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| BridgeError::Auth(format!("Time error: {}", e)))?
            .as_secs();

        let mut pruned = 0;
        
        {
            let mut cache = self.cache.write().await;
            let expired_keys: Vec<String> = cache
                .iter()
                .filter(|(_, v)| v.expires_at < now)
                .map(|(k, _)| k.clone())
                .collect();
            
            for key in expired_keys {
                cache.remove(&key);
                pruned += 1;
            }
        }
        
        {
            let mut invariant_table = self.invariant_table.write().await;
            let mut expired_invariants = Vec::new();
            
            for key in invariant_table.keys() {
                if !self.cache.read().await.contains_key(key) {
                    expired_invariants.push(key.clone());
                }
            }
            
            for key in expired_invariants {
                invariant_table.remove(&key);
            }
        }

        Ok(pruned)
    }
}

impl Default for SapEcacheWarmer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_warmup() {
        let warmer = SapEcacheWarmer::new();
        
        let request = WarmupRequest {
            batch_id: "batch_001".to_string(),
            entries: vec![
                CacheEntryRequest {
                    key: "key1".to_string(),
                    value: "value1".to_string(),
                    invariant: 1,
                },
                CacheEntryRequest {
                    key: "key2".to_string(),
                    value: "value2".to_string(),
                    invariant: 1,
                },
            ],
            priority: 1,
        };
        
        let result = warmer.warmup(request).await.unwrap();
        assert!(result.success);
        assert_eq!(result.entries_loaded, 2);
        assert!(result.invariant_preserved);
    }

    #[tokio::test]
    async fn test_invariant_violation() {
        let warmer = SapEcacheWarmer::new();
        
        let request1 = WarmupRequest {
            batch_id: "batch_001".to_string(),
            entries: vec![
                CacheEntryRequest {
                    key: "key1".to_string(),
                    value: "value1".to_string(),
                    invariant: 1,
                },
            ],
            priority: 1,
        };
        
        warmer.warmup(request1).await.unwrap();
        
        let request2 = WarmupRequest {
            batch_id: "batch_002".to_string(),
            entries: vec![
                CacheEntryRequest {
                    key: "key1".to_string(),
                    value: "different_value".to_string(),
                    invariant: 2,
                },
            ],
            priority: 1,
        };
        
        let result = warmer.warmup(request2).await.unwrap();
        assert!(!result.success);
        assert!(!result.invariant_preserved);
    }
}