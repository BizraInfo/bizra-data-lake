// src/synapse.rs - Redis (Synapse) State Persistence
//
// BIZRA State Management Layer
// =============================
// - FATE escalation queue persistence
// - Receipt storage and retrieval
// - Distributed locking for multi-instance
// - Metrics and health tracking

use anyhow::{Context, Result};
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;
use tracing::{debug, error, info, instrument, warn};

/// Redis key prefixes for namespacing
const KEY_PREFIX_FATE: &str = "bizra:fate:";
const KEY_PREFIX_RECEIPT: &str = "bizra:receipt:";
const KEY_PREFIX_METRICS: &str = "bizra:metrics:";
const KEY_PREFIX_LOCK: &str = "bizra:lock:";

/// Default TTL for receipts (30 days)
const RECEIPT_TTL_SECS: u64 = 30 * 24 * 60 * 60;

/// Default TTL for FATE escalations (7 days)
const FATE_TTL_SECS: u64 = 7 * 24 * 60 * 60;

/// Lock TTL (30 seconds)
const LOCK_TTL_SECS: u64 = 30;

/// Synapse client for Redis state management
#[derive(Clone)]
pub struct SynapseClient {
    conn: ConnectionManager,
    available: bool,
}

impl SynapseClient {
    /// Create new Synapse client from environment
    #[instrument]
    pub async fn from_env() -> Result<Self> {
        let url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        
        Self::connect(&url).await
    }
    
    /// Connect to Redis
    #[instrument(skip(url))]
    pub async fn connect(url: &str) -> Result<Self> {
        info!(url = %url, "Connecting to Synapse (Redis)");
        
        let client = Client::open(url)
            .context("Failed to create Redis client")?;
        
        match ConnectionManager::new(client).await {
            Ok(conn) => {
                info!("✅ Synapse connection established");
                Ok(Self { conn, available: true })
            }
            Err(e) => {
                warn!(error = %e, "Synapse unavailable, using fallback mode");
                // Create a dummy connection for fallback mode
                let dummy_client = Client::open("redis://127.0.0.1:6379")?;
                let conn = ConnectionManager::new(dummy_client).await
                    .unwrap_or_else(|_| panic!("Cannot create even dummy connection"));
                Ok(Self { conn, available: false })
            }
        }
    }
    
    /// Check if Synapse is available
    pub fn is_available(&self) -> bool {
        self.available
    }
    
    // ================================================================
    // FATE Escalation Queue
    // ================================================================
    
    /// Push escalation to queue
    #[instrument(skip(self, escalation))]
    pub async fn push_fate_escalation<T: Serialize>(&mut self, escalation_id: &str, escalation: &T) -> Result<()> {
        if !self.available {
            debug!("Synapse unavailable, escalation stored in memory only");
            return Ok(());
        }
        
        let key = format!("{}{}", KEY_PREFIX_FATE, escalation_id);
        let value = serde_json::to_string(escalation)?;
        
        self.conn
            .set_ex::<_, _, ()>(&key, &value, FATE_TTL_SECS)
            .await
            .context("Failed to store FATE escalation")?;
        
        // Also add to the pending queue
        self.conn
            .lpush::<_, _, ()>("bizra:fate:pending", escalation_id)
            .await
            .context("Failed to add to pending queue")?;
        
        debug!(escalation_id, "FATE escalation persisted to Synapse");
        Ok(())
    }
    
    /// Get escalation by ID
    #[instrument(skip(self))]
    pub async fn get_fate_escalation<T: DeserializeOwned>(&mut self, escalation_id: &str) -> Result<Option<T>> {
        if !self.available {
            return Ok(None);
        }
        
        let key = format!("{}{}", KEY_PREFIX_FATE, escalation_id);
        let value: Option<String> = self.conn.get(&key).await?;
        
        match value {
            Some(v) => Ok(Some(serde_json::from_str(&v)?)),
            None => Ok(None),
        }
    }
    
    /// Get pending escalation count
    #[instrument(skip(self))]
    pub async fn pending_escalation_count(&mut self) -> Result<usize> {
        if !self.available {
            return Ok(0);
        }
        
        let count: usize = self.conn.llen("bizra:fate:pending").await?;
        Ok(count)
    }
    
    /// Pop next pending escalation ID
    #[instrument(skip(self))]
    pub async fn pop_pending_escalation(&mut self) -> Result<Option<String>> {
        if !self.available {
            return Ok(None);
        }
        
        let id: Option<String> = self.conn.rpop("bizra:fate:pending", None).await?;
        Ok(id)
    }
    
    /// Mark escalation as resolved
    #[instrument(skip(self))]
    pub async fn resolve_escalation(&mut self, escalation_id: &str, resolution: &str) -> Result<()> {
        if !self.available {
            return Ok(());
        }
        
        let key = format!("{}{}:resolution", KEY_PREFIX_FATE, escalation_id);
        self.conn
            .set_ex::<_, _, ()>(&key, resolution, FATE_TTL_SECS)
            .await?;
        
        // Move from pending to resolved
        self.conn
            .lrem::<_, _, ()>("bizra:fate:pending", 1, escalation_id)
            .await?;
        self.conn
            .lpush::<_, _, ()>("bizra:fate:resolved", escalation_id)
            .await?;
        
        debug!(escalation_id, "FATE escalation resolved");
        Ok(())
    }
    
    // ================================================================
    // Receipt Storage
    // ================================================================
    
    /// Store receipt
    #[instrument(skip(self, receipt))]
    pub async fn store_receipt<T: Serialize>(&mut self, receipt_id: &str, receipt: &T) -> Result<()> {
        if !self.available {
            debug!("Synapse unavailable, receipt stored locally only");
            return Ok(());
        }
        
        let key = format!("{}{}", KEY_PREFIX_RECEIPT, receipt_id);
        let value = serde_json::to_string(receipt)?;
        
        self.conn
            .set_ex::<_, _, ()>(&key, &value, RECEIPT_TTL_SECS)
            .await
            .context("Failed to store receipt")?;
        
        // Add to receipt index (score = timestamp, member = receipt_id)
        let score = chrono::Utc::now().timestamp() as f64;
        let _: () = self.conn
            .zadd("bizra:receipts:index", receipt_id, score)
            .await?;
        
        debug!(receipt_id, "Receipt persisted to Synapse");
        Ok(())
    }
    
    /// Get receipt by ID
    #[instrument(skip(self))]
    pub async fn get_receipt<T: DeserializeOwned>(&mut self, receipt_id: &str) -> Result<Option<T>> {
        if !self.available {
            return Ok(None);
        }
        
        let key = format!("{}{}", KEY_PREFIX_RECEIPT, receipt_id);
        let value: Option<String> = self.conn.get(&key).await?;
        
        match value {
            Some(v) => Ok(Some(serde_json::from_str(&v)?)),
            None => Ok(None),
        }
    }
    
    /// Get recent receipts (last N)
    #[instrument(skip(self))]
    pub async fn recent_receipts(&mut self, count: isize) -> Result<Vec<String>> {
        if !self.available {
            return Ok(vec![]);
        }
        
        let ids: Vec<String> = self.conn
            .zrevrange("bizra:receipts:index", 0, count - 1)
            .await?;
        
        Ok(ids)
    }
    
    // ================================================================
    // Distributed Locking
    // ================================================================
    
    /// Acquire distributed lock
    #[instrument(skip(self))]
    pub async fn acquire_lock(&mut self, resource: &str) -> Result<bool> {
        if !self.available {
            return Ok(true); // No locking in fallback mode
        }
        
        let key = format!("{}{}", KEY_PREFIX_LOCK, resource);
        let lock_id = uuid::Uuid::new_v4().to_string();
        
        let acquired: bool = self.conn
            .set_options(
                &key,
                &lock_id,
                redis::SetOptions::default()
                    .with_expiration(redis::SetExpiry::EX(LOCK_TTL_SECS))
                    .conditional_set(redis::ExistenceCheck::NX),
            )
            .await
            .unwrap_or(false);
        
        Ok(acquired)
    }
    
    /// Release distributed lock
    #[instrument(skip(self))]
    pub async fn release_lock(&mut self, resource: &str) -> Result<()> {
        if !self.available {
            return Ok(());
        }
        
        let key = format!("{}{}", KEY_PREFIX_LOCK, resource);
        self.conn.del::<_, ()>(&key).await?;
        Ok(())
    }
    
    // ================================================================
    // Metrics Counters
    // ================================================================
    
    /// Increment a metric counter
    #[instrument(skip(self))]
    pub async fn incr_metric(&mut self, metric: &str) -> Result<i64> {
        if !self.available {
            return Ok(0);
        }
        
        let key = format!("{}{}", KEY_PREFIX_METRICS, metric);
        let value: i64 = self.conn.incr(&key, 1).await?;
        Ok(value)
    }
    
    /// Get metric value
    #[instrument(skip(self))]
    pub async fn get_metric(&mut self, metric: &str) -> Result<i64> {
        if !self.available {
            return Ok(0);
        }
        
        let key = format!("{}{}", KEY_PREFIX_METRICS, metric);
        let value: i64 = self.conn.get(&key).await.unwrap_or(0);
        Ok(value)
    }
    
    // ================================================================
    // Health Check
    // ================================================================
    
    /// Ping Redis
    #[instrument(skip(self))]
    pub async fn ping(&mut self) -> Result<bool> {
        if !self.available {
            return Ok(false);
        }
        
        let pong: String = redis::cmd("PING")
            .query_async(&mut self.conn)
            .await
            .unwrap_or_else(|_| "FAIL".to_string());
        
        Ok(pong == "PONG")
    }
}

/// Create Synapse client with fallback
pub async fn synapse_client() -> SynapseClient {
    match SynapseClient::from_env().await {
        Ok(client) => client,
        Err(e) => {
            error!(error = %e, "Failed to create Synapse client, running in degraded mode");
            // Return a client in unavailable mode
            SynapseClient {
                conn: ConnectionManager::new(Client::open("redis://127.0.0.1:6379").unwrap())
                    .await
                    .expect("Fallback connection failed"),
                available: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_key_prefixes() {
        assert!(KEY_PREFIX_FATE.starts_with("bizra:"));
        assert!(KEY_PREFIX_RECEIPT.starts_with("bizra:"));
        assert!(KEY_PREFIX_METRICS.starts_with("bizra:"));
        assert!(KEY_PREFIX_LOCK.starts_with("bizra:"));
    }
    
    #[test]
    fn test_ttl_values() {
        // Receipt TTL should be 30 days
        assert_eq!(RECEIPT_TTL_SECS, 30 * 24 * 60 * 60);
        // FATE TTL should be 7 days
        assert_eq!(FATE_TTL_SECS, 7 * 24 * 60 * 60);
        // Lock TTL should be 30 seconds
        assert_eq!(LOCK_TTL_SECS, 30);
    }
}
