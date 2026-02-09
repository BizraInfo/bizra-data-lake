//! Harberger Memory Rent
//!
//! TRICK 6: Economic spam prevention via time-based memory rent.
//!
//! Principles:
//! - Users pay rent proportional to memory × time
//! - Unpaid rent results in eviction (data deletion)
//! - Harberger tax: self-assessed value, anyone can buy at that price
//!
//! This prevents spam by making memory usage economically expensive.

use std::collections::BTreeMap;
use parking_lot::RwLock;

/// Rent slot containing data and payment info
#[derive(Debug, Clone)]
pub struct RentSlot {
    /// Unique slot ID
    pub id: u32,
    /// Data stored in slot
    pub data: Vec<u8>,
    /// Self-assessed value (in wei)
    pub assessed_value: u64,
    /// Rent paid until (timestamp seconds)
    pub rent_paid_until: u64,
    /// Owner address (for eviction refund)
    pub owner: [u8; 20],
    /// Creation timestamp
    pub created_at: u64,
}

/// Rent error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RentError {
    SlotNotFound,
    InsufficientPayment,
    SlotExpired,
    InvalidDuration,
}

/// Harberger rent controller
pub struct HarbergerRent {
    /// Active slots
    slots: RwLock<BTreeMap<u32, RentSlot>>,
    /// Next slot ID
    next_id: std::sync::atomic::AtomicU32,
    /// Base rent rate (wei per byte per second)
    base_rate: u64,
    /// Grace period (seconds)
    grace_period: u64,
    /// Tax rate (basis points, e.g., 1000 = 10%)
    tax_rate: u64,
}

impl HarbergerRent {
    /// Create new rent controller
    pub fn new(base_rate: u64, grace_period: u64) -> Self {
        Self {
            slots: RwLock::new(BTreeMap::new()),
            next_id: std::sync::atomic::AtomicU32::new(1),
            base_rate,
            grace_period,
            tax_rate: 1000, // 10%
        }
    }

    /// Calculate rent for given size and duration
    pub fn calculate_rent(&self, size_bytes: usize, duration_secs: u64) -> u64 {
        (size_bytes as u64) * self.base_rate * duration_secs
    }

    /// Calculate Harberger tax on assessed value
    pub fn calculate_tax(&self, assessed_value: u64, duration_secs: u64) -> u64 {
        // Annual tax rate applied proportionally
        let annual_tax = assessed_value * self.tax_rate / 10000;
        annual_tax * duration_secs / (365 * 24 * 3600)
    }

    /// Rent a new slot
    pub fn rent_slot(
        &self,
        data: Vec<u8>,
        duration_secs: u64,
        assessed_value: u64,
        owner: [u8; 20],
        payment: u64,
    ) -> Result<u32, RentError> {
        if duration_secs == 0 {
            return Err(RentError::InvalidDuration);
        }

        let rent_cost = self.calculate_rent(data.len(), duration_secs);
        let tax_cost = self.calculate_tax(assessed_value, duration_secs);
        let total_cost = rent_cost + tax_cost;

        if payment < total_cost {
            return Err(RentError::InsufficientPayment);
        }

        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let now = now_secs();

        let slot = RentSlot {
            id,
            data,
            assessed_value,
            rent_paid_until: now + duration_secs,
            owner,
            created_at: now,
        };

        let mut slots = self.slots.write();
        slots.insert(id, slot);

        Ok(id)
    }

    /// Extend rent for existing slot
    pub fn extend_rent(
        &self,
        slot_id: u32,
        additional_secs: u64,
        payment: u64,
    ) -> Result<(), RentError> {
        let mut slots = self.slots.write();
        let slot = slots.get_mut(&slot_id).ok_or(RentError::SlotNotFound)?;

        let rent_cost = self.calculate_rent(slot.data.len(), additional_secs);
        let tax_cost = self.calculate_tax(slot.assessed_value, additional_secs);
        let total_cost = rent_cost + tax_cost;

        if payment < total_cost {
            return Err(RentError::InsufficientPayment);
        }

        slot.rent_paid_until += additional_secs;
        Ok(())
    }

    /// Check if slot is expired (past grace period)
    pub fn is_expired(&self, slot_id: u32) -> bool {
        let slots = self.slots.read();
        if let Some(slot) = slots.get(&slot_id) {
            let now = now_secs();
            now > slot.rent_paid_until + self.grace_period
        } else {
            true // Non-existent slots are considered expired
        }
    }

    /// Evict expired slot (returns owner for refund processing)
    pub fn evict(&self, slot_id: u32) -> Option<[u8; 20]> {
        if !self.is_expired(slot_id) {
            return None;
        }

        let mut slots = self.slots.write();
        slots.remove(&slot_id).map(|slot| slot.owner)
    }

    /// Run eviction pass on all slots
    pub fn evict_all_expired(&self) -> Vec<(u32, [u8; 20])> {
        let expired: Vec<u32> = {
            let slots = self.slots.read();
            let now = now_secs();
            slots
                .iter()
                .filter(|(_, slot)| now > slot.rent_paid_until + self.grace_period)
                .map(|(id, _)| *id)
                .collect()
        };

        let mut evicted = Vec::new();
        for id in expired {
            if let Some(owner) = self.evict(id) {
                evicted.push((id, owner));
            }
        }

        evicted
    }

    /// Get slot data (if not expired)
    pub fn get_data(&self, slot_id: u32) -> Option<Vec<u8>> {
        if self.is_expired(slot_id) {
            return None;
        }

        let slots = self.slots.read();
        slots.get(&slot_id).map(|slot| slot.data.clone())
    }

    /// Buy slot from current owner (Harberger mechanism)
    pub fn force_buy(
        &self,
        slot_id: u32,
        new_owner: [u8; 20],
        payment: u64,
    ) -> Result<[u8; 20], RentError> {
        let mut slots = self.slots.write();
        let slot = slots.get_mut(&slot_id).ok_or(RentError::SlotNotFound)?;

        // Payment must be >= assessed value
        if payment < slot.assessed_value {
            return Err(RentError::InsufficientPayment);
        }

        let previous_owner = slot.owner;
        slot.owner = new_owner;
        // Assessed value updated to purchase price
        slot.assessed_value = payment;

        Ok(previous_owner)
    }

    /// Get statistics
    pub fn stats(&self) -> HarbergerStats {
        let slots = self.slots.read();
        let now = now_secs();

        let total_slots = slots.len();
        let total_bytes: usize = slots.values().map(|s| s.data.len()).sum();
        let total_value: u64 = slots.values().map(|s| s.assessed_value).sum();
        let expired_count = slots
            .values()
            .filter(|s| now > s.rent_paid_until + self.grace_period)
            .count();

        HarbergerStats {
            total_slots,
            total_bytes,
            total_assessed_value: total_value,
            expired_slots: expired_count,
            base_rate: self.base_rate,
            grace_period: self.grace_period,
            tax_rate: self.tax_rate,
        }
    }
}

/// Harberger rent statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct HarbergerStats {
    pub total_slots: usize,
    pub total_bytes: usize,
    pub total_assessed_value: u64,
    pub expired_slots: usize,
    pub base_rate: u64,
    pub grace_period: u64,
    pub tax_rate: u64,
}

/// Get current time in seconds
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harberger_rent_new() {
        let rent = HarbergerRent::new(1000, 3600);
        assert_eq!(rent.base_rate, 1000);
        assert_eq!(rent.grace_period, 3600);
    }

    #[test]
    fn test_calculate_rent() {
        let rent = HarbergerRent::new(100, 0); // 100 wei/byte/sec
        let cost = rent.calculate_rent(1000, 3600); // 1KB for 1 hour
        assert_eq!(cost, 100 * 1000 * 3600);
    }

    #[test]
    fn test_rent_slot() {
        let rent = HarbergerRent::new(1, 0);
        let data = vec![0u8; 100];
        let owner = [1u8; 20];

        let cost = rent.calculate_rent(100, 3600);
        let tax = rent.calculate_tax(1000, 3600);
        let total = cost + tax;

        let result = rent.rent_slot(data, 3600, 1000, owner, total);
        assert!(result.is_ok());
    }

    #[test]
    fn test_insufficient_payment() {
        let rent = HarbergerRent::new(1000, 0);
        let data = vec![0u8; 100];
        let owner = [1u8; 20];

        let result = rent.rent_slot(data, 3600, 1000, owner, 1); // 1 wei payment
        assert_eq!(result, Err(RentError::InsufficientPayment));
    }

    #[test]
    fn test_get_data_not_expired() {
        let rent = HarbergerRent::new(1, 3600);
        let data = vec![42u8; 10];
        let owner = [1u8; 20];

        let payment = rent.calculate_rent(10, 7200) + rent.calculate_tax(100, 7200);
        let id = rent
            .rent_slot(data.clone(), 7200, 100, owner, payment)
            .unwrap();

        let retrieved = rent.get_data(id);
        assert_eq!(retrieved, Some(data));
    }
}
