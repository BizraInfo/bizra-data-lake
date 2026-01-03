use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Transaction {
    pub id: String,
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub status: TransactionStatus,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum TransactionStatus {
    Pending,
    Approved,
    Rejected,
}

impl Transaction {
    pub fn new(from: String, to: String, amount: f64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from,
            to,
            amount,
            status: TransactionStatus::Pending,
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Deserialize)]
pub struct SubmitTransactionRequest {
    pub from: String,
    pub to: String,
    pub amount: f64,
}

#[derive(Serialize)]
pub struct SubmitTransactionResponse {
    pub id: String,
    pub status: TransactionStatus,
}
