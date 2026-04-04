use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use reqwest::Client;
use serde_json::Value;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub mod monetization;
mod transaction; // Expose monetization module

use transaction::{
    SubmitTransactionRequest, SubmitTransactionResponse, Transaction, TransactionStatus,
};

use monetization::{get_roadmap_projections, EngineeringAPI};

#[derive(Clone)]
struct AppState {
    transactions: Arc<DashMap<String, Transaction>>,
    ai_client: Client,
    pricing_model: Arc<EngineeringAPI>, // Cached pricing model
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with better configuration
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "finance_v1=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Initializing finance service...");

    let transactions: Arc<DashMap<String, Transaction>> = Arc::new(DashMap::new());

    // Load pricing model into memory (High Performance / No DB hit for query)
    let pricing_model = Arc::new(EngineeringAPI::default());

    // Create a shared HTTP client for AI verification to avoid connection overhead
    let ai_client = Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;

    let state = AppState {
        transactions,
        ai_client,
        pricing_model,
    };

    let app = Router::new()
        .route("/transactions", post(submit_transaction))
        .route("/transactions/:id", get(get_transaction))
        .route("/pricing", get(get_pricing_structure))
        .route("/projections", get(get_financial_projections))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], 3002)); // Port 3002 for Finance Service
    info!("Starting Finance Service (MaaSP) on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Finance Service (MaaSP) Active on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Endpoint: Expose the Engineering API Pricing Structure
async fn get_pricing_structure(State(state): State<AppState>) -> Json<EngineeringAPI> {
    info!("Serving MaaSP Pricing Structure");
    Json((*state.pricing_model).clone())
}

/// Endpoint: Expose the 5-Year Financial Projections (Transparency)
async fn get_financial_projections() -> Json<serde_json::Value> {
    let (year_1, year_5) = get_roadmap_projections();
    info!("Serving Financial Projections");
    Json(serde_json::json!({
        "year_1_conservative": year_1,
        "year_5_optimistic": year_5,
        "methodology": "Based on 20% MoM growth in formal verification market"
    }))
}

async fn submit_transaction(
    State(state): State<AppState>,
    Json(payload): Json<SubmitTransactionRequest>,
) -> Result<Json<SubmitTransactionResponse>, StatusCode> {
    let transactions = &state.transactions;
    // Input validation
    if payload.amount <= 0.0 {
        return Err(StatusCode::BAD_REQUEST);
    }

    if payload.from.is_empty() || payload.to.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut tx = Transaction::new(payload.from, payload.to, payload.amount);
    info!(
        "Processing transaction: {} -> {} amount: {}",
        tx.from, tx.to, tx.amount
    );

    let status = if tx.amount > 1000.0 {
        info!("Transaction amount > 1000, requiring AI verification");
        match verify_with_ai(&state.ai_client, &tx).await {
            Ok(true) => {
                info!("AI verification approved transaction {}", tx.id);
                TransactionStatus::Approved
            }
            Ok(false) => {
                warn!("AI verification rejected transaction {}", tx.id);
                TransactionStatus::Rejected
            }
            Err(e) => {
                error!("AI verification failed for transaction {}: {}", tx.id, e);
                TransactionStatus::Pending
            }
        }
    } else {
        info!("Transaction amount <= 1000, auto-approved");
        TransactionStatus::Approved
    };

    tx.status = status;
    transactions.insert(tx.id.clone(), tx.clone());

    Ok(Json(SubmitTransactionResponse {
        id: tx.id,
        status: tx.status,
    }))
}

async fn get_transaction(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Transaction>, StatusCode> {
    // Validate ID format (UUID)
    if uuid::Uuid::parse_str(&id).is_err() {
        return Err(StatusCode::BAD_REQUEST);
    }

    match state.transactions.get(&id) {
        Some(tx) => {
            info!("Retrieved transaction {} successfully", id);
            Ok(Json(tx.clone()))
        }
        None => {
            warn!("Transaction {} not found", id);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

async fn verify_with_ai(client: &Client, tx: &Transaction) -> Result<bool, anyhow::Error> {
    let prompt = format!(
        "Verify this financial transaction for fraud or compliance: from {} to {} amount {}. Respond with 'approved' or 'rejected'.",
        tx.from, tx.to, tx.amount
    );

    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&serde_json::json!({
            "model": "llama2",
            "prompt": prompt,
            "stream": false
        }))
        .send()
        .await?;

    // Check for successful response
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "AI service returned error: {}",
            response.status()
        ));
    }

    let result: Value = response.json().await?;

    // More robust response parsing
    let response_text = result
        .get("response")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if response_text.is_empty() {
        return Err(anyhow::anyhow!("Empty response from AI service"));
    }

    let response_lower = response_text.to_lowercase();
    let approved = response_lower.contains("approved");

    info!(
        "AI verification result for transaction {}: {} (response: '{}')",
        tx.id, approved, response_text
    );

    Ok(approved)
}
