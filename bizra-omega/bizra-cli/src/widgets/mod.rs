//! BIZRA Custom Widgets
//!
//! Specialized TUI widgets for the sovereign node interface.

mod agent_card;
mod fate_gauge;
mod ghost_feed;
mod header;
mod parliament_panel;
mod receipt_detail;
mod receipt_rail;
mod status_bar;
mod substrate_panel;
mod trust_rail;

pub use agent_card::AgentCard;
// TUI scaffolding -- re-enabled when FATE gauge panel returns to dashboard
#[allow(unused_imports)]
pub use fate_gauge::FateGauge;
pub use ghost_feed::GhostFeed;
pub use header::Header;
pub use parliament_panel::ParliamentPanel;
pub use receipt_detail::ReceiptDetail;
pub use receipt_rail::ReceiptRail;
pub use status_bar::StatusBar;
pub use substrate_panel::SubstratePanel;
pub use trust_rail::TrustRail;
