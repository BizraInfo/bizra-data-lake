//! BIZRA Installer — Alpha-100 Sprint 1
//!
//! Library crate exposing the Alpha-100 bootstrap modules for the BIZRA
//! sovereign installer. Provides policy hash canonicalization, install
//! configuration management, provider detection, binary acquisition, and
//! the Alpha-100 subcommand handler.
//!
//! Giants: Torvalds (Unix philosophy), Pike (Go CLI patterns), Stallman (GNU)

pub mod alpha100;
pub mod binary_fetch;
pub mod config;
pub mod device_profile;
pub mod hardware_detect;
pub mod health_check;
pub mod i18n;
pub mod install_flow;
pub mod install_receipt;
pub mod model_cache;
pub mod policy;
pub mod profiles;
pub mod provider;
pub mod self_update;
pub mod urp;
