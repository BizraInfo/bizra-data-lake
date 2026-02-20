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
pub mod policy;
pub mod provider;
