//! Bytecode Ingestion Layer
//!
//! Trait-based abstraction for feeding contracts into the SNR pipeline.
//!
//! Implementations:
//! - `StaticSource` — in-memory contracts for testing and batch analysis
//! - `FileSource` — read hex-encoded bytecodes from a file
//!
//! Design principle: the ingestion layer is pure I/O — no analysis logic.
//! The Hunter orchestrates ingestion → pipeline → findings.

use std::path::Path;

/// A contract ready for analysis.
#[derive(Debug, Clone)]
pub struct Contract {
    /// 20-byte Ethereum address
    pub address: [u8; 20],
    /// Raw EVM bytecode
    pub bytecode: Vec<u8>,
    /// Chain identifier (1 = mainnet)
    pub chain_id: u64,
}

/// Source of contract bytecodes.
///
/// Implementations provide contracts to the scanner in batches.
/// The trait is object-safe for dynamic dispatch.
pub trait BytecodeSource {
    /// Drain all currently available contracts from this source.
    fn drain(&mut self) -> Vec<Contract>;

    /// Whether this source can produce more contracts.
    fn is_exhausted(&self) -> bool;

    /// Human-readable description (for logging).
    fn describe(&self) -> &str;
}

// ─── StaticSource ──────────────────────────────────────────────────────────

/// In-memory contract source for testing and CLI piping.
pub struct StaticSource {
    contracts: Vec<Contract>,
    drained: bool,
}

impl StaticSource {
    /// Create from pre-loaded contracts.
    pub fn new(contracts: Vec<Contract>) -> Self {
        Self {
            contracts,
            drained: false,
        }
    }

    /// Convenience: create from a single (address, bytecode) pair.
    pub fn single(address: [u8; 20], bytecode: Vec<u8>) -> Self {
        Self::new(vec![Contract {
            address,
            bytecode,
            chain_id: 1,
        }])
    }

    /// Convenience: create from pairs of (address, bytecode).
    pub fn from_pairs(pairs: Vec<([u8; 20], Vec<u8>)>) -> Self {
        Self::new(
            pairs
                .into_iter()
                .map(|(address, bytecode)| Contract {
                    address,
                    bytecode,
                    chain_id: 1,
                })
                .collect(),
        )
    }
}

impl BytecodeSource for StaticSource {
    fn drain(&mut self) -> Vec<Contract> {
        if self.drained {
            return Vec::new();
        }
        self.drained = true;
        std::mem::take(&mut self.contracts)
    }

    fn is_exhausted(&self) -> bool {
        self.drained
    }

    fn describe(&self) -> &str {
        "static"
    }
}

// ─── FileSource ────────────────────────────────────────────────────────────

/// File-based bytecode source.
///
/// Reads a text file with one contract per line. Supported formats:
/// - `<hex_address> <hex_bytecode>` (address is 40 hex chars, no 0x)
/// - `<hex_bytecode>` (address auto-generated from line number)
///
/// Lines starting with `#` are comments. Empty lines are skipped.
pub struct FileSource {
    contracts: Vec<Contract>,
    drained: bool,
}

impl FileSource {
    /// Load contracts from a file path.
    ///
    /// Returns an error if the file cannot be read or contains invalid hex.
    pub fn load(path: &Path, chain_id: u64) -> Result<Self, FileSourceError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| FileSourceError::Io(path.display().to_string(), e))?;

        let mut contracts = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let contract = parse_contract_line(trimmed, line_num, chain_id)
                .map_err(|e| FileSourceError::Parse(line_num + 1, e))?;
            contracts.push(contract);
        }

        Ok(Self {
            contracts,
            drained: false,
        })
    }
}

impl BytecodeSource for FileSource {
    fn drain(&mut self) -> Vec<Contract> {
        if self.drained {
            return Vec::new();
        }
        self.drained = true;
        std::mem::take(&mut self.contracts)
    }

    fn is_exhausted(&self) -> bool {
        self.drained
    }

    fn describe(&self) -> &str {
        "file"
    }
}

/// Errors from file-based ingestion.
#[derive(Debug)]
pub enum FileSourceError {
    Io(String, std::io::Error),
    Parse(usize, String),
}

impl std::fmt::Display for FileSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "cannot read {path}: {e}"),
            Self::Parse(line, msg) => write!(f, "line {line}: {msg}"),
        }
    }
}

impl std::error::Error for FileSourceError {}

/// Parse a single line into a Contract.
fn parse_contract_line(line: &str, line_num: usize, chain_id: u64) -> Result<Contract, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    match parts.len() {
        // Just bytecode — synthetic address from line number
        1 => {
            let hex = parts[0].strip_prefix("0x").unwrap_or(parts[0]);
            let bytecode = hex::decode(hex).map_err(|e| format!("invalid bytecode hex: {e}"))?;

            let mut address = [0u8; 20];
            let idx = (line_num as u32).to_be_bytes();
            address[16..20].copy_from_slice(&idx);

            Ok(Contract {
                address,
                bytecode,
                chain_id,
            })
        }
        // Address + bytecode
        2 => {
            let addr_hex = parts[0].strip_prefix("0x").unwrap_or(parts[0]);
            if addr_hex.len() != 40 {
                return Err(format!(
                    "address must be 40 hex chars, got {}",
                    addr_hex.len()
                ));
            }
            let addr_bytes =
                hex::decode(addr_hex).map_err(|e| format!("invalid address hex: {e}"))?;
            let mut address = [0u8; 20];
            address.copy_from_slice(&addr_bytes);

            let bc_hex = parts[1].strip_prefix("0x").unwrap_or(parts[1]);
            let bytecode = hex::decode(bc_hex).map_err(|e| format!("invalid bytecode hex: {e}"))?;

            Ok(Contract {
                address,
                bytecode,
                chain_id,
            })
        }
        n => Err(format!("expected 1 or 2 fields, got {n}")),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_static_source_single() {
        let mut src = StaticSource::single([1u8; 20], vec![0x60, 0x00, 0x55]);
        assert!(!src.is_exhausted());

        let contracts = src.drain();
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].address, [1u8; 20]);
        assert_eq!(contracts[0].bytecode, vec![0x60, 0x00, 0x55]);
        assert!(src.is_exhausted());

        // Second drain returns empty
        assert!(src.drain().is_empty());
    }

    #[test]
    fn test_static_source_from_pairs() {
        let mut src =
            StaticSource::from_pairs(vec![([1u8; 20], vec![0x00]), ([2u8; 20], vec![0x01])]);
        let contracts = src.drain();
        assert_eq!(contracts.len(), 2);
    }

    #[test]
    fn test_file_source_bytecode_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contracts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# comment line").unwrap();
        writeln!(f, "6000556001f1").unwrap();
        writeln!(f).unwrap(); // empty line
        writeln!(f, "0x600160005500").unwrap();
        drop(f);

        let mut src = FileSource::load(&path, 1).unwrap();
        let contracts = src.drain();
        assert_eq!(contracts.len(), 2);
        assert_eq!(
            contracts[0].bytecode,
            vec![0x60, 0x00, 0x55, 0x60, 0x01, 0xf1]
        );
        assert_eq!(
            contracts[1].bytecode,
            vec![0x60, 0x01, 0x60, 0x00, 0x55, 0x00]
        );
    }

    #[test]
    fn test_file_source_address_and_bytecode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contracts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "1234567890abcdef1234567890abcdef12345678 6000556001f1").unwrap();
        drop(f);

        let mut src = FileSource::load(&path, 42).unwrap();
        let contracts = src.drain();
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].chain_id, 42);
        assert_eq!(contracts[0].address[0..4], [0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_file_source_invalid_hex() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.txt");
        std::fs::write(&path, "not_hex_data_gg").unwrap();

        let result = FileSource::load(&path, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_contract_line_three_fields() {
        let result = parse_contract_line("a b c", 0, 1);
        assert!(result.is_err());
    }
}
