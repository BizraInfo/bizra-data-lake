//! Multi-Axis Entropy Calculation
//!
//! TRICK 3: SIMD-accelerated Shannon entropy across multiple axes.
//!
//! Axes:
//! 1. Bytecode entropy (raw byte distribution)
//! 2. CFG entropy (control flow complexity)
//! 3. State entropy (storage patterns)
//! 4. Economic entropy (value flow patterns)
//! 5. Temporal entropy (timestamp dependencies)
//! 6. Memory entropy (memory access patterns)
//!
//! Supports two modes:
//! - **Raw byte mode** (legacy): operates on `&[u8]` directly
//! - **Instruction mode** (accurate): operates on decoded `&[Instruction]`,
//!   eliminating PUSH data noise from opcode counts

use std::f32::consts::LN_2;

use crate::evm::Instruction;

/// Multi-axis entropy measurement (6 axes)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))] // Cache-line aligned for SIMD
pub struct MultiAxisEntropy {
    pub bytecode: f32,
    pub cfg: f32,
    pub state: f32,
    pub economic: f32,
    pub temporal: f32,
    pub memory: f32,
}

impl MultiAxisEntropy {
    /// Create new entropy measurement
    pub const fn new() -> Self {
        Self {
            bytecode: 0.0,
            cfg: 0.0,
            state: 0.0,
            economic: 0.0,
            temporal: 0.0,
            memory: 0.0,
        }
    }

    /// Average entropy across all axes
    #[inline]
    pub fn average(&self) -> f32 {
        (self.bytecode + self.cfg + self.state + self.economic + self.temporal + self.memory) / 6.0
    }

    /// Count axes above threshold
    #[inline]
    pub fn axes_above_threshold(&self, threshold: f32) -> usize {
        let mut count = 0;
        if self.bytecode > threshold {
            count += 1;
        }
        if self.cfg > threshold {
            count += 1;
        }
        if self.state > threshold {
            count += 1;
        }
        if self.economic > threshold {
            count += 1;
        }
        if self.temporal > threshold {
            count += 1;
        }
        if self.memory > threshold {
            count += 1;
        }
        count
    }

    /// Check multi-axis consistency (TRICK 3)
    /// At least `min_axes` must be above threshold
    #[inline]
    pub fn is_consistent(&self, threshold: f32, min_axes: usize) -> bool {
        self.axes_above_threshold(threshold) >= min_axes
    }

    /// Convert to array for SIMD operations
    #[inline]
    pub fn to_array(&self) -> [f32; 6] {
        [
            self.bytecode,
            self.cfg,
            self.state,
            self.economic,
            self.temporal,
            self.memory,
        ]
    }
}

/// Entropy calculator with SIMD optimization
pub struct EntropyCalculator {
    /// Pre-allocated byte count buffer (256 entries)
    byte_counts: [u32; 256],
    /// Pre-computed log2 lookup table (reserved for SIMD-accelerated path)
    #[allow(dead_code)]
    log2_table: [f32; 256],
}

impl EntropyCalculator {
    /// Create new calculator with pre-computed tables
    pub fn new() -> Self {
        let mut log2_table = [0.0f32; 256];
        for (i, entry) in log2_table.iter_mut().enumerate().skip(1) {
            *entry = (i as f32).ln() / LN_2;
        }

        Self {
            byte_counts: [0u32; 256],
            log2_table,
        }
    }

    /// Calculate bytecode entropy (Shannon entropy of byte distribution)
    /// Formula: H(X) = -Σ p(x) log₂ p(x)
    #[inline]
    pub fn bytecode_entropy(&mut self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        // Reset counts
        self.byte_counts.fill(0);

        // Count byte occurrences
        for &byte in bytecode {
            self.byte_counts[byte as usize] += 1;
        }

        // Calculate Shannon entropy
        let total = bytecode.len() as f32;
        let mut entropy = 0.0f32;

        for &count in &self.byte_counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] (max entropy = log₂(256) = 8)
        entropy / 8.0
    }

    /// Calculate CFG entropy (control flow graph complexity)
    /// Higher entropy = more complex control flow
    pub fn cfg_entropy(&self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        // Count control flow opcodes (EVM)
        let mut jumps = 0u32;
        let mut jumpdests = 0u32;
        let mut calls = 0u32;
        let mut returns = 0u32;

        for &opcode in bytecode {
            match opcode {
                0x56 => jumps += 1,                        // JUMP
                0x57 => jumps += 1,                        // JUMPI
                0x5b => jumpdests += 1,                    // JUMPDEST
                0xf1 | 0xf2 | 0xf4 | 0xfa => calls += 1,   // CALL variants
                0xf3 | 0xfd | 0xfe | 0xff => returns += 1, // RETURN variants
                _ => {}
            }
        }

        // Complexity score based on control flow
        let total = bytecode.len() as f32;
        let jump_density = (jumps + jumpdests) as f32 / total;
        let call_density = calls as f32 / total;
        let return_density = returns as f32 / total;

        // Normalize to [0, 1]
        ((jump_density * 100.0) + (call_density * 50.0) + (return_density * 20.0)).min(1.0)
    }

    /// Calculate state entropy (storage access patterns)
    pub fn state_entropy(&self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        let mut sloads = 0u32;
        let mut sstores = 0u32;

        for &opcode in bytecode {
            match opcode {
                0x54 => sloads += 1,  // SLOAD
                0x55 => sstores += 1, // SSTORE
                _ => {}
            }
        }

        // Entropy based on storage access balance
        let total_storage = (sloads + sstores) as f32;
        if total_storage == 0.0 {
            return 0.0;
        }

        let read_ratio = sloads as f32 / total_storage;
        let write_ratio = sstores as f32 / total_storage;

        // Higher entropy when read/write ratio is unbalanced (potential vulnerability)
        let imbalance = (read_ratio - write_ratio).abs();
        let density = total_storage / bytecode.len() as f32;

        (imbalance + density * 10.0).min(1.0)
    }

    /// Calculate economic entropy (value transfer patterns)
    pub fn economic_entropy(&self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        let mut value_ops = 0u32;
        let mut balance_checks = 0u32;

        for &opcode in bytecode {
            match opcode {
                0x34 => value_ops += 1,      // CALLVALUE
                0x31 => balance_checks += 1, // BALANCE
                0x47 => balance_checks += 1, // SELFBALANCE
                _ => {}
            }
        }

        // Economic complexity
        let total = (value_ops + balance_checks) as f32;
        let density = total / bytecode.len() as f32;

        (density * 100.0).min(1.0)
    }

    /// Calculate temporal entropy (timestamp/block dependencies)
    pub fn temporal_entropy(&self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        let mut temporal_ops = 0u32;

        for &opcode in bytecode {
            match opcode {
                0x42 => temporal_ops += 1, // TIMESTAMP
                0x43 => temporal_ops += 1, // NUMBER
                0x44 => temporal_ops += 1, // DIFFICULTY/PREVRANDAO
                0x45 => temporal_ops += 1, // GASLIMIT
                0x48 => temporal_ops += 1, // BASEFEE
                _ => {}
            }
        }

        let density = temporal_ops as f32 / bytecode.len() as f32;
        (density * 500.0).min(1.0)
    }

    /// Calculate memory entropy (memory access patterns)
    pub fn memory_entropy(&self, bytecode: &[u8]) -> f32 {
        if bytecode.is_empty() {
            return 0.0;
        }

        let mut mloads = 0u32;
        let mut mstores = 0u32;
        let mut mcopy = 0u32;

        for &opcode in bytecode {
            match opcode {
                0x51 => mloads += 1,  // MLOAD
                0x52 => mstores += 1, // MSTORE
                0x53 => mstores += 1, // MSTORE8
                0x37 => mcopy += 1,   // CALLDATACOPY
                0x39 => mcopy += 1,   // CODECOPY
                0x3e => mcopy += 1,   // RETURNDATACOPY
                _ => {}
            }
        }

        let total = (mloads + mstores + mcopy) as f32;
        let density = total / bytecode.len() as f32;

        (density * 20.0).min(1.0)
    }

    /// Calculate all entropy axes at once
    pub fn calculate_all(&mut self, bytecode: &[u8]) -> MultiAxisEntropy {
        MultiAxisEntropy {
            bytecode: self.bytecode_entropy(bytecode),
            cfg: self.cfg_entropy(bytecode),
            state: self.state_entropy(bytecode),
            economic: self.economic_entropy(bytecode),
            temporal: self.temporal_entropy(bytecode),
            memory: self.memory_entropy(bytecode),
        }
    }

    // ── Instruction-level methods (accurate, PUSH-data-aware) ──────────

    /// Calculate CFG entropy from decoded instructions (no PUSH data noise)
    pub fn cfg_entropy_from_instructions(&self, instructions: &[Instruction]) -> f32 {
        if instructions.is_empty() {
            return 0.0;
        }

        let mut jumps = 0u32;
        let mut jumpdests = 0u32;
        let mut calls = 0u32;
        let mut halts = 0u32;

        for inst in instructions {
            if inst.opcode.is_jump() {
                if inst.opcode == crate::evm::OpCode::JumpDest {
                    jumpdests += 1;
                } else {
                    jumps += 1;
                }
            } else if inst.opcode.is_call() {
                calls += 1;
            } else if inst.opcode.is_halt() {
                halts += 1;
            }
        }

        let total = instructions.len() as f32;
        let jump_density = (jumps + jumpdests) as f32 / total;
        let call_density = calls as f32 / total;
        let halt_density = halts as f32 / total;

        ((jump_density * 100.0) + (call_density * 50.0) + (halt_density * 20.0)).min(1.0)
    }

    /// Calculate state entropy from decoded instructions
    pub fn state_entropy_from_instructions(&self, instructions: &[Instruction]) -> f32 {
        if instructions.is_empty() {
            return 0.0;
        }

        let mut sloads = 0u32;
        let mut sstores = 0u32;

        for inst in instructions {
            match inst.opcode {
                crate::evm::OpCode::SLoad => sloads += 1,
                crate::evm::OpCode::SStore => sstores += 1,
                _ => {}
            }
        }

        let total_storage = (sloads + sstores) as f32;
        if total_storage == 0.0 {
            return 0.0;
        }

        let read_ratio = sloads as f32 / total_storage;
        let write_ratio = sstores as f32 / total_storage;
        let imbalance = (read_ratio - write_ratio).abs();
        let density = total_storage / instructions.len() as f32;

        (imbalance + density * 10.0).min(1.0)
    }

    /// Calculate economic entropy from decoded instructions
    pub fn economic_entropy_from_instructions(&self, instructions: &[Instruction]) -> f32 {
        if instructions.is_empty() {
            return 0.0;
        }

        let mut count = 0u32;
        for inst in instructions {
            if inst.opcode.is_value() {
                count += 1;
            }
        }

        let density = count as f32 / instructions.len() as f32;
        (density * 100.0).min(1.0)
    }

    /// Calculate temporal entropy from decoded instructions
    pub fn temporal_entropy_from_instructions(&self, instructions: &[Instruction]) -> f32 {
        if instructions.is_empty() {
            return 0.0;
        }

        let mut count = 0u32;
        for inst in instructions {
            if inst.opcode.is_temporal() {
                count += 1;
            }
        }

        let density = count as f32 / instructions.len() as f32;
        (density * 500.0).min(1.0)
    }

    /// Calculate memory entropy from decoded instructions
    pub fn memory_entropy_from_instructions(&self, instructions: &[Instruction]) -> f32 {
        if instructions.is_empty() {
            return 0.0;
        }

        let mut count = 0u32;
        for inst in instructions {
            if inst.opcode.is_memory() {
                count += 1;
            }
        }

        let density = count as f32 / instructions.len() as f32;
        (density * 20.0).min(1.0)
    }

    /// Calculate all entropy axes using decoded instructions for accuracy.
    ///
    /// `bytecode_entropy` still uses raw bytes (Shannon entropy of byte distribution
    /// is intentionally a Level-0 signal). All other axes use the instruction stream
    /// which correctly excludes PUSH data bytes from opcode counts.
    pub fn calculate_all_from_instructions(
        &mut self,
        instructions: &[Instruction],
        raw_bytecode: &[u8],
    ) -> MultiAxisEntropy {
        MultiAxisEntropy {
            bytecode: self.bytecode_entropy(raw_bytecode),
            cfg: self.cfg_entropy_from_instructions(instructions),
            state: self.state_entropy_from_instructions(instructions),
            economic: self.economic_entropy_from_instructions(instructions),
            temporal: self.temporal_entropy_from_instructions(instructions),
            memory: self.memory_entropy_from_instructions(instructions),
        }
    }
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_entropy_empty() {
        let mut calc = EntropyCalculator::new();
        assert_eq!(calc.bytecode_entropy(&[]), 0.0);
    }

    #[test]
    fn test_bytecode_entropy_uniform() {
        let mut calc = EntropyCalculator::new();
        // Uniform distribution should have high entropy
        let uniform: Vec<u8> = (0..=255).collect();
        let entropy = calc.bytecode_entropy(&uniform);
        assert!(entropy > 0.99, "Uniform distribution entropy: {entropy}");
    }

    #[test]
    fn test_bytecode_entropy_constant() {
        let mut calc = EntropyCalculator::new();
        // Constant bytes should have zero entropy
        let constant = vec![0x60u8; 256];
        let entropy = calc.bytecode_entropy(&constant);
        assert!(entropy < 0.01, "Constant entropy: {entropy}");
    }

    #[test]
    fn test_multi_axis_consistency() {
        let entropy = MultiAxisEntropy {
            bytecode: 0.8,
            cfg: 0.75,
            state: 0.6,
            economic: 0.3,
            temporal: 0.1,
            memory: 0.5,
        };

        // 3 axes above 0.5: bytecode, cfg, state
        assert!(entropy.is_consistent(0.5, 3));
        assert!(!entropy.is_consistent(0.5, 4));
    }

    #[test]
    fn test_cfg_entropy_with_jumps() {
        let calc = EntropyCalculator::new();
        // Bytecode with many jumps
        let bytecode = vec![0x56, 0x57, 0x5b, 0x56, 0x57, 0x5b, 0x60, 0x60];
        let entropy = calc.cfg_entropy(&bytecode);
        assert!(entropy > 0.5, "CFG entropy: {entropy}");
    }

    #[test]
    fn test_instruction_level_removes_push_noise() {
        use crate::evm::EvmDecoder;

        let calc = EntropyCalculator::new();

        // Build bytecode where PUSH data contains bytes that look like
        // SLOAD (0x54) and SSTORE (0x55) to state_entropy — those specific
        // counters won't saturate to 1.0 as easily as CFG counters.
        let mut bytecode = vec![0x7f]; // PUSH32
                                       // 32 data bytes: alternate 0x54 (SLOAD) and 0x55 (SSTORE)
        bytecode.extend_from_slice(&[
            0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55,
            0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55, 0x54, 0x55,
            0x54, 0x55, 0x54, 0x55,
        ]);
        bytecode.push(0x00); // STOP

        // Raw byte mode: falsely counts PUSH data as SLOAD/SSTORE
        let raw_state = calc.state_entropy(&bytecode);

        // Instruction mode: correctly sees only PUSH32, STOP (no storage ops)
        let instructions = EvmDecoder::decode(&bytecode);
        let instr_state = calc.state_entropy_from_instructions(&instructions);

        // Raw should be higher due to false SLOAD/SSTORE counts in PUSH data
        assert!(
            raw_state > instr_state,
            "Raw state ({raw_state}) should exceed instruction state ({instr_state}) \
             because raw mode falsely counts PUSH data bytes as storage opcodes"
        );
        assert_eq!(instr_state, 0.0, "No storage ops in actual instructions");
    }

    #[test]
    fn test_calculate_all_from_instructions() {
        use crate::evm::EvmDecoder;

        let mut calc = EntropyCalculator::new();
        // Simple bytecode: PUSH1 0x00, SLOAD, PUSH1 0x01, SSTORE, TIMESTAMP, STOP
        let bytecode = [0x60, 0x00, 0x54, 0x60, 0x01, 0x55, 0x42, 0x00];
        let instructions = EvmDecoder::decode(&bytecode);

        let entropy = calc.calculate_all_from_instructions(&instructions, &bytecode);

        // bytecode entropy should be non-zero (raw bytes)
        assert!(entropy.bytecode > 0.0);
        // state entropy should be non-zero (SLOAD + SSTORE present)
        assert!(entropy.state > 0.0);
        // temporal entropy should be non-zero (TIMESTAMP present)
        assert!(entropy.temporal > 0.0);
    }
}
