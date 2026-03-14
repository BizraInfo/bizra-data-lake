//! EVM Instruction Decoder
//!
//! Zero-copy linear-sweep decoder that disambiguates opcodes from PUSH data.
//! Bridges the SAPE Symbolic layer: raw bytes → structured instructions.
//!
//! Performance target: <10µs per 4KB contract.

/// EVM opcode with semantic predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // 0x00s: Stop and Arithmetic
    Stop = 0x00,
    Add = 0x01,
    Mul = 0x02,
    Sub = 0x03,
    Div = 0x04,
    SDiv = 0x05,
    Mod = 0x06,
    SMod = 0x07,
    AddMod = 0x08,
    MulMod = 0x09,
    Exp = 0x0a,
    SignExtend = 0x0b,

    // 0x10s: Comparison & Bitwise
    Lt = 0x10,
    Gt = 0x11,
    Slt = 0x12,
    Sgt = 0x13,
    Eq = 0x14,
    IsZero = 0x15,
    And = 0x16,
    Or = 0x17,
    Xor = 0x18,
    Not = 0x19,
    Byte = 0x1a,
    Shl = 0x1b,
    Shr = 0x1c,
    Sar = 0x1d,

    // 0x20: Keccak256
    Keccak256 = 0x20,

    // 0x30s: Environmental
    Address = 0x30,
    Balance = 0x31,
    Origin = 0x32,
    Caller = 0x33,
    CallValue = 0x34,
    CallDataLoad = 0x35,
    CallDataSize = 0x36,
    CallDataCopy = 0x37,
    CodeSize = 0x38,
    CodeCopy = 0x39,
    GasPrice = 0x3a,
    ExtCodeSize = 0x3b,
    ExtCodeCopy = 0x3c,
    ReturnDataSize = 0x3d,
    ReturnDataCopy = 0x3e,
    ExtCodeHash = 0x3f,

    // 0x40s: Block
    BlockHash = 0x40,
    Coinbase = 0x41,
    Timestamp = 0x42,
    Number = 0x43,
    Difficulty = 0x44, // PREVRANDAO post-merge
    GasLimit = 0x45,
    ChainId = 0x46,
    SelfBalance = 0x47,
    BaseFee = 0x48,
    BlobHash = 0x49,
    BlobBaseFee = 0x4a,

    // 0x50s: Stack, Memory, Storage, Flow
    Pop = 0x50,
    MLoad = 0x51,
    MStore = 0x52,
    MStore8 = 0x53,
    SLoad = 0x54,
    SStore = 0x55,
    Jump = 0x56,
    JumpI = 0x57,
    Pc = 0x58,
    MSize = 0x59,
    Gas = 0x5a,
    JumpDest = 0x5b,
    TLoad = 0x5c,
    TStore = 0x5d,
    MCopy = 0x5e,
    Push0 = 0x5f,

    // 0x60-0x7f: PUSH1-PUSH32
    Push1 = 0x60,
    Push2 = 0x61,
    Push3 = 0x62,
    Push4 = 0x63,
    Push5 = 0x64,
    Push6 = 0x65,
    Push7 = 0x66,
    Push8 = 0x67,
    Push9 = 0x68,
    Push10 = 0x69,
    Push11 = 0x6a,
    Push12 = 0x6b,
    Push13 = 0x6c,
    Push14 = 0x6d,
    Push15 = 0x6e,
    Push16 = 0x6f,
    Push17 = 0x70,
    Push18 = 0x71,
    Push19 = 0x72,
    Push20 = 0x73,
    Push21 = 0x74,
    Push22 = 0x75,
    Push23 = 0x76,
    Push24 = 0x77,
    Push25 = 0x78,
    Push26 = 0x79,
    Push27 = 0x7a,
    Push28 = 0x7b,
    Push29 = 0x7c,
    Push30 = 0x7d,
    Push31 = 0x7e,
    Push32 = 0x7f,

    // 0x80-0x8f: DUP1-DUP16
    Dup1 = 0x80,
    Dup2 = 0x81,
    Dup3 = 0x82,
    Dup4 = 0x83,
    Dup5 = 0x84,
    Dup6 = 0x85,
    Dup7 = 0x86,
    Dup8 = 0x87,
    Dup9 = 0x88,
    Dup10 = 0x89,
    Dup11 = 0x8a,
    Dup12 = 0x8b,
    Dup13 = 0x8c,
    Dup14 = 0x8d,
    Dup15 = 0x8e,
    Dup16 = 0x8f,

    // 0x90-0x9f: SWAP1-SWAP16
    Swap1 = 0x90,
    Swap2 = 0x91,
    Swap3 = 0x92,
    Swap4 = 0x93,
    Swap5 = 0x94,
    Swap6 = 0x95,
    Swap7 = 0x96,
    Swap8 = 0x97,
    Swap9 = 0x98,
    Swap10 = 0x99,
    Swap11 = 0x9a,
    Swap12 = 0x9b,
    Swap13 = 0x9c,
    Swap14 = 0x9d,
    Swap15 = 0x9e,
    Swap16 = 0x9f,

    // 0xa0-0xa4: LOG0-LOG4
    Log0 = 0xa0,
    Log1 = 0xa1,
    Log2 = 0xa2,
    Log3 = 0xa3,
    Log4 = 0xa4,

    // 0xf0s: System
    Create = 0xf0,
    Call = 0xf1,
    CallCode = 0xf2,
    Return = 0xf3,
    DelegateCall = 0xf4,
    Create2 = 0xf5,
    StaticCall = 0xfa,
    Revert = 0xfd,
    Invalid = 0xfe,
    SelfDestruct = 0xff,
}

impl OpCode {
    /// Construct from raw byte. Unrecognized opcodes map to Invalid.
    #[inline]
    pub fn from_byte(byte: u8) -> Self {
        // Safety: all defined variants match their discriminant.
        // Undefined opcodes (gaps) are mapped to Invalid.
        match byte {
            0x00..=0x0b => unsafe { std::mem::transmute(byte) },
            0x10..=0x1d => unsafe { std::mem::transmute(byte) },
            0x20 => Self::Keccak256,
            0x30..=0x3f => unsafe { std::mem::transmute(byte) },
            0x40..=0x4a => unsafe { std::mem::transmute(byte) },
            0x50..=0x5f => unsafe { std::mem::transmute(byte) },
            0x60..=0x7f => unsafe { std::mem::transmute(byte) },
            0x80..=0x8f => unsafe { std::mem::transmute(byte) },
            0x90..=0x9f => unsafe { std::mem::transmute(byte) },
            0xa0..=0xa4 => unsafe { std::mem::transmute(byte) },
            0xf0..=0xf5 => unsafe { std::mem::transmute(byte) },
            0xfa => Self::StaticCall,
            0xfd => Self::Revert,
            0xfe => Self::Invalid,
            0xff => Self::SelfDestruct,
            _ => Self::Invalid,
        }
    }

    /// True if this is a PUSH instruction (PUSH0-PUSH32)
    #[inline]
    pub fn is_push(&self) -> bool {
        let b = *self as u8;
        b >= 0x5f && b <= 0x7f
    }

    /// Number of immediate data bytes following this instruction.
    /// PUSH1=1, PUSH2=2, ..., PUSH32=32. All others=0.
    #[inline]
    pub fn push_size(&self) -> usize {
        let b = *self as u8;
        if b >= 0x60 && b <= 0x7f {
            (b - 0x5f) as usize
        } else {
            0
        }
    }

    /// True for JUMP, JUMPI, JUMPDEST
    #[inline]
    pub fn is_jump(&self) -> bool {
        matches!(self, Self::Jump | Self::JumpI | Self::JumpDest)
    }

    /// True for CALL, CALLCODE, DELEGATECALL, STATICCALL
    #[inline]
    pub fn is_call(&self) -> bool {
        matches!(
            self,
            Self::Call | Self::CallCode | Self::DelegateCall | Self::StaticCall
        )
    }

    /// True for RETURN, REVERT, INVALID, SELFDESTRUCT, STOP
    #[inline]
    pub fn is_halt(&self) -> bool {
        matches!(
            self,
            Self::Return | Self::Revert | Self::Invalid | Self::SelfDestruct | Self::Stop
        )
    }

    /// True for SLOAD, SSTORE
    #[inline]
    pub fn is_storage(&self) -> bool {
        matches!(self, Self::SLoad | Self::SStore)
    }

    /// True for CALLVALUE, BALANCE, SELFBALANCE
    #[inline]
    pub fn is_value(&self) -> bool {
        matches!(self, Self::CallValue | Self::Balance | Self::SelfBalance)
    }

    /// True for TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT, BASEFEE
    #[inline]
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::Timestamp | Self::Number | Self::Difficulty | Self::GasLimit | Self::BaseFee
        )
    }

    /// True for MLOAD, MSTORE, MSTORE8, CALLDATACOPY, CODECOPY, RETURNDATACOPY, MCOPY
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(
            self,
            Self::MLoad
                | Self::MStore
                | Self::MStore8
                | Self::CallDataCopy
                | Self::CodeCopy
                | Self::ReturnDataCopy
                | Self::MCopy
        )
    }
}

/// A decoded EVM instruction with zero-copy reference to push data.
#[derive(Debug, Clone, Copy)]
pub struct Instruction<'a> {
    /// Byte offset of this instruction in the original bytecode
    pub offset: u32,
    /// The opcode
    pub opcode: OpCode,
    /// Immediate data (non-empty only for PUSH1-PUSH32)
    pub push_data: &'a [u8],
}

/// Zero-copy linear-sweep EVM bytecode decoder.
pub struct EvmDecoder;

impl EvmDecoder {
    /// Decode bytecode into a vector of instructions.
    ///
    /// Linear sweep: O(n) single pass. PUSH1-PUSH32 skips N data bytes.
    /// Truncated PUSH at end of bytecode is handled gracefully.
    #[inline]
    pub fn decode<'a>(bytecode: &'a [u8]) -> Vec<Instruction<'a>> {
        if bytecode.is_empty() {
            return Vec::new();
        }

        // Pre-allocate: average EVM instruction is ~2.5 bytes (many PUSH ops)
        let estimated_count = bytecode.len() * 2 / 5;
        let mut instructions = Vec::with_capacity(estimated_count.max(16));
        let len = bytecode.len();
        let mut pc = 0usize;

        while pc < len {
            let opcode = OpCode::from_byte(bytecode[pc]);
            let push_size = opcode.push_size();

            let data_start = pc + 1;
            // Clamp to bytecode length for truncated PUSH at end
            let data_end = (data_start + push_size).min(len);
            let push_data = if push_size > 0 && data_start < len {
                &bytecode[data_start..data_end]
            } else {
                &[] as &[u8]
            };

            instructions.push(Instruction {
                offset: pc as u32,
                opcode,
                push_data,
            });

            pc = data_end;
            if push_size == 0 {
                pc = data_start; // advance past the opcode byte
            }
        }

        instructions
    }

    /// Decode and also produce an instruction map (bit-vector).
    ///
    /// The map has one entry per byte: `true` if that byte offset is an
    /// instruction start, `false` if it is PUSH data or past the end.
    /// Enables O(1) "is this offset an opcode?" lookups.
    pub fn decode_with_map<'a>(bytecode: &'a [u8]) -> (Vec<Instruction<'a>>, Vec<bool>) {
        let instructions = Self::decode(bytecode);
        let mut map = vec![false; bytecode.len()];
        for inst in &instructions {
            let offset = inst.offset as usize;
            if offset < map.len() {
                map[offset] = true;
            }
        }
        (instructions, map)
    }

    /// Count only executable opcodes (excludes PUSH data bytes).
    /// Useful for normalizing density calculations.
    #[inline]
    pub fn opcode_count(instructions: &[Instruction]) -> usize {
        instructions.len()
    }
}

/// Scan an instruction stream for suspicious opcode bigrams.
///
/// Returns `(vuln_pattern, offset)` of the highest-signal match,
/// or `None` if no known dangerous pattern is found.
pub fn detect_opcode_sequences(instructions: &[Instruction]) -> Option<(SequencePattern, u32)> {
    let mut best: Option<(SequencePattern, u32, u8)> = None; // (pattern, offset, priority)

    for window in instructions.windows(2) {
        let a = window[0].opcode;
        let b = window[1].opcode;

        let candidate = match (a, b) {
            // SSTORE followed by external CALL → classic reentrancy
            (OpCode::SStore, op) if op.is_call() => {
                Some((SequencePattern::SStoreBeforeCall, window[0].offset, 10))
            }
            // CALLVALUE followed by conditional jump → access control / payable guard
            (OpCode::CallValue, OpCode::JumpI) => {
                Some((SequencePattern::CallValueJumpi, window[0].offset, 6))
            }
            // TIMESTAMP into SSTORE → time-dependent state mutation
            (OpCode::Timestamp, OpCode::SStore) => {
                Some((SequencePattern::TimestampSStore, window[0].offset, 7))
            }
            // External CALL result not checked (CALL → POP discards success flag)
            (op, OpCode::Pop) if op.is_call() => {
                Some((SequencePattern::UncheckedCall, window[0].offset, 8))
            }
            // DELEGATECALL to dynamic target (after SLOAD or CALLDATALOAD)
            (OpCode::SLoad | OpCode::CallDataLoad, OpCode::DelegateCall) => {
                Some((SequencePattern::DynamicDelegateCall, window[0].offset, 9))
            }
            _ => None,
        };

        if let Some((pattern, offset, priority)) = candidate {
            if best
                .as_ref()
                .map_or(true, |(_, _, best_p)| priority > *best_p)
            {
                best = Some((pattern, offset, priority));
            }
        }
    }

    best.map(|(pattern, offset, _)| (pattern, offset))
}

/// Known dangerous opcode sequence patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePattern {
    /// SSTORE → CALL (reentrancy)
    SStoreBeforeCall,
    /// CALLVALUE → JUMPI (access control / payable check)
    CallValueJumpi,
    /// TIMESTAMP → SSTORE (front-running / time manipulation)
    TimestampSStore,
    /// CALL → POP (unchecked external call return)
    UncheckedCall,
    /// SLOAD/CALLDATALOAD → DELEGATECALL (dynamic delegate target)
    DynamicDelegateCall,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty() {
        let instructions = EvmDecoder::decode(&[]);
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_decode_simple_opcodes() {
        // STOP, ADD, MUL
        let bytecode = [0x00, 0x01, 0x02];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 3);
        assert_eq!(instructions[0].opcode, OpCode::Stop);
        assert_eq!(instructions[0].offset, 0);
        assert_eq!(instructions[1].opcode, OpCode::Add);
        assert_eq!(instructions[1].offset, 1);
        assert_eq!(instructions[2].opcode, OpCode::Mul);
        assert_eq!(instructions[2].offset, 2);
    }

    #[test]
    fn test_push1_skips_data() {
        // PUSH1 0xAB, STOP
        let bytecode = [0x60, 0xAB, 0x00];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].opcode, OpCode::Push1);
        assert_eq!(instructions[0].push_data, &[0xAB]);
        assert_eq!(instructions[0].offset, 0);
        assert_eq!(instructions[1].opcode, OpCode::Stop);
        assert_eq!(instructions[1].offset, 2);
    }

    #[test]
    fn test_push32_skips_32_bytes() {
        // PUSH32 + 32 data bytes + STOP
        let mut bytecode = vec![0x7f]; // PUSH32
        bytecode.extend_from_slice(&[0xFF; 32]);
        bytecode.push(0x00); // STOP
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].opcode, OpCode::Push32);
        assert_eq!(instructions[0].push_data.len(), 32);
        assert_eq!(instructions[1].opcode, OpCode::Stop);
        assert_eq!(instructions[1].offset, 33);
    }

    #[test]
    fn test_push_data_not_counted_as_opcode() {
        // PUSH1 0x56, STOP — 0x56 is JUMP but here it's data, NOT an opcode
        let bytecode = [0x60, 0x56, 0x00];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 2);
        // The 0x56 byte is push data, not decoded as JUMP
        assert_eq!(instructions[0].opcode, OpCode::Push1);
        assert_eq!(instructions[1].opcode, OpCode::Stop);
        assert_eq!(instructions[1].offset, 2);
    }

    #[test]
    fn test_truncated_push_at_end() {
        // PUSH4 at end with only 2 data bytes available
        let bytecode = [0x63, 0xAA, 0xBB];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].opcode, OpCode::Push4);
        // Only get the 2 available bytes, not 4
        assert_eq!(instructions[0].push_data, &[0xAA, 0xBB]);
    }

    #[test]
    fn test_instruction_map() {
        // PUSH1 0xAB, JUMP (0x56)
        let bytecode = [0x60, 0xAB, 0x56];
        let (instructions, map) = EvmDecoder::decode_with_map(&bytecode);
        assert_eq!(instructions.len(), 2);
        assert!(map[0]); // offset 0: PUSH1 (instruction)
        assert!(!map[1]); // offset 1: 0xAB (push data)
        assert!(map[2]); // offset 2: JUMP (instruction)
    }

    #[test]
    fn test_push0() {
        // PUSH0 doesn't have data bytes
        let bytecode = [0x5f, 0x00];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].opcode, OpCode::Push0);
        assert!(instructions[0].push_data.is_empty());
        assert_eq!(instructions[1].opcode, OpCode::Stop);
    }

    #[test]
    fn test_sstore_call_sequence_detected() {
        // SSTORE, CALL → reentrancy pattern
        let bytecode = [0x55, 0xf1];
        let instructions = EvmDecoder::decode(&bytecode);
        let result = detect_opcode_sequences(&instructions);
        assert!(result.is_some());
        let (pattern, offset) = result.unwrap();
        assert_eq!(pattern, SequencePattern::SStoreBeforeCall);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_timestamp_sstore_sequence_detected() {
        // TIMESTAMP, SSTORE → front-running pattern
        let bytecode = [0x42, 0x55];
        let instructions = EvmDecoder::decode(&bytecode);
        let result = detect_opcode_sequences(&instructions);
        assert!(result.is_some());
        let (pattern, _) = result.unwrap();
        assert_eq!(pattern, SequencePattern::TimestampSStore);
    }

    #[test]
    fn test_no_false_positive_on_push_data() {
        // PUSH1 0x55 (data=SSTORE opcode), PUSH1 0xF1 (data=CALL opcode), STOP
        // The 0x55 and 0xF1 are data, NOT opcodes — should NOT trigger sequence detection
        let bytecode = [0x60, 0x55, 0x60, 0xF1, 0x00];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 3); // PUSH1, PUSH1, STOP
        let result = detect_opcode_sequences(&instructions);
        assert!(result.is_none()); // No dangerous pattern in PUSH1, PUSH1, STOP
    }

    #[test]
    fn test_realistic_bytecode_snippet() {
        // A typical EVM snippet: PUSH1 0x80, PUSH1 0x40, MSTORE, CALLVALUE, JUMPI...
        let bytecode = [
            0x60, 0x80, // PUSH1 0x80
            0x60, 0x40, // PUSH1 0x40
            0x52, // MSTORE
            0x34, // CALLVALUE
            0x80, // DUP1
            0x15, // ISZERO
            0x60, 0x0f, // PUSH1 0x0f
            0x57, // JUMPI
        ];
        let instructions = EvmDecoder::decode(&bytecode);
        assert_eq!(instructions.len(), 8);
        assert_eq!(instructions[0].opcode, OpCode::Push1);
        assert_eq!(instructions[2].opcode, OpCode::MStore);
        assert_eq!(instructions[2].offset, 4);
        assert_eq!(instructions[3].opcode, OpCode::CallValue);
        assert_eq!(instructions[3].offset, 5);
    }

    #[test]
    fn test_opcode_predicates() {
        assert!(OpCode::Push1.is_push());
        assert!(OpCode::Push32.is_push());
        assert!(OpCode::Push0.is_push());
        assert!(!OpCode::Stop.is_push());

        assert_eq!(OpCode::Push1.push_size(), 1);
        assert_eq!(OpCode::Push32.push_size(), 32);
        assert_eq!(OpCode::Push0.push_size(), 0);
        assert_eq!(OpCode::Stop.push_size(), 0);

        assert!(OpCode::Jump.is_jump());
        assert!(OpCode::JumpI.is_jump());
        assert!(OpCode::JumpDest.is_jump());
        assert!(!OpCode::Stop.is_jump());

        assert!(OpCode::Call.is_call());
        assert!(OpCode::DelegateCall.is_call());
        assert!(OpCode::StaticCall.is_call());
        assert!(!OpCode::Create.is_call());

        assert!(OpCode::SLoad.is_storage());
        assert!(OpCode::SStore.is_storage());
        assert!(!OpCode::MLoad.is_storage());

        assert!(OpCode::CallValue.is_value());
        assert!(OpCode::Balance.is_value());
        assert!(OpCode::SelfBalance.is_value());

        assert!(OpCode::Timestamp.is_temporal());
        assert!(OpCode::Number.is_temporal());
        assert!(OpCode::BaseFee.is_temporal());

        assert!(OpCode::MLoad.is_memory());
        assert!(OpCode::MStore.is_memory());
        assert!(OpCode::CallDataCopy.is_memory());
    }
}
