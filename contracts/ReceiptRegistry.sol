// SPDX-License-Identifier: MIT
// BIZRA Native Chain - Receipt Registry Contract
// Anchors all BIZRA receipts to the native blockchain for immutable evidence

pragma solidity ^0.8.19;

import "./interfaces/IIhsanOracle.sol";

/**
 * @title ReceiptRegistry
 * @notice Anchors BIZRA execution and rejection receipts to the native chain
 * @dev All receipts are immutable once anchored - append-only storage
 *
 * Receipt Types:
 * - Execution: Successful PAT/SAT flow completion
 * - Rejection: SAT consensus failure
 * - Quarantine: Pending human review
 * - IhsanFailure: Score below threshold (0.95)
 */
contract ReceiptRegistry {
    // ═══════════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════════

    enum ReceiptType {
        Execution,      // Successful execution
        Rejection,      // SAT rejection
        Quarantine,     // Pending review
        IhsanFailure,   // Ihsan threshold failure
        SynergyDetection,
        CompoundDiscovery,
        ExplosionModeEntry,
        ExplosionModeExit
    }

    struct Receipt {
        bytes32 receiptId;          // Unique receipt identifier
        ReceiptType receiptType;    // Type of receipt
        bytes32 requestId;          // Original request ID
        uint256 timestamp;          // Block timestamp
        bytes32 taskSummaryHash;    // SHA256 of task (privacy)
        bytes32 integrityHash;      // Full content hash
        uint96 ihsanScore;          // Ihsan score (scaled by 1e4)
        uint96 ihsanThreshold;      // Threshold applied (scaled by 1e4)
        address anchoredBy;         // Node that anchored this receipt
        uint8 satApprovers;         // SAT validators that approved (0-5)
        uint8 satRejectors;         // SAT validators that rejected (0-5)
        bool isAnchored;            // Anchoring status
    }

    struct ReceiptBatch {
        bytes32[] receiptIds;
        uint256 batchTimestamp;
        bytes32 batchMerkleRoot;
        address batchAnchor;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice Ihsan Oracle for score validation
    IIhsanOracle public ihsanOracle;

    /// @notice All anchored receipts by ID
    mapping(bytes32 => Receipt) public receipts;

    /// @notice Receipt IDs by type for querying
    mapping(ReceiptType => bytes32[]) public receiptsByType;

    /// @notice Receipt IDs by anchoring node
    mapping(address => bytes32[]) public receiptsByNode;

    /// @notice Request ID to receipt ID mapping
    mapping(bytes32 => bytes32) public requestToReceipt;

    /// @notice Total receipts anchored
    uint256 public totalReceipts;

    /// @notice Total receipts by type
    mapping(ReceiptType => uint256) public countByType;

    /// @notice Receipt batches for efficient anchoring
    ReceiptBatch[] public batches;

    /// @notice Authorized anchor nodes (PAT/SAT nodes)
    mapping(address => bool) public authorizedAnchors;

    /// @notice Genesis block reference
    bytes32 public immutable GENESIS_HASH;

    /// @notice Admin for adding authorized anchors
    address public admin;

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event ReceiptAnchored(
        bytes32 indexed receiptId,
        ReceiptType indexed receiptType,
        bytes32 requestId,
        bytes32 integrityHash,
        uint96 ihsanScore,
        address indexed anchoredBy,
        uint256 timestamp
    );

    event BatchAnchored(
        uint256 indexed batchIndex,
        bytes32 merkleRoot,
        uint256 receiptCount,
        address indexed anchoredBy
    );

    event AnchorAuthorized(address indexed node);
    event AnchorRevoked(address indexed node);

    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error UnauthorizedAnchor();
    error ReceiptAlreadyAnchored();
    error InvalidReceiptId();
    error InvalidIhsanScore();
    error OracleValidationFailed();
    error BatchTooLarge();
    error EmptyBatch();

    // ═══════════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════════

    modifier onlyAuthorizedAnchor() {
        if (!authorizedAnchors[msg.sender]) revert UnauthorizedAnchor();
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor(bytes32 _genesisHash, address _ihsanOracle) {
        GENESIS_HASH = _genesisHash;
        ihsanOracle = IIhsanOracle(_ihsanOracle);
        admin = msg.sender;
        authorizedAnchors[msg.sender] = true;
        emit AnchorAuthorized(msg.sender);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ANCHOR FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Anchor a single receipt to the chain
     * @param receiptId Unique receipt identifier
     * @param receiptType Type of receipt
     * @param requestId Original request ID
     * @param taskSummaryHash SHA256 of task summary
     * @param integrityHash Full content integrity hash
     * @param ihsanScore Ihsan score (scaled by 1e4, e.g., 9500 = 0.95)
     * @param ihsanThreshold Threshold applied
     * @param satApprovers Number of SAT approvers
     * @param satRejectors Number of SAT rejectors
     */
    function anchorReceipt(
        bytes32 receiptId,
        ReceiptType receiptType,
        bytes32 requestId,
        bytes32 taskSummaryHash,
        bytes32 integrityHash,
        uint96 ihsanScore,
        uint96 ihsanThreshold,
        uint8 satApprovers,
        uint8 satRejectors
    ) external onlyAuthorizedAnchor {
        if (receiptId == bytes32(0)) revert InvalidReceiptId();
        if (receipts[receiptId].isAnchored) revert ReceiptAlreadyAnchored();
        if (ihsanScore > 10000) revert InvalidIhsanScore(); // Max 1.0 scaled

        // Validate Ihsan score against oracle if execution receipt
        if (receiptType == ReceiptType.Execution) {
            if (!ihsanOracle.validateScore(ihsanScore, ihsanThreshold)) {
                revert OracleValidationFailed();
            }
        }

        Receipt memory receipt = Receipt({
            receiptId: receiptId,
            receiptType: receiptType,
            requestId: requestId,
            timestamp: block.timestamp,
            taskSummaryHash: taskSummaryHash,
            integrityHash: integrityHash,
            ihsanScore: ihsanScore,
            ihsanThreshold: ihsanThreshold,
            anchoredBy: msg.sender,
            satApprovers: satApprovers,
            satRejectors: satRejectors,
            isAnchored: true
        });

        receipts[receiptId] = receipt;
        receiptsByType[receiptType].push(receiptId);
        receiptsByNode[msg.sender].push(receiptId);

        if (requestId != bytes32(0)) {
            requestToReceipt[requestId] = receiptId;
        }

        totalReceipts++;
        countByType[receiptType]++;

        emit ReceiptAnchored(
            receiptId,
            receiptType,
            requestId,
            integrityHash,
            ihsanScore,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Anchor multiple receipts in a batch with Merkle root
     * @param receiptIds Array of receipt IDs
     * @param merkleRoot Merkle root of all receipt hashes
     */
    function anchorBatch(
        bytes32[] calldata receiptIds,
        bytes32 merkleRoot
    ) external onlyAuthorizedAnchor {
        if (receiptIds.length == 0) revert EmptyBatch();
        if (receiptIds.length > 1000) revert BatchTooLarge();

        ReceiptBatch memory batch = ReceiptBatch({
            receiptIds: receiptIds,
            batchTimestamp: block.timestamp,
            batchMerkleRoot: merkleRoot,
            batchAnchor: msg.sender
        });

        batches.push(batch);

        emit BatchAnchored(
            batches.length - 1,
            merkleRoot,
            receiptIds.length,
            msg.sender
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Get receipt by ID
     */
    function getReceipt(bytes32 receiptId) external view returns (Receipt memory) {
        return receipts[receiptId];
    }

    /**
     * @notice Check if receipt exists and is anchored
     */
    function isAnchored(bytes32 receiptId) external view returns (bool) {
        return receipts[receiptId].isAnchored;
    }

    /**
     * @notice Get receipt by request ID
     */
    function getReceiptByRequest(bytes32 requestId) external view returns (Receipt memory) {
        bytes32 receiptId = requestToReceipt[requestId];
        return receipts[receiptId];
    }

    /**
     * @notice Get count of receipts by type
     */
    function getCountByType(ReceiptType receiptType) external view returns (uint256) {
        return countByType[receiptType];
    }

    /**
     * @notice Get receipt IDs by type (paginated)
     */
    function getReceiptsByType(
        ReceiptType receiptType,
        uint256 offset,
        uint256 limit
    ) external view returns (bytes32[] memory) {
        bytes32[] storage typeReceipts = receiptsByType[receiptType];
        uint256 total = typeReceipts.length;

        if (offset >= total) {
            return new bytes32[](0);
        }

        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }

        bytes32[] memory result = new bytes32[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            result[i - offset] = typeReceipts[i];
        }

        return result;
    }

    /**
     * @notice Get total batch count
     */
    function getBatchCount() external view returns (uint256) {
        return batches.length;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Authorize a new anchor node
     */
    function authorizeAnchor(address node) external onlyAdmin {
        authorizedAnchors[node] = true;
        emit AnchorAuthorized(node);
    }

    /**
     * @notice Revoke anchor authorization
     */
    function revokeAnchor(address node) external onlyAdmin {
        authorizedAnchors[node] = false;
        emit AnchorRevoked(node);
    }

    /**
     * @notice Update Ihsan Oracle address
     */
    function setIhsanOracle(address _oracle) external onlyAdmin {
        ihsanOracle = IIhsanOracle(_oracle);
    }

    /**
     * @notice Transfer admin role
     */
    function transferAdmin(address newAdmin) external onlyAdmin {
        admin = newAdmin;
    }
}
