// SPDX-License-Identifier: MIT
// BIZRA Native Chain - Ihsan Oracle Contract
// On-chain enforcement of the Ihsan (Excellence) ethical framework

pragma solidity ^0.8.19;

import "./interfaces/IIhsanOracle.sol";

/**
 * @title IhsanOracle
 * @notice On-chain oracle for Ihsan score validation and calculation
 * @dev Implements the 8-dimension ethical scoring system from ihsan_v1.yaml
 *
 * Dimensions (weights sum to 1.0, represented as 1e4):
 * 0. correctness: 0.22 (2200)
 * 1. safety: 0.22 (2200)
 * 2. user_benefit: 0.14 (1400)
 * 3. efficiency: 0.12 (1200)
 * 4. auditability: 0.12 (1200)
 * 5. anti_centralization: 0.08 (800)
 * 6. robustness: 0.06 (600)
 * 7. adl_fairness: 0.04 (400)
 */
contract IhsanOracle is IIhsanOracle {
    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTANTS (from ihsan_v1.yaml)
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice Dimension weights (scaled by 1e4)
    uint96 public constant WEIGHT_CORRECTNESS = 2200;
    uint96 public constant WEIGHT_SAFETY = 2200;
    uint96 public constant WEIGHT_USER_BENEFIT = 1400;
    uint96 public constant WEIGHT_EFFICIENCY = 1200;
    uint96 public constant WEIGHT_AUDITABILITY = 1200;
    uint96 public constant WEIGHT_ANTI_CENTRALIZATION = 800;
    uint96 public constant WEIGHT_ROBUSTNESS = 600;
    uint96 public constant WEIGHT_ADL_FAIRNESS = 400;

    /// @notice Weights array for iteration
    uint96[8] public WEIGHTS = [
        WEIGHT_CORRECTNESS,
        WEIGHT_SAFETY,
        WEIGHT_USER_BENEFIT,
        WEIGHT_EFFICIENCY,
        WEIGHT_AUDITABILITY,
        WEIGHT_ANTI_CENTRALIZATION,
        WEIGHT_ROBUSTNESS,
        WEIGHT_ADL_FAIRNESS
    ];

    /// @notice Environment thresholds (scaled by 1e4)
    uint96 public constant THRESHOLD_DEV = 8000;        // 0.80
    uint96 public constant THRESHOLD_CI = 9000;         // 0.90
    uint96 public constant THRESHOLD_STAGING = 9500;    // 0.95
    uint96 public constant THRESHOLD_PRODUCTION = 9500; // 0.95

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice Current active environment (0=dev, 1=ci, 2=staging, 3=production)
    uint8 public currentEnvironment;

    /// @notice Admin for environment changes
    address public admin;

    /// @notice Score validation records for audit
    mapping(bytes32 => ScoreValidation) public validationRecords;

    struct ScoreValidation {
        uint96 score;
        uint96 threshold;
        uint8 environment;
        bool passed;
        uint256 timestamp;
        address validator;
    }

    /// @notice Dimension names for readability
    string[8] public DIMENSION_NAMES = [
        "correctness",
        "safety",
        "user_benefit",
        "efficiency",
        "auditability",
        "anti_centralization",
        "robustness",
        "adl_fairness"
    ];

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event ScoreValidated(
        bytes32 indexed requestId,
        uint96 score,
        uint96 threshold,
        bool passed,
        address indexed validator
    );

    event EnvironmentChanged(uint8 indexed oldEnv, uint8 indexed newEnv);

    event ScoreCalculated(
        bytes32 indexed requestId,
        uint96[8] dimensionScores,
        uint96 compositeScore
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error InvalidDimensionIndex();
    error InvalidEnvironment();
    error ScoreBelowThreshold(uint96 score, uint96 threshold);

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor() {
        admin = msg.sender;
        currentEnvironment = 3; // Production by default
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CORE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Validate that an Ihsan score meets threshold
     * @param score The Ihsan score (scaled by 1e4)
     * @param threshold The threshold to validate against
     * @return valid True if score >= threshold
     */
    function validateScore(
        uint96 score,
        uint96 threshold
    ) external view override returns (bool valid) {
        return score >= threshold;
    }

    /**
     * @notice Validate and record a score with audit trail
     * @param requestId Request identifier for tracking
     * @param score The Ihsan score (scaled by 1e4)
     * @return passed True if score passes current threshold
     */
    function validateAndRecord(
        bytes32 requestId,
        uint96 score
    ) external returns (bool passed) {
        uint96 threshold = _getThresholdForEnv(currentEnvironment);
        passed = score >= threshold;

        validationRecords[requestId] = ScoreValidation({
            score: score,
            threshold: threshold,
            environment: currentEnvironment,
            passed: passed,
            timestamp: block.timestamp,
            validator: msg.sender
        });

        emit ScoreValidated(requestId, score, threshold, passed, msg.sender);

        return passed;
    }

    /**
     * @notice Get threshold for environment
     * @param environment Environment identifier (0=dev, 1=ci, 2=staging, 3=production)
     */
    function getThreshold(uint8 environment) external view override returns (uint96 threshold) {
        return _getThresholdForEnv(environment);
    }

    /**
     * @notice Get dimension weight
     * @param dimension Dimension index (0-7)
     */
    function getDimensionWeight(uint8 dimension) external view override returns (uint96 weight) {
        if (dimension >= 8) revert InvalidDimensionIndex();
        return WEIGHTS[dimension];
    }

    /**
     * @notice Calculate Ihsan score from dimension scores
     * @param dimensionScores Array of 8 dimension scores (each 0-10000)
     * @return score Weighted composite score
     */
    function calculateScore(
        uint96[8] calldata dimensionScores
    ) external view override returns (uint96 score) {
        return _calculateComposite(dimensionScores);
    }

    /**
     * @notice Calculate and record Ihsan score with audit trail
     * @param requestId Request identifier
     * @param dimensionScores Array of 8 dimension scores
     * @return score Composite score
     * @return passed Whether it passes current threshold
     */
    function calculateAndRecord(
        bytes32 requestId,
        uint96[8] calldata dimensionScores
    ) external returns (uint96 score, bool passed) {
        score = _calculateComposite(dimensionScores);
        uint96 threshold = _getThresholdForEnv(currentEnvironment);
        passed = score >= threshold;

        validationRecords[requestId] = ScoreValidation({
            score: score,
            threshold: threshold,
            environment: currentEnvironment,
            passed: passed,
            timestamp: block.timestamp,
            validator: msg.sender
        });

        emit ScoreCalculated(requestId, dimensionScores, score);
        emit ScoreValidated(requestId, score, threshold, passed, msg.sender);

        return (score, passed);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Get current active threshold
     */
    function getCurrentThreshold() external view returns (uint96) {
        return _getThresholdForEnv(currentEnvironment);
    }

    /**
     * @notice Get all dimension weights
     */
    function getAllWeights() external view returns (uint96[8] memory) {
        return WEIGHTS;
    }

    /**
     * @notice Verify weights sum to 1.0 (invariant check)
     */
    function verifyWeightsInvariant() external pure returns (bool valid, uint256 sum) {
        sum = uint256(WEIGHT_CORRECTNESS) +
              uint256(WEIGHT_SAFETY) +
              uint256(WEIGHT_USER_BENEFIT) +
              uint256(WEIGHT_EFFICIENCY) +
              uint256(WEIGHT_AUDITABILITY) +
              uint256(WEIGHT_ANTI_CENTRALIZATION) +
              uint256(WEIGHT_ROBUSTNESS) +
              uint256(WEIGHT_ADL_FAIRNESS);
        valid = sum == 10000; // 1.0 scaled by 1e4
        return (valid, sum);
    }

    /**
     * @notice Get validation record
     */
    function getValidationRecord(bytes32 requestId) external view returns (ScoreValidation memory) {
        return validationRecords[requestId];
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Set current environment
     */
    function setEnvironment(uint8 environment) external {
        require(msg.sender == admin, "Not admin");
        if (environment > 3) revert InvalidEnvironment();

        uint8 oldEnv = currentEnvironment;
        currentEnvironment = environment;

        emit EnvironmentChanged(oldEnv, environment);
    }

    /**
     * @notice Transfer admin role
     */
    function transferAdmin(address newAdmin) external {
        require(msg.sender == admin, "Not admin");
        admin = newAdmin;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function _getThresholdForEnv(uint8 environment) internal pure returns (uint96) {
        if (environment == 0) return THRESHOLD_DEV;
        if (environment == 1) return THRESHOLD_CI;
        if (environment == 2) return THRESHOLD_STAGING;
        return THRESHOLD_PRODUCTION; // Default to production
    }

    function _calculateComposite(uint96[8] calldata scores) internal view returns (uint96) {
        uint256 weightedSum = 0;
        for (uint8 i = 0; i < 8; i++) {
            weightedSum += uint256(scores[i]) * uint256(WEIGHTS[i]);
        }
        // Divide by 10000 to normalize (weights are in 1e4)
        return uint96(weightedSum / 10000);
    }
}
