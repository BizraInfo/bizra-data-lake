// SPDX-License-Identifier: MIT
// BIZRA Native Chain - Ihsan Oracle Interface

pragma solidity ^0.8.19;

/**
 * @title IIhsanOracle
 * @notice Interface for Ihsan score validation oracle
 */
interface IIhsanOracle {
    /**
     * @notice Validate that an Ihsan score meets threshold
     * @param score The Ihsan score (scaled by 1e4)
     * @param threshold The threshold to validate against
     * @return valid True if score >= threshold
     */
    function validateScore(uint96 score, uint96 threshold) external view returns (bool valid);

    /**
     * @notice Get current threshold for environment
     * @param environment Environment identifier (0=dev, 1=ci, 2=staging, 3=production)
     * @return threshold Current threshold
     */
    function getThreshold(uint8 environment) external view returns (uint96 threshold);

    /**
     * @notice Get dimension weight
     * @param dimension Dimension index (0-7)
     * @return weight Weight scaled by 1e4
     */
    function getDimensionWeight(uint8 dimension) external view returns (uint96 weight);

    /**
     * @notice Calculate Ihsan score from dimension scores
     * @param dimensionScores Array of 8 dimension scores
     * @return score Weighted composite score
     */
    function calculateScore(uint96[8] calldata dimensionScores) external view returns (uint96 score);
}
