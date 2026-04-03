// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title BLOOMToken (BZT)
 * @author BIZRA Foundation
 * @notice BIZRA BLOOM Token - Soulbound Impact Growth Token
 * @dev ERC-5192 Soulbound implementation with:
 *      - Non-transferable (soulbound to address)
 *      - Minted through Proof-of-Impact attestations
 *      - Quadratic voting power: votes = sqrt(tokens) * weight
 *      - Decay mechanism for inactive holders
 *      - No pre-mine (fair launch)
 *
 * Based on ERC-5192: Minimal Soulbound NFTs
 * Reference: https://eips.ethereum.org/EIPS/eip-5192
 *
 * Genesis hash: 9dfa0bd5375ee06120e72c04618b407b2cf184f110075a573984a4b185f25974
 */

import "./interfaces/IIhsanOracle.sol";

/**
 * @title IERC5192
 * @dev Minimal Soulbound interface (locked tokens)
 */
interface IERC5192 {
    /// @notice Emitted when the locking status is changed to locked
    event Locked(uint256 tokenId);

    /// @notice Emitted when the locking status is changed to unlocked
    event Unlocked(uint256 tokenId);

    /// @notice Returns the locking status of a token
    function locked(uint256 tokenId) external view returns (bool);
}

/**
 * @title BLOOMToken
 * @notice BIZRA BLOOM governance token (BZT) - Soulbound
 */
contract BLOOMToken is IERC5192 {
    // ========================================================================
    // STATE VARIABLES
    // ========================================================================

    string public constant name = "BIZRA BLOOM Token";
    string public constant symbol = "BZT";
    uint8 public constant decimals = 18;

    /// @notice Total supply (grows with impact)
    uint256 private _totalSupply;

    /// @notice Total impact score accumulated globally
    uint256 public totalImpact;

    /// @notice BLOOM minted per impact point (0.001 BLOOM = 10^15 wei)
    uint256 public constant BLOOM_RATE = 1e15;

    /// @notice Minimum impact threshold for minting
    uint256 public constant MIN_IMPACT_THRESHOLD = 10;

    /// @notice Governance voting weight multiplier
    uint256 public constant GOVERNANCE_WEIGHT = 2;

    /// @notice Decay rate per epoch if inactive (basis points: 100 = 1%)
    uint256 public constant DECAY_RATE_BPS = 100;

    /// @notice Epochs before decay starts
    uint256 public constant DECAY_GRACE_EPOCHS = 30;

    /// @notice Minimum BLOOM to create governance proposal
    uint256 public constant PROPOSAL_THRESHOLD = 100 * 10**18;

    /// @notice Genesis timestamp
    uint256 public immutable genesisTime;

    /// @notice Account balances
    mapping(address => uint256) private _balances;

    /// @notice Last activity epoch for each account
    mapping(address => uint256) public lastActivityEpoch;

    /// @notice Impact score per account
    mapping(address => uint256) public impactScores;

    /// @notice Impact category for minting
    enum ImpactCategory {
        Education,      // 1.2x multiplier
        Healthcare,     // 1.5x multiplier
        Environment,    // 1.3x multiplier
        Economic,       // 1.1x multiplier
        Governance,     // 1.0x multiplier
        Technical,      // 1.4x multiplier
        Community       // 1.1x multiplier
    }

    /// @notice Category multipliers (basis points: 10000 = 1.0x)
    mapping(ImpactCategory => uint256) public categoryMultipliers;

    /// @notice Authorized minters (Proof-of-Impact validators)
    mapping(address => bool) public authorizedMinters;

    /// @notice Ihsan Oracle for ethics gate
    IIhsanOracle public ihsanOracle;

    /// @notice Admin address
    address public admin;

    /// @notice Minimum Ihsan score for minting (basis points: 9000 = 0.90)
    uint256 public constant MIN_IHSAN_FOR_MINT_BPS = 9000;

    // ========================================================================
    // EVENTS
    // ========================================================================

    event Transfer(address indexed from, address indexed to, uint256 value);
    event BloomMinted(
        address indexed recipient,
        uint256 amount,
        uint256 impactScore,
        ImpactCategory category,
        bytes32 evidenceHash
    );
    event ImpactRecorded(address indexed account, uint256 score, ImpactCategory category);
    event DecayApplied(address indexed account, uint256 decayAmount, uint256 epochsInactive);
    event MinterAuthorized(address indexed minter, bool authorized);
    event OracleUpdated(address indexed oldOracle, address indexed newOracle);
    event AdminRenounced(address indexed oldAdmin);

    // ========================================================================
    // ERRORS
    // ========================================================================

    error SoulboundTokenNonTransferable();
    error BelowImpactThreshold(uint256 score, uint256 threshold);
    error IhsanBelowThreshold(uint256 score, uint256 threshold);
    error OnlyAdmin();
    error OnlyAuthorizedMinter();
    error ZeroAddress();
    error InvalidCategory();

    // ========================================================================
    // MODIFIERS
    // ========================================================================

    modifier onlyAdmin() {
        if (msg.sender != admin) revert OnlyAdmin();
        _;
    }

    modifier onlyMinter() {
        if (!authorizedMinters[msg.sender]) revert OnlyAuthorizedMinter();
        _;
    }

    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================

    /**
     * @notice Deploy BLOOM token
     * @param _ihsanOracle Address of the Ihsan Oracle contract
     */
    constructor(address _ihsanOracle) {
        genesisTime = block.timestamp;
        admin = msg.sender;
        ihsanOracle = IIhsanOracle(_ihsanOracle);

        // Admin is initial authorized minter
        authorizedMinters[msg.sender] = true;

        // Initialize category multipliers
        categoryMultipliers[ImpactCategory.Education] = 12000;   // 1.2x
        categoryMultipliers[ImpactCategory.Healthcare] = 15000;  // 1.5x
        categoryMultipliers[ImpactCategory.Environment] = 13000; // 1.3x
        categoryMultipliers[ImpactCategory.Economic] = 11000;    // 1.1x
        categoryMultipliers[ImpactCategory.Governance] = 10000;  // 1.0x
        categoryMultipliers[ImpactCategory.Technical] = 14000;   // 1.4x
        categoryMultipliers[ImpactCategory.Community] = 11000;   // 1.1x
    }

    // ========================================================================
    // ERC-5192 SOULBOUND INTERFACE
    // ========================================================================

    /**
     * @notice All BLOOM tokens are permanently locked (soulbound)
     * @param tokenId Token ID (ignored - all tokens are locked)
     * @return Always returns true (all tokens locked)
     */
    function locked(uint256 tokenId) external pure override returns (bool) {
        return true; // All BLOOM tokens are soulbound
    }

    // ========================================================================
    // VIEW FUNCTIONS
    // ========================================================================

    /**
     * @notice Get total supply
     */
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }

    /**
     * @notice Get balance of account (after decay calculation)
     * @param account Address to query
     * @return Effective balance after decay
     */
    function balanceOf(address account) external view returns (uint256) {
        return _effectiveBalance(account);
    }

    /**
     * @notice Get raw balance without decay
     * @param account Address to query
     * @return Raw balance
     */
    function rawBalanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }

    /**
     * @notice Calculate governance voting power (quadratic)
     * @param account Address to calculate power for
     * @return Voting power
     */
    function votingPower(address account) external view returns (uint256) {
        uint256 balance = _effectiveBalance(account);
        if (balance == 0) return 0;

        // Quadratic voting: sqrt(tokens) * weight
        uint256 sqrtBalance = _sqrt(balance);
        return (sqrtBalance * GOVERNANCE_WEIGHT) / 1e9; // Adjust for decimals
    }

    /**
     * @notice Check if account can create governance proposals
     * @param account Address to check
     * @return True if balance >= PROPOSAL_THRESHOLD
     */
    function canCreateProposal(address account) external view returns (bool) {
        return _effectiveBalance(account) >= PROPOSAL_THRESHOLD;
    }

    /**
     * @notice Get current epoch
     * @return Epoch number (days since genesis)
     */
    function currentEpoch() external view returns (uint256) {
        return _currentEpoch();
    }

    /**
     * @notice Calculate pending decay for an account
     * @param account Address to check
     * @return Decay amount
     */
    function pendingDecay(address account) external view returns (uint256) {
        return _calculateDecay(account);
    }

    // ========================================================================
    // MINTING FUNCTIONS (Proof-of-Impact)
    // ========================================================================

    /**
     * @notice Mint BLOOM tokens from verified impact
     * @param recipient Address to receive BLOOM
     * @param impactScore Base impact score
     * @param category Impact category for multiplier
     * @param evidenceHash Hash of impact evidence (IPFS CID, etc.)
     * @return mintedAmount Amount of BLOOM minted
     */
    function mintFromImpact(
        address recipient,
        uint256 impactScore,
        ImpactCategory category,
        bytes32 evidenceHash
    ) external onlyMinter returns (uint256 mintedAmount) {
        if (recipient == address(0)) revert ZeroAddress();
        if (impactScore < MIN_IMPACT_THRESHOLD) {
            revert BelowImpactThreshold(impactScore, MIN_IMPACT_THRESHOLD);
        }

        // Check Ihsan score if oracle is set
        if (address(ihsanOracle) != address(0)) {
            uint256 ihsanScore = ihsanOracle.getIhsanScore(recipient);
            if (ihsanScore < MIN_IHSAN_FOR_MINT_BPS) {
                revert IhsanBelowThreshold(ihsanScore, MIN_IHSAN_FOR_MINT_BPS);
            }
        }

        // Apply decay before minting
        _applyDecay(recipient);

        // Calculate mint amount with category multiplier
        uint256 multiplier = categoryMultipliers[category];
        uint256 adjustedScore = (impactScore * multiplier) / 10000;
        mintedAmount = adjustedScore * BLOOM_RATE;

        // Update state
        _balances[recipient] += mintedAmount;
        _totalSupply += mintedAmount;
        totalImpact += adjustedScore;
        impactScores[recipient] += adjustedScore;
        lastActivityEpoch[recipient] = _currentEpoch();

        emit Transfer(address(0), recipient, mintedAmount);
        emit BloomMinted(recipient, mintedAmount, impactScore, category, evidenceHash);
        emit ImpactRecorded(recipient, adjustedScore, category);
    }

    /**
     * @notice Batch mint for multiple recipients
     * @param recipients Array of recipient addresses
     * @param impactScores Array of impact scores
     * @param categories Array of impact categories
     * @param evidenceHashes Array of evidence hashes
     */
    function batchMintFromImpact(
        address[] calldata recipients,
        uint256[] calldata impactScores,
        ImpactCategory[] calldata categories,
        bytes32[] calldata evidenceHashes
    ) external onlyMinter {
        require(
            recipients.length == impactScores.length &&
            impactScores.length == categories.length &&
            categories.length == evidenceHashes.length,
            "Array length mismatch"
        );

        for (uint256 i = 0; i < recipients.length; i++) {
            this.mintFromImpact(
                recipients[i],
                impactScores[i],
                categories[i],
                evidenceHashes[i]
            );
        }
    }

    // ========================================================================
    // SOULBOUND RESTRICTIONS
    // ========================================================================

    /**
     * @notice Transfer is disabled (soulbound token)
     * @dev Always reverts
     */
    function transfer(address, uint256) external pure returns (bool) {
        revert SoulboundTokenNonTransferable();
    }

    /**
     * @notice TransferFrom is disabled (soulbound token)
     * @dev Always reverts
     */
    function transferFrom(address, address, uint256) external pure returns (bool) {
        revert SoulboundTokenNonTransferable();
    }

    /**
     * @notice Approve is disabled (soulbound token)
     * @dev Always reverts
     */
    function approve(address, uint256) external pure returns (bool) {
        revert SoulboundTokenNonTransferable();
    }

    /**
     * @notice Allowance always returns 0 (soulbound token)
     */
    function allowance(address, address) external pure returns (uint256) {
        return 0;
    }

    // ========================================================================
    // ADMIN FUNCTIONS
    // ========================================================================

    /**
     * @notice Authorize or revoke a minter
     * @param minter Address to authorize
     * @param authorized Whether to authorize or revoke
     */
    function setMinter(address minter, bool authorized) external onlyAdmin {
        authorizedMinters[minter] = authorized;
        emit MinterAuthorized(minter, authorized);
    }

    /**
     * @notice Update Ihsan Oracle address
     * @param newOracle New oracle address
     */
    function setIhsanOracle(address newOracle) external onlyAdmin {
        address oldOracle = address(ihsanOracle);
        ihsanOracle = IIhsanOracle(newOracle);
        emit OracleUpdated(oldOracle, newOracle);
    }

    /**
     * @notice Renounce admin role (irreversible)
     */
    function renounceAdmin() external onlyAdmin {
        emit AdminRenounced(admin);
        admin = address(0);
    }

    // ========================================================================
    // DECAY FUNCTIONS
    // ========================================================================

    /**
     * @notice Apply decay to an account (public, anyone can call)
     * @param account Address to apply decay to
     */
    function applyDecay(address account) external {
        _applyDecay(account);
    }

    // ========================================================================
    // INTERNAL FUNCTIONS
    // ========================================================================

    function _currentEpoch() internal view returns (uint256) {
        return (block.timestamp - genesisTime) / 1 days;
    }

    function _calculateDecay(address account) internal view returns (uint256) {
        uint256 balance = _balances[account];
        if (balance == 0) return 0;

        uint256 lastEpoch = lastActivityEpoch[account];
        uint256 currentEpoch_ = _currentEpoch();

        if (currentEpoch_ <= lastEpoch + DECAY_GRACE_EPOCHS) return 0;

        uint256 epochsInactive = currentEpoch_ - lastEpoch - DECAY_GRACE_EPOCHS;

        // Decay = balance * rate * epochs / 10000
        uint256 decay = (balance * DECAY_RATE_BPS * epochsInactive) / 10000;

        // Cap at total balance
        return decay > balance ? balance : decay;
    }

    function _effectiveBalance(address account) internal view returns (uint256) {
        uint256 balance = _balances[account];
        uint256 decay = _calculateDecay(account);
        return balance > decay ? balance - decay : 0;
    }

    function _applyDecay(address account) internal {
        uint256 decay = _calculateDecay(account);
        if (decay > 0) {
            _balances[account] -= decay;
            _totalSupply -= decay; // Decay is burned

            uint256 epochsInactive = _currentEpoch() - lastActivityEpoch[account] - DECAY_GRACE_EPOCHS;
            emit DecayApplied(account, decay, epochsInactive);
        }
    }

    /**
     * @notice Integer square root using Babylonian method
     * @param x Input value
     * @return y Square root
     */
    function _sqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;

        uint256 z = (x + 1) / 2;
        y = x;

        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }
}
