// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SEEDToken (BZC)
 * @author BIZRA Foundation
 * @notice BIZRA SEED Token - Stable Utility Token with Proof-of-Impact Integration
 * @dev ERC-20 implementation with:
 *      - Fixed supply (1 billion SEED)
 *      - Deflationary burns (transaction fees, penalties)
 *      - Staking mechanism with QR rewards
 *      - Integration with ReceiptRegistry for fee burns
 *      - Harberger tax integration
 *
 * Based on constitution/token_constitution_v1.yaml
 * Genesis hash: 9dfa0bd5375ee06120e72c04618b407b2cf184f110075a573984a4b185f25974
 */

import "./interfaces/IIhsanOracle.sol";

/**
 * @title IERC20
 * @dev Standard ERC-20 interface
 */
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

/**
 * @title SEEDToken
 * @notice BIZRA SEED utility token (BZC)
 */
contract SEEDToken is IERC20 {
    // ========================================================================
    // STATE VARIABLES
    // ========================================================================

    string public constant name = "BIZRA SEED Token";
    string public constant symbol = "BZC";
    uint8 public constant decimals = 18;

    /// @notice Initial and maximum supply: 1 billion SEED
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000 * 10**18;

    /// @notice Total supply (decreases with burns)
    uint256 private _totalSupply;

    /// @notice Total tokens burned
    uint256 public totalBurned;

    /// @notice Total tokens staked
    uint256 public totalStaked;

    /// @notice Reserve pool balance
    uint256 public reservePool;

    /// @notice Annual staking reward rate (basis points: 500 = 5%)
    uint256 public constant STAKING_RATE_BPS = 500;

    /// @notice Minimum validator stake: 10,000 SEED
    uint256 public constant MIN_VALIDATOR_STAKE = 10_000 * 10**18;

    /// @notice Epochs per year (1 epoch = 1 day)
    uint256 public constant EPOCHS_PER_YEAR = 365;

    /// @notice Reserve ratio (basis points: 2000 = 20%)
    uint256 public constant RESERVE_RATIO_BPS = 2000;

    /// @notice Genesis timestamp
    uint256 public immutable genesisTime;

    /// @notice Account balances
    mapping(address => uint256) private _balances;

    /// @notice Allowances
    mapping(address => mapping(address => uint256)) private _allowances;

    /// @notice Staking info
    struct StakeInfo {
        uint256 amount;
        uint256 startEpoch;
        uint256 lastClaimEpoch;
    }
    mapping(address => StakeInfo) public stakes;

    /// @notice Ihsan Oracle for ethics gate
    IIhsanOracle public ihsanOracle;

    /// @notice Receipt Registry for fee burns
    address public receiptRegistry;

    /// @notice Admin address (for initial setup, renounced after)
    address public admin;

    /// @notice Minimum Ihsan score for transfers (basis points: 9000 = 0.90)
    uint256 public constant MIN_IHSAN_BPS = 9000;

    // ========================================================================
    // EVENTS
    // ========================================================================

    event TokensBurned(address indexed burner, uint256 amount, string reason);
    event TokensStaked(address indexed staker, uint256 amount, uint256 epoch);
    event TokensUnstaked(address indexed staker, uint256 amount, uint256 rewardsClaimed);
    event RewardsClaimed(address indexed staker, uint256 amount, uint256 epochs);
    event OracleUpdated(address indexed oldOracle, address indexed newOracle);
    event RegistryUpdated(address indexed oldRegistry, address indexed newRegistry);
    event AdminRenounced(address indexed oldAdmin);

    // ========================================================================
    // ERRORS
    // ========================================================================

    error InsufficientBalance();
    error InsufficientAllowance();
    error InsufficientStake();
    error BelowMinimumStake();
    error IhsanBelowThreshold(uint256 score, uint256 threshold);
    error OnlyAdmin();
    error OnlyReceiptRegistry();
    error ZeroAddress();
    error NoRewardsToClaim();

    // ========================================================================
    // MODIFIERS
    // ========================================================================

    modifier onlyAdmin() {
        if (msg.sender != admin) revert OnlyAdmin();
        _;
    }

    modifier onlyReceiptRegistry() {
        if (msg.sender != receiptRegistry) revert OnlyReceiptRegistry();
        _;
    }

    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================

    /**
     * @notice Deploy SEED token with genesis distribution
     * @param _ihsanOracle Address of the Ihsan Oracle contract
     * @param _receiptRegistry Address of the Receipt Registry contract
     * @param _treasury Treasury address for initial distribution
     */
    constructor(
        address _ihsanOracle,
        address _receiptRegistry,
        address _treasury
    ) {
        if (_treasury == address(0)) revert ZeroAddress();

        genesisTime = block.timestamp;
        admin = msg.sender;

        // Set integrations (can be address(0) initially)
        ihsanOracle = IIhsanOracle(_ihsanOracle);
        receiptRegistry = _receiptRegistry;

        // Mint initial supply to treasury
        _totalSupply = INITIAL_SUPPLY;
        _balances[_treasury] = INITIAL_SUPPLY;

        // Calculate reserve
        reservePool = (INITIAL_SUPPLY * RESERVE_RATIO_BPS) / 10000;

        emit Transfer(address(0), _treasury, INITIAL_SUPPLY);
    }

    // ========================================================================
    // ERC-20 STANDARD FUNCTIONS
    // ========================================================================

    /// @inheritdoc IERC20
    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }

    /// @inheritdoc IERC20
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    /// @inheritdoc IERC20
    function transfer(address to, uint256 amount) external override returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    /// @inheritdoc IERC20
    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    /// @inheritdoc IERC20
    function approve(address spender, uint256 amount) external override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    /// @inheritdoc IERC20
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external override returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        if (currentAllowance < amount) revert InsufficientAllowance();

        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }

        _transfer(from, to, amount);
        return true;
    }

    // ========================================================================
    // STAKING FUNCTIONS
    // ========================================================================

    /**
     * @notice Stake SEED tokens
     * @param amount Amount to stake (must be >= MIN_VALIDATOR_STAKE for validators)
     */
    function stake(uint256 amount) external {
        if (_balances[msg.sender] < amount) revert InsufficientBalance();

        uint256 currentEpoch = _currentEpoch();

        // Claim any existing rewards first
        if (stakes[msg.sender].amount > 0) {
            _claimRewards(msg.sender);
        }

        // Update stake
        stakes[msg.sender].amount += amount;
        stakes[msg.sender].startEpoch = currentEpoch;
        stakes[msg.sender].lastClaimEpoch = currentEpoch;

        // Transfer to staking (internal accounting)
        _balances[msg.sender] -= amount;
        totalStaked += amount;

        emit TokensStaked(msg.sender, amount, currentEpoch);
    }

    /**
     * @notice Unstake SEED tokens
     * @param amount Amount to unstake
     */
    function unstake(uint256 amount) external {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        if (stakeInfo.amount < amount) revert InsufficientStake();

        // Claim rewards first
        uint256 rewards = _claimRewards(msg.sender);

        // Update stake
        stakeInfo.amount -= amount;
        totalStaked -= amount;

        // Return tokens
        _balances[msg.sender] += amount;

        emit TokensUnstaked(msg.sender, amount, rewards);
    }

    /**
     * @notice Claim staking rewards
     * @return rewards Amount of rewards claimed
     */
    function claimRewards() external returns (uint256 rewards) {
        rewards = _claimRewards(msg.sender);
        if (rewards == 0) revert NoRewardsToClaim();
    }

    /**
     * @notice Calculate pending rewards for a staker
     * @param staker Address of the staker
     * @return Pending rewards amount
     */
    function pendingRewards(address staker) external view returns (uint256) {
        return _calculateRewards(staker);
    }

    /**
     * @notice Check if address has validator-level stake
     * @param staker Address to check
     * @return True if stake >= MIN_VALIDATOR_STAKE
     */
    function isValidator(address staker) external view returns (bool) {
        return stakes[staker].amount >= MIN_VALIDATOR_STAKE;
    }

    // ========================================================================
    // BURN FUNCTIONS
    // ========================================================================

    /**
     * @notice Burn tokens (deflationary mechanism)
     * @param amount Amount to burn
     * @param reason Reason for burn (for logging)
     */
    function burn(uint256 amount, string calldata reason) external {
        _burn(msg.sender, amount, reason);
    }

    /**
     * @notice Burn tokens from allowance (for fee collection)
     * @param from Address to burn from
     * @param amount Amount to burn
     * @param reason Reason for burn
     */
    function burnFrom(address from, uint256 amount, string calldata reason) external {
        uint256 currentAllowance = _allowances[from][msg.sender];
        if (currentAllowance < amount) revert InsufficientAllowance();

        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }

        _burn(from, amount, reason);
    }

    /**
     * @notice Burn transaction fees (called by ReceiptRegistry)
     * @param from Address paying the fee
     * @param amount Fee amount to burn
     */
    function burnTransactionFee(address from, uint256 amount) external onlyReceiptRegistry {
        _burn(from, amount, "transaction_fee");
    }

    // ========================================================================
    // ADMIN FUNCTIONS
    // ========================================================================

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
     * @notice Update Receipt Registry address
     * @param newRegistry New registry address
     */
    function setReceiptRegistry(address newRegistry) external onlyAdmin {
        address oldRegistry = receiptRegistry;
        receiptRegistry = newRegistry;
        emit RegistryUpdated(oldRegistry, newRegistry);
    }

    /**
     * @notice Renounce admin role (irreversible)
     */
    function renounceAdmin() external onlyAdmin {
        emit AdminRenounced(admin);
        admin = address(0);
    }

    // ========================================================================
    // VIEW FUNCTIONS
    // ========================================================================

    /**
     * @notice Get current epoch number
     * @return Current epoch (days since genesis)
     */
    function currentEpoch() external view returns (uint256) {
        return _currentEpoch();
    }

    /**
     * @notice Get circulating supply (total - staked - reserve)
     * @return Circulating supply
     */
    function circulatingSupply() external view returns (uint256) {
        return _totalSupply - totalStaked - reservePool;
    }

    /**
     * @notice Get stake info for an address
     * @param staker Address to query
     * @return amount Staked amount
     * @return startEpoch Epoch when stake started
     * @return lastClaimEpoch Last epoch rewards were claimed
     */
    function getStakeInfo(address staker)
        external
        view
        returns (
            uint256 amount,
            uint256 startEpoch,
            uint256 lastClaimEpoch
        )
    {
        StakeInfo storage info = stakes[staker];
        return (info.amount, info.startEpoch, info.lastClaimEpoch);
    }

    // ========================================================================
    // INTERNAL FUNCTIONS
    // ========================================================================

    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal {
        if (from == address(0) || to == address(0)) revert ZeroAddress();
        if (_balances[from] < amount) revert InsufficientBalance();

        // Optional: Check Ihsan score for large transfers
        // (disabled if oracle not set)
        if (address(ihsanOracle) != address(0) && amount > 10000 * 10**18) {
            uint256 ihsanScore = ihsanOracle.getIhsanScore(from);
            if (ihsanScore < MIN_IHSAN_BPS) {
                revert IhsanBelowThreshold(ihsanScore, MIN_IHSAN_BPS);
            }
        }

        unchecked {
            _balances[from] -= amount;
            _balances[to] += amount;
        }

        emit Transfer(from, to, amount);
    }

    function _approve(
        address owner,
        address spender,
        uint256 amount
    ) internal {
        if (owner == address(0) || spender == address(0)) revert ZeroAddress();

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _burn(address from, uint256 amount, string memory reason) internal {
        if (_balances[from] < amount) revert InsufficientBalance();

        unchecked {
            _balances[from] -= amount;
            _totalSupply -= amount;
        }

        totalBurned += amount;

        emit Transfer(from, address(0), amount);
        emit TokensBurned(from, amount, reason);
    }

    function _currentEpoch() internal view returns (uint256) {
        return (block.timestamp - genesisTime) / 1 days;
    }

    function _calculateRewards(address staker) internal view returns (uint256) {
        StakeInfo storage stakeInfo = stakes[staker];
        if (stakeInfo.amount == 0) return 0;

        uint256 currentEpoch_ = _currentEpoch();
        uint256 epochsStaked = currentEpoch_ - stakeInfo.lastClaimEpoch;
        if (epochsStaked == 0) return 0;

        // Reward = Staked * (Annual Rate / Epochs Per Year) * Epochs
        // Using basis points: 500 bps = 5%
        uint256 annualReward = (stakeInfo.amount * STAKING_RATE_BPS) / 10000;
        uint256 epochReward = (annualReward * epochsStaked) / EPOCHS_PER_YEAR;

        return epochReward;
    }

    function _claimRewards(address staker) internal returns (uint256) {
        uint256 rewards = _calculateRewards(staker);
        if (rewards == 0) return 0;

        // Check if reserve has enough for rewards
        if (rewards > reservePool) {
            rewards = reservePool;
        }

        if (rewards > 0) {
            reservePool -= rewards;
            _balances[staker] += rewards;
            stakes[staker].lastClaimEpoch = _currentEpoch();

            emit RewardsClaimed(staker, rewards, _currentEpoch() - stakes[staker].lastClaimEpoch);
        }

        return rewards;
    }
}
