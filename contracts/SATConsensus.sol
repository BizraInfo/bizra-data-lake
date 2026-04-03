// SPDX-License-Identifier: MIT
// BIZRA Native Chain - SAT Consensus Contract
// On-chain Byzantine fault-tolerant consensus for SAT validation

pragma solidity ^0.8.19;

import "./interfaces/IIhsanOracle.sol";

/**
 * @title SATConsensus
 * @notice On-chain consensus mechanism for SAT (System Agentic Team) validation
 * @dev Implements 3/5 Byzantine fault-tolerant consensus
 *
 * SAT Guardians (5):
 * - PoiVerifier: Proof-of-Impact validation
 * - ResourceAllocator: Compute optimization
 * - RiskGuardian: Security monitoring
 * - GovernanceEngine: Policy enforcement
 * - EvidenceEngine: Audit trail generation
 *
 * Consensus Requirements:
 * - Standard requests: 3/5 approval (60%)
 * - Explosion mode: 4/5 approval (80%)
 * - Critical actions: 5/5 approval (100%)
 */
contract SATConsensus {
    // ═══════════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════════

    enum VoteType {
        Approve,
        Reject,
        Abstain
    }

    enum ConsensusLevel {
        Standard,      // 3/5 required
        Elevated,      // 4/5 required (explosion mode)
        Critical       // 5/5 required
    }

    enum ProposalStatus {
        Pending,       // Awaiting votes
        Approved,      // Consensus reached - approved
        Rejected,      // Consensus reached - rejected
        Expired,       // Voting period expired
        Executed       // Action completed
    }

    struct Proposal {
        bytes32 proposalId;        // Unique proposal identifier
        bytes32 requestId;         // Original request ID
        address proposer;          // Node that created proposal
        ConsensusLevel level;      // Required consensus level
        ProposalStatus status;     // Current status
        bytes32 taskHash;          // Hash of task being validated
        uint96 ihsanScore;         // Pre-calculated Ihsan score
        uint256 createdAt;         // Creation timestamp
        uint256 expiresAt;         // Expiration timestamp
        uint8 approvalsCount;      // Current approval count
        uint8 rejectionsCount;     // Current rejection count
        uint8 abstentionsCount;    // Current abstention count
        string[] rejectionCodes;   // Codes from rejecting validators
    }

    struct Vote {
        bytes32 validatorId;       // SAT agent that voted
        VoteType voteType;         // Approve/Reject/Abstain
        uint256 timestamp;         // When vote was cast
        string rejectionCode;      // If rejected, the reason code
        bytes32 evidenceHash;      // Supporting evidence hash
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice All proposals by ID
    mapping(bytes32 => Proposal) public proposals;

    /// @notice Votes for each proposal
    mapping(bytes32 => mapping(bytes32 => Vote)) public votes;

    /// @notice Which validators voted on which proposals
    mapping(bytes32 => bytes32[]) public proposalVoters;

    /// @notice Registered SAT validators
    mapping(bytes32 => bool) public isValidator;
    bytes32[] public validators;

    /// @notice Validator addresses
    mapping(bytes32 => address) public validatorAddresses;

    /// @notice Ihsan Oracle for score validation
    IIhsanOracle public ihsanOracle;

    /// @notice Agent Registry for validator lookup
    address public agentRegistry;

    /// @notice Voting period (seconds)
    uint256 public votingPeriod;

    /// @notice Total proposals created
    uint256 public totalProposals;

    /// @notice Admin address
    address public admin;

    // Consensus thresholds (out of 5)
    uint8 public constant THRESHOLD_STANDARD = 3;
    uint8 public constant THRESHOLD_ELEVATED = 4;
    uint8 public constant THRESHOLD_CRITICAL = 5;
    uint8 public constant TOTAL_VALIDATORS = 5;

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event ProposalCreated(
        bytes32 indexed proposalId,
        bytes32 indexed requestId,
        ConsensusLevel level,
        address indexed proposer,
        uint256 expiresAt
    );

    event VoteCast(
        bytes32 indexed proposalId,
        bytes32 indexed validatorId,
        VoteType voteType,
        string rejectionCode
    );

    event ConsensusReached(
        bytes32 indexed proposalId,
        ProposalStatus indexed status,
        uint8 approvals,
        uint8 rejections
    );

    event ProposalExpired(bytes32 indexed proposalId);

    event ValidatorRegistered(bytes32 indexed validatorId, address indexed nodeAddress);
    event ValidatorRemoved(bytes32 indexed validatorId);

    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error NotValidator();
    error ProposalNotFound();
    error ProposalNotPending();
    error AlreadyVoted();
    error VotingExpired();
    error InvalidConsensusLevel();
    error ValidatorAlreadyExists();
    error ValidatorNotFound();
    error TooManyValidators();

    // ═══════════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════════

    modifier onlyValidator(bytes32 validatorId) {
        if (!isValidator[validatorId]) revert NotValidator();
        if (validatorAddresses[validatorId] != msg.sender) revert NotValidator();
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    modifier proposalPending(bytes32 proposalId) {
        if (proposals[proposalId].createdAt == 0) revert ProposalNotFound();
        if (proposals[proposalId].status != ProposalStatus.Pending) revert ProposalNotPending();
        if (block.timestamp > proposals[proposalId].expiresAt) {
            _expireProposal(proposalId);
            revert VotingExpired();
        }
        _;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor(address _ihsanOracle, uint256 _votingPeriod) {
        ihsanOracle = IIhsanOracle(_ihsanOracle);
        votingPeriod = _votingPeriod;
        admin = msg.sender;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATOR MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Register a SAT validator
     * @param validatorId Unique validator identifier
     * @param nodeAddress Address of the node running this validator
     */
    function registerValidator(
        bytes32 validatorId,
        address nodeAddress
    ) external onlyAdmin {
        if (isValidator[validatorId]) revert ValidatorAlreadyExists();
        if (validators.length >= TOTAL_VALIDATORS) revert TooManyValidators();

        isValidator[validatorId] = true;
        validatorAddresses[validatorId] = nodeAddress;
        validators.push(validatorId);

        emit ValidatorRegistered(validatorId, nodeAddress);
    }

    /**
     * @notice Remove a SAT validator
     */
    function removeValidator(bytes32 validatorId) external onlyAdmin {
        if (!isValidator[validatorId]) revert ValidatorNotFound();

        isValidator[validatorId] = false;
        validatorAddresses[validatorId] = address(0);

        // Remove from array
        for (uint256 i = 0; i < validators.length; i++) {
            if (validators[i] == validatorId) {
                validators[i] = validators[validators.length - 1];
                validators.pop();
                break;
            }
        }

        emit ValidatorRemoved(validatorId);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PROPOSAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Create a new validation proposal
     * @param proposalId Unique proposal identifier
     * @param requestId Original request ID
     * @param level Required consensus level
     * @param taskHash Hash of task being validated
     * @param ihsanScore Pre-calculated Ihsan score
     */
    function createProposal(
        bytes32 proposalId,
        bytes32 requestId,
        ConsensusLevel level,
        bytes32 taskHash,
        uint96 ihsanScore
    ) external returns (bytes32) {
        if (proposals[proposalId].createdAt != 0) {
            proposalId = keccak256(abi.encodePacked(proposalId, block.timestamp, msg.sender));
        }

        Proposal memory proposal = Proposal({
            proposalId: proposalId,
            requestId: requestId,
            proposer: msg.sender,
            level: level,
            status: ProposalStatus.Pending,
            taskHash: taskHash,
            ihsanScore: ihsanScore,
            createdAt: block.timestamp,
            expiresAt: block.timestamp + votingPeriod,
            approvalsCount: 0,
            rejectionsCount: 0,
            abstentionsCount: 0,
            rejectionCodes: new string[](0)
        });

        proposals[proposalId] = proposal;
        totalProposals++;

        emit ProposalCreated(proposalId, requestId, level, msg.sender, proposal.expiresAt);

        return proposalId;
    }

    /**
     * @notice Cast a vote on a proposal
     * @param proposalId Proposal to vote on
     * @param validatorId SAT validator casting the vote
     * @param voteType Approve/Reject/Abstain
     * @param rejectionCode Code if rejecting
     * @param evidenceHash Supporting evidence hash
     */
    function castVote(
        bytes32 proposalId,
        bytes32 validatorId,
        VoteType voteType,
        string calldata rejectionCode,
        bytes32 evidenceHash
    ) external onlyValidator(validatorId) proposalPending(proposalId) {
        // Check not already voted
        if (votes[proposalId][validatorId].timestamp != 0) revert AlreadyVoted();

        // Record vote
        votes[proposalId][validatorId] = Vote({
            validatorId: validatorId,
            voteType: voteType,
            timestamp: block.timestamp,
            rejectionCode: rejectionCode,
            evidenceHash: evidenceHash
        });

        proposalVoters[proposalId].push(validatorId);

        // Update counts
        Proposal storage proposal = proposals[proposalId];
        if (voteType == VoteType.Approve) {
            proposal.approvalsCount++;
        } else if (voteType == VoteType.Reject) {
            proposal.rejectionsCount++;
            // Store rejection code
            string[] memory newCodes = new string[](proposal.rejectionCodes.length + 1);
            for (uint256 i = 0; i < proposal.rejectionCodes.length; i++) {
                newCodes[i] = proposal.rejectionCodes[i];
            }
            newCodes[proposal.rejectionCodes.length] = rejectionCode;
            proposal.rejectionCodes = newCodes;
        } else {
            proposal.abstentionsCount++;
        }

        emit VoteCast(proposalId, validatorId, voteType, rejectionCode);

        // Check if consensus reached
        _checkConsensus(proposalId);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSENSUS LOGIC
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Check if consensus has been reached
     */
    function _checkConsensus(bytes32 proposalId) internal {
        Proposal storage proposal = proposals[proposalId];

        uint8 threshold = _getThreshold(proposal.level);
        uint8 totalVotes = proposal.approvalsCount + proposal.rejectionsCount + proposal.abstentionsCount;

        // Check for approval consensus
        if (proposal.approvalsCount >= threshold) {
            proposal.status = ProposalStatus.Approved;
            emit ConsensusReached(
                proposalId,
                ProposalStatus.Approved,
                proposal.approvalsCount,
                proposal.rejectionsCount
            );
            return;
        }

        // Check for rejection consensus (if enough rejections to make approval impossible)
        uint8 remainingVotes = TOTAL_VALIDATORS - totalVotes;
        if (proposal.approvalsCount + remainingVotes < threshold) {
            proposal.status = ProposalStatus.Rejected;
            emit ConsensusReached(
                proposalId,
                ProposalStatus.Rejected,
                proposal.approvalsCount,
                proposal.rejectionsCount
            );
            return;
        }

        // If all votes cast but no consensus
        if (totalVotes == TOTAL_VALIDATORS) {
            // Default to rejected if threshold not met
            if (proposal.approvalsCount < threshold) {
                proposal.status = ProposalStatus.Rejected;
                emit ConsensusReached(
                    proposalId,
                    ProposalStatus.Rejected,
                    proposal.approvalsCount,
                    proposal.rejectionsCount
                );
            }
        }
    }

    function _getThreshold(ConsensusLevel level) internal pure returns (uint8) {
        if (level == ConsensusLevel.Standard) return THRESHOLD_STANDARD;
        if (level == ConsensusLevel.Elevated) return THRESHOLD_ELEVATED;
        return THRESHOLD_CRITICAL;
    }

    function _expireProposal(bytes32 proposalId) internal {
        proposals[proposalId].status = ProposalStatus.Expired;
        emit ProposalExpired(proposalId);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Get proposal details
     */
    function getProposal(bytes32 proposalId) external view returns (Proposal memory) {
        return proposals[proposalId];
    }

    /**
     * @notice Get vote for a proposal by validator
     */
    function getVote(bytes32 proposalId, bytes32 validatorId) external view returns (Vote memory) {
        return votes[proposalId][validatorId];
    }

    /**
     * @notice Check if proposal passed
     */
    function proposalPassed(bytes32 proposalId) external view returns (bool) {
        return proposals[proposalId].status == ProposalStatus.Approved;
    }

    /**
     * @notice Get all validators who voted on a proposal
     */
    function getProposalVoters(bytes32 proposalId) external view returns (bytes32[] memory) {
        return proposalVoters[proposalId];
    }

    /**
     * @notice Get current validator count
     */
    function getValidatorCount() external view returns (uint256) {
        return validators.length;
    }

    /**
     * @notice Get all registered validators
     */
    function getValidators() external view returns (bytes32[] memory) {
        return validators;
    }

    /**
     * @notice Calculate required approvals for consensus level
     */
    function getRequiredApprovals(ConsensusLevel level) external pure returns (uint8) {
        return _getThreshold(level);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Set voting period
     */
    function setVotingPeriod(uint256 _votingPeriod) external onlyAdmin {
        votingPeriod = _votingPeriod;
    }

    /**
     * @notice Set Ihsan Oracle
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

    /**
     * @notice Force expire a proposal (emergency)
     */
    function forceExpire(bytes32 proposalId) external onlyAdmin {
        if (proposals[proposalId].status == ProposalStatus.Pending) {
            _expireProposal(proposalId);
        }
    }
}
