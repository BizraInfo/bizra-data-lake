// SPDX-License-Identifier: MIT
// BIZRA Native Chain - Agent Registry Contract
// On-chain identity and bonding for PAT/SAT agents

pragma solidity ^0.8.19;

/**
 * @title AgentRegistry
 * @notice On-chain registry for BIZRA agent identity and bonding
 * @dev Manages PAT (7 agents) and SAT (5 guardians) registration
 *
 * PAT Agents:
 * - MasterReasoner: Strategic thinking
 * - MemoryArchitect: Knowledge organization
 * - CreativeSynthesizer: Writing/ideation
 * - DataAnalyzer: Pattern recognition
 * - Communicator: External communications
 * - ExecutionPlanner: Task planning
 * - EthicsGuardian: Safety/bias detection
 *
 * SAT Guardians:
 * - PoiVerifier: Proof-of-Impact validation
 * - ResourceAllocator: Compute optimization
 * - RiskGuardian: Security monitoring
 * - GovernanceEngine: Policy enforcement
 * - EvidenceEngine: Audit trail generation
 */
contract AgentRegistry {
    // ═══════════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════════

    enum AgentTeam {
        PAT,  // Personal Agentic Team (7 agents)
        SAT   // System Agentic Team (5 guardians)
    }

    enum AgentStatus {
        Inactive,    // Not registered
        Active,      // Currently operational
        Suspended,   // Temporarily suspended
        Revoked      // Permanently revoked
    }

    struct Agent {
        bytes32 agentId;          // Unique agent identifier
        AgentTeam team;           // PAT or SAT
        string name;              // Human-readable name
        string specialty;         // Agent specialty/role
        address nodeAddress;      // Node running this agent
        uint256 bondAmount;       // ADL tokens bonded
        uint256 registeredAt;     // Registration timestamp
        uint256 lastHeartbeat;    // Last activity timestamp
        AgentStatus status;       // Current status
        uint96 reputationScore;   // Reputation (0-10000)
        uint64 tasksCompleted;    // Total tasks completed
        uint64 consensusVotes;    // SAT votes participated in
        bool isValidator;         // Can participate in SAT consensus
    }

    struct AgentCapabilities {
        bool canSpawnSubAgents;
        uint8 maxSubAgents;
        bool canAccessMcp;
        bool canDelegateA2a;
        bool canEscalate;
        string[] mcpToolsAllowed;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice All registered agents
    mapping(bytes32 => Agent) public agents;

    /// @notice Agent capabilities
    mapping(bytes32 => AgentCapabilities) public capabilities;

    /// @notice Agents by team
    mapping(AgentTeam => bytes32[]) public agentsByTeam;

    /// @notice Node to agents mapping
    mapping(address => bytes32[]) public nodeAgents;

    /// @notice Active validators (SAT agents that can vote)
    bytes32[] public validators;

    /// @notice Minimum bond required for registration
    uint256 public minBond;

    /// @notice Heartbeat timeout (seconds)
    uint256 public heartbeatTimeout;

    /// @notice Total agents registered
    uint256 public totalAgents;

    /// @notice Admin address
    address public admin;

    /// @notice ADL token address for bonding
    address public adlToken;

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event AgentRegistered(
        bytes32 indexed agentId,
        AgentTeam indexed team,
        string name,
        address indexed nodeAddress,
        uint256 bondAmount
    );

    event AgentStatusChanged(
        bytes32 indexed agentId,
        AgentStatus oldStatus,
        AgentStatus newStatus
    );

    event AgentHeartbeat(bytes32 indexed agentId, uint256 timestamp);

    event BondSlashed(
        bytes32 indexed agentId,
        uint256 slashAmount,
        string reason
    );

    event ValidatorAdded(bytes32 indexed agentId);
    event ValidatorRemoved(bytes32 indexed agentId);

    event ReputationUpdated(
        bytes32 indexed agentId,
        uint96 oldScore,
        uint96 newScore
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error AgentAlreadyExists();
    error AgentNotFound();
    error InsufficientBond();
    error InvalidTeam();
    error NotAdmin();
    error NotAgentNode();
    error AgentNotActive();
    error HeartbeatExpired();
    error InvalidStatus();

    // ═══════════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════════

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }

    modifier onlyAgentNode(bytes32 agentId) {
        if (agents[agentId].nodeAddress != msg.sender) revert NotAgentNode();
        _;
    }

    modifier agentExists(bytes32 agentId) {
        if (agents[agentId].registeredAt == 0) revert AgentNotFound();
        _;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor(address _adlToken, uint256 _minBond) {
        admin = msg.sender;
        adlToken = _adlToken;
        minBond = _minBond;
        heartbeatTimeout = 300; // 5 minutes default
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // REGISTRATION FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Register a new agent
     * @param agentId Unique agent identifier
     * @param team PAT or SAT
     * @param name Human-readable name
     * @param specialty Agent specialty
     * @param bondAmount ADL tokens to bond
     */
    function registerAgent(
        bytes32 agentId,
        AgentTeam team,
        string calldata name,
        string calldata specialty,
        uint256 bondAmount
    ) external {
        if (agents[agentId].registeredAt != 0) revert AgentAlreadyExists();
        if (bondAmount < minBond) revert InsufficientBond();

        // Validate team constraints
        if (team == AgentTeam.PAT && agentsByTeam[AgentTeam.PAT].length >= 7) {
            revert InvalidTeam(); // Max 7 PAT agents
        }
        if (team == AgentTeam.SAT && agentsByTeam[AgentTeam.SAT].length >= 5) {
            revert InvalidTeam(); // Max 5 SAT agents
        }

        // Transfer bond (requires approval)
        // In production: IERC20(adlToken).transferFrom(msg.sender, address(this), bondAmount);

        Agent memory agent = Agent({
            agentId: agentId,
            team: team,
            name: name,
            specialty: specialty,
            nodeAddress: msg.sender,
            bondAmount: bondAmount,
            registeredAt: block.timestamp,
            lastHeartbeat: block.timestamp,
            status: AgentStatus.Active,
            reputationScore: 5000, // Start at 50%
            tasksCompleted: 0,
            consensusVotes: 0,
            isValidator: team == AgentTeam.SAT // SAT agents are validators
        });

        agents[agentId] = agent;
        agentsByTeam[team].push(agentId);
        nodeAgents[msg.sender].push(agentId);
        totalAgents++;

        if (team == AgentTeam.SAT) {
            validators.push(agentId);
            emit ValidatorAdded(agentId);
        }

        emit AgentRegistered(agentId, team, name, msg.sender, bondAmount);
    }

    /**
     * @notice Set agent capabilities
     */
    function setCapabilities(
        bytes32 agentId,
        bool canSpawnSubAgents,
        uint8 maxSubAgents,
        bool canAccessMcp,
        bool canDelegateA2a,
        bool canEscalate,
        string[] calldata mcpToolsAllowed
    ) external onlyAgentNode(agentId) agentExists(agentId) {
        capabilities[agentId] = AgentCapabilities({
            canSpawnSubAgents: canSpawnSubAgents,
            maxSubAgents: maxSubAgents,
            canAccessMcp: canAccessMcp,
            canDelegateA2a: canDelegateA2a,
            canEscalate: canEscalate,
            mcpToolsAllowed: mcpToolsAllowed
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LIFECYCLE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Send heartbeat to prove agent is alive
     */
    function heartbeat(bytes32 agentId) external onlyAgentNode(agentId) agentExists(agentId) {
        if (agents[agentId].status != AgentStatus.Active) revert AgentNotActive();

        agents[agentId].lastHeartbeat = block.timestamp;
        emit AgentHeartbeat(agentId, block.timestamp);
    }

    /**
     * @notice Update agent status
     */
    function setStatus(
        bytes32 agentId,
        AgentStatus newStatus
    ) external onlyAdmin agentExists(agentId) {
        AgentStatus oldStatus = agents[agentId].status;
        agents[agentId].status = newStatus;
        emit AgentStatusChanged(agentId, oldStatus, newStatus);
    }

    /**
     * @notice Record task completion
     */
    function recordTaskCompletion(bytes32 agentId) external onlyAgentNode(agentId) agentExists(agentId) {
        agents[agentId].tasksCompleted++;
        agents[agentId].lastHeartbeat = block.timestamp;
    }

    /**
     * @notice Record consensus vote participation
     */
    function recordConsensusVote(bytes32 agentId) external agentExists(agentId) {
        if (agents[agentId].team != AgentTeam.SAT) revert InvalidTeam();
        agents[agentId].consensusVotes++;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // REPUTATION & SLASHING
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Update agent reputation
     * @param agentId Agent identifier
     * @param delta Change in reputation (positive or negative)
     */
    function updateReputation(
        bytes32 agentId,
        int96 delta
    ) external onlyAdmin agentExists(agentId) {
        uint96 oldScore = agents[agentId].reputationScore;
        int96 newScoreInt = int96(oldScore) + delta;

        // Clamp to valid range
        if (newScoreInt < 0) newScoreInt = 0;
        if (newScoreInt > 10000) newScoreInt = 10000;

        uint96 newScore = uint96(uint256(int256(newScoreInt)));
        agents[agentId].reputationScore = newScore;

        emit ReputationUpdated(agentId, oldScore, newScore);
    }

    /**
     * @notice Slash agent bond for misbehavior
     * @param agentId Agent identifier
     * @param slashPercent Percentage to slash (0-100)
     * @param reason Reason for slashing
     */
    function slashBond(
        bytes32 agentId,
        uint8 slashPercent,
        string calldata reason
    ) external onlyAdmin agentExists(agentId) {
        require(slashPercent <= 100, "Invalid slash percent");

        uint256 slashAmount = (agents[agentId].bondAmount * slashPercent) / 100;
        agents[agentId].bondAmount -= slashAmount;

        // Transfer slashed amount to treasury
        // In production: IERC20(adlToken).transfer(treasury, slashAmount);

        emit BondSlashed(agentId, slashAmount, reason);

        // Auto-suspend if bond falls below minimum
        if (agents[agentId].bondAmount < minBond) {
            AgentStatus oldStatus = agents[agentId].status;
            agents[agentId].status = AgentStatus.Suspended;
            emit AgentStatusChanged(agentId, oldStatus, AgentStatus.Suspended);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Get agent details
     */
    function getAgent(bytes32 agentId) external view returns (Agent memory) {
        return agents[agentId];
    }

    /**
     * @notice Check if agent is active and healthy
     */
    function isAgentHealthy(bytes32 agentId) external view returns (bool) {
        Agent storage agent = agents[agentId];
        if (agent.status != AgentStatus.Active) return false;
        if (block.timestamp - agent.lastHeartbeat > heartbeatTimeout) return false;
        return true;
    }

    /**
     * @notice Get all PAT agents
     */
    function getPATAgents() external view returns (bytes32[] memory) {
        return agentsByTeam[AgentTeam.PAT];
    }

    /**
     * @notice Get all SAT agents
     */
    function getSATAgents() external view returns (bytes32[] memory) {
        return agentsByTeam[AgentTeam.SAT];
    }

    /**
     * @notice Get active validators for consensus
     */
    function getActiveValidators() external view returns (bytes32[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < validators.length; i++) {
            if (agents[validators[i]].status == AgentStatus.Active) {
                count++;
            }
        }

        bytes32[] memory active = new bytes32[](count);
        uint256 j = 0;
        for (uint256 i = 0; i < validators.length; i++) {
            if (agents[validators[i]].status == AgentStatus.Active) {
                active[j] = validators[i];
                j++;
            }
        }

        return active;
    }

    /**
     * @notice Get validator count
     */
    function getValidatorCount() external view returns (uint256) {
        return validators.length;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Set minimum bond amount
     */
    function setMinBond(uint256 _minBond) external onlyAdmin {
        minBond = _minBond;
    }

    /**
     * @notice Set heartbeat timeout
     */
    function setHeartbeatTimeout(uint256 _timeout) external onlyAdmin {
        heartbeatTimeout = _timeout;
    }

    /**
     * @notice Transfer admin role
     */
    function transferAdmin(address newAdmin) external onlyAdmin {
        admin = newAdmin;
    }
}
