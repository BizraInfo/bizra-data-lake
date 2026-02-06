/**
 * PAT/SAT Agent Minting System — TypeScript Interfaces
 *
 * Personal Agentic Team (PAT) and Shared Agentic Team (SAT) type definitions
 * for the BIZRA ecosystem frontend and API layer.
 *
 * Standing on Giants:
 * - General Magic (1990): Telescript primitives for mobile agents
 * - Shannon (1948): SNR-based signal quality in agent communication
 * - Lamport (1982): Byzantine fault tolerance in agent consensus
 * - Bernstein (2012): Ed25519 for agent identity signatures
 * - Al-Ghazali (1095): Maqasid al-Shariah for FATE gate ethics
 * - Anthropic (2023): Constitutional AI for Ihsan threshold
 */

// =============================================================================
// CONSTANTS
// =============================================================================

/** PAT team size — 7 personal agents (mastermind council) */
export const PAT_TEAM_SIZE = 7;

/** SAT team size — 5 shared agents (public utility) */
export const SAT_TEAM_SIZE = 5;

/** Minimum Ihsan threshold for agent minting */
export const AGENT_MINT_IHSAN_THRESHOLD = 0.95;

/** Maximum delegation depth for agent permits */
export const MAX_AGENT_DELEGATION_DEPTH = 7;

// =============================================================================
// AGENT ROLES
// =============================================================================

/**
 * Personal Agentic Team (PAT) agent roles — 7 mastermind council members
 */
export type PATRole =
  | 'strategist'    // Strategic planning and high-level decision making
  | 'researcher'    // Deep research and knowledge synthesis
  | 'developer'     // Code implementation and technical execution
  | 'analyst'       // Data analysis and pattern recognition
  | 'reviewer'      // Quality assurance and code review
  | 'executor'      // Task execution and operational work
  | 'guardian';     // Security monitoring and ethics enforcement

/**
 * Shared Agentic Team (SAT) agent roles — 5 public utility agents
 */
export type SATRole =
  | 'validator'     // Transaction and state validation in the pool
  | 'oracle'        // External data and truth verification service
  | 'mediator'      // Dispute resolution and arbitration service
  | 'archivist'     // Knowledge preservation and retrieval service
  | 'sentinel';     // Network security and threat detection service

/** All PAT roles in canonical order */
export const PAT_ROLES: readonly PATRole[] = [
  'strategist',
  'researcher',
  'developer',
  'analyst',
  'reviewer',
  'executor',
  'guardian',
] as const;

/** All SAT roles in canonical order */
export const SAT_ROLES: readonly SATRole[] = [
  'validator',
  'oracle',
  'mediator',
  'archivist',
  'sentinel',
] as const;

/** PAT role descriptions */
export const PAT_ROLE_DESCRIPTIONS: Record<PATRole, string> = {
  strategist: 'Strategic planning and high-level decision making',
  researcher: 'Deep research and knowledge synthesis',
  developer: 'Code implementation and technical execution',
  analyst: 'Data analysis and pattern recognition',
  reviewer: 'Quality assurance and code review',
  executor: 'Task execution and operational work',
  guardian: 'Security monitoring and ethics enforcement',
};

/** SAT role descriptions */
export const SAT_ROLE_DESCRIPTIONS: Record<SATRole, string> = {
  validator: 'Transaction and state validation in the pool',
  oracle: 'External data and truth verification service',
  mediator: 'Dispute resolution and arbitration service',
  archivist: 'Knowledge preservation and retrieval service',
  sentinel: 'Network security and threat detection service',
};

/** Minimum stake requirements for SAT roles */
export const SAT_MINIMUM_STAKES: Record<SATRole, number> = {
  validator: 1000,  // High stake for validation
  oracle: 500,      // Medium stake for oracle
  mediator: 750,    // Medium-high for mediation
  archivist: 250,   // Lower stake for storage
  sentinel: 500,    // Medium stake for security
};

// =============================================================================
// AGENT CAPABILITIES
// =============================================================================

/**
 * Capabilities that can be granted to agents via permits
 */
export type AgentCapability =
  // Movement & Communication
  | 'go'            // Travel to other places (Telescript go())
  | 'meet'          // Participate in meetings (Telescript meet())
  | 'network'       // Access network resources

  // Computation & Storage
  | 'compute'       // Access computational resources
  | 'store'         // Access storage resources
  | 'execute'       // Execute code or actions
  | 'inference'     // Access inference tier (EDGE/LOCAL/POOL)

  // Reasoning & Analysis
  | 'reason'        // Perform reasoning and decision making
  | 'search'        // Search and retrieve information
  | 'validate'      // Validate transactions or state

  // Delegation & Authority
  | 'delegate'      // Delegate permits to sub-agents
  | 'access_pool'   // Access the resource pool
  | 'consensus'     // Participate in consensus

  // Security & Monitoring
  | 'monitor'       // Monitor system state and events
  | 'veto'          // Veto operations (Guardian power)
  | 'alert'         // Generate alerts

  // SAT-Specific
  | 'attest'        // Create attestations
  | 'arbitrate'     // Arbitrate disputes
  | 'replicate';    // Replicate data across nodes

/** Default capabilities for PAT roles */
export const PAT_DEFAULT_CAPABILITIES: Record<PATRole, AgentCapability[]> = {
  strategist: ['reason', 'delegate', 'access_pool'],
  researcher: ['search', 'reason', 'store'],
  developer: ['execute', 'compute', 'store'],
  analyst: ['reason', 'compute', 'inference'],
  reviewer: ['reason', 'validate', 'meet'],
  executor: ['execute', 'network', 'go'],
  guardian: ['validate', 'monitor', 'veto'],
};

/** Default capabilities for SAT roles */
export const SAT_DEFAULT_CAPABILITIES: Record<SATRole, AgentCapability[]> = {
  validator: ['validate', 'consensus', 'attest'],
  oracle: ['network', 'search', 'attest'],
  mediator: ['reason', 'meet', 'arbitrate'],
  archivist: ['store', 'search', 'replicate'],
  sentinel: ['monitor', 'veto', 'alert'],
};

// =============================================================================
// RESOURCE LIMITS
// =============================================================================

/**
 * Resource limits for agent permits
 */
export interface AgentResourceLimits {
  /** Maximum CPU millicores (1000 = 1 core) */
  cpuMillicores: number;
  /** Maximum memory in bytes */
  memoryBytes: number;
  /** Maximum storage in bytes */
  storageBytes: number;
  /** Maximum network bandwidth (bytes/sec) */
  networkBps: number;
  /** Maximum inference tokens per request */
  inferenceTokens: number;
  /** Time-to-live in seconds */
  ttlSeconds: number;
  /** Maximum concurrent operations */
  maxConcurrentOps: number;
}

/** Default resource limits */
export const DEFAULT_RESOURCE_LIMITS: AgentResourceLimits = {
  cpuMillicores: 100,
  memoryBytes: 64 * 1024 * 1024,      // 64 MB
  storageBytes: 256 * 1024 * 1024,    // 256 MB
  networkBps: 1024 * 1024,            // 1 MB/s
  inferenceTokens: 4096,
  ttlSeconds: 3600,
  maxConcurrentOps: 10,
};

// =============================================================================
// STANDING ON GIANTS ATTESTATION
// =============================================================================

/**
 * A single intellectual foundation entry
 */
export interface IntellectualFoundation {
  /** Name of the giant */
  giantName: string;
  /** Specific contribution being used */
  contribution: string;
  /** Citation or reference */
  citation?: string;
  /** How it's used in this agent */
  usageInAgent: string;
}

/**
 * A provenance record tracking knowledge origin
 */
export interface ProvenanceRecord {
  /** Source of knowledge/action */
  source: string;
  /** Action taken */
  action: string;
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Hash of the data/knowledge (hex) */
  contentHash: string;
}

/**
 * StandingOnGiantsAttestation — Required attribution chain for every agent
 *
 * Every agent MUST cite its intellectual foundations. This is a core requirement
 * of the BIZRA ecosystem to honor the giants whose work made this possible.
 */
export interface StandingOnGiantsAttestation {
  /** Attestation ID (UUID) */
  id: string;
  /** Agent this attestation is for */
  agentId: string;
  /** Intellectual foundations cited (minimum 5 required) */
  foundations: IntellectualFoundation[];
  /** Knowledge provenance chain */
  provenance: ProvenanceRecord[];
  /** Attestation timestamp (ISO 8601) */
  attestedAt: string;
  /** Blake3 hash for integrity (hex) */
  attestationHash: string;
}

/** Universal foundations all agents must cite */
export const UNIVERSAL_FOUNDATIONS: IntellectualFoundation[] = [
  {
    giantName: 'Claude Shannon',
    contribution: 'Information Theory — SNR-based signal quality',
    citation: 'A Mathematical Theory of Communication, 1948',
    usageInAgent: 'Signal quality assessment',
  },
  {
    giantName: 'Leslie Lamport',
    contribution: 'Byzantine Fault Tolerance — Distributed consensus',
    citation: 'Byzantine Generals Problem, 1982',
    usageInAgent: 'Fault-tolerant operation',
  },
  {
    giantName: 'Daniel J. Bernstein',
    contribution: 'Ed25519 — Fast and secure digital signatures',
    citation: 'High-speed high-security signatures, 2012',
    usageInAgent: 'Identity and signing',
  },
  {
    giantName: 'General Magic',
    contribution: 'Telescript — Mobile agent primitives',
    citation: 'Telescript Technology, 1994',
    usageInAgent: 'Agent architecture',
  },
  {
    giantName: 'Anthropic',
    contribution: 'Constitutional AI — Principle-governed systems',
    citation: 'Constitutional AI, 2022',
    usageInAgent: 'Ihsan threshold enforcement',
  },
];

// =============================================================================
// AGENT CAPABILITY CARD
// =============================================================================

/**
 * AgentCapabilityCard — Extends Telescript Permit with BIZRA-specific attributes
 *
 * The capability card defines what an agent is allowed to do, including
 * resource limits, valid locations, and required Ihsan threshold.
 */
export interface AgentCapabilityCard {
  /** Unique card identifier (UUID) */
  id: string;
  /** Agent this card is issued to */
  agentId: string;
  /** Human node that issued this card */
  issuerNodeId: string;
  /** Granted capabilities */
  capabilities: AgentCapability[];
  /** Resource limits */
  limits: AgentResourceLimits;
  /** Ihsan threshold requirement (>= 0.95) */
  ihsanRequirement: number;
  /** Valid places (empty = all) */
  validPlaces: string[];
  /** Card creation timestamp (ISO 8601) */
  createdAt: string;
  /** Card expiration timestamp (ISO 8601) */
  expiresAt: string;
  /** Standing on Giants attestation (required) */
  giantsAttestation: StandingOnGiantsAttestation;
  /** Blake3 hash for integrity (hex) */
  cardHash: string;
}

// =============================================================================
// AGENT MINT REQUEST
// =============================================================================

/**
 * Agent type classification
 */
export type AgentType =
  | { type: 'PAT'; role: PATRole }
  | { type: 'SAT'; role: SATRole };

/**
 * AgentMintRequest — Request to mint a new agent
 *
 * This is the primary interface for creating new agents. The request must
 * include proper Standing on Giants attestation.
 */
export interface AgentMintRequest {
  /** Request ID (UUID) */
  requestId: string;
  /** Human node requesting the mint */
  requesterNodeId: string;
  /** Requester's public key (Ed25519, hex) */
  requesterPublicKey: string;
  /** Agent type (PAT or SAT) */
  agentType: AgentType;
  /** Requested capabilities */
  requestedCapabilities: AgentCapability[];
  /** Requested resource limits */
  requestedLimits: AgentResourceLimits;
  /** Standing on Giants attestation (required) */
  giantsAttestation: StandingOnGiantsAttestation;
  /** Request timestamp (ISO 8601) */
  requestedAt: string;
  /** Request signature (Ed25519, hex) */
  signature: string;
  /** Request hash (Blake3, hex) */
  requestHash: string;
}

// =============================================================================
// MINTED AGENT
// =============================================================================

/**
 * Agent state enum
 */
export type AgentState =
  | 'pending'       // Agent is pending activation
  | 'active'        // Agent is active
  | 'paused'        // Agent is paused
  | 'suspended'     // Agent is suspended (Ihsan violation)
  | 'terminated';   // Agent is terminated

/**
 * MintedAgent — A fully minted and active agent
 */
export interface MintedAgent {
  /** Agent ID (UUID) */
  id: string;
  /** Agent's Ed25519 public key (hex) */
  publicKey: string;
  /** Agent name */
  name: string;
  /** Agent's capability card */
  capabilityCard: AgentCapabilityCard;
  /** Current Ihsan score */
  ihsanScore: number;
  /** Current state */
  state: AgentState;
  /** Stake deposited (for SAT agents) */
  stake: number;
  /** Impact score (for earnings calculation) */
  impactScore: number;
  /** Creation timestamp (ISO 8601) */
  createdAt: string;
  /** Last activity timestamp (ISO 8601) */
  lastActivity: string;
  /** Identity block hash (links to human node's authority chain, hex) */
  identityBlockHash: string;
}

// =============================================================================
// PERSONAL AGENTIC TEAM (PAT)
// =============================================================================

/**
 * PersonalAgentTeam — The 7-agent mastermind council
 *
 * Each human node has their own PAT - a private team of 7 specialized agents
 * that work together to assist the user.
 */
export interface PersonalAgentTeam {
  /** Team ID (UUID) */
  id: string;
  /** Owner's node ID */
  ownerNodeId: string;
  /** Owner's public key (Ed25519, hex) */
  ownerPublicKey: string;
  /** The 7 PAT agents (keyed by role) */
  agents: Record<PATRole, MintedAgent>;
  /** Team creation timestamp (ISO 8601) */
  createdAt: string;
  /** Team-level Ihsan score (aggregate) */
  teamIhsanScore: number;
  /** Team-level authority chain hash (hex) */
  authorityChainHash: string;
}

/**
 * Check if a PAT is complete (all 7 agents minted)
 */
export function isPATComplete(pat: PersonalAgentTeam): boolean {
  return PAT_ROLES.every(role => pat.agents[role] !== undefined);
}

/**
 * Calculate a PAT's aggregate Ihsan score
 */
export function calculatePATIhsan(pat: PersonalAgentTeam): number {
  const agents = Object.values(pat.agents);
  if (agents.length === 0) return 0;
  const sum = agents.reduce((acc, agent) => acc + agent.ihsanScore, 0);
  return sum / agents.length;
}

// =============================================================================
// SHARED AGENTIC TEAM (SAT)
// =============================================================================

/**
 * SharedAgentTeam — The 5-agent public utility contribution
 *
 * Each human node contributes a SAT to the resource pool. These agents
 * provide services to other users and earn tokens based on usage.
 */
export interface SharedAgentTeam {
  /** Team ID (UUID) */
  id: string;
  /** Contributor's node ID */
  contributorNodeId: string;
  /** Contributor's public key (Ed25519, hex) */
  contributorPublicKey: string;
  /** The 5 SAT agents (keyed by role) */
  agents: Record<SATRole, MintedAgent>;
  /** Pool registration ID (after registration) */
  poolRegistrationId?: string;
  /** Total stake deposited */
  totalStake: number;
  /** Earnings accumulated */
  totalEarnings: number;
  /** Team creation timestamp (ISO 8601) */
  createdAt: string;
  /** Team-level Ihsan score (aggregate) */
  teamIhsanScore: number;
  /** Pool governance rules hash (hex) */
  governanceHash: string;
}

/**
 * Check if a SAT is complete (all 5 agents minted)
 */
export function isSATComplete(sat: SharedAgentTeam): boolean {
  return SAT_ROLES.every(role => sat.agents[role] !== undefined);
}

/**
 * Check if a SAT meets all stake requirements
 */
export function satMeetsStakeRequirements(sat: SharedAgentTeam): boolean {
  return SAT_ROLES.every(role => {
    const agent = sat.agents[role];
    if (!agent) return false;
    return agent.stake >= SAT_MINIMUM_STAKES[role];
  });
}

/**
 * Calculate SAT earnings for a usage
 * 90% to contributor, 10% to pool
 */
export function calculateSATEarnings(usageValue: number): number {
  return Math.floor((usageValue * 90) / 100);
}

// =============================================================================
// IDENTITY BLOCK
// =============================================================================

/**
 * A link in the authority chain
 */
export interface AuthorityLink {
  /** Authority name */
  name: string;
  /** Authority public key (Ed25519, hex) */
  publicKey: string;
  /** Delegation depth */
  depth: number;
  /** Link hash (Blake3, hex) */
  linkHash: string;
}

/**
 * AgentIdentityBlock — KNOWLEDGE_BLOCK type for agent identity
 *
 * Links the agent to its parent (human node) and establishes the
 * authority chain back to Genesis.
 */
export interface AgentIdentityBlock {
  /** Block type identifier */
  blockType: 'KNOWLEDGE_BLOCK';
  /** Block version */
  version: number;
  /** Agent ID (UUID) */
  agentId: string;
  /** Agent's public key (Ed25519, hex) */
  agentPublicKey: string;
  /** Parent (human node) public key (Ed25519, hex) */
  parentPublicKey: string;
  /** Parent node ID */
  parentNodeId: string;
  /** Authority chain proof */
  authorityChain: AuthorityLink[];
  /** Block creation timestamp (ISO 8601) */
  createdAt: string;
  /** Standing on Giants attestation */
  giantsAttestation: StandingOnGiantsAttestation;
  /** Block hash (Blake3, hex) */
  blockHash: string;
  /** Parent signature (Ed25519, hex) */
  parentSignature: string;
}

// =============================================================================
// ACTION ATTESTATION
// =============================================================================

/**
 * Types of actions agents can perform
 */
export type ActionType =
  | 'reason'        // Reasoning/inference
  | 'execute'       // Code execution
  | 'retrieve'      // Data retrieval
  | 'store'         // Data storage
  | 'validate'      // Validation
  | 'communicate'   // Communication
  | 'travel'        // Travel between places
  | 'meet'          // Meeting with another agent
  | 'consensus'     // Consensus participation
  | 'pool_service'; // Pool service provision

/**
 * A citation of a giant's work in an action
 */
export interface GiantCitation {
  /** Giant's name */
  giantName: string;
  /** Specific contribution used */
  contribution: string;
  /** How it was applied */
  application: string;
}

/**
 * ActionAttestation — Immutable record of agent activity
 *
 * Every action an agent takes must cite the relevant giants whose
 * work enabled that action (Standing on Giants protocol).
 */
export interface ActionAttestation {
  /** Attestation ID (UUID) */
  id: string;
  /** Agent that performed the action */
  agentId: string;
  /** Action type */
  actionType: ActionType;
  /** Action description */
  description: string;
  /** Input data hash (Blake3, hex) */
  inputHash: string;
  /** Output data hash (Blake3, hex) */
  outputHash: string;
  /** Giants cited for this action (at least 1 required) */
  giantsCited: GiantCitation[];
  /** Provenance records */
  provenance: ProvenanceRecord[];
  /** Action timestamp (ISO 8601) */
  timestamp: string;
  /** Ihsan score at time of action */
  ihsanScore: number;
  /** Attestation hash (Blake3, hex) */
  attestationHash: string;
}

// =============================================================================
// POOL USAGE TRACKING
// =============================================================================

/**
 * Resource usage breakdown
 */
export interface ResourceUsage {
  /** CPU milliseconds used */
  cpuMs: number;
  /** Memory bytes peak */
  memoryBytesPeak: number;
  /** Network bytes transferred */
  networkBytes: number;
  /** Storage bytes used */
  storageBytes: number;
  /** Inference tokens used */
  inferenceTokens: number;
}

/**
 * PoolUsageRecord — Tracks SAT agent usage in the resource pool
 */
export interface PoolUsageRecord {
  /** Record ID (UUID) */
  id: string;
  /** SAT agent that provided the service */
  providerAgentId: string;
  /** Consumer agent/node */
  consumerId: string;
  /** Service type */
  serviceType: SATRole;
  /** Usage duration in milliseconds */
  durationMs: number;
  /** Resources consumed */
  resourcesConsumed: ResourceUsage;
  /** Value earned (tokens) */
  valueEarned: number;
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Related attestation */
  attestationId: string;
}

// =============================================================================
// API TYPES
// =============================================================================

/**
 * Response for PAT minting
 */
export interface MintPATResponse {
  success: boolean;
  pat?: PersonalAgentTeam;
  error?: string;
}

/**
 * Response for SAT minting
 */
export interface MintSATResponse {
  success: boolean;
  sat?: SharedAgentTeam;
  poolRegistrationId?: string;
  error?: string;
}

/**
 * Response for single agent minting
 */
export interface MintAgentResponse {
  success: boolean;
  agent?: MintedAgent;
  identityBlock?: AgentIdentityBlock;
  error?: string;
}

/**
 * Request to mint a complete PAT
 */
export interface MintPATRequest {
  /** Requester's node ID */
  nodeId: string;
  /** Ed25519 signature of the request */
  signature: string;
}

/**
 * Request to mint a complete SAT
 */
export interface MintSATRequest {
  /** Requester's node ID */
  nodeId: string;
  /** Stakes for each role */
  stakes: Record<SATRole, number>;
  /** Ed25519 signature of the request */
  signature: string;
}

// =============================================================================
// TELESCRIPT INTEGRATION
// =============================================================================

/**
 * Maps PAT/SAT capabilities to Telescript Permit capabilities
 *
 * This provides integration with the existing bizra-telescript
 * Agent/Permit system.
 */
export const CAPABILITY_TO_TELESCRIPT: Record<AgentCapability, string> = {
  go: 'Go',
  meet: 'Meet',
  network: 'Network',
  compute: 'Compute',
  store: 'Store',
  execute: 'Compute',
  inference: 'Inference',
  reason: 'Compute',
  search: 'Network',
  validate: 'Compute',
  delegate: 'Delegate',
  access_pool: 'Network',
  consensus: 'Meet',
  monitor: 'Network',
  veto: 'SelfModify',
  alert: 'Network',
  attest: 'Create',
  arbitrate: 'Meet',
  replicate: 'Store',
};

/**
 * Create a Telescript-compatible permit from a capability card
 */
export function toTelescriptPermit(card: AgentCapabilityCard): {
  capabilities: string[];
  limits: {
    cpuMillicores: number;
    memoryBytes: number;
    storageBytes: number;
    networkBps: number;
    inferenceTokens: number;
    ttlSeconds: number;
  };
  ihsanRequirement: number;
} {
  const capabilities = card.capabilities.map(cap => CAPABILITY_TO_TELESCRIPT[cap]);
  const uniqueCapabilities = [...new Set(capabilities)];

  return {
    capabilities: uniqueCapabilities,
    limits: {
      cpuMillicores: card.limits.cpuMillicores,
      memoryBytes: card.limits.memoryBytes,
      storageBytes: card.limits.storageBytes,
      networkBps: card.limits.networkBps,
      inferenceTokens: card.limits.inferenceTokens,
      ttlSeconds: card.limits.ttlSeconds,
    },
    ihsanRequirement: card.ihsanRequirement,
  };
}
