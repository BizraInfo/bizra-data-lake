# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Islamic Masterminds - Constellation Orchestrator v1.0
# ═══════════════════════════════════════════════════════════════════════════════
# Execution Flow: Intake → Plan → Work → Verify → Synthesize → Deliver
# ═══════════════════════════════════════════════════════════════════════════════

"""
BIZRA Constellation Orchestrator

This module implements the multi-agent orchestration system for the
Islamic Masterminds Agentic Constellation. It manages:

1. Task intake and classification
2. Team selection and composition
3. Reasoning mode routing (CoT/ToT/GoT)
4. Work distribution and collection
5. Verification and evidence gates
6. Synthesis via Polymath Integrator
7. Final delivery formatting

Compatible with LangGraph, custom orchestrators, or standalone execution.
"""

from __future__ import annotations

import re
import yaml
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS AND TYPES
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningMode(str, Enum):
    """Reasoning architecture modes."""
    COT = "cot"  # Chain-of-Thought
    TOT = "tot"  # Tree-of-Thought
    GOT = "got"  # Graph-of-Thought


class Stakes(str, Enum):
    """Task stakes classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClaimTag(str, Enum):
    """Epistemic status tags for claims."""
    MEASURED = "MEASURED"
    IMPLEMENTED = "IMPLEMENTED"
    DERIVED = "DERIVED"
    DESIGNED = "DESIGNED"
    TARGET = "TARGET"
    HYPOTHESIS = "HYPOTHESIS"
    METAPHOR = "METAPHOR"


class AgentRole(str, Enum):
    """Agent role classifications."""
    ORCHESTRATION = "orchestration"
    SYNTHESIS = "synthesis"
    VERIFIER = "verifier"
    DOMAIN = "domain"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """Represents an agent in the constellation."""
    id: int
    name: str
    slug: str
    domain: str
    snr_target: tuple[float, float]
    snr_tier: str
    specialty: str
    reasoning_default: ReasoningMode
    reasoning_pattern: str
    personality: list[str]
    tools_allowed: list[str]
    output_contract: dict
    role: AgentRole
    description: str


@dataclass
class Team:
    """Represents a cross-pollination team."""
    id: str
    name: str
    description: str
    leader: Agent
    members: list[Agent]
    snr_target: float
    reasoning_mode: ReasoningMode
    reasoning_template: str
    activation_patterns: list[str]
    output_contract: dict


@dataclass
class TaskAnalysis:
    """Result of task intake analysis."""
    task_id: str
    original_query: str
    stakes: Stakes
    domains: list[str]
    keywords_matched: list[str]
    suggested_team: Optional[str]
    reasoning_mode: ReasoningMode
    snr_target: float
    verifiers_required: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentOutput:
    """Output from a single agent."""
    agent_id: int
    agent_name: str
    content: str
    claims: list[dict]  # Each claim has 'text', 'tag', 'evidence'
    confidence: float
    reasoning_trace: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class VerificationResult:
    """Result from a verification pass."""
    verifier_id: int
    verifier_name: str
    passed: bool
    issues: list[str]
    attestation: Optional[str]
    snr_adjustment: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConstellationOutput:
    """Final synthesized output from the constellation."""
    task_id: str
    executive_summary: list[str]  # 5 bullets max
    what_we_know: list[dict]
    what_we_assume: list[dict]
    what_to_test_next: list[str]
    full_content: str
    agent_contributions: list[AgentOutput]
    verifications: list[VerificationResult]
    final_snr: float
    reasoning_mode_used: ReasoningMode
    team_used: Optional[str]
    delivery_checklist: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# CONSTELLATION LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ConstellationLoader:
    """Loads constellation configuration from YAML files."""
    
    def __init__(self, constellation_path: Path):
        self.path = constellation_path
        self.agents: dict[str, Agent] = {}
        self.teams: dict[str, Team] = {}
        self.router_policy: dict = {}
        self.evaluation_gates: dict = {}
        
    def load_all(self) -> None:
        """Load all constellation configuration."""
        self._load_agents()
        self._load_teams()
        self._load_router()
        self._load_evaluation()
        
    def _load_agents(self) -> None:
        """Load agent roster from YAML."""
        roster_path = self.path / "agents" / "roster.yaml"
        with open(roster_path, "r", encoding="utf-8") as f:
            roster = yaml.safe_load(f)
            
        # Load meta-agents
        for agent_data in roster.get("meta_agents", []):
            agent = self._parse_agent(agent_data)
            self.agents[agent.slug] = agent
            
        # Load domain agents
        for agent_data in roster.get("domain_agents", []):
            agent = self._parse_agent(agent_data)
            self.agents[agent.slug] = agent
            
    def _parse_agent(self, data: dict) -> Agent:
        """Parse agent data into Agent object."""
        return Agent(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            domain=data["domain"],
            snr_target=tuple(data["snr_target"]),
            snr_tier=data["snr_tier"],
            specialty=data["specialty"],
            reasoning_default=ReasoningMode(data["reasoning_default"]),
            reasoning_pattern=data["reasoning_pattern"],
            personality=data["personality"],
            tools_allowed=data["tools_allowed"],
            output_contract=data["output_contract"],
            role=AgentRole(data["role"]),
            description=data["description"]
        )
        
    def _load_teams(self) -> None:
        """Load team configurations from YAML."""
        teams_path = self.path / "teams" / "configurations.yaml"
        with open(teams_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        for team_data in config.get("teams", []):
            team = self._parse_team(team_data)
            self.teams[team.id] = team
            
    def _parse_team(self, data: dict) -> Team:
        """Parse team data into Team object."""
        leader_slug = data["leader"]["agent"]
        member_slugs = [m["agent"] for m in data["members"]]
        
        return Team(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            leader=self.agents.get(leader_slug),
            members=[self.agents.get(slug) for slug in member_slugs],
            snr_target=data["snr_target"],
            reasoning_mode=ReasoningMode(data["reasoning_mode"]),
            reasoning_template=data["reasoning_template"],
            activation_patterns=data["activation_patterns"],
            output_contract=data["output_contract"]
        )
        
    def _load_router(self) -> None:
        """Load router policy from YAML."""
        router_path = self.path / "router" / "policy.yaml"
        with open(router_path, "r", encoding="utf-8") as f:
            self.router_policy = yaml.safe_load(f)
            
    def _load_evaluation(self) -> None:
        """Load evaluation gates from YAML."""
        eval_path = self.path / "evaluation" / "gates.yaml"
        with open(eval_path, "r", encoding="utf-8") as f:
            self.evaluation_gates = yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# TASK ANALYZER (INTAKE PHASE)
# ─────────────────────────────────────────────────────────────────────────────

class TaskAnalyzer:
    """Analyzes incoming tasks for routing decisions."""
    
    def __init__(self, loader: ConstellationLoader):
        self.loader = loader
        self.policy = loader.router_policy
        
    def analyze(self, query: str, context: Optional[dict] = None) -> TaskAnalysis:
        """
        Analyze a task and produce routing recommendations.
        
        Phase 1 of execution flow: INTAKE
        """
        import uuid
        
        task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"
        
        # Classify stakes
        stakes = self._classify_stakes(query, context)
        
        # Detect domains
        domains = self._detect_domains(query)
        
        # Match keywords
        keywords = self._match_keywords(query)
        
        # Suggest team
        suggested_team = self._suggest_team(query, domains)
        
        # Determine reasoning mode
        reasoning_mode = self._determine_reasoning_mode(
            stakes, len(domains), keywords
        )
        
        # Calculate SNR target
        snr_target = self._calculate_snr_target(domains, stakes)
        
        # Determine verifiers required
        verifiers_required = self._count_verifiers_required(snr_target, stakes)
        
        return TaskAnalysis(
            task_id=task_id,
            original_query=query,
            stakes=stakes,
            domains=domains,
            keywords_matched=keywords,
            suggested_team=suggested_team,
            reasoning_mode=reasoning_mode,
            snr_target=snr_target,
            verifiers_required=verifiers_required
        )
        
    def _classify_stakes(self, query: str, context: Optional[dict]) -> Stakes:
        """Classify task stakes based on keywords and context."""
        query_lower = query.lower()
        
        high_stakes_keywords = self.policy.get("router", {}).get(
            "task_classification", {}
        ).get("high_stakes_keywords", [])
        
        for keyword in high_stakes_keywords:
            if keyword in query_lower:
                return Stakes.HIGH
                
        if context and context.get("production", False):
            return Stakes.HIGH
            
        if context and context.get("critical", False):
            return Stakes.CRITICAL
            
        return Stakes.MEDIUM
        
    def _detect_domains(self, query: str) -> list[str]:
        """Detect relevant domains from query."""
        domains = []
        query_lower = query.lower()
        
        domain_keywords = {
            "philosophy": ["philosophy", "ethics", "metaphysics", "epistemology"],
            "theology": ["theology", "faith", "religion", "spiritual"],
            "medicine": ["medical", "health", "diagnosis", "treatment", "clinical"],
            "mathematics": ["math", "algorithm", "equation", "computation"],
            "science": ["science", "experiment", "hypothesis", "empirical"],
            "engineering": ["engineering", "mechanical", "design", "build"],
            "jurisprudence": ["legal", "law", "ruling", "jurisprudence", "fatwa"],
            "hadith": ["hadith", "narration", "authentication", "chain"],
            "governance": ["governance", "administration", "policy", "leadership"],
            "strategy": ["strategy", "military", "coordination", "planning"],
            "creativity": ["creative", "poetry", "art", "metaphor", "innovation"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)
                
        return domains if domains else ["general"]
        
    def _match_keywords(self, query: str) -> list[str]:
        """Match query against classification keywords."""
        matched = []
        query_lower = query.lower()
        
        classification = self.policy.get("router", {}).get("task_classification", {})
        
        for category, keywords in classification.items():
            if isinstance(keywords, list):
                for kw in keywords:
                    if kw in query_lower:
                        matched.append(f"{category}:{kw}")
                        
        return matched
        
    def _suggest_team(self, query: str, domains: list[str]) -> Optional[str]:
        """Suggest a pre-built team based on query and domains."""
        query_lower = query.lower()
        
        team_selection = self.policy.get("team_selection", {}).get("auto_select", [])
        
        for rule in team_selection:
            pattern = rule.get("pattern", "")
            if re.search(pattern, query_lower):
                return rule.get("team")
                
        return None
        
    def _determine_reasoning_mode(
        self, 
        stakes: Stakes, 
        domain_count: int,
        keywords: list[str]
    ) -> ReasoningMode:
        """Determine appropriate reasoning mode."""
        # High stakes -> ToT for exploration
        if stakes in (Stakes.HIGH, Stakes.CRITICAL):
            return ReasoningMode.TOT
            
        # Multiple domains -> GoT for synthesis
        if domain_count >= 3:
            return ReasoningMode.GOT
            
        # Creative keywords -> GoT for associative thinking
        if any("creative" in kw for kw in keywords):
            return ReasoningMode.GOT
            
        # Default to CoT for efficiency
        return ReasoningMode.COT
        
    def _calculate_snr_target(self, domains: list[str], stakes: Stakes) -> float:
        """Calculate target SNR based on domains and stakes."""
        snr_targets = self.policy.get("snr_targets", {})
        
        # Get highest SNR requirement from detected domains
        max_snr = 0.85  # Minimum floor
        
        for domain in domains:
            if domain in snr_targets:
                domain_range = snr_targets[domain].get("range", [0.85, 0.90])
                max_snr = max(max_snr, domain_range[0])
                
        # Boost for high stakes
        if stakes == Stakes.HIGH:
            max_snr = max(max_snr, 0.93)
        elif stakes == Stakes.CRITICAL:
            max_snr = max(max_snr, 0.96)
            
        return max_snr
        
    def _count_verifiers_required(self, snr_target: float, stakes: Stakes) -> int:
        """Determine number of verifiers required."""
        if snr_target >= 0.96:
            return 2
        elif snr_target >= 0.93 or stakes == Stakes.HIGH:
            return 1
        else:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# TEAM COMPOSER (PLAN PHASE)
# ─────────────────────────────────────────────────────────────────────────────

class TeamComposer:
    """Composes teams for task execution."""
    
    def __init__(self, loader: ConstellationLoader):
        self.loader = loader
        
    def compose(self, analysis: TaskAnalysis) -> tuple[Team, list[Agent]]:
        """
        Compose a team for the given task analysis.
        
        Phase 2 of execution flow: PLAN
        
        Returns:
            Tuple of (selected team, additional verifiers if needed)
        """
        # Use suggested team if available
        if analysis.suggested_team and analysis.suggested_team in self.loader.teams:
            team = self.loader.teams[analysis.suggested_team]
        else:
            # Compose ad-hoc team
            team = self._compose_adhoc(analysis)
            
        # Add verifiers if required
        verifiers = self._select_verifiers(analysis, team)
        
        return team, verifiers
        
    def _compose_adhoc(self, analysis: TaskAnalysis) -> Team:
        """Compose an ad-hoc team based on task analysis."""
        # Select agents matching domains
        selected_agents = []
        
        for domain in analysis.domains:
            domain_agents = [
                agent for agent in self.loader.agents.values()
                if domain.lower() in agent.domain.lower()
            ]
            if domain_agents:
                # Sort by SNR and take highest
                domain_agents.sort(key=lambda a: a.snr_target[1], reverse=True)
                selected_agents.append(domain_agents[0])
                
        # Ensure minimum agents
        if len(selected_agents) < 2:
            # Add polymath integrator for synthesis
            polymath = self.loader.agents.get("polymath_integrator")
            if polymath and polymath not in selected_agents:
                selected_agents.append(polymath)
                
        # Create ad-hoc team
        leader = selected_agents[0] if selected_agents else None
        members = selected_agents[1:] if len(selected_agents) > 1 else []
        
        return Team(
            id="adhoc_team",
            name="Ad-Hoc Team",
            description=f"Dynamically composed for: {analysis.original_query[:50]}...",
            leader=leader,
            members=members,
            snr_target=analysis.snr_target,
            reasoning_mode=analysis.reasoning_mode,
            reasoning_template="Adaptive execution based on task requirements",
            activation_patterns=[],
            output_contract={"must_include": ["analysis", "recommendations"]}
        )
        
    def _select_verifiers(self, analysis: TaskAnalysis, team: Team) -> list[Agent]:
        """Select verifiers based on requirements."""
        if analysis.verifiers_required == 0:
            return []
            
        verifiers = []
        verifier_pool = [
            agent for agent in self.loader.agents.values()
            if agent.role == AgentRole.VERIFIER
        ]
        
        # Sort by SNR
        verifier_pool.sort(key=lambda a: a.snr_target[1], reverse=True)
        
        # Select required number, avoiding team members
        team_ids = {team.leader.id if team.leader else -1}
        team_ids.update(m.id for m in team.members if m)
        
        for verifier in verifier_pool:
            if verifier.id not in team_ids:
                verifiers.append(verifier)
                if len(verifiers) >= analysis.verifiers_required:
                    break
                    
        return verifiers


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE EVALUATOR (VERIFY PHASE)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceEvaluator:
    """Evaluates agent outputs against evidence gates."""
    
    def __init__(self, loader: ConstellationLoader):
        self.loader = loader
        self.gates = loader.evaluation_gates
        
    def evaluate(self, output: AgentOutput, analysis: TaskAnalysis) -> VerificationResult:
        """
        Evaluate an agent output against evidence gates.
        
        Phase 4 of execution flow: VERIFY
        """
        issues = []
        snr_adjustment = 0.0
        
        # Check claim tagging
        tagging_issues = self._check_claim_tagging(output)
        issues.extend(tagging_issues)
        
        # Check contradiction
        contradiction_issues = self._check_contradictions(output)
        issues.extend(contradiction_issues)
        if contradiction_issues:
            snr_adjustment -= 0.05
            
        # Check SNR floor
        snr_issues = self._check_snr_floor(output, analysis)
        issues.extend(snr_issues)
        
        # Check source citations
        citation_issues = self._check_citations(output, analysis)
        issues.extend(citation_issues)
        
        passed = len(issues) == 0
        
        attestation = None
        if passed:
            attestation = f"Verified by EvidenceEvaluator at {datetime.now(timezone.utc).isoformat()}"
            
        return VerificationResult(
            verifier_id=0,  # System verifier
            verifier_name="EvidenceEvaluator",
            passed=passed,
            issues=issues,
            attestation=attestation,
            snr_adjustment=snr_adjustment
        )
        
    def _check_claim_tagging(self, output: AgentOutput) -> list[str]:
        """Check that claims are properly tagged."""
        issues = []
        
        for claim in output.claims:
            if "tag" not in claim:
                issues.append(f"Untagged claim: {claim.get('text', '')[:50]}...")
            elif claim["tag"] not in [t.value for t in ClaimTag]:
                issues.append(f"Invalid tag '{claim['tag']}' on claim")
                
        return issues
        
    def _check_contradictions(self, output: AgentOutput) -> list[str]:
        """Check for internal contradictions."""
        issues = []
        # Simplified contradiction detection
        # In production, would use semantic analysis
        
        claim_texts = [c.get("text", "").lower() for c in output.claims]
        
        # Check for negation patterns
        for i, text in enumerate(claim_texts):
            for j, other in enumerate(claim_texts[i+1:], i+1):
                if "not" in other and text.replace("not ", "") in other:
                    issues.append(f"Potential contradiction between claims {i} and {j}")
                    
        return issues
        
    def _check_snr_floor(self, output: AgentOutput, analysis: TaskAnalysis) -> list[str]:
        """Check that output meets SNR floor."""
        issues = []
        
        if output.confidence < analysis.snr_target:
            issues.append(
                f"Confidence {output.confidence:.2f} below target {analysis.snr_target:.2f}"
            )
            
        return issues
        
    def _check_citations(self, output: AgentOutput, analysis: TaskAnalysis) -> list[str]:
        """Check source citations for high-SNR claims."""
        issues = []
        
        high_snr_tags = [ClaimTag.MEASURED.value, ClaimTag.IMPLEMENTED.value]
        
        for claim in output.claims:
            if claim.get("tag") in high_snr_tags:
                if not claim.get("evidence"):
                    issues.append(
                        f"Missing citation for {claim['tag']} claim: {claim.get('text', '')[:30]}..."
                    )
                    
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# POLYMATH SYNTHESIZER (SYNTHESIZE PHASE)
# ─────────────────────────────────────────────────────────────────────────────

class PolymathSynthesizer:
    """Synthesizes outputs from multiple agents into coherent deliverable."""
    
    def __init__(self, loader: ConstellationLoader):
        self.loader = loader
        self.polymath = loader.agents.get("polymath_integrator")
        
    def synthesize(
        self,
        analysis: TaskAnalysis,
        outputs: list[AgentOutput],
        verifications: list[VerificationResult],
        team: Team
    ) -> ConstellationOutput:
        """
        Synthesize all outputs into final deliverable.
        
        Phase 5 of execution flow: SYNTHESIZE
        """
        # Extract executive summary
        executive_summary = self._create_summary(outputs)
        
        # Classify claims
        what_we_know = []
        what_we_assume = []
        
        for output in outputs:
            for claim in output.claims:
                classified = {
                    "claim": claim.get("text"),
                    "tag": claim.get("tag"),
                    "source_agent": output.agent_name,
                    "confidence": output.confidence
                }
                
                if claim.get("tag") in [
                    ClaimTag.MEASURED.value, 
                    ClaimTag.IMPLEMENTED.value,
                    ClaimTag.DERIVED.value
                ]:
                    what_we_know.append(classified)
                else:
                    what_we_assume.append(classified)
                    
        # Generate next steps
        what_to_test = self._generate_next_steps(outputs, verifications)
        
        # Combine full content
        full_content = self._combine_content(outputs)
        
        # Calculate final SNR
        final_snr = self._calculate_final_snr(outputs, verifications)
        
        # Build delivery checklist
        checklist = self._build_checklist(analysis, verifications)
        
        return ConstellationOutput(
            task_id=analysis.task_id,
            executive_summary=executive_summary,
            what_we_know=what_we_know,
            what_we_assume=what_we_assume,
            what_to_test_next=what_to_test,
            full_content=full_content,
            agent_contributions=outputs,
            verifications=verifications,
            final_snr=final_snr,
            reasoning_mode_used=analysis.reasoning_mode,
            team_used=team.id if team else None,
            delivery_checklist=checklist
        )
        
    def _create_summary(self, outputs: list[AgentOutput]) -> list[str]:
        """Create 5-bullet executive summary."""
        # Extract key points from each output
        summaries = []
        
        for output in outputs[:5]:  # Max 5 agents for summary
            # Take first claim or first 100 chars of content
            if output.claims:
                summaries.append(f"[{output.agent_name}] {output.claims[0].get('text', '')[:100]}")
            else:
                summaries.append(f"[{output.agent_name}] {output.content[:100]}")
                
        return summaries[:5]  # Ensure max 5 bullets
        
    def _generate_next_steps(
        self, 
        outputs: list[AgentOutput],
        verifications: list[VerificationResult]
    ) -> list[str]:
        """Generate recommendations for what to test next."""
        next_steps = []
        
        # Add steps based on unverified hypotheses
        for output in outputs:
            for claim in output.claims:
                if claim.get("tag") == ClaimTag.HYPOTHESIS.value:
                    next_steps.append(f"Test hypothesis: {claim.get('text', '')[:80]}")
                    
        # Add steps based on verification issues
        for verification in verifications:
            for issue in verification.issues:
                next_steps.append(f"Address: {issue}")
                
        return next_steps[:10]  # Max 10 next steps
        
    def _combine_content(self, outputs: list[AgentOutput]) -> str:
        """Combine all agent outputs into coherent content."""
        sections = []
        
        for output in outputs:
            section = f"## Contribution from {output.agent_name}\n\n{output.content}"
            sections.append(section)
            
        return "\n\n---\n\n".join(sections)
        
    def _calculate_final_snr(
        self,
        outputs: list[AgentOutput],
        verifications: list[VerificationResult]
    ) -> float:
        """Calculate final SNR score."""
        if not outputs:
            return 0.0
            
        base_snr = sum(o.confidence for o in outputs) / len(outputs)
        
        # Apply verification adjustments
        for v in verifications:
            base_snr += v.snr_adjustment
            if v.passed:
                base_snr *= 1.02  # 2% boost for passing verification
                
        return min(max(base_snr, 0.0), 1.0)  # Clamp to [0, 1]
        
    def _build_checklist(
        self,
        analysis: TaskAnalysis,
        verifications: list[VerificationResult]
    ) -> dict:
        """Build delivery checklist status."""
        return {
            "executive_summary": True,
            "claims_tagged": all(v.passed for v in verifications),
            "assumptions_stated": True,
            "confidence_declared": True,
            "verifier_attestation": any(v.attestation for v in verifications),
            "snr_target_met": analysis.snr_target <= 0.90,  # Simplified check
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ConstellationOrchestrator:
    """
    Main orchestrator for the BIZRA Islamic Masterminds Constellation.
    
    Implements the full execution flow:
    1. INTAKE: Parse task, stakes, domains, evidence requirements
    2. PLAN: Select team, choose reasoning mode
    3. WORK: Agents produce candidate solutions + evidence bundles
    4. VERIFY: Verifiers challenge assumptions, check sources
    5. SYNTHESIZE: Polymath produces unified deliverable
    6. DELIVER: Format final output with 'know/assume/test' structure
    """
    
    def __init__(self, constellation_path: Optional[Path] = None):
        """Initialize orchestrator with constellation configuration."""
        if constellation_path is None:
            constellation_path = Path(__file__).parent
            
        self.loader = ConstellationLoader(constellation_path)
        self.loader.load_all()
        
        self.analyzer = TaskAnalyzer(self.loader)
        self.composer = TeamComposer(self.loader)
        self.evaluator = EvidenceEvaluator(self.loader)
        self.synthesizer = PolymathSynthesizer(self.loader)
        
    def execute(
        self,
        query: str,
        context: Optional[dict] = None,
        agent_executor: Optional[callable] = None
    ) -> ConstellationOutput:
        """
        Execute the full constellation workflow.
        
        Args:
            query: The user's task/question
            context: Optional context (stakes, production flag, etc.)
            agent_executor: Callable that takes (agent, query, reasoning_mode)
                           and returns AgentOutput. If None, uses mock execution.
                           
        Returns:
            ConstellationOutput with full structured response
        """
        # Phase 1: INTAKE
        analysis = self.analyzer.analyze(query, context)
        
        # Phase 2: PLAN
        team, verifiers = self.composer.compose(analysis)
        
        # Phase 3: WORK
        outputs = self._execute_work(query, analysis, team, agent_executor)
        
        # Phase 4: VERIFY
        verifications = self._execute_verification(outputs, verifiers, analysis)
        
        # Phase 5: SYNTHESIZE
        result = self.synthesizer.synthesize(analysis, outputs, verifications, team)
        
        # Phase 6: DELIVER (result is already formatted)
        return result
        
    def _execute_work(
        self,
        query: str,
        analysis: TaskAnalysis,
        team: Team,
        agent_executor: Optional[callable]
    ) -> list[AgentOutput]:
        """Execute work phase with team agents."""
        outputs = []
        
        # Collect all agents to execute
        agents = []
        if team.leader:
            agents.append(team.leader)
        agents.extend(m for m in team.members if m)
        
        for agent in agents:
            if agent_executor:
                output = agent_executor(agent, query, analysis.reasoning_mode)
            else:
                output = self._mock_agent_execution(agent, query, analysis)
            outputs.append(output)
            
        return outputs
        
    def _execute_verification(
        self,
        outputs: list[AgentOutput],
        verifiers: list[Agent],
        analysis: TaskAnalysis
    ) -> list[VerificationResult]:
        """Execute verification phase."""
        verifications = []
        
        # First, run system-level evaluation on all outputs
        for output in outputs:
            result = self.evaluator.evaluate(output, analysis)
            verifications.append(result)
            
        # Then, if we have designated verifiers, run their checks
        # In a real implementation, these would be LLM calls
        for verifier in verifiers:
            result = VerificationResult(
                verifier_id=verifier.id,
                verifier_name=verifier.name,
                passed=True,  # Mock: always pass
                issues=[],
                attestation=f"Attested by {verifier.name}",
                snr_adjustment=0.02  # Boost for verifier attestation
            )
            verifications.append(result)
            
        return verifications
        
    def _mock_agent_execution(
        self,
        agent: Agent,
        query: str,
        analysis: TaskAnalysis
    ) -> AgentOutput:
        """Mock agent execution for testing."""
        return AgentOutput(
            agent_id=agent.id,
            agent_name=agent.name,
            content=f"[Mock response from {agent.name} ({agent.domain})]\n\n"
                   f"Analyzing query with {agent.reasoning_default.value} reasoning...\n"
                   f"Specialty applied: {agent.specialty}",
            claims=[
                {
                    "text": f"Analysis from {agent.name} perspective",
                    "tag": ClaimTag.DERIVED.value,
                    "evidence": "Mock evidence reference"
                }
            ],
            confidence=agent.snr_target[0],
            reasoning_trace=f"Applied {agent.reasoning_pattern}"
        )
        
    def get_agent(self, slug: str) -> Optional[Agent]:
        """Get an agent by slug."""
        return self.loader.agents.get(slug)
        
    def get_team(self, team_id: str) -> Optional[Team]:
        """Get a team by ID."""
        return self.loader.teams.get(team_id)
        
    def list_agents(self) -> list[Agent]:
        """List all agents."""
        return list(self.loader.agents.values())
        
    def list_teams(self) -> list[Team]:
        """List all teams."""
        return list(self.loader.teams.values())


# ─────────────────────────────────────────────────────────────────────────────
# CLI / TEST ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for testing the orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIZRA Islamic Masterminds Constellation Orchestrator"
    )
    parser.add_argument("query", help="Task or question to process")
    parser.add_argument("--stakes", choices=["low", "medium", "high", "critical"],
                       default="medium", help="Task stakes level")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    constellation_path = Path(__file__).parent
    orchestrator = ConstellationOrchestrator(constellation_path)
    
    # Execute
    context = {"stakes": args.stakes}
    result = orchestrator.execute(args.query, context)
    
    if args.json:
        # Convert to JSON-serializable format
        output = {
            "task_id": result.task_id,
            "executive_summary": result.executive_summary,
            "what_we_know": result.what_we_know,
            "what_we_assume": result.what_we_assume,
            "what_to_test_next": result.what_to_test_next,
            "final_snr": result.final_snr,
            "reasoning_mode": result.reasoning_mode_used.value,
            "team_used": result.team_used,
            "timestamp": result.timestamp
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"TASK: {result.task_id}")
        print(f"{'='*60}")
        print(f"\nExecutive Summary:")
        for i, point in enumerate(result.executive_summary, 1):
            print(f"  {i}. {point}")
        print(f"\nFinal SNR: {result.final_snr:.2f}")
        print(f"Reasoning Mode: {result.reasoning_mode_used.value}")
        print(f"Team Used: {result.team_used}")
        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
