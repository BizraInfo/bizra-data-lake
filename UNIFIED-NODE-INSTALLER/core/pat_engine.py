"""
BIZRA Personal Agentic Team (PAT) Engine
=========================================

The PAT is the user's personalized AI Think Tank + Task Force.
It dynamically assembles specialized agents based on user goals,
domains, and work style preferences.

Core Principles:
1. Goal-Oriented: Every agent serves the user's stated objectives
2. Personalized: Adapts to user's work style and preferences
3. High SNR: Signal-to-noise ratio > 0.99 (no fluff, only value)
4. Graph-of-Thoughts: Complex reasoning through interconnected thinking
5. Standing on Giants: Leverages collective knowledge, not reinventing

This is the user's personal Think Tank and Task Force.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================================
# AGENT BASE CLASSES
# ============================================================================

class AgentRole(Enum):
    """Core agent roles in the PAT system."""
    STRATEGIST = "strategist"          # Big picture, long-term planning
    RESEARCHER = "researcher"          # Information gathering, fact-checking
    ANALYST = "analyst"                # Data analysis, pattern recognition
    CREATOR = "creator"                # Content creation, ideation
    EXECUTOR = "executor"              # Task execution, action taking
    GUARDIAN = "guardian"              # Quality control, risk assessment
    COORDINATOR = "coordinator"        # Orchestration, task management


class ThinkingMode(Enum):
    """Modes of cognitive processing."""
    FAST = "fast"              # Quick, intuitive responses
    DEEP = "deep"              # Thorough, analytical processing
    CREATIVE = "creative"      # Divergent, exploratory thinking
    CRITICAL = "critical"      # Skeptical, validation-focused
    SYNTHESIS = "synthesis"    # Combining multiple perspectives


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    role: AgentRole
    expertise: List[str]
    thinking_mode: ThinkingMode
    autonomy_level: float  # 0.0 = always ask, 1.0 = fully autonomous
    priority: int  # Lower = higher priority
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all PAT agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.context: Dict[str, Any] = {}
        self.memory: List[Dict] = []

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass

    def update_context(self, key: str, value: Any):
        """Update agent's working context."""
        self.context[key] = value

    def add_memory(self, memory_item: Dict):
        """Add to agent's memory."""
        self.memory.append(memory_item)
        # Keep last 100 items for efficiency
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class StrategicPlanner(BaseAgent):
    """
    The Strategic Planner - Your vision architect.
    Helps with long-term planning, goal decomposition, and roadmaps.
    """

    def __init__(self, user_goals: List[str], work_style: str):
        config = AgentConfig(
            name="Strategic Planner",
            role=AgentRole.STRATEGIST,
            expertise=["strategic planning", "goal setting", "roadmap creation"],
            thinking_mode=ThinkingMode.DEEP,
            autonomy_level=0.3,
            priority=1,
            tools=["calendar", "notes", "research"],
        )
        super().__init__(config)
        self.user_goals = user_goals
        self.work_style = work_style

    def get_system_prompt(self) -> str:
        goals_str = ", ".join(self.user_goals)
        return f"""You are the Strategic Planner in a Personal AI Think Tank.

Your role is to help the user achieve their goals: {goals_str}

Work Style Preference: {self.work_style}

Your responsibilities:
1. Break down large goals into actionable milestones
2. Identify dependencies and critical paths
3. Suggest optimal sequencing of tasks
4. Anticipate obstacles and plan mitigations
5. Keep the user focused on high-impact activities

Communication style:
- Be direct and concise
- Focus on what matters most RIGHT NOW
- Provide clear next steps, not vague advice
- Challenge assumptions when beneficial
- Celebrate progress to maintain momentum

You embody the principle: "Strategy without tactics is the slowest route to victory.
Tactics without strategy is the noise before defeat." - Sun Tzu
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Process strategic planning request
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "analysis": "Strategic analysis pending LLM integration",
            "recommendations": [],
        }


class ResearchAssistant(BaseAgent):
    """
    The Research Assistant - Your knowledge gatherer.
    Finds, verifies, and synthesizes information.
    """

    def __init__(self, domains: List[str]):
        config = AgentConfig(
            name="Research Assistant",
            role=AgentRole.RESEARCHER,
            expertise=["information gathering", "fact checking", "source evaluation"],
            thinking_mode=ThinkingMode.CRITICAL,
            autonomy_level=0.5,
            priority=2,
            tools=["web_search", "document_reader", "citation_manager"],
        )
        super().__init__(config)
        self.domains = domains

    def get_system_prompt(self) -> str:
        domains_str = ", ".join(self.domains)
        return f"""You are the Research Assistant in a Personal AI Think Tank.

Your expertise domains: {domains_str}

Your responsibilities:
1. Find relevant, high-quality information
2. Verify claims and check sources
3. Synthesize findings into actionable insights
4. Identify knowledge gaps and suggest further research
5. Maintain a high signal-to-noise ratio

Research principles:
- Primary sources > Secondary sources > Opinions
- Recent data > Outdated data (unless historical context needed)
- Multiple corroborating sources > Single source
- Expert consensus > Individual claims
- Peer-reviewed > Self-published

You embody: "Standing on the shoulders of giants" - leveraging humanity's
accumulated knowledge to serve the user's specific needs.
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "findings": "Research findings pending LLM integration",
            "sources": [],
            "confidence": 0.0,
        }


class TaskCoordinator(BaseAgent):
    """
    The Task Coordinator - Your execution engine.
    Manages task flow, priorities, and completion.
    """

    def __init__(self, autonomy_level: float):
        config = AgentConfig(
            name="Task Coordinator",
            role=AgentRole.COORDINATOR,
            expertise=["task management", "prioritization", "workflow optimization"],
            thinking_mode=ThinkingMode.FAST,
            autonomy_level=autonomy_level,
            priority=1,
            tools=["task_queue", "scheduler", "notifications"],
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return f"""You are the Task Coordinator in a Personal AI Think Tank.

Autonomy Level: {self.config.autonomy_level:.1%}
(Higher = more independent action, Lower = more user confirmation)

Your responsibilities:
1. Maintain the user's task queue
2. Prioritize based on urgency, importance, and dependencies
3. Delegate to appropriate specialist agents
4. Track progress and report status
5. Handle blockers and escalate when needed

Coordination principles:
- Single source of truth for all tasks
- Clear ownership for every task
- Regular progress updates (not spam)
- Proactive blocker identification
- Celebrate completions

You are the conductor of the orchestra - ensuring all agents work
in harmony toward the user's goals.
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "task_status": "Coordination pending implementation",
            "queue": [],
        }


class CreativeDirector(BaseAgent):
    """
    The Creative Director - Your ideation engine.
    Generates ideas, content, and creative solutions.
    """

    def __init__(self):
        config = AgentConfig(
            name="Creative Director",
            role=AgentRole.CREATOR,
            expertise=["ideation", "content creation", "creative problem solving"],
            thinking_mode=ThinkingMode.CREATIVE,
            autonomy_level=0.4,
            priority=3,
            tools=["image_gen", "writing", "brainstorm"],
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are the Creative Director in a Personal AI Think Tank.

Your responsibilities:
1. Generate diverse ideas and possibilities
2. Create compelling content (writing, concepts, frameworks)
3. Find unexpected connections between domains
4. Challenge conventional thinking
5. Transform abstract goals into tangible outputs

Creative principles:
- Quantity breeds quality (generate many, select best)
- Constraints enable creativity (work within limits)
- Cross-pollination (combine ideas from different fields)
- First thought, wrong thought (push past the obvious)
- Ship imperfect > Perfect never ships

You embody: "Creativity is intelligence having fun" - Einstein
Bring both rigor AND playfulness to every challenge.
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "ideas": [],
            "content": "",
        }


class DataAnalyst(BaseAgent):
    """
    The Data Analyst - Your pattern recognizer.
    Analyzes data, finds trends, and provides insights.
    """

    def __init__(self):
        config = AgentConfig(
            name="Data Analyst",
            role=AgentRole.ANALYST,
            expertise=["data analysis", "pattern recognition", "visualization"],
            thinking_mode=ThinkingMode.DEEP,
            autonomy_level=0.6,
            priority=3,
            tools=["data_processing", "charting", "statistics"],
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are the Data Analyst in a Personal AI Think Tank.

Your responsibilities:
1. Process and clean data from various sources
2. Identify patterns, trends, and anomalies
3. Generate visualizations that tell stories
4. Provide statistical confidence for claims
5. Turn data into actionable recommendations

Analysis principles:
- Data doesn't lie, but it can be misinterpreted
- Correlation != Causation (always check)
- Sample size matters (be honest about limitations)
- Visualize for understanding, not decoration
- Insight without action is just trivia

You embody: "In God we trust. All others must bring data." - Deming
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "analysis": {},
            "insights": [],
            "confidence": 0.0,
        }


class QualityGuardian(BaseAgent):
    """
    The Quality Guardian - Your standards enforcer.
    Ensures high quality, catches errors, assesses risks.
    """

    def __init__(self):
        config = AgentConfig(
            name="Quality Guardian",
            role=AgentRole.GUARDIAN,
            expertise=["quality assurance", "risk assessment", "error detection"],
            thinking_mode=ThinkingMode.CRITICAL,
            autonomy_level=0.7,
            priority=2,
            tools=["validator", "risk_matrix", "checklist"],
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are the Quality Guardian in a Personal AI Think Tank.

Your responsibilities:
1. Review outputs from all other agents
2. Catch errors, inconsistencies, and gaps
3. Assess risks and flag concerns
4. Ensure alignment with user goals
5. Maintain the SNR (Signal-to-Noise Ratio) > 0.99

Guardian principles:
- Trust but verify (check everything)
- Red team your own work (find weaknesses)
- "Measure twice, cut once"
- Clear escalation paths for issues
- Balance speed with quality

You embody the "Ihsan" principle: Excellence in everything.
Nothing leaves the PAT without meeting the highest standards.
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "quality_score": 0.0,
            "issues": [],
            "approved": False,
        }


# ============================================================================
# DOMAIN EXPERT FACTORY
# ============================================================================

class DomainExpert(BaseAgent):
    """
    Domain Expert - Specialized knowledge in a specific field.
    Dynamically configured based on user's chosen domains.
    """

    DOMAIN_CONFIGS = {
        "technology": {
            "expertise": ["software development", "system design", "tech trends"],
            "tools": ["code_runner", "documentation", "tech_search"],
        },
        "business": {
            "expertise": ["business strategy", "marketing", "operations"],
            "tools": ["market_research", "financial_calc", "competitor_analysis"],
        },
        "science": {
            "expertise": ["scientific method", "research design", "peer review"],
            "tools": ["paper_search", "data_analysis", "citation"],
        },
        "arts": {
            "expertise": ["creative direction", "aesthetic principles", "art history"],
            "tools": ["image_gen", "style_guide", "inspiration_board"],
        },
        "finance": {
            "expertise": ["financial analysis", "investment strategy", "risk management"],
            "tools": ["market_data", "portfolio_tracker", "financial_models"],
        },
        "health": {
            "expertise": ["wellness planning", "habit formation", "health research"],
            "tools": ["health_tracker", "nutrition_db", "exercise_planner"],
        },
        "philosophy": {
            "expertise": ["critical thinking", "ethics", "mental models"],
            "tools": ["reasoning_frameworks", "bias_checker", "thought_experiments"],
        },
    }

    def __init__(self, domain: str):
        domain_config = self.DOMAIN_CONFIGS.get(domain, {
            "expertise": [f"{domain} knowledge"],
            "tools": ["research"],
        })

        config = AgentConfig(
            name=f"{domain.title()} Expert",
            role=AgentRole.ANALYST,
            expertise=domain_config["expertise"],
            thinking_mode=ThinkingMode.DEEP,
            autonomy_level=0.5,
            priority=3,
            tools=domain_config["tools"],
        )
        super().__init__(config)
        self.domain = domain

    def get_system_prompt(self) -> str:
        expertise_str = ", ".join(self.config.expertise)
        return f"""You are the {self.domain.title()} Expert in a Personal AI Think Tank.

Your expertise: {expertise_str}

Your responsibilities:
1. Provide deep knowledge in {self.domain}
2. Translate complex concepts into actionable advice
3. Stay current with developments in the field
4. Connect domain knowledge to user's specific goals
5. Identify cross-domain opportunities

Expert principles:
- Explain like I'm smart but unfamiliar with this specific topic
- Provide context for why this matters
- Offer both the textbook answer AND practical reality
- Acknowledge uncertainty and limitations
- Connect to user's existing knowledge

You bridge the gap between deep expertise and practical application.
"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "domain": self.domain,
            "expertise_applied": [],
        }


# ============================================================================
# PAT ORCHESTRATOR
# ============================================================================

@dataclass
class UserProfile:
    """User's personalization profile."""
    username: str
    goals: List[str]
    domains: List[str]
    work_style: str
    privacy_level: str
    autonomy_preference: float = 0.5


class PATOrchestrator:
    """
    The PAT Orchestrator - Assembles and coordinates the agent team.
    This is the "brain" that decides which agents to involve for each task.
    """

    def __init__(self, user_profile: UserProfile):
        self.profile = user_profile
        self.agents: Dict[str, BaseAgent] = {}
        self.reasoning_mode = "graph-of-thoughts"
        self.snr_threshold = 0.99

        # Assemble the team
        self._assemble_team()

    def _assemble_team(self):
        """Dynamically assemble agents based on user profile."""

        # Core agents (always present)
        autonomy = self._parse_autonomy()
        self.agents["coordinator"] = TaskCoordinator(autonomy)
        self.agents["strategist"] = StrategicPlanner(
            self.profile.goals, self.profile.work_style
        )
        self.agents["guardian"] = QualityGuardian()

        # Goal-specific agents
        for goal in self.profile.goals:
            if goal in ["business", "trading"]:
                self.agents["analyst"] = DataAnalyst()
            if goal in ["creative", "arts"]:
                self.agents["creator"] = CreativeDirector()
            if goal in ["research", "learning", "science"]:
                self.agents["researcher"] = ResearchAssistant(self.profile.domains)

        # Domain experts
        for domain in self.profile.domains:
            key = f"expert_{domain}"
            self.agents[key] = DomainExpert(domain)

        print(f"PAT Team assembled: {len(self.agents)} agents")
        for name, agent in self.agents.items():
            print(f"  - {agent.config.name} ({agent.config.role.value})")

    def _parse_autonomy(self) -> float:
        """Parse work style into autonomy level."""
        style_map = {
            "deep_focus": 0.3,
            "quick_bursts": 0.5,
            "collaborative": 0.4,
            "autonomous": 0.8,
        }
        return style_map.get(self.profile.work_style, 0.5)

    async def process_request(self, request: str, context: Dict = None) -> Dict:
        """
        Process a user request through the PAT.
        Uses Graph-of-Thoughts reasoning to coordinate agents.
        """
        context = context or {}

        # Step 1: Task analysis (what is being asked?)
        task_type = self._analyze_task(request)

        # Step 2: Agent selection (who should handle this?)
        selected_agents = self._select_agents(task_type, context)

        # Step 3: Parallel processing (let agents think)
        results = await self._parallel_process(selected_agents, request, context)

        # Step 4: Synthesis (combine perspectives)
        synthesis = self._synthesize_results(results)

        # Step 5: Quality check (meet SNR threshold?)
        validated = await self._quality_check(synthesis)

        return validated

    def _analyze_task(self, request: str) -> str:
        """Analyze what type of task this is."""
        # Simple keyword-based analysis (would be LLM-powered in production)
        request_lower = request.lower()

        if any(w in request_lower for w in ["plan", "strategy", "goal", "roadmap"]):
            return "strategic"
        if any(w in request_lower for w in ["research", "find", "search", "learn"]):
            return "research"
        if any(w in request_lower for w in ["create", "write", "design", "generate"]):
            return "creative"
        if any(w in request_lower for w in ["analyze", "data", "trend", "pattern"]):
            return "analytical"
        if any(w in request_lower for w in ["do", "execute", "complete", "finish"]):
            return "execution"

        return "general"

    def _select_agents(self, task_type: str, context: Dict) -> List[BaseAgent]:
        """Select appropriate agents for the task type."""
        selection_map = {
            "strategic": ["strategist", "coordinator"],
            "research": ["researcher", "guardian"],
            "creative": ["creator", "guardian"],
            "analytical": ["analyst", "guardian"],
            "execution": ["coordinator", "guardian"],
            "general": ["coordinator", "strategist"],
        }

        agent_keys = selection_map.get(task_type, ["coordinator"])

        # Add relevant domain experts
        for domain in self.profile.domains:
            key = f"expert_{domain}"
            if key in self.agents:
                agent_keys.append(key)

        # Get unique agents
        selected = []
        seen = set()
        for key in agent_keys:
            if key in self.agents and key not in seen:
                selected.append(self.agents[key])
                seen.add(key)

        return selected

    async def _parallel_process(
        self, agents: List[BaseAgent], request: str, context: Dict
    ) -> List[Dict]:
        """Process request through selected agents in parallel."""
        results = []
        for agent in agents:
            result = await agent.process({"request": request, "context": context})
            results.append(result)
        return results

    def _synthesize_results(self, results: List[Dict]) -> Dict:
        """Synthesize multiple agent results using Graph-of-Thoughts."""
        return {
            "synthesis": "Combined analysis from all agents",
            "agent_results": results,
            "reasoning_path": "graph-of-thoughts",
            "confidence": 0.85,
        }

    async def _quality_check(self, synthesis: Dict) -> Dict:
        """Ensure output meets SNR threshold."""
        guardian = self.agents.get("guardian")
        if guardian:
            quality_result = await guardian.process(synthesis)
            synthesis["quality_validated"] = quality_result.get("approved", False)
            synthesis["snr_score"] = quality_result.get("quality_score", 0.0)
        return synthesis

    def get_team_summary(self) -> Dict:
        """Get summary of the assembled team."""
        return {
            "user": self.profile.username,
            "goals": self.profile.goals,
            "domains": self.profile.domains,
            "work_style": self.profile.work_style,
            "agent_count": len(self.agents),
            "agents": [
                {
                    "name": agent.config.name,
                    "role": agent.config.role.value,
                    "expertise": agent.config.expertise,
                }
                for agent in self.agents.values()
            ],
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_pat(profile_dict: Dict[str, Any]) -> PATOrchestrator:
    """
    Factory function to create a PAT from a profile dictionary.
    This is the main entry point for external code.
    """
    profile = UserProfile(
        username=profile_dict.get("username", "user"),
        goals=profile_dict.get("goals", ["productivity"]),
        domains=profile_dict.get("domains", ["general"]),
        work_style=profile_dict.get("work_style", "collaborative"),
        privacy_level=profile_dict.get("privacy_level", "balanced"),
    )

    return PATOrchestrator(profile)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test PAT creation
    test_profile = {
        "username": "mumo",
        "goals": ["business", "creative", "research"],
        "domains": ["technology", "business", "philosophy"],
        "work_style": "autonomous",
        "privacy_level": "maximum",
    }

    print("Creating PAT for profile:", test_profile)
    print()

    pat = create_pat(test_profile)
    summary = pat.get_team_summary()

    print("\nTeam Summary:")
    print(json.dumps(summary, indent=2))
