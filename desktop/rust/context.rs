// bizra-agent/src/context.rs
// ============================================================
// Context Assembly — building rich context from memory
// ============================================================
// The Scholar's primary tool. Takes a user message and
// assembles everything the agent team needs to respond
// with genuine personalization.
//
// This is where "My AI knows me" becomes tangible.
// The context assembler pulls from:
//   - User profile traits (who they are)
//   - Recent insights (what we've learned)
//   - Fragment patterns (behavioral cues)
//   - Temporal context (conversation history shape)
//   - Current إحسان (system health)
// ============================================================

use crate::types::*;
use bizra_memory::{
    MemoryPipeline, Confidence,
};
use bizra_hooks::IhsanScore;

// ============================================================
// CONTEXT ASSEMBLY CONFIG
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct ContextConfig {
    /// Maximum profile traits to include
    pub max_traits: usize,
    /// Maximum insights to include
    pub max_insights: usize,
    /// Minimum confidence for trait inclusion
    pub min_trait_confidence: Confidence,
    /// Minimum إحسان to assemble context
    pub min_ihsan: IhsanScore,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_traits: 12,
            max_insights: 6,
            min_trait_confidence: Confidence::MEDIUM,
            min_ihsan: IhsanScore::new(9500),
        }
    }
}

// ============================================================
// CONTEXT ASSEMBLER
// ============================================================

pub struct ContextAssembler {
    config: ContextConfig,
    assemblies_count: u64,
    avg_richness: f32,
}

impl ContextAssembler {
    pub fn new() -> Self {
        Self::with_config(ContextConfig::default())
    }

    pub fn with_config(config: ContextConfig) -> Self {
        Self {
            config,
            assemblies_count: 0,
            avg_richness: 0.0,
        }
    }

    /// Assemble full context for a message
    /// Pulls from memory pipeline to build rich agent context
    pub fn assemble(
        &mut self,
        message: &Message,
        pipeline: &mut MemoryPipeline,
        current_ihsan: IhsanScore,
    ) -> AgentContext {
        let mut ctx = AgentContext::empty(message.id, message.timestamp);
        ctx.current_ihsan = current_ihsan;

        // إحسان gate: degraded system gets empty context (safe fallback)
        if current_ihsan.raw() < self.config.min_ihsan.raw() {
            return ctx;
        }

        // 1. Pull profile traits
        let traits = pipeline.query_profile();
        for (key, value, confidence) in traits.iter().take(self.config.max_traits) {
            if confidence.raw() >= self.config.min_trait_confidence.raw() {
                ctx.add_trait(key, value, *confidence);
            }
        }

        // 2. Pull insights
        let insights = pipeline.query_insights();
        for insight in insights.iter().take(self.config.max_insights) {
            ctx.add_insight(insight.content.as_str());
        }

        // 3. Get knows-me score
        ctx.knows_me_score = pipeline.knows_me_score();

        // 4. Get session depth
        let health = pipeline.health();
        ctx.session_history_depth = health.sessions_tracked;

        // Update assembly stats
        self.assemblies_count += 1;
        let richness = ctx.richness();
        self.avg_richness = self.avg_richness
            + (richness - self.avg_richness) / self.assemblies_count as f32;

        ctx
    }

    /// Quick context for internal agent-to-agent messages
    /// Lighter weight — only critical traits
    pub fn assemble_light(
        &mut self,
        message: &Message,
        pipeline: &mut MemoryPipeline,
        current_ihsan: IhsanScore,
    ) -> AgentContext {
        let mut ctx = AgentContext::empty(message.id, message.timestamp);
        ctx.current_ihsan = current_ihsan;

        if current_ihsan.raw() < self.config.min_ihsan.raw() {
            return ctx;
        }

        // Only top traits for speed
        let traits = pipeline.query_profile();
        for (key, value, confidence) in traits.iter().take(4) {
            if confidence.raw() >= Confidence::HIGH.raw() {
                ctx.add_trait(key, value, *confidence);
            }
        }

        ctx.knows_me_score = pipeline.knows_me_score();
        ctx
    }

    pub fn assemblies_count(&self) -> u64 {
        self.assemblies_count
    }

    pub fn avg_richness(&self) -> f32 {
        self.avg_richness
    }
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// INTENT CLASSIFIER — Navigator's primary tool
// ============================================================
// Classifies user intent to route tasks to the right agents.
// MVP: keyword-based heuristics
// Production: LLM-powered classification via FFI

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UserIntent {
    /// Asking a question
    Question = 0,
    /// Requesting content creation
    Create = 1,
    /// Requesting code / technical help
    Code = 2,
    /// Conversational / social
    Chat = 3,
    /// Requesting analysis or research
    Analyze = 4,
    /// Requesting planning or strategy
    Plan = 5,
    /// Requesting modification of something
    Modify = 6,
    /// Unclear intent — need clarification
    Ambiguous = 7,
}

impl UserIntent {
    /// Which agent roles should be consulted for this intent?
    pub fn required_agents(&self) -> &'static [AgentRole] {
        match self {
            Self::Question => &[AgentRole::Scholar, AgentRole::Diplomat],
            Self::Create => &[AgentRole::Artisan, AgentRole::Diplomat, AgentRole::Mentor],
            Self::Code => &[AgentRole::Artisan, AgentRole::Scholar],
            Self::Chat => &[AgentRole::Diplomat, AgentRole::Oracle],
            Self::Analyze => &[AgentRole::Scholar, AgentRole::Oracle],
            Self::Plan => &[AgentRole::Oracle, AgentRole::Scholar, AgentRole::Navigator],
            Self::Modify => &[AgentRole::Artisan, AgentRole::Scholar],
            Self::Ambiguous => &[AgentRole::Navigator, AgentRole::Diplomat],
        }
    }

    /// What task kinds does this intent generate?
    pub fn task_pipeline(&self) -> &'static [TaskKind] {
        match self {
            Self::Question => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::AdaptStyle],
            Self::Create => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::SafetyCheck, TaskKind::AdaptStyle],
            Self::Code => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::SafetyCheck],
            Self::Chat => &[TaskKind::GenerateResponse, TaskKind::AdaptStyle, TaskKind::ProactiveSuggest],
            Self::Analyze => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::AdaptStyle],
            Self::Plan => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::ProactiveSuggest],
            Self::Modify => &[TaskKind::RetrieveContext, TaskKind::GenerateResponse, TaskKind::SafetyCheck],
            Self::Ambiguous => &[TaskKind::ClassifyIntent],
        }
    }
}

/// Simple intent classifier
/// MVP: keyword matching
/// Production: LLM via FFI
pub struct IntentClassifier;

impl IntentClassifier {
    /// Classify user intent from message content
    pub fn classify(content: &str) -> (UserIntent, Confidence) {
        let lower = content.to_lowercase();

        // Code indicators
        if lower.contains("code") || lower.contains("function")
            || lower.contains("implement") || lower.contains("debug")
            || lower.contains("compile") || lower.contains("crate")
            || lower.contains("script") || lower.contains("program")
        {
            return (UserIntent::Code, Confidence::HIGH);
        }

        // Creation indicators
        if lower.contains("create") || lower.contains("build")
            || lower.contains("make") || lower.contains("generate")
            || lower.contains("write") || lower.contains("design")
            || lower.contains("draft")
        {
            return (UserIntent::Create, Confidence::HIGH);
        }

        // Analysis indicators
        if lower.contains("analyze") || lower.contains("compare")
            || lower.contains("evaluate") || lower.contains("assess")
            || lower.contains("review") || lower.contains("examine")
        {
            return (UserIntent::Analyze, Confidence::MEDIUM);
        }

        // Planning indicators
        if lower.contains("plan") || lower.contains("strategy")
            || lower.contains("roadmap") || lower.contains("schedule")
            || lower.contains("next steps")
        {
            return (UserIntent::Plan, Confidence::MEDIUM);
        }

        // Modification indicators
        if lower.contains("fix") || lower.contains("change")
            || lower.contains("update") || lower.contains("modify")
            || lower.contains("edit") || lower.contains("refactor")
        {
            return (UserIntent::Modify, Confidence::MEDIUM);
        }

        // Question indicators
        if lower.contains('?') || lower.starts_with("what")
            || lower.starts_with("how") || lower.starts_with("why")
            || lower.starts_with("when") || lower.starts_with("where")
            || lower.starts_with("who") || lower.starts_with("can")
            || lower.starts_with("does") || lower.starts_with("is")
        {
            return (UserIntent::Question, Confidence::HIGH);
        }

        // Chat indicators
        if lower.starts_with("hi") || lower.starts_with("hello")
            || lower.starts_with("hey") || lower.contains("thanks")
            || lower.contains("thank you") || lower.len() < 20
        {
            return (UserIntent::Chat, Confidence::MEDIUM);
        }

        // Default: ambiguous
        (UserIntent::Ambiguous, Confidence::LOW)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_code_intent() {
        let (intent, conf) = IntentClassifier::classify("Help me implement a hash map in Rust");
        assert_eq!(intent, UserIntent::Code);
        assert!(conf.raw() >= Confidence::HIGH.raw());
    }

    #[test]
    fn classify_create_intent() {
        let (intent, _) = IntentClassifier::classify("Create a presentation about AI ethics");
        assert_eq!(intent, UserIntent::Create);
    }

    #[test]
    fn classify_question_intent() {
        let (intent, _) = IntentClassifier::classify("What is the difference between TCP and UDP?");
        assert_eq!(intent, UserIntent::Question);
    }

    #[test]
    fn classify_analysis_intent() {
        let (intent, _) = IntentClassifier::classify("Analyze the performance of this algorithm");
        assert_eq!(intent, UserIntent::Analyze);
    }

    #[test]
    fn classify_plan_intent() {
        let (intent, _) = IntentClassifier::classify("Help me plan the roadmap for Q3");
        assert_eq!(intent, UserIntent::Plan);
    }

    #[test]
    fn classify_chat_intent() {
        let (intent, _) = IntentClassifier::classify("Hello!");
        assert_eq!(intent, UserIntent::Chat);
    }

    #[test]
    fn classify_ambiguous_intent() {
        let (intent, conf) = IntentClassifier::classify("The weather is nice today and I feel productive");
        assert_eq!(intent, UserIntent::Ambiguous);
        assert!(conf.raw() <= Confidence::LOW.raw());
    }

    #[test]
    fn code_intent_requires_artisan() {
        let intent = UserIntent::Code;
        let agents = intent.required_agents();
        assert!(agents.contains(&AgentRole::Artisan));
    }

    #[test]
    fn every_intent_has_agents() {
        for role_val in 0..=7 {
            let intent = match role_val {
                0 => UserIntent::Question,
                1 => UserIntent::Create,
                2 => UserIntent::Code,
                3 => UserIntent::Chat,
                4 => UserIntent::Analyze,
                5 => UserIntent::Plan,
                6 => UserIntent::Modify,
                _ => UserIntent::Ambiguous,
            };
            assert!(!intent.required_agents().is_empty());
            assert!(!intent.task_pipeline().is_empty());
        }
    }

    #[test]
    fn context_assembler_empty_on_degraded_ihsan() {
        let mut assembler = ContextAssembler::new();
        let mut pipeline = MemoryPipeline::new();
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Test",
            1000,
            IhsanScore::new(9900),
        );

        let ctx = assembler.assemble(&msg, &mut pipeline, IhsanScore::new(9000));
        assert!(ctx.traits.is_empty());
        assert!(ctx.insight_summaries.is_empty());
    }

    #[test]
    fn context_assembler_with_populated_memory() {
        use bizra_memory::{MemoryFragment, FragmentId, FragmentKind, Confidence as MemConf};
        use bizra_hooks::ComponentId;

        let mut assembler = ContextAssembler::new();
        let mut pipeline = MemoryPipeline::new();

        // Populate memory
        for i in 1..=5 {
            let frag = MemoryFragment::new(
                FragmentId::new(1, i),
                FragmentKind::Style,
                &format!("Style preference {}", i),
                MemConf::HIGH,
                ComponentId::new("test", "1.0"),
                1000,
                IhsanScore::new(9900),
            );
            pipeline.ingest(frag, 1000).unwrap();
        }
        pipeline.force_synthesis(2000);

        let msg = Message::inbound(
            MessageId::new(1, 1),
            "How should I structure my project?",
            3000,
            IhsanScore::new(9900),
        );

        let ctx = assembler.assemble(&msg, &mut pipeline, IhsanScore::new(9900));
        assert!(ctx.knows_me_score > 0.0);
        assert_eq!(assembler.assemblies_count(), 1);
    }
}
