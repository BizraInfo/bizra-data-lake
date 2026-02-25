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
use bizra_hooks::IhsanScore;
use bizra_memory::{Confidence, MemoryPipeline};

// Shim: the omega MemoryPipeline does not have query_profile, query_insights,
// knows_me_score, or health methods. The context assembler uses a simplified
// path through the pipeline's existing API.

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
            min_trait_confidence: Confidence::inferred(0),
            min_ihsan: IhsanScore::from_raw(9500),
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
    /// Pulls from memory pipeline to build rich agent context.
    ///
    /// The omega MemoryPipeline exposes query_facts(AtomKind, now) and
    /// query_insights(Option<SynthesisMethod>). We use those to populate context.
    pub fn assemble(
        &mut self,
        message: &Message,
        pipeline: &mut MemoryPipeline,
        current_ihsan: IhsanScore,
    ) -> AgentContext {
        use bizra_memory::AtomKind;

        let mut ctx = AgentContext::empty(message.id, message.timestamp);
        ctx.current_ihsan = current_ihsan;

        // إحسان gate: degraded system gets empty context (safe fallback)
        if current_ihsan.raw() < self.config.min_ihsan.raw() {
            return ctx;
        }

        // 1. Pull profile-like traits from memory pipeline facts
        let now = message.timestamp;
        let kinds = [
            AtomKind::Fact,
            AtomKind::Preference,
            AtomKind::Expertise,
            AtomKind::Pattern,
        ];
        let mut trait_count = 0;
        for kind in &kinds {
            if trait_count >= self.config.max_traits {
                break;
            }
            let facts = pipeline.query_facts(*kind, now);
            for (text, conf) in facts.iter().take(self.config.max_traits - trait_count) {
                if *conf >= self.config.min_trait_confidence.base {
                    ctx.add_trait(&format!("{kind:?}"), text, Confidence::new(*conf, now));
                    trait_count += 1;
                }
            }
        }

        // 2. Pull insights
        let insights = pipeline.query_insights(None);
        for (text, _conf) in insights.iter().take(self.config.max_insights) {
            ctx.add_insight(text);
        }

        // 3. Derive knows-me score from profile snapshot
        let profile = pipeline.profile();
        ctx.knows_me_score = profile.completeness();

        // 4. Session depth from profile
        ctx.session_history_depth = profile.section_count() as u16;

        // Update assembly stats
        self.assemblies_count += 1;
        let richness = ctx.richness();
        self.avg_richness =
            self.avg_richness + (richness - self.avg_richness) / self.assemblies_count as f32;

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
        use bizra_memory::AtomKind;

        let mut ctx = AgentContext::empty(message.id, message.timestamp);
        ctx.current_ihsan = current_ihsan;

        if current_ihsan.raw() < self.config.min_ihsan.raw() {
            return ctx;
        }

        // Only top facts for speed
        let now = message.timestamp;
        let high_conf = Confidence::stated(0).base;
        let facts = pipeline.query_facts(AtomKind::Fact, now);
        for (text, conf) in facts.iter().take(4) {
            if *conf >= high_conf {
                ctx.add_trait("fact", text, Confidence::new(*conf, now));
            }
        }

        let profile = pipeline.profile();
        ctx.knows_me_score = profile.completeness();
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
            Self::Question => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::AdaptStyle,
            ],
            Self::Create => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::SafetyCheck,
                TaskKind::AdaptStyle,
            ],
            Self::Code => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::SafetyCheck,
            ],
            Self::Chat => &[
                TaskKind::GenerateResponse,
                TaskKind::AdaptStyle,
                TaskKind::ProactiveSuggest,
            ],
            Self::Analyze => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::AdaptStyle,
            ],
            Self::Plan => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::ProactiveSuggest,
            ],
            Self::Modify => &[
                TaskKind::RetrieveContext,
                TaskKind::GenerateResponse,
                TaskKind::SafetyCheck,
            ],
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
        if lower.contains("code")
            || lower.contains("function")
            || lower.contains("implement")
            || lower.contains("debug")
            || lower.contains("compile")
            || lower.contains("crate")
            || lower.contains("script")
            || lower.contains("program")
        {
            return (UserIntent::Code, Confidence::stated(0));
        }

        // Creation indicators
        if lower.contains("create")
            || lower.contains("build")
            || lower.contains("make")
            || lower.contains("generate")
            || lower.contains("write")
            || lower.contains("design")
            || lower.contains("draft")
        {
            return (UserIntent::Create, Confidence::stated(0));
        }

        // Analysis indicators
        if lower.contains("analyze")
            || lower.contains("compare")
            || lower.contains("evaluate")
            || lower.contains("assess")
            || lower.contains("review")
            || lower.contains("examine")
        {
            return (UserIntent::Analyze, Confidence::inferred(0));
        }

        // Planning indicators
        if lower.contains("plan")
            || lower.contains("strategy")
            || lower.contains("roadmap")
            || lower.contains("schedule")
            || lower.contains("next steps")
        {
            return (UserIntent::Plan, Confidence::inferred(0));
        }

        // Modification indicators
        if lower.contains("fix")
            || lower.contains("change")
            || lower.contains("update")
            || lower.contains("modify")
            || lower.contains("edit")
            || lower.contains("refactor")
        {
            return (UserIntent::Modify, Confidence::inferred(0));
        }

        // Question indicators
        if lower.contains('?')
            || lower.starts_with("what")
            || lower.starts_with("how")
            || lower.starts_with("why")
            || lower.starts_with("when")
            || lower.starts_with("where")
            || lower.starts_with("who")
            || lower.starts_with("can")
            || lower.starts_with("does")
            || lower.starts_with("is")
        {
            return (UserIntent::Question, Confidence::stated(0));
        }

        // Chat indicators
        if lower.starts_with("hi")
            || lower.starts_with("hello")
            || lower.starts_with("hey")
            || lower.contains("thanks")
            || lower.contains("thank you")
            || lower.len() < 20
        {
            return (UserIntent::Chat, Confidence::inferred(0));
        }

        // Default: ambiguous
        (UserIntent::Ambiguous, Confidence::speculative(0))
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
        assert!(conf.base >= Confidence::stated(0).base);
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
        let (intent, conf) =
            IntentClassifier::classify("The weather is nice today and I feel productive");
        assert_eq!(intent, UserIntent::Ambiguous);
        assert!(conf.base <= Confidence::speculative(0).base);
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
            IhsanScore::from_raw(9900),
        );

        let ctx = assembler.assemble(&msg, &mut pipeline, IhsanScore::from_raw(9000));
        assert!(ctx.traits.is_empty());
        assert!(ctx.insight_summaries.is_empty());
    }

    #[test]
    fn context_assembler_with_populated_memory() {
        use bizra_memory::FragmentKind;

        let mut assembler = ContextAssembler::new();
        let mut pipeline = MemoryPipeline::new();

        // Populate memory via pipeline's ingest (FragmentKind, content, session, turn, timestamp)
        for i in 1..=5u32 {
            pipeline
                .ingest(
                    FragmentKind::UserMessage,
                    &format!("I prefer style preference {i}"),
                    1,
                    i,
                    1000 + i as u64,
                )
                .unwrap();
        }
        // Extract atoms from ingested fragments
        pipeline.extract(2000);
        pipeline.force_synthesize(2000);

        let msg = Message::inbound(
            MessageId::new(1, 1),
            "How should I structure my project?",
            3000,
            IhsanScore::from_raw(9900),
        );

        let ctx = assembler.assemble(&msg, &mut pipeline, IhsanScore::from_raw(9900));
        assert!(ctx.knows_me_score > 0.0);
        assert_eq!(assembler.assemblies_count(), 1);
    }
}
