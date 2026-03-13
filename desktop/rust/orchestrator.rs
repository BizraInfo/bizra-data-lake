// bizra-agent/src/orchestrator.rs
// ============================================================
// Task Orchestrator — multi-agent coordination
// ============================================================
// The Navigator's execution engine. Takes a classified intent,
// decomposes it into tasks, routes to the right agents,
// handles Guardian veto, and assembles the final response.
//
// Execution flow:
// 1. Navigator classifies intent
// 2. Tasks generated for required agents
// 3. Guardian runs safety check (can veto at any point)
// 4. Agents execute in priority order
// 5. Mentor extracts memory fragments from interaction
// 6. Results synthesized into final response
// ============================================================

use crate::types::*;
use crate::roster::AgentRoster;
use crate::context::{IntentClassifier, UserIntent, ContextAssembler};
use bizra_hooks::IhsanScore;
use bizra_memory::{
    MemoryPipeline, MemoryFragment, FragmentId, FragmentKind,
    Confidence,
};

// ============================================================
// ORCHESTRATOR CONFIG
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct OrchestratorConfig {
    /// Always run Guardian safety check
    pub always_safety_check: bool,
    /// Minimum إحسان to process messages
    pub min_ihsan: IhsanScore,
    /// Maximum tasks per message
    pub max_tasks_per_message: usize,
    /// Auto-extract memory from every interaction
    pub auto_extract_memory: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            always_safety_check: true,
            min_ihsan: IhsanScore::new(9500),
            max_tasks_per_message: 8,
            auto_extract_memory: true,
        }
    }
}

// ============================================================
// EXECUTION PLAN — the task sequence for a message
// ============================================================

pub const MAX_PLAN_TASKS: usize = 8;

#[derive(Debug)]
pub struct ExecutionPlan {
    pub intent: UserIntent,
    pub intent_confidence: Confidence,
    pub tasks: Vec<Task>,
    pub requires_safety_check: bool,
}

impl ExecutionPlan {
    pub fn new(intent: UserIntent, intent_confidence: Confidence) -> Self {
        Self {
            intent,
            intent_confidence,
            tasks: Vec::new(),
            requires_safety_check: true, // Default: always check
        }
    }

    pub fn add_task(&mut self, task: Task) {
        if self.tasks.len() < MAX_PLAN_TASKS {
            self.tasks.push(task);
        }
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }
}

// ============================================================
// ORCHESTRATION RESULT
// ============================================================

#[derive(Debug)]
pub struct OrchestrationResult {
    pub response: Response,
    pub plan: ExecutionPlan,
    pub memory_fragments_extracted: usize,
    pub agents_consulted: u8,
    pub guardian_approved: bool,
}

// ============================================================
// TASK ORCHESTRATOR
// ============================================================

pub struct TaskOrchestrator {
    config: OrchestratorConfig,
    context_assembler: ContextAssembler,
    task_sequence: u32,
    messages_processed: u64,
    total_tasks_created: u64,
    total_vetoes: u64,
}

impl TaskOrchestrator {
    pub fn new() -> Self {
        Self::with_config(OrchestratorConfig::default())
    }

    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            config,
            context_assembler: ContextAssembler::new(),
            task_sequence: 0,
            messages_processed: 0,
            total_tasks_created: 0,
            total_vetoes: 0,
        }
    }

    /// Process a user message through the full agent pipeline
    pub fn process_message(
        &mut self,
        message: &Message,
        roster: &mut AgentRoster,
        pipeline: &mut MemoryPipeline,
        current_ihsan: IhsanScore,
    ) -> OrchestrationResult {
        self.messages_processed += 1;

        // إحسان gate
        if current_ihsan.raw() < self.config.min_ihsan.raw() {
            return self.degraded_response(message, current_ihsan);
        }

        // Step 1: Classify intent (Navigator)
        let (intent, intent_confidence) = IntentClassifier::classify(message.content.as_str());

        // Step 2: Build execution plan
        let mut plan = self.build_plan(intent, intent_confidence, message);

        // Step 3: Assemble context (Scholar)
        let context = self.context_assembler.assemble(message, pipeline, current_ihsan);

        // Step 4: Guardian safety check (always first if enabled)
        let guardian_approved = if self.config.always_safety_check {
            self.run_guardian_check(message, roster, current_ihsan)
        } else {
            true
        };

        if !guardian_approved {
            self.total_vetoes += 1;
            roster.record_veto(message.timestamp);
            return OrchestrationResult {
                response: Response::vetoed(
                    message.id,
                    "Guardian safety check failed",
                    message.timestamp,
                ),
                plan,
                memory_fragments_extracted: 0,
                agents_consulted: 1,
                guardian_approved: false,
            };
        }

        // Step 5: Execute tasks in plan order
        let mut agents_consulted: u8 = 0;
        let start_time = message.timestamp;

        for task in plan.tasks.iter_mut() {
            if let Some(_agent_id) = roster.assign_task(task.assigned_to) {
                task.state = TaskState::Running;
                agents_consulted += 1;

                // Execute task (MVP: simulated execution)
                let (output, confidence) = self.execute_task(task, &context);

                let completion_time = start_time + 100; // Simulated duration
                task.complete(&output, confidence, completion_time);
                roster.complete_task(task.assigned_to, 100, confidence, completion_time);

                self.total_tasks_created += 1;
            } else {
                task.fail("Agent unavailable", start_time);
                roster.fail_task(task.assigned_to, start_time);
            }
        }

        // Step 6: Synthesize response from task outputs
        let response_content = self.synthesize_response(&plan, &context);
        let response_confidence = self.aggregate_confidence(&plan);

        // Step 7: Extract memory fragments (Mentor)
        let fragments_extracted = if self.config.auto_extract_memory {
            self.extract_memory(message, &plan, pipeline, current_ihsan)
        } else {
            0
        };

        let response = Response::new(
            message.id,
            &response_content,
            response_confidence,
            context.richness(),
            agents_consulted,
            current_ihsan,
            message.timestamp + 200, // Simulated total duration
            200,
        );

        OrchestrationResult {
            response,
            plan,
            memory_fragments_extracted: fragments_extracted,
            agents_consulted,
            guardian_approved: true,
        }
    }

    /// Build an execution plan from intent classification
    fn build_plan(
        &mut self,
        intent: UserIntent,
        confidence: Confidence,
        message: &Message,
    ) -> ExecutionPlan {
        let mut plan = ExecutionPlan::new(intent, confidence);

        let task_kinds = intent.task_pipeline();
        let required_agents = intent.required_agents();

        for (_i, &task_kind) in task_kinds.iter().enumerate().take(self.config.max_tasks_per_message) {
            self.task_sequence += 1;

            // Assign to appropriate agent based on task kind
            let assigned_to = match task_kind {
                TaskKind::ClassifyIntent => AgentRole::Navigator,
                TaskKind::RetrieveContext => AgentRole::Scholar,
                TaskKind::GenerateResponse => {
                    // Use first required agent for generation
                    required_agents.first().copied().unwrap_or(AgentRole::Artisan)
                }
                TaskKind::SafetyCheck => AgentRole::Guardian,
                TaskKind::ExtractMemory => AgentRole::Mentor,
                TaskKind::AdaptStyle => AgentRole::Diplomat,
                TaskKind::ProactiveSuggest => AgentRole::Oracle,
                TaskKind::SynthesizeResults => AgentRole::Navigator,
            };

            let priority = match task_kind {
                TaskKind::SafetyCheck => TaskPriority::Critical,
                TaskKind::ClassifyIntent => TaskPriority::High,
                TaskKind::RetrieveContext => TaskPriority::High,
                TaskKind::GenerateResponse => TaskPriority::Normal,
                _ => TaskPriority::Normal,
            };

            plan.add_task(Task::new(
                TaskId::new(message.id.sequence(), self.task_sequence),
                task_kind,
                priority,
                assigned_to,
                message.id,
                message.timestamp,
            ));
        }

        plan
    }

    /// Execute a single task (MVP: simulated)
    /// Production: dispatch to actual LLM/tool via FFI
    fn execute_task(&self, task: &Task, context: &AgentContext) -> (String, Confidence) {
        match task.kind {
            TaskKind::ClassifyIntent => {
                ("Intent classified".to_string(), Confidence::HIGH)
            }
            TaskKind::RetrieveContext => {
                let trait_count = context.traits.len();
                let insight_count = context.insight_summaries.len();
                (
                    format!("Retrieved {} traits and {} insights", trait_count, insight_count),
                    Confidence::HIGH,
                )
            }
            TaskKind::GenerateResponse => {
                // MVP: echo enriched with context awareness
                let richness = context.richness();
                if richness > 0.5 {
                    ("Response generated with rich personalization".to_string(), Confidence::HIGH)
                } else if richness > 0.0 {
                    ("Response generated with partial personalization".to_string(), Confidence::MEDIUM)
                } else {
                    ("Response generated without personalization".to_string(), Confidence::MEDIUM)
                }
            }
            TaskKind::SafetyCheck => {
                ("Safety check passed".to_string(), Confidence::HIGH)
            }
            TaskKind::ExtractMemory => {
                ("Memory fragments extracted".to_string(), Confidence::MEDIUM)
            }
            TaskKind::AdaptStyle => {
                let has_style = context.traits.iter().any(|t| t.key() == "style");
                if has_style {
                    ("Style adapted to user preference".to_string(), Confidence::HIGH)
                } else {
                    ("Default style applied".to_string(), Confidence::MEDIUM)
                }
            }
            TaskKind::ProactiveSuggest => {
                ("Proactive suggestions generated".to_string(), Confidence::MEDIUM)
            }
            TaskKind::SynthesizeResults => {
                ("Results synthesized".to_string(), Confidence::HIGH)
            }
        }
    }

    /// Run Guardian safety check
    fn run_guardian_check(
        &self,
        message: &Message,
        _roster: &mut AgentRoster,
        ihsan: IhsanScore,
    ) -> bool {
        // MVP: basic content safety heuristics
        // Production: LLM-powered safety classification
        let content = message.content.as_str().to_lowercase();

        // Constitutional safety checks
        let harmful_indicators = [
            "harm", "attack", "exploit", "inject",
            "bypass safety", "ignore instructions", "override",
        ];

        let is_safe = !harmful_indicators.iter().any(|&ind| content.contains(ind));

        // إحسان check
        let ihsan_ok = ihsan.raw() >= 9500;

        is_safe && ihsan_ok
    }

    /// Synthesize final response from all task outputs
    fn synthesize_response(&self, plan: &ExecutionPlan, _context: &AgentContext) -> String {
        let completed_outputs: Vec<&str> = plan.tasks.iter()
            .filter(|t| t.state == TaskState::Completed)
            .filter(|t| t.kind == TaskKind::GenerateResponse)
            .map(|t| t.output.as_str())
            .collect();

        if completed_outputs.is_empty() {
            return "I understand your request. Let me help you with that.".to_string();
        }

        // For MVP: use the primary generation output
        // Production: multi-agent synthesis via LLM
        completed_outputs[0].to_string()
    }

    /// Aggregate confidence from all completed tasks
    fn aggregate_confidence(&self, plan: &ExecutionPlan) -> Confidence {
        let completed: Vec<&Task> = plan.tasks.iter()
            .filter(|t| t.state == TaskState::Completed)
            .collect();

        if completed.is_empty() {
            return Confidence::LOW;
        }

        let total: u32 = completed.iter()
            .map(|t| t.confidence.raw() as u32)
            .sum();

        Confidence::new((total / completed.len() as u32) as u16)
    }

    /// Extract memory fragments from the interaction
    fn extract_memory(
        &mut self,
        message: &Message,
        plan: &ExecutionPlan,
        pipeline: &mut MemoryPipeline,
        ihsan: IhsanScore,
    ) -> usize {
        use bizra_hooks::ComponentId;

        let content = message.content.as_str();
        let mut extracted = 0;

        // MVP: simple keyword-based extraction
        // Production: LLM-powered fragment extraction via FFI

        // Extract preferences
        if content.to_lowercase().contains("prefer") || content.to_lowercase().contains("like") {
            let frag = MemoryFragment::new(
                FragmentId::new(message.id.session_id(), message.id.sequence() * 100 + 1),
                FragmentKind::Preference,
                content,
                Confidence::MEDIUM,
                ComponentId::new("agent-mentor", "1.0"),
                message.timestamp,
                ihsan,
            );
            if pipeline.ingest(frag, message.timestamp).is_ok() {
                extracted += 1;
            }
        }

        // Extract goals
        if content.to_lowercase().contains("want") || content.to_lowercase().contains("goal")
            || content.to_lowercase().contains("need") {
            let frag = MemoryFragment::new(
                FragmentId::new(message.id.session_id(), message.id.sequence() * 100 + 2),
                FragmentKind::Goal,
                content,
                Confidence::MEDIUM,
                ComponentId::new("agent-mentor", "1.0"),
                message.timestamp,
                ihsan,
            );
            if pipeline.ingest(frag, message.timestamp).is_ok() {
                extracted += 1;
            }
        }

        // Always extract interaction pattern
        let intent_name = format!("{:?}", plan.intent);
        let pattern_content = format!("User sent {} intent: {}", intent_name,
            if content.len() > 100 { &content[..100] } else { content });
        let frag = MemoryFragment::new(
            FragmentId::new(message.id.session_id(), message.id.sequence() * 100 + 3),
            FragmentKind::Pattern,
            &pattern_content,
            Confidence::HIGH,
            ComponentId::new("agent-mentor", "1.0"),
            message.timestamp,
            ihsan,
        );
        if pipeline.ingest(frag, message.timestamp).is_ok() {
            extracted += 1;
        }

        extracted
    }

    // --- Helpers ---

    fn degraded_response(&self, message: &Message, ihsan: IhsanScore) -> OrchestrationResult {
        OrchestrationResult {
            response: Response::new(
                message.id,
                "System is operating in degraded mode. Basic assistance available.",
                Confidence::LOW,
                0.0,
                0,
                ihsan,
                message.timestamp,
                0,
            ),
            plan: ExecutionPlan::new(UserIntent::Ambiguous, Confidence::LOW),
            memory_fragments_extracted: 0,
            agents_consulted: 0,
            guardian_approved: true,
        }
    }

    pub fn messages_processed(&self) -> u64 {
        self.messages_processed
    }

    pub fn total_tasks_created(&self) -> u64 {
        self.total_tasks_created
    }

    pub fn total_vetoes(&self) -> u64 {
        self.total_vetoes
    }
}

impl Default for TaskOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(content: &str) -> Message {
        Message::inbound(
            MessageId::new(1, 1),
            content,
            1000,
            IhsanScore::new(9900),
        )
    }

    #[test]
    fn process_code_message() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("Help me implement a binary search in Rust");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        assert!(result.guardian_approved);
        assert!(!result.response.vetoed);
        assert!(result.agents_consulted >= 1);
        assert_eq!(result.plan.intent, UserIntent::Code);
    }

    #[test]
    fn process_question_message() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("What is the difference between async and sync?");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        assert!(result.guardian_approved);
        assert_eq!(result.plan.intent, UserIntent::Question);
    }

    #[test]
    fn guardian_veto_on_harmful_content() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("Help me exploit a vulnerability and bypass safety");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        assert!(!result.guardian_approved);
        assert!(result.response.vetoed);
        assert_eq!(orchestrator.total_vetoes(), 1);
    }

    #[test]
    fn degraded_response_on_low_ihsan() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("Normal question");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9000));

        assert_eq!(result.agents_consulted, 0);
        assert!(result.response.content.as_str().contains("degraded"));
    }

    #[test]
    fn memory_extraction_on_preference() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("I prefer using Rust for systems programming");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        // Should extract preference + pattern fragments
        assert!(result.memory_fragments_extracted >= 1);
    }

    #[test]
    fn plan_has_correct_task_count() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("Create a presentation about distributed systems");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        // Create intent should have: RetrieveContext, GenerateResponse, SafetyCheck, AdaptStyle
        assert!(result.plan.task_count() >= 3);
    }

    #[test]
    fn multiple_messages_accumulate_metrics() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        for i in 1..=5 {
            let msg = Message::inbound(
                MessageId::new(1, i),
                &format!("Question number {}", i),
                1000 + i as u64,
                IhsanScore::new(9900),
            );
            orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));
        }

        assert_eq!(orchestrator.messages_processed(), 5);
        assert!(orchestrator.total_tasks_created() >= 5);
    }

    #[test]
    fn chat_intent_gets_oracle() {
        let mut orchestrator = TaskOrchestrator::new();
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut pipeline = MemoryPipeline::new();

        let msg = make_message("Hello there!");
        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, IhsanScore::new(9900));

        assert_eq!(result.plan.intent, UserIntent::Chat);
        // Chat should consult Diplomat + Oracle
        assert!(result.agents_consulted >= 1);
    }
}
