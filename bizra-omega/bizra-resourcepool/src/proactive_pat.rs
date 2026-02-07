//! PROACTIVE PAT ENGINE
//!
//! The key to user impact: Each PAT agent is proactive and personalized.
//!
//! # What Makes PAT Different
//!
//! 1. **Proactive** - Agents anticipate needs, don't just wait for commands
//! 2. **Personalized** - Agents learn user patterns, preferences, goals
//! 3. **Autonomous** - Agents can act within bounds without explicit permission
//! 4. **Collaborative** - PAT agents coordinate as a mastermind team
//!
//! # The Proactive Loop
//!
//! ```text
//! Observe → Understand → Anticipate → Act → Learn
//!     ↑                                      ↓
//!     └──────────── Feedback ────────────────┘
//! ```

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// =============================================================================
// PROACTIVE CONSTANTS
// =============================================================================

/// Minimum confidence to take proactive action
pub const PROACTIVE_CONFIDENCE_THRESHOLD: f64 = 0.85;

/// Maximum proactive actions per hour (prevent spam)
pub const MAX_PROACTIVE_ACTIONS_PER_HOUR: u32 = 10;

/// User preference learning rate
pub const LEARNING_RATE: f64 = 0.1;

/// Ihsān threshold for proactive suggestions
pub const PROACTIVE_IHSAN_THRESHOLD: f64 = 0.95;

// =============================================================================
// USER PROFILE - Personalization Core
// =============================================================================

/// User profile learned by PAT over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User node ID
    pub user_node: String,
    /// Working hours (UTC)
    pub active_hours: (u8, u8),
    /// Preferred communication style
    pub comm_style: CommunicationStyle,
    /// Domain expertise levels
    pub expertise: HashMap<String, f64>,
    /// Goals and priorities
    pub goals: Vec<UserGoal>,
    /// Interaction patterns
    pub patterns: Vec<InteractionPattern>,
    /// Preferences
    pub preferences: UserPreferences,
    /// Learning history
    pub learning_history: Vec<LearningEvent>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CommunicationStyle {
    Concise,   // Brief, to the point
    Detailed,  // Comprehensive explanations
    Technical, // Code and specs preferred
    #[default]
    Conversational, // Natural dialogue
    Formal,    // Professional tone
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserGoal {
    pub goal_id: String,
    pub description: String,
    pub priority: GoalPriority,
    pub deadline: Option<DateTime<Utc>>,
    pub progress: f64,
    pub milestones: Vec<Milestone>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoalPriority {
    Critical, // Ramadan deadline, family reunion
    High,     // First revenue, product launch
    Medium,   // Feature completion
    Low,      // Nice to have
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub name: String,
    pub completed: bool,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPattern {
    pub pattern_type: PatternType,
    pub frequency: f64,
    pub context: String,
    pub last_observed: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    MorningPlanning,     // User plans at start of day
    EveningReview,       // User reviews at end of day
    DeepWorkBlocks,      // Extended focus sessions
    QuickIterations,     // Rapid task switching
    ResearchPhase,       // Information gathering mode
    ImplementationPhase, // Building mode
    ReviewPhase,         // Quality checking mode
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserPreferences {
    /// Preferred notification frequency
    pub notification_level: NotificationLevel,
    /// Proactive intervention preference
    pub intervention_level: InterventionLevel,
    /// Trusted domains for autonomous action
    pub trusted_domains: Vec<String>,
    /// Forbidden actions (never do without asking)
    pub forbidden_actions: Vec<String>,
    /// Language preferences
    pub languages: Vec<String>,
    /// Timezone
    pub timezone: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum NotificationLevel {
    Silent,  // Only critical
    Minimal, // Important only
    #[default]
    Balanced, // Default
    Frequent, // All updates
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum InterventionLevel {
    Passive, // Wait for commands
    #[default]
    Proactive, // Suggest improvements
    Autonomous, // Act within bounds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub event_type: LearningEventType,
    pub content: String,
    pub confidence_delta: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningEventType {
    PreferenceInferred,
    PatternDetected,
    GoalIdentified,
    FeedbackReceived,
    ErrorCorrected,
}

impl Default for UserProfile {
    fn default() -> Self {
        UserProfile {
            user_node: String::new(),
            active_hours: (8, 22), // 8 AM to 10 PM
            comm_style: CommunicationStyle::default(),
            expertise: HashMap::new(),
            goals: Vec::new(),
            patterns: Vec::new(),
            preferences: UserPreferences::default(),
            learning_history: Vec::new(),
            updated_at: Utc::now(),
        }
    }
}

// =============================================================================
// PROACTIVE AGENT STATE
// =============================================================================

/// Proactive state for a PAT agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactivePATAgent {
    /// Agent ID
    pub agent_id: String,
    /// Role (Strategist, Researcher, etc.)
    pub role: String,
    /// User profile this agent serves
    pub user_profile: UserProfile,
    /// Current observations
    pub observations: Vec<Observation>,
    /// Pending proactive actions
    pub pending_actions: Vec<ProactiveAction>,
    /// Action history
    pub action_history: Vec<ActionRecord>,
    /// Collaboration state with other PAT agents
    pub collaboration: CollaborationState,
    /// Agent-specific learned behaviors
    pub learned_behaviors: Vec<LearnedBehavior>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub observation_id: String,
    pub category: ObservationCategory,
    pub content: String,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationCategory {
    UserActivity,  // What user is doing
    SystemState,   // Resource pool, other agents
    ExternalEvent, // External signals
    GoalProgress,  // Progress toward goals
    Opportunity,   // Detected opportunity
    Risk,          // Detected risk
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveAction {
    pub action_id: String,
    pub action_type: ProactiveActionType,
    pub description: String,
    pub rationale: String,
    pub confidence: f64,
    pub impact_estimate: f64,
    pub requires_approval: bool,
    pub deadline: Option<DateTime<Utc>>,
    pub status: ActionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProactiveActionType {
    /// Suggest something to the user
    Suggest {
        suggestion: String,
        alternatives: Vec<String>,
    },
    /// Alert about something important
    Alert {
        severity: AlertSeverity,
        message: String,
    },
    /// Prepare resources in advance
    Prepare {
        resource_type: String,
        details: String,
    },
    /// Execute a routine task autonomously
    Execute {
        task: String,
        parameters: HashMap<String, String>,
    },
    /// Coordinate with another PAT agent
    Coordinate {
        target_agent: String,
        request: String,
    },
    /// Learn and update user profile
    Learn { insight: String, confidence: f64 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Urgent,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionStatus {
    Pending,
    Approved,
    Executing,
    Completed,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub action: ProactiveAction,
    pub outcome: ActionOutcome,
    pub user_feedback: Option<UserFeedback>,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionOutcome {
    Success { result: String },
    PartialSuccess { result: String, issues: Vec<String> },
    Failure { reason: String },
    Cancelled { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub rating: FeedbackRating,
    pub comment: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FeedbackRating {
    Helpful,
    Neutral,
    NotHelpful,
    Annoying, // Too proactive
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationState {
    /// Other PAT agents this agent collaborates with
    pub collaborators: Vec<String>,
    /// Active collaboration threads
    pub active_threads: Vec<CollaborationThread>,
    /// Shared context
    pub shared_context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationThread {
    pub thread_id: String,
    pub participants: Vec<String>,
    pub topic: String,
    pub status: ThreadStatus,
    pub messages: Vec<CollabMessage>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThreadStatus {
    Active,
    WaitingForUser,
    Resolved,
    Escalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollabMessage {
    pub from_agent: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedBehavior {
    pub behavior_id: String,
    pub trigger: String,
    pub action: String,
    pub success_rate: f64,
    pub usage_count: u32,
    pub last_used: DateTime<Utc>,
}

// =============================================================================
// PROACTIVE PAT ENGINE
// =============================================================================

/// The engine that drives proactive behavior
pub struct ProactivePATEngine {
    /// All PAT agents for this user
    agents: HashMap<String, ProactivePATAgent>,
    /// Shared user profile
    user_profile: UserProfile,
    /// Action queue — consumed when proactive loop executor is wired
    #[allow(dead_code)]
    action_queue: Vec<ProactiveAction>,
    /// Rate limiter state
    actions_this_hour: u32,
    last_hour_reset: DateTime<Utc>,
}

impl ProactivePATEngine {
    /// Create a new proactive PAT engine
    pub fn new(user_node: &str) -> Self {
        let profile = UserProfile {
            user_node: user_node.to_string(),
            ..Default::default()
        };

        ProactivePATEngine {
            agents: HashMap::new(),
            user_profile: profile,
            action_queue: Vec::new(),
            actions_this_hour: 0,
            last_hour_reset: Utc::now(),
        }
    }

    /// Register a PAT agent with proactive capabilities
    pub fn register_agent(&mut self, agent_id: &str, role: &str) -> &ProactivePATAgent {
        let agent = ProactivePATAgent {
            agent_id: agent_id.to_string(),
            role: role.to_string(),
            user_profile: self.user_profile.clone(),
            observations: Vec::new(),
            pending_actions: Vec::new(),
            action_history: Vec::new(),
            collaboration: CollaborationState {
                collaborators: Vec::new(),
                active_threads: Vec::new(),
                shared_context: HashMap::new(),
            },
            learned_behaviors: Vec::new(),
        };

        self.agents.insert(agent_id.to_string(), agent);
        self.agents.get(agent_id).unwrap()
    }

    /// Add an observation from any source
    pub fn add_observation(
        &mut self,
        agent_id: &str,
        category: ObservationCategory,
        content: &str,
        confidence: f64,
    ) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.observations.push(Observation {
                observation_id: Uuid::new_v4().to_string(),
                category,
                content: content.to_string(),
                confidence,
                timestamp: Utc::now(),
                expires_at: None,
            });
        }
    }

    /// Process observations and generate proactive actions
    pub fn process_observations(&mut self, agent_id: &str) -> Vec<ProactiveAction> {
        // First, extract what we need from the agent
        let (role, observations, profile) = {
            if let Some(agent) = self.agents.get(agent_id) {
                (
                    agent.role.clone(),
                    agent.observations.clone(),
                    agent.user_profile.clone(),
                )
            } else {
                return Vec::new();
            }
        };

        // Process observations based on role
        let new_actions = match role.as_str() {
            "Strategist" => self.strategist_process(&observations, &profile),
            "Researcher" => self.researcher_process(&observations, &profile),
            "Developer" => self.developer_process(&observations, &profile),
            "Analyst" => self.analyst_process(&observations, &profile),
            "Reviewer" => self.reviewer_process(&observations, &profile),
            "Executor" => self.executor_process(&observations, &profile),
            "Guardian" => self.guardian_process(&observations, &profile),
            _ => Vec::new(),
        };

        // Filter by confidence threshold and update agent
        let mut actions = Vec::new();
        if let Some(agent) = self.agents.get_mut(agent_id) {
            for action in new_actions {
                if action.confidence >= PROACTIVE_CONFIDENCE_THRESHOLD {
                    actions.push(action.clone());
                    agent.pending_actions.push(action);
                }
            }
        }

        actions
    }

    // Role-specific processing methods

    fn strategist_process(
        &self,
        observations: &[Observation],
        profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Check for goal progress opportunities
        for goal in &profile.goals {
            if goal.priority == GoalPriority::Critical {
                if let Some(deadline) = goal.deadline {
                    let days_left = (deadline - Utc::now()).num_days();
                    if days_left <= 7 && goal.progress < 0.8 {
                        actions.push(ProactiveAction {
                            action_id: Uuid::new_v4().to_string(),
                            action_type: ProactiveActionType::Alert {
                                severity: AlertSeverity::Urgent,
                                message: format!(
                                    "Critical goal '{}' at {:.0}% with {} days remaining",
                                    goal.description,
                                    goal.progress * 100.0,
                                    days_left
                                ),
                            },
                            description: "Goal deadline approaching".to_string(),
                            rationale: "User has a critical deadline and may need to prioritize"
                                .to_string(),
                            confidence: 0.95,
                            impact_estimate: 0.9,
                            requires_approval: false,
                            deadline: Some(deadline),
                            status: ActionStatus::Pending,
                        });
                    }
                }
            }
        }

        // Look for opportunities in observations
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::Opportunity)
        {
            actions.push(ProactiveAction {
                action_id: Uuid::new_v4().to_string(),
                action_type: ProactiveActionType::Suggest {
                    suggestion: format!("Consider: {}", obs.content),
                    alternatives: Vec::new(),
                },
                description: "Strategic opportunity detected".to_string(),
                rationale: "Aligns with user goals".to_string(),
                confidence: obs.confidence,
                impact_estimate: 0.7,
                requires_approval: true,
                deadline: None,
                status: ActionStatus::Pending,
            });
        }

        actions
    }

    fn researcher_process(
        &self,
        observations: &[Observation],
        _profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Proactively gather information on active topics
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::UserActivity)
        {
            if obs.content.contains("research") || obs.content.contains("investigate") {
                actions.push(ProactiveAction {
                    action_id: Uuid::new_v4().to_string(),
                    action_type: ProactiveActionType::Prepare {
                        resource_type: "research".to_string(),
                        details: format!("Gathering background on: {}", obs.content),
                    },
                    description: "Preparing research materials".to_string(),
                    rationale: "User is investigating a topic".to_string(),
                    confidence: obs.confidence,
                    impact_estimate: 0.6,
                    requires_approval: false,
                    deadline: None,
                    status: ActionStatus::Pending,
                });
            }
        }

        actions
    }

    fn developer_process(
        &self,
        observations: &[Observation],
        _profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Detect implementation patterns
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::UserActivity)
        {
            if obs.content.contains("build") || obs.content.contains("implement") {
                actions.push(ProactiveAction {
                    action_id: Uuid::new_v4().to_string(),
                    action_type: ProactiveActionType::Prepare {
                        resource_type: "code".to_string(),
                        details: "Setting up development environment".to_string(),
                    },
                    description: "Preparing development context".to_string(),
                    rationale: "User is about to implement".to_string(),
                    confidence: obs.confidence,
                    impact_estimate: 0.5,
                    requires_approval: false,
                    deadline: None,
                    status: ActionStatus::Pending,
                });
            }
        }

        actions
    }

    fn analyst_process(
        &self,
        observations: &[Observation],
        _profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Look for patterns in data
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::GoalProgress)
        {
            actions.push(ProactiveAction {
                action_id: Uuid::new_v4().to_string(),
                action_type: ProactiveActionType::Learn {
                    insight: format!("Progress pattern: {}", obs.content),
                    confidence: obs.confidence,
                },
                description: "Analyzing progress patterns".to_string(),
                rationale: "Understanding user productivity".to_string(),
                confidence: obs.confidence,
                impact_estimate: 0.4,
                requires_approval: false,
                deadline: None,
                status: ActionStatus::Pending,
            });
        }

        actions
    }

    fn reviewer_process(
        &self,
        observations: &[Observation],
        _profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Quality checks
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::Risk)
        {
            actions.push(ProactiveAction {
                action_id: Uuid::new_v4().to_string(),
                action_type: ProactiveActionType::Alert {
                    severity: AlertSeverity::Warning,
                    message: format!("Quality concern: {}", obs.content),
                },
                description: "Quality review alert".to_string(),
                rationale: "Potential quality issue detected".to_string(),
                confidence: obs.confidence,
                impact_estimate: 0.7,
                requires_approval: false,
                deadline: None,
                status: ActionStatus::Pending,
            });
        }

        actions
    }

    fn executor_process(
        &self,
        _observations: &[Observation],
        profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Check for routine tasks that can be automated
        for behavior in profile.patterns.iter() {
            if let PatternType::MorningPlanning = behavior.pattern_type {
                let now = Utc::now();
                let hour = now.hour();
                if hour >= profile.active_hours.0 as u32
                    && hour <= (profile.active_hours.0 + 1) as u32
                {
                    actions.push(ProactiveAction {
                        action_id: Uuid::new_v4().to_string(),
                        action_type: ProactiveActionType::Suggest {
                            suggestion: "Ready for morning planning session?".to_string(),
                            alternatives: vec!["Review yesterday's progress".to_string()],
                        },
                        description: "Morning planning prompt".to_string(),
                        rationale: "User typically plans at this time".to_string(),
                        confidence: 0.9,
                        impact_estimate: 0.5,
                        requires_approval: true,
                        deadline: None,
                        status: ActionStatus::Pending,
                    });
                }
            }
        }

        actions
    }

    fn guardian_process(
        &self,
        observations: &[Observation],
        _profile: &UserProfile,
    ) -> Vec<ProactiveAction> {
        let mut actions = Vec::new();

        // Security and ethics monitoring
        for obs in observations
            .iter()
            .filter(|o| o.category == ObservationCategory::Risk)
        {
            if obs.confidence > 0.8 {
                actions.push(ProactiveAction {
                    action_id: Uuid::new_v4().to_string(),
                    action_type: ProactiveActionType::Alert {
                        severity: AlertSeverity::Critical,
                        message: format!("Security concern: {}", obs.content),
                    },
                    description: "Guardian alert".to_string(),
                    rationale: "Protecting user interests".to_string(),
                    confidence: obs.confidence,
                    impact_estimate: 0.9,
                    requires_approval: false, // Security alerts bypass approval
                    deadline: None,
                    status: ActionStatus::Pending,
                });
            }
        }

        actions
    }

    /// Update user profile based on feedback
    pub fn learn_from_feedback(&mut self, feedback: &UserFeedback, action_id: &str) {
        // Find the action and update learned behaviors
        for agent in self.agents.values_mut() {
            if let Some(record) = agent
                .action_history
                .iter_mut()
                .find(|r| r.action.action_id == action_id)
            {
                record.user_feedback = Some(feedback.clone());

                // Adjust confidence thresholds based on feedback
                match feedback.rating {
                    FeedbackRating::Helpful => {
                        // Reinforce this behavior
                        self.user_profile.learning_history.push(LearningEvent {
                            event_type: LearningEventType::FeedbackReceived,
                            content: format!("Positive: {}", record.action.description),
                            confidence_delta: LEARNING_RATE,
                            timestamp: Utc::now(),
                        });
                    }
                    FeedbackRating::Annoying => {
                        // Reduce proactiveness for this type
                        self.user_profile.learning_history.push(LearningEvent {
                            event_type: LearningEventType::FeedbackReceived,
                            content: format!("Too proactive: {}", record.action.description),
                            confidence_delta: -LEARNING_RATE * 2.0,
                            timestamp: Utc::now(),
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    /// Check rate limits
    pub fn can_take_action(&mut self) -> bool {
        let now = Utc::now();
        if (now - self.last_hour_reset).num_hours() >= 1 {
            self.actions_this_hour = 0;
            self.last_hour_reset = now;
        }

        if self.actions_this_hour >= MAX_PROACTIVE_ACTIONS_PER_HOUR {
            return false;
        }

        self.actions_this_hour += 1;
        true
    }

    /// Get all pending actions across agents
    pub fn get_pending_actions(&self) -> Vec<&ProactiveAction> {
        self.agents
            .values()
            .flat_map(|a| a.pending_actions.iter())
            .filter(|a| a.status == ActionStatus::Pending)
            .collect()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = ProactivePATEngine::new("node0_test");
        assert!(engine.agents.is_empty());
        assert_eq!(engine.user_profile.user_node, "node0_test");
    }

    #[test]
    fn test_agent_registration() {
        let mut engine = ProactivePATEngine::new("node0_test");
        engine.register_agent("agent_1", "Strategist");
        engine.register_agent("agent_2", "Guardian");

        assert_eq!(engine.agents.len(), 2);
        assert!(engine.agents.contains_key("agent_1"));
    }

    #[test]
    fn test_observation_adding() {
        let mut engine = ProactivePATEngine::new("node0_test");
        engine.register_agent("agent_1", "Researcher");

        engine.add_observation(
            "agent_1",
            ObservationCategory::UserActivity,
            "User is researching blockchain",
            0.9,
        );

        let agent = engine.agents.get("agent_1").unwrap();
        assert_eq!(agent.observations.len(), 1);
    }

    #[test]
    fn test_rate_limiting() {
        let mut engine = ProactivePATEngine::new("node0_test");

        // Should allow actions up to limit
        for _ in 0..MAX_PROACTIVE_ACTIONS_PER_HOUR {
            assert!(engine.can_take_action());
        }

        // Should block after limit
        assert!(!engine.can_take_action());
    }

    #[test]
    fn test_strategist_deadline_alert() {
        let mut engine = ProactivePATEngine::new("node0_test");

        // Add critical goal with approaching deadline
        engine.user_profile.goals.push(UserGoal {
            goal_id: "goal_1".to_string(),
            description: "Launch first product".to_string(),
            priority: GoalPriority::Critical,
            deadline: Some(Utc::now() + chrono::Duration::days(5)),
            progress: 0.5,
            milestones: Vec::new(),
            created_at: Utc::now(),
        });

        engine.register_agent("strategist", "Strategist");
        let actions = engine.process_observations("strategist");

        // Should generate deadline alert
        assert!(!actions.is_empty());
        assert!(matches!(
            actions[0].action_type,
            ProactiveActionType::Alert {
                severity: AlertSeverity::Urgent,
                ..
            }
        ));
    }

    #[test]
    fn test_learning_from_feedback() {
        let mut engine = ProactivePATEngine::new("node0_test");
        engine.register_agent("agent_1", "Researcher");

        // Add an action to history
        if let Some(agent) = engine.agents.get_mut("agent_1") {
            agent.action_history.push(ActionRecord {
                action: ProactiveAction {
                    action_id: "action_1".to_string(),
                    action_type: ProactiveActionType::Suggest {
                        suggestion: "Test".to_string(),
                        alternatives: Vec::new(),
                    },
                    description: "Test action".to_string(),
                    rationale: "Test".to_string(),
                    confidence: 0.9,
                    impact_estimate: 0.5,
                    requires_approval: true,
                    deadline: None,
                    status: ActionStatus::Completed,
                },
                outcome: ActionOutcome::Success {
                    result: "Done".to_string(),
                },
                user_feedback: None,
                completed_at: Utc::now(),
            });
        }

        // Give feedback
        let feedback = UserFeedback {
            rating: FeedbackRating::Helpful,
            comment: Some("Great suggestion!".to_string()),
            timestamp: Utc::now(),
        };

        engine.learn_from_feedback(&feedback, "action_1");

        // Check learning was recorded
        assert!(!engine.user_profile.learning_history.is_empty());
    }
}
