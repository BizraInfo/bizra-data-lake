// bizra-agent/src/roster.rs
// ============================================================
// Agent Roster — managing the PAT (Personal Agent Team)
// ============================================================
// Each user gets 7 specialized agents. The roster tracks:
// - Agent state (active, idle, busy, degraded)
// - Per-agent performance metrics
// - Readiness for task assignment
// - إحسان per-agent
//
// Think of this as the team manager's dashboard.
// ============================================================

use crate::types::*;
use bizra_hooks::IhsanScore;
use bizra_memory::Confidence;

// ============================================================
// AGENT STATE
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AgentState {
    /// Ready to accept tasks
    Idle = 0,
    /// Currently processing a task
    Busy = 1,
    /// Temporarily unavailable
    Suspended = 2,
    /// Degraded — low إحسان, limited capability
    Degraded = 3,
}

// ============================================================
// AGENT ENTRY — one agent in the roster
// ============================================================

#[derive(Debug, Clone)]
pub struct AgentEntry {
    pub id: AgentId,
    pub role: AgentRole,
    pub state: AgentState,
    pub ihsan: IhsanScore,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_vetoed: u64,
    pub avg_response_time_us: u64,
    pub avg_confidence: f32,
    pub last_active: u64,
    pub created_at: u64,
}

impl AgentEntry {
    pub fn new(role: AgentRole, user_hash: u32, timestamp: u64) -> Self {
        Self {
            id: AgentId::new(role, user_hash),
            role,
            state: AgentState::Idle,
            ihsan: IhsanScore::from_raw(9900),
            tasks_completed: 0,
            tasks_failed: 0,
            tasks_vetoed: 0,
            avg_response_time_us: 0,
            avg_confidence: 0.0,
            last_active: timestamp,
            created_at: timestamp,
        }
    }

    /// Is this agent available for a new task?
    pub fn is_available(&self) -> bool {
        self.state == AgentState::Idle
    }

    /// Record a completed task
    pub fn record_completion(&mut self, duration_us: u64, confidence: Confidence, timestamp: u64) {
        self.tasks_completed += 1;
        self.last_active = timestamp;

        // Running average of response time
        let total = self.tasks_completed + self.tasks_failed;
        self.avg_response_time_us = self.avg_response_time_us
            + (duration_us.saturating_sub(self.avg_response_time_us)) / total.max(1);

        // Running average of confidence
        self.avg_confidence =
            self.avg_confidence + (confidence.base - self.avg_confidence) / total as f32;
    }

    /// Record a failed task
    pub fn record_failure(&mut self, timestamp: u64) {
        self.tasks_failed += 1;
        self.last_active = timestamp;
    }

    /// Record a veto (Guardian only typically)
    pub fn record_veto(&mut self, timestamp: u64) {
        self.tasks_vetoed += 1;
        self.last_active = timestamp;
    }

    /// Task success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.tasks_completed + self.tasks_failed;
        if total == 0 {
            return 1.0;
        }
        self.tasks_completed as f32 / total as f32
    }

    /// Update إحسان and potentially degrade
    pub fn update_ihsan(&mut self, score: IhsanScore) {
        self.ihsan = score;
        if score.raw() < 9500 {
            self.state = AgentState::Degraded;
        } else if self.state == AgentState::Degraded {
            self.state = AgentState::Idle;
        }
    }
}

// ============================================================
// AGENT ROSTER — the full team
// ============================================================

pub const PAT_SIZE: usize = 7; // Personal Agent Team = 7 agents

pub struct AgentRoster {
    agents: [AgentEntry; PAT_SIZE],
    user_hash: u32,
    #[allow(dead_code)]
    created_at: u64,
    total_tasks_routed: u64,
}

impl AgentRoster {
    /// Create a new PAT for a user
    pub fn new(user_hash: u32, timestamp: u64) -> Self {
        let roles = AgentRole::all();
        let agents = core::array::from_fn(|i| AgentEntry::new(roles[i], user_hash, timestamp));

        Self {
            agents,
            user_hash,
            created_at: timestamp,
            total_tasks_routed: 0,
        }
    }

    /// Get agent by role
    pub fn get(&self, role: AgentRole) -> &AgentEntry {
        &self.agents[role as usize]
    }

    /// Get mutable agent by role
    pub fn get_mut(&mut self, role: AgentRole) -> &mut AgentEntry {
        &mut self.agents[role as usize]
    }

    /// Check if a role is available for task assignment
    pub fn is_available(&self, role: AgentRole) -> bool {
        self.agents[role as usize].is_available()
    }

    /// Mark agent as busy
    pub fn mark_busy(&mut self, role: AgentRole) {
        self.agents[role as usize].state = AgentState::Busy;
    }

    /// Mark agent as idle
    pub fn mark_idle(&mut self, role: AgentRole) {
        if self.agents[role as usize].state == AgentState::Busy {
            self.agents[role as usize].state = AgentState::Idle;
        }
    }

    /// Suspend an agent
    pub fn suspend(&mut self, role: AgentRole) {
        self.agents[role as usize].state = AgentState::Suspended;
    }

    /// Resume a suspended agent
    pub fn resume(&mut self, role: AgentRole) {
        if self.agents[role as usize].state == AgentState::Suspended {
            self.agents[role as usize].state = AgentState::Idle;
        }
    }

    /// Assign a task to an agent
    pub fn assign_task(&mut self, role: AgentRole) -> Option<AgentId> {
        let agent = &mut self.agents[role as usize];
        if agent.is_available() {
            agent.state = AgentState::Busy;
            self.total_tasks_routed += 1;
            Some(agent.id)
        } else {
            None
        }
    }

    /// Complete a task for an agent
    pub fn complete_task(
        &mut self,
        role: AgentRole,
        duration_us: u64,
        confidence: Confidence,
        timestamp: u64,
    ) {
        let agent = &mut self.agents[role as usize];
        agent.record_completion(duration_us, confidence, timestamp);
        agent.state = AgentState::Idle;
    }

    /// Fail a task for an agent
    pub fn fail_task(&mut self, role: AgentRole, timestamp: u64) {
        let agent = &mut self.agents[role as usize];
        agent.record_failure(timestamp);
        agent.state = AgentState::Idle;
    }

    /// Record a veto by Guardian
    pub fn record_veto(&mut self, timestamp: u64) {
        self.agents[AgentRole::Guardian as usize].record_veto(timestamp);
    }

    /// Update إحسان for all agents
    pub fn update_ihsan_all(&mut self, score: IhsanScore) {
        for agent in self.agents.iter_mut() {
            agent.update_ihsan(score);
        }
    }

    /// How many agents are available?
    pub fn available_count(&self) -> usize {
        self.agents.iter().filter(|a| a.is_available()).count()
    }

    /// How many agents are degraded?
    pub fn degraded_count(&self) -> usize {
        self.agents
            .iter()
            .filter(|a| a.state == AgentState::Degraded)
            .count()
    }

    /// All agents
    pub fn all(&self) -> &[AgentEntry; PAT_SIZE] {
        &self.agents
    }

    /// Team health: weighted average of agent success rates
    pub fn team_health(&self) -> f32 {
        let weighted_sum: f32 = self
            .agents
            .iter()
            .map(|a| a.success_rate() * a.role.consensus_weight())
            .sum();
        let weight_sum: f32 = self.agents.iter().map(|a| a.role.consensus_weight()).sum();
        if weight_sum == 0.0 {
            return 1.0;
        }
        weighted_sum / weight_sum
    }

    pub fn user_hash(&self) -> u32 {
        self.user_hash
    }

    pub fn total_tasks_routed(&self) -> u64 {
        self.total_tasks_routed
    }

    /// Roster snapshot for observability
    pub fn snapshot(&self) -> RosterSnapshot {
        RosterSnapshot {
            user_hash: self.user_hash,
            agents_available: self.available_count() as u8,
            agents_degraded: self.degraded_count() as u8,
            total_tasks_routed: self.total_tasks_routed,
            team_health: self.team_health(),
            guardian_vetoes: self.agents[AgentRole::Guardian as usize].tasks_vetoed,
        }
    }
}

// ============================================================
// ROSTER SNAPSHOT
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct RosterSnapshot {
    pub user_hash: u32,
    pub agents_available: u8,
    pub agents_degraded: u8,
    pub total_tasks_routed: u64,
    pub team_health: f32,
    pub guardian_vetoes: u64,
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roster_creates_seven_agents() {
        let roster = AgentRoster::new(0xBEEF, 1000);
        assert_eq!(roster.all().len(), 7);
    }

    #[test]
    fn all_agents_start_idle() {
        let roster = AgentRoster::new(0xBEEF, 1000);
        for agent in roster.all() {
            assert_eq!(agent.state, AgentState::Idle);
            assert!(agent.is_available());
        }
        assert_eq!(roster.available_count(), 7);
    }

    #[test]
    fn assign_task_marks_busy() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let agent_id = roster.assign_task(AgentRole::Scholar);

        assert!(agent_id.is_some());
        assert!(!roster.is_available(AgentRole::Scholar));
        assert_eq!(roster.available_count(), 6);
    }

    #[test]
    fn cannot_assign_to_busy_agent() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.assign_task(AgentRole::Artisan);
        let second = roster.assign_task(AgentRole::Artisan);
        assert!(second.is_none());
    }

    #[test]
    fn complete_task_marks_idle() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.assign_task(AgentRole::Scholar);

        roster.complete_task(AgentRole::Scholar, 500, Confidence::stated(0), 1500);

        assert!(roster.is_available(AgentRole::Scholar));
        assert_eq!(roster.get(AgentRole::Scholar).tasks_completed, 1);
    }

    #[test]
    fn fail_task_tracks_failure() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.assign_task(AgentRole::Oracle);
        roster.fail_task(AgentRole::Oracle, 1500);

        let oracle = roster.get(AgentRole::Oracle);
        assert_eq!(oracle.tasks_failed, 1);
        assert!(oracle.is_available()); // Should be idle after failure
    }

    #[test]
    fn guardian_veto_tracking() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.record_veto(1500);
        roster.record_veto(2000);

        assert_eq!(roster.get(AgentRole::Guardian).tasks_vetoed, 2);
        assert_eq!(roster.snapshot().guardian_vetoes, 2);
    }

    #[test]
    fn ihsan_degradation() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.update_ihsan_all(IhsanScore::from_raw(9000));

        assert_eq!(roster.degraded_count(), 7);
        for agent in roster.all() {
            assert_eq!(agent.state, AgentState::Degraded);
        }

        // Recovery
        roster.update_ihsan_all(IhsanScore::from_raw(9900));
        assert_eq!(roster.degraded_count(), 0);
    }

    #[test]
    fn suspend_and_resume() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.suspend(AgentRole::Oracle);

        assert!(!roster.is_available(AgentRole::Oracle));
        assert_eq!(roster.available_count(), 6);

        roster.resume(AgentRole::Oracle);
        assert!(roster.is_available(AgentRole::Oracle));
    }

    #[test]
    fn team_health_starts_perfect() {
        let roster = AgentRoster::new(0xBEEF, 1000);
        assert!((roster.team_health() - 1.0).abs() < 0.001);
    }

    #[test]
    fn team_health_degrades_with_failures() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);

        // Simulate some failures
        for _ in 0..5 {
            roster.assign_task(AgentRole::Scholar);
            roster.fail_task(AgentRole::Scholar, 1500);
        }

        assert!(roster.team_health() < 1.0);
    }

    #[test]
    fn roster_snapshot() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        roster.assign_task(AgentRole::Navigator);
        roster.complete_task(AgentRole::Navigator, 100, Confidence::stated(0), 1100);

        let snap = roster.snapshot();
        assert_eq!(snap.user_hash, 0xBEEF);
        assert_eq!(snap.total_tasks_routed, 1);
        assert_eq!(snap.agents_available, 7);
    }

    #[test]
    fn agent_success_rate() {
        let mut entry = AgentEntry::new(AgentRole::Scholar, 0xBEEF, 1000);
        entry.record_completion(100, Confidence::stated(0), 1100);
        entry.record_completion(200, Confidence::inferred(0), 1200);
        entry.record_failure(1300);

        assert!((entry.success_rate() - 0.6667).abs() < 0.01);
    }
}
