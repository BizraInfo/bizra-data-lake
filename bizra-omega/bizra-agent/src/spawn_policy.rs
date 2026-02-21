// bizra-agent/src/spawn_policy.rs
// ============================================================
// Spawn Policy — hard limits for sub-agent safety
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct SpawnPolicy {
    pub max_depth: u8,
    pub max_children_per_agent: u8,
    pub max_total_active: u16,
    pub permit_degradation: f32,
    pub guardian_on_spawn: bool,
}

impl Default for SpawnPolicy {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_children_per_agent: 5,
            max_total_active: 20,
            permit_degradation: 0.5,
            guardian_on_spawn: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnDenied {
    MaxDepthReached,
    MaxChildrenReached,
    GlobalLimitReached,
    GuardianDenied,
    PermitInsufficient,
}

impl SpawnDenied {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxDepthReached => "max_depth_reached",
            Self::MaxChildrenReached => "max_children_reached",
            Self::GlobalLimitReached => "global_limit_reached",
            Self::GuardianDenied => "guardian_denied",
            Self::PermitInsufficient => "permit_insufficient",
        }
    }
}
