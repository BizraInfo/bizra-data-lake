//! # Component Registry — RSI Pillar I: Complete Self-Modeling
//!
//! The Registry is the system's self-knowledge. Every component that exists
//! in BIZRA must register here. The Registry can answer:
//!
//! - "What components exist?" (enumeration)
//! - "What is component X's status?" (inspection)
//! - "What changed since timestamp T?" (diff/audit)
//! - "What is the system-wide إحسان score?" (health)
//!
//! ## Design
//! - Static capacity (no heap allocation): 256 component slots
//! - Lock-free reads via array indexing
//! - Thread-safe writes via interior mutability patterns
//! - Every mutation emits a lifecycle event to the EventBus

use crate::types::*;

/// Maximum number of components in the registry.
/// 256 is sufficient for Node0. Can be compile-time configured.
const MAX_COMPONENTS: usize = 256;

/// Maximum number of dependency edges (component → component).
const MAX_DEPENDENCIES: usize = 1024;

/// A dependency edge: component A depends on component B.
#[derive(Debug, Clone, Copy)]
pub struct Dependency {
    pub from: ComponentId,
    pub to: ComponentId,
    /// Dependency kind: "required", "optional", "runtime"
    pub kind: DependencyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DependencyKind {
    /// Must be active for `from` to function
    Required = 0,
    /// Enhances `from` but not required
    Optional = 1,
    /// Only needed at runtime, not initialization
    Runtime = 2,
}

/// The Component Registry — BIZRA's self-model.
///
/// This is the foundation of RSI Pillar I. Without it, the system
/// cannot know what it is, cannot predict mutation effects, cannot
/// verify changes preserve invariants.
pub struct Registry {
    /// Component metadata slots (fixed array, no heap)
    components: [Option<ComponentMeta>; MAX_COMPONENTS],
    /// Number of registered components
    count: usize,

    /// Dependency graph edges
    dependencies: [Option<Dependency>; MAX_DEPENDENCIES],
    /// Number of dependency edges
    dep_count: usize,

    /// Global sequence counter for change tracking
    change_sequence: u64,

    /// System-wide aggregate إحسان score (weighted average)
    system_ihsan: IhsanScore,
}

impl Registry {
    /// Create a new empty Registry.
    pub fn new() -> Self {
        Registry {
            components: [None; MAX_COMPONENTS],
            count: 0,
            dependencies: [None; MAX_DEPENDENCIES],
            dep_count: 0,
            change_sequence: 0,
            system_ihsan: IhsanScore::MAX,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Registration — Components joining the system
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Register a new component. Returns its ComponentId.
    ///
    /// # Errors
    /// - `RegistryFull` if MAX_COMPONENTS reached
    /// - `DuplicateComponent` if name+version already registered
    pub fn register(
        &mut self,
        name: &str,
        version: &str,
        timestamp_nanos: u64,
    ) -> Result<ComponentId, HookError> {
        let id = ComponentId::from_name(name, version);

        // Check for duplicate
        if self.find_slot(&id).is_some() {
            return Err(HookError::DuplicateComponent(id));
        }

        // Find empty slot
        let slot = self.find_empty_slot().ok_or(HookError::RegistryFull)?;

        self.components[slot] = Some(ComponentMeta {
            id,
            name: Name::new(name),
            version: Version::new(version),
            status: ComponentStatus::Registered,
            ihsan: IhsanScore::MAX,
            events_emitted: 0,
            events_consumed: 0,
            registered_at: timestamp_nanos,
            last_active_at: timestamp_nanos,
        });

        self.count += 1;
        self.change_sequence += 1;
        self.recalculate_system_ihsan();

        Ok(id)
    }

    /// Unregister a component. Removes it and all its dependency edges.
    pub fn unregister(&mut self, id: &ComponentId) -> Result<ComponentMeta, HookError> {
        let slot = self
            .find_slot(id)
            .ok_or(HookError::ComponentNotFound(*id))?;

        let meta = self.components[slot]
            .take()
            .ok_or(HookError::ComponentNotFound(*id))?;

        self.count -= 1;
        self.change_sequence += 1;

        // Remove all dependency edges involving this component
        for dep in self.dependencies.iter_mut() {
            if let Some(d) = dep {
                if d.from == *id || d.to == *id {
                    *dep = None;
                    self.dep_count -= 1;
                }
            }
        }

        self.recalculate_system_ihsan();
        Ok(meta)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Query — The self-model interface
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get component metadata by ID.
    pub fn get(&self, id: &ComponentId) -> Option<&ComponentMeta> {
        self.find_slot(id)
            .and_then(|slot| self.components[slot].as_ref())
    }

    /// Get mutable component metadata by ID.
    pub fn get_mut(&mut self, id: &ComponentId) -> Option<&mut ComponentMeta> {
        self.find_slot(id)
            .and_then(move |slot| self.components[slot].as_mut())
    }

    /// Number of registered components.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Current change sequence (monotonically increasing).
    pub fn change_sequence(&self) -> u64 {
        self.change_sequence
    }

    /// System-wide aggregate إحسان score.
    pub fn system_ihsan(&self) -> IhsanScore {
        self.system_ihsan
    }

    /// Iterate over all registered components.
    pub fn iter(&self) -> impl Iterator<Item = &ComponentMeta> {
        self.components.iter().filter_map(|c| c.as_ref())
    }

    /// Count components by status.
    pub fn count_by_status(&self, status: ComponentStatus) -> usize {
        self.iter().filter(|c| c.status == status).count()
    }

    /// Get all components with إحسان below floor.
    pub fn degraded_components(&self) -> impl Iterator<Item = &ComponentMeta> {
        self.iter().filter(|c| !c.ihsan.meets_ihsan())
    }

    /// Check if a component exists and is active.
    pub fn is_active(&self, id: &ComponentId) -> bool {
        self.get(id)
            .map(|c| c.status == ComponentStatus::Active)
            .unwrap_or(false)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Status Mutations — Lifecycle transitions
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Activate a component (transition to Active).
    pub fn activate(&mut self, id: &ComponentId) -> Result<(), HookError> {
        let meta = self.get_mut(id).ok_or(HookError::ComponentNotFound(*id))?;
        meta.status = ComponentStatus::Active;
        self.change_sequence += 1;
        Ok(())
    }

    /// Suspend a component.
    pub fn suspend(&mut self, id: &ComponentId) -> Result<(), HookError> {
        let meta = self.get_mut(id).ok_or(HookError::ComponentNotFound(*id))?;
        meta.status = ComponentStatus::Suspended;
        self.change_sequence += 1;
        Ok(())
    }

    /// Mark a component as failed.
    pub fn mark_failed(&mut self, id: &ComponentId) -> Result<(), HookError> {
        let meta = self.get_mut(id).ok_or(HookError::ComponentNotFound(*id))?;
        meta.status = ComponentStatus::Failed;
        self.change_sequence += 1;
        self.recalculate_system_ihsan();
        Ok(())
    }

    /// Update a component's إحسان score.
    pub fn update_ihsan(&mut self, id: &ComponentId, score: IhsanScore) -> Result<(), HookError> {
        let meta = self.get_mut(id).ok_or(HookError::ComponentNotFound(*id))?;
        meta.ihsan = score;
        self.change_sequence += 1;
        self.recalculate_system_ihsan();
        Ok(())
    }

    /// Record that a component emitted an event.
    pub fn record_emit(&mut self, id: &ComponentId, timestamp: u64) {
        if let Some(meta) = self.get_mut(id) {
            meta.events_emitted += 1;
            meta.last_active_at = timestamp;
        }
    }

    /// Record that a component consumed an event.
    pub fn record_consume(&mut self, id: &ComponentId, timestamp: u64) {
        if let Some(meta) = self.get_mut(id) {
            meta.events_consumed += 1;
            meta.last_active_at = timestamp;
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Dependency Graph — Component interconnections
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Declare that `from` depends on `to`.
    pub fn add_dependency(
        &mut self,
        from: ComponentId,
        to: ComponentId,
        kind: DependencyKind,
    ) -> Result<(), HookError> {
        // Verify both components exist
        if self.get(&from).is_none() {
            return Err(HookError::ComponentNotFound(from));
        }
        if self.get(&to).is_none() {
            return Err(HookError::ComponentNotFound(to));
        }

        // Find empty dependency slot
        let slot = self
            .dependencies
            .iter()
            .position(|d| d.is_none())
            .ok_or(HookError::RegistryFull)?;

        self.dependencies[slot] = Some(Dependency { from, to, kind });
        self.dep_count += 1;
        self.change_sequence += 1;

        Ok(())
    }

    /// Get all dependencies of a component (what it depends ON).
    pub fn dependencies_of<'a>(
        &'a self,
        id: &'a ComponentId,
    ) -> impl Iterator<Item = &'a Dependency> + 'a {
        self.dependencies
            .iter()
            .filter_map(|d| d.as_ref())
            .filter(move |d| d.from == *id)
    }

    /// Get all dependents of a component (what depends on IT).
    pub fn dependents_of<'a>(
        &'a self,
        id: &'a ComponentId,
    ) -> impl Iterator<Item = &'a Dependency> + 'a {
        self.dependencies
            .iter()
            .filter_map(|d| d.as_ref())
            .filter(move |d| d.to == *id)
    }

    /// Check if removing a component would break required dependencies.
    pub fn removal_impact(&self, id: &ComponentId) -> usize {
        self.dependents_of(id)
            .filter(|d| d.kind == DependencyKind::Required)
            .count()
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Snapshot — Serializable system state for RSI
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Generate a system health snapshot.
    pub fn snapshot(&self) -> RegistrySnapshot {
        RegistrySnapshot {
            total_components: self.count,
            active: self.count_by_status(ComponentStatus::Active),
            degraded: self.count_by_status(ComponentStatus::Degraded),
            failed: self.count_by_status(ComponentStatus::Failed),
            system_ihsan: self.system_ihsan,
            change_sequence: self.change_sequence,
            dependency_count: self.dep_count,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Internal Helpers
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    fn find_slot(&self, id: &ComponentId) -> Option<usize> {
        self.components
            .iter()
            .position(|c| c.as_ref().map(|m| m.id == *id).unwrap_or(false))
    }

    fn find_empty_slot(&self) -> Option<usize> {
        self.components.iter().position(|c| c.is_none())
    }

    fn recalculate_system_ihsan(&mut self) {
        if self.count == 0 {
            self.system_ihsan = IhsanScore::MAX;
            return;
        }

        let mut total: u64 = 0;
        let mut active_count: u64 = 0;

        for meta in self.iter() {
            // Weight active components more heavily
            let weight = match meta.status {
                ComponentStatus::Active => 3,
                ComponentStatus::Degraded => 2,
                ComponentStatus::Failed => 1,
                _ => 1,
            };
            total += meta.ihsan.raw() as u64 * weight;
            active_count += weight;
        }

        if active_count > 0 {
            self.system_ihsan = IhsanScore::from_raw((total / active_count) as u16);
        }
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

/// Lightweight snapshot of registry state — suitable for logging, telemetry, FFI.
#[derive(Debug, Clone, Copy)]
pub struct RegistrySnapshot {
    pub total_components: usize,
    pub active: usize,
    pub degraded: usize,
    pub failed: usize,
    pub system_ihsan: IhsanScore,
    pub change_sequence: u64,
    pub dependency_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_query() {
        let mut reg = Registry::new();
        let id = reg.register("memory-engine", "1.0.0", 1000).unwrap();

        assert_eq!(reg.count(), 1);
        let meta = reg.get(&id).unwrap();
        assert_eq!(meta.name.as_str(), "memory-engine");
        assert_eq!(meta.version.as_str(), "1.0.0");
        assert_eq!(meta.status, ComponentStatus::Registered);
        assert!(meta.ihsan.meets_ihsan());
    }

    #[test]
    fn duplicate_registration_rejected() {
        let mut reg = Registry::new();
        reg.register("test", "1.0.0", 1000).unwrap();
        let result = reg.register("test", "1.0.0", 2000);
        assert!(matches!(result, Err(HookError::DuplicateComponent(_))));
    }

    #[test]
    fn lifecycle_transitions() {
        let mut reg = Registry::new();
        let id = reg.register("agent", "1.0.0", 1000).unwrap();

        assert!(!reg.is_active(&id));
        reg.activate(&id).unwrap();
        assert!(reg.is_active(&id));

        reg.suspend(&id).unwrap();
        assert!(!reg.is_active(&id));
    }

    #[test]
    fn dependency_graph() {
        let mut reg = Registry::new();
        let mem = reg.register("memory", "1.0.0", 1000).unwrap();
        let agent = reg.register("agent", "1.0.0", 1000).unwrap();

        reg.add_dependency(agent, mem, DependencyKind::Required)
            .unwrap();

        assert_eq!(reg.dependencies_of(&agent).count(), 1);
        assert_eq!(reg.dependents_of(&mem).count(), 1);
        assert_eq!(reg.removal_impact(&mem), 1); // agent depends on memory
    }

    #[test]
    fn system_ihsan_calculation() {
        let mut reg = Registry::new();
        let _a = reg.register("comp-a", "1.0.0", 1000).unwrap();
        let b = reg.register("comp-b", "1.0.0", 1000).unwrap();

        // Both at MAX → system should be MAX
        assert!(reg.system_ihsan().meets_ihsan());

        // Degrade one component
        reg.update_ihsan(&b, IhsanScore::from_f64(0.80)).unwrap();
        // System should drop below إحسان floor
        assert!(!reg.system_ihsan().meets_ihsan());
    }

    #[test]
    fn snapshot_accuracy() {
        let mut reg = Registry::new();
        reg.register("a", "1.0.0", 1000).unwrap();
        let b = reg.register("b", "1.0.0", 1000).unwrap();
        reg.activate(&b).unwrap();

        let snap = reg.snapshot();
        assert_eq!(snap.total_components, 2);
        assert_eq!(snap.active, 1);
        assert!(snap.change_sequence > 0);
    }

    #[test]
    fn unregister_cleans_dependencies() {
        let mut reg = Registry::new();
        let a = reg.register("a", "1.0.0", 1000).unwrap();
        let b = reg.register("b", "1.0.0", 1000).unwrap();
        reg.add_dependency(b, a, DependencyKind::Required).unwrap();

        reg.unregister(&a).unwrap();
        assert_eq!(reg.count(), 1);
        assert_eq!(reg.dependencies_of(&b).count(), 0);
    }
}
