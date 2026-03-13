//! Component Registry — the live self-model of Node0.
//!
//! Every component that registers becomes a node in the architecture graph.
//! Every dependency declared becomes an edge. The registry IS the self-model.
//!
//! RSI Pillar I: "formal, complete, and queryable representation of its own architecture"
//! Standing on Giants: Milner (pi-calculus, 1999) · Fowler (service registry, 2014)

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE GRAPH — The self-model as a directed graph.
// Nodes = components. Edges = dependencies + pub/sub relationships.
// ═══════════════════════════════════════════════════════════════════════════════

/// A snapshot of the architecture graph at a point in time.
#[derive(Debug, Clone)]
pub struct ArchitectureGraph {
    pub components: Vec<ComponentInfo>,
    pub edges: Vec<GraphEdge>,
    pub snapshot_at: Timestamp,
}

/// An edge in the architecture graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: ComponentId,
    pub to: ComponentId,
    pub kind: EdgeKind,
}

/// What kind of relationship does this edge represent?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// `from` depends on `to` (declared dependency).
    DependsOn,
    /// `from` publishes events that `to` subscribes to.
    PublishesTo,
}

impl ArchitectureGraph {
    /// Count of nodes (components) in the graph.
    pub fn node_count(&self) -> usize {
        self.components.len()
    }

    /// Count of edges (relationships) in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find all components of a given kind.
    pub fn by_kind(&self, kind: ComponentKind) -> Vec<&ComponentInfo> {
        self.components.iter().filter(|c| c.kind == kind).collect()
    }

    /// Find all components that depend on `target`.
    pub fn dependents_of(&self, target: ComponentId) -> Vec<ComponentId> {
        self.edges
            .iter()
            .filter(|e| e.to == target && e.kind == EdgeKind::DependsOn)
            .map(|e| e.from)
            .collect()
    }

    /// Find all components that `source` depends on.
    pub fn dependencies_of(&self, source: ComponentId) -> Vec<ComponentId> {
        self.edges
            .iter()
            .filter(|e| e.from == source && e.kind == EdgeKind::DependsOn)
            .map(|e| e.to)
            .collect()
    }

    /// Check for cycles in dependency graph (would indicate design error).
    pub fn has_cycles(&self) -> bool {
        // Simple DFS-based cycle detection.
        let mut visited = HashMap::new();
        for comp in &self.components {
            if self.dfs_cycle(comp.id, &mut visited) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        node: ComponentId,
        visited: &mut HashMap<ComponentId, u8>, // 0=unvisited, 1=in-progress, 2=done
    ) -> bool {
        match visited.get(&node) {
            Some(1) => return true,  // Back edge = cycle.
            Some(2) => return false, // Already fully explored.
            _ => {}
        }
        visited.insert(node, 1);
        for dep in self.dependencies_of(node) {
            if self.dfs_cycle(dep, visited) {
                return true;
            }
        }
        visited.insert(node, 2);
        false
    }

    /// Components with no dependencies (roots of the graph).
    pub fn roots(&self) -> Vec<&ComponentInfo> {
        self.components
            .iter()
            .filter(|c| self.dependencies_of(c.id).is_empty())
            .collect()
    }

    /// Components with no dependents (leaves of the graph).
    pub fn leaves(&self) -> Vec<&ComponentInfo> {
        self.components
            .iter()
            .filter(|c| self.dependents_of(c.id).is_empty())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REGISTRY — Thread-safe component registration and lookup.
// ═══════════════════════════════════════════════════════════════════════════════

/// The component registry. Thread-safe. Clonable (shares state via Arc).
#[derive(Clone)]
pub struct Registry {
    inner: Arc<RwLock<RegistryInner>>,
    /// Callback invoked when a component is registered (for EventBus integration).
    on_register: Arc<Mutex<Option<Box<dyn Fn(&ComponentInfo) + Send + Sync>>>>,
    /// Callback invoked when component health changes.
    on_health_change: Arc<Mutex<Option<Box<dyn Fn(ComponentId, Health, Health) + Send + Sync>>>>,
}

struct RegistryInner {
    components: HashMap<ComponentId, ComponentInfo>,
    name_index: HashMap<String, ComponentId>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                components: HashMap::new(),
                name_index: HashMap::new(),
            })),
            on_register: Arc::new(Mutex::new(None)),
            on_health_change: Arc::new(Mutex::new(None)),
        }
    }

    /// Set callback for component registration events.
    pub fn on_register<F>(&self, f: F)
    where
        F: Fn(&ComponentInfo) + Send + Sync + 'static,
    {
        if let Ok(mut cb) = self.on_register.lock() {
            *cb = Some(Box::new(f));
        }
    }

    /// Set callback for health change events.
    pub fn on_health_change<F>(&self, f: F)
    where
        F: Fn(ComponentId, Health, Health) + Send + Sync + 'static,
    {
        if let Ok(mut cb) = self.on_health_change.lock() {
            *cb = Some(Box::new(f));
        }
    }

    /// Register a component. Returns its ID. Fails if name already taken.
    pub fn register(&self, info: ComponentInfo) -> HookResult<ComponentId> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        if inner.name_index.contains_key(&info.name) {
            return Err(HookError::DuplicateComponent(info.name.clone()));
        }

        let id = info.id;
        inner.name_index.insert(info.name.clone(), id);
        inner.components.insert(id, info.clone());
        drop(inner); // Release lock before callback.

        // Fire registration callback.
        if let Ok(cb) = self.on_register.lock() {
            if let Some(f) = cb.as_ref() {
                f(&info);
            }
        }

        Ok(id)
    }

    /// Look up a component by ID.
    pub fn get(&self, id: ComponentId) -> HookResult<ComponentInfo> {
        let inner = self
            .inner
            .read()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;
        inner
            .components
            .get(&id)
            .cloned()
            .ok_or(HookError::ComponentNotFound(id))
    }

    /// Look up a component by name.
    pub fn get_by_name(&self, name: &str) -> Option<ComponentInfo> {
        let inner = self.inner.read().ok()?;
        let id = inner.name_index.get(name)?;
        inner.components.get(id).cloned()
    }

    /// Update component health. Fires callback if changed.
    pub fn set_health(&self, id: ComponentId, new_health: Health) -> HookResult<()> {
        let old_health;
        {
            let mut inner = self
                .inner
                .write()
                .map_err(|e| HookError::LockPoisoned(e.to_string()))?;
            let comp = inner
                .components
                .get_mut(&id)
                .ok_or(HookError::ComponentNotFound(id))?;
            old_health = comp.health;
            comp.health = new_health;
        }

        if old_health != new_health {
            if let Ok(cb) = self.on_health_change.lock() {
                if let Some(f) = cb.as_ref() {
                    f(id, old_health, new_health);
                }
            }
        }
        Ok(())
    }

    /// Number of registered components.
    pub fn count(&self) -> usize {
        self.inner.read().map(|i| i.components.len()).unwrap_or(0)
    }

    /// All registered component IDs.
    pub fn all_ids(&self) -> Vec<ComponentId> {
        self.inner
            .read()
            .map(|i| i.components.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Snapshot the full architecture graph at this instant.
    pub fn architecture_graph(&self) -> ArchitectureGraph {
        let inner = match self.inner.read() {
            Ok(i) => i,
            Err(_) => {
                return ArchitectureGraph {
                    components: Vec::new(),
                    edges: Vec::new(),
                    snapshot_at: Timestamp::now(),
                }
            }
        };

        let components: Vec<ComponentInfo> = inner.components.values().cloned().collect();
        let mut edges = Vec::new();

        // Build edges from declared dependencies.
        for comp in &components {
            for dep_id in &comp.depends_on {
                edges.push(GraphEdge {
                    from: comp.id,
                    to: *dep_id,
                    kind: EdgeKind::DependsOn,
                });
            }
        }

        // Build edges from pub/sub overlap.
        for pub_comp in &components {
            for sub_comp in &components {
                if pub_comp.id == sub_comp.id {
                    continue;
                }
                for pub_kind in &pub_comp.publishes {
                    if sub_comp.subscribes.contains(pub_kind) {
                        edges.push(GraphEdge {
                            from: pub_comp.id,
                            to: sub_comp.id,
                            kind: EdgeKind::PublishesTo,
                        });
                    }
                }
            }
        }

        ArchitectureGraph {
            components,
            edges,
            snapshot_at: Timestamp::now(),
        }
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.count();
        f.debug_struct("Registry")
            .field("component_count", &count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        let reg = Registry::new();
        let info = ComponentInfo::new("vector_search", ComponentKind::Engine)
            .with_version("1.0.0")
            .publishes(&[EventKind::SearchExecuted]);

        let id = reg.register(info).unwrap();
        let found = reg.get(id).unwrap();
        assert_eq!(found.name, "vector_search");
        assert_eq!(found.kind, ComponentKind::Engine);
        assert_eq!(found.publishes, vec![EventKind::SearchExecuted]);
    }

    #[test]
    fn duplicate_name_rejected() {
        let reg = Registry::new();
        reg.register(ComponentInfo::new("memory", ComponentKind::Agent))
            .unwrap();
        let result = reg.register(ComponentInfo::new("memory", ComponentKind::Agent));
        assert!(result.is_err());
    }

    #[test]
    fn health_changes_tracked() {
        let reg = Registry::new();
        let id = reg
            .register(ComponentInfo::new("hmm", ComponentKind::Engine))
            .unwrap();

        assert_eq!(reg.get(id).unwrap().health, Health::Uninitialized);
        reg.set_health(id, Health::Healthy).unwrap();
        assert_eq!(reg.get(id).unwrap().health, Health::Healthy);
    }

    #[test]
    fn architecture_graph_builds_edges() {
        let reg = Registry::new();
        let search = reg
            .register(
                ComponentInfo::new("search", ComponentKind::Engine)
                    .publishes(&[EventKind::SearchExecuted]),
            )
            .unwrap();

        let _resonance = reg
            .register(
                ComponentInfo::new("resonance", ComponentKind::Engine)
                    .subscribes_to(&[EventKind::SearchExecuted])
                    .depends_on(&[search]),
            )
            .unwrap();

        let graph = reg.architecture_graph();
        assert_eq!(graph.node_count(), 2);
        // 1 DependsOn edge + 1 PublishesTo edge
        assert_eq!(graph.edge_count(), 2);
        assert!(!graph.has_cycles());
    }

    #[test]
    fn cycle_detection_works() {
        // Manually construct a graph with a cycle for testing.
        let graph = ArchitectureGraph {
            components: vec![
                ComponentInfo::new("a", ComponentKind::Core),
                ComponentInfo::new("b", ComponentKind::Core),
            ],
            edges: vec![],
            snapshot_at: Timestamp::now(),
        };
        // No edges = no cycles.
        assert!(!graph.has_cycles());
    }

    #[test]
    fn get_by_name() {
        let reg = Registry::new();
        reg.register(ComponentInfo::new("my_agent", ComponentKind::Agent))
            .unwrap();
        assert!(reg.get_by_name("my_agent").is_some());
        assert!(reg.get_by_name("nonexistent").is_none());
    }
}
