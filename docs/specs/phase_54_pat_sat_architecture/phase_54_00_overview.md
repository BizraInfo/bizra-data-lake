# Phase 54.0: PAT/SAT Dual-Team Architecture — Overview

> Standing on Giants: Minsky (Society of Mind, 1986) · Dijkstra (separation of concerns, 1974) · Lamport (Byzantine fault tolerance, 1982) · Al-Ghazali (Ihsan self-governance, 1095) · General Magic (mobile agents, 1994) · Nakamoto (decentralized consensus, 2008) · Shannon (resource-information duality, 1948)

## 1. Problem Statement

Today's blockchain ecosystems have a fatal architectural flaw: every node connects
directly to the network. This creates:

- **Attack surface explosion**: each node is a direct entry point
- **No separation of concerns**: user logic and system logic share the same runtime
- **Resource starvation**: user tasks compete with consensus/validation tasks
- **No self-correction**: the system cannot optimize itself independently of users
- **Security theater**: validators and users operate in the same trust boundary

BIZRA solves this with a **dual-team architecture** that splits every node into two
independent agentic teams with fundamentally different mandates.

## 2. The Split: PAT vs SAT

```
                    ┌─────────────────────────────────┐
                    │         BIZRA NODE (12 Agents)   │
                    │                                   │
                    │  ┌─────────┐    ┌─────────┐      │
                    │  │  PAT-7  │◄──►│  SAT-5  │      │
                    │  │ (User)  │    │(System) │      │
                    │  └────┬────┘    └────┬────┘      │
                    │       │              │            │
                    │       ▼              ▼            │
                    │   User Goals    System Health     │
                    │   User Data     Resource Pool     │
                    │   User Tasks    Security Gate     │
                    └──────────────────┬───────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  Universal        │
                              │  Resource Pool    │
                              │  (URP)            │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │   BIZRA Network   │
                              └──────────────────┘
```

### PAT — Personal Agentic Team (7 Agents)

**Mandate**: Serve the USER. Only the user. Forever.

| Property | Value |
|----------|-------|
| Count | 7 per user |
| Created | At user onboarding (minted) |
| Loyalty | User only — never the system |
| Growth | Adapts to user's goals, dreams, skills over time |
| Personality | Customized, personalized, evolves with user |
| Communication | User talks directly to PAT |
| Mode | Scales from `reactive` to `proactive_partner` based on trust |
| Persistence | Lives as long as the user account exists |
| Data access | User's data only (sovereign boundary) |

### SAT — System Agentic Team (5 Agents)

**Mandate**: Serve the SYSTEM. Self-sustainable. Self-correcting.

| Property | Value |
|----------|-------|
| Count | 5 per user (contributed to system) |
| Created | Simultaneously with PAT at onboarding |
| Loyalty | System only — never any individual user |
| Growth | Optimizes for system-wide health metrics |
| Personality | None — pure function, constitutional constraints only |
| Communication | User NEVER talks to SAT directly. PAT talks to SAT when needed. |
| Mode | Always `proactive_partner` (24/7 daemon) |
| Persistence | Lives in the Universal Resource Pool |
| Data access | System-wide metrics, resource inventory, security state |

## 3. The 12-Agent Node Composition

When a new user joins BIZRA:

```pseudocode
FUNCTION onboard_new_user(identity: UserIdentity) -> Node:
    # Step 1: Mint PAT-7 (belongs to user)
    pat = mint_pat_team(identity)          # 7 agents, user-sovereign

    # Step 2: Mint SAT-5 (belongs to system)
    sat = mint_sat_team(identity.node_id)  # 5 agents, system-sovereign

    # Step 3: PAT stays with user
    node.attach_pat(pat)

    # Step 4: SAT moves to Universal Resource Pool
    urp.register_sat_team(sat)             # SAT joins the collective
    urp.pledge_resources(node.hardware)    # Node pledges compute/storage

    # Step 5: Node connects through URP (NEVER directly to network)
    node.network_gateway = urp.get_gateway()

    RETURN node
```

## 4. The Security Innovation

### Traditional Blockchain (Broken)
```
Node A ──────► Network ◄────── Node B
Node C ──────► Network ◄────── Attacker
                                  ↑
                          Direct access to network!
```

### BIZRA Architecture (Fixed)
```
User A ──► PAT-7 ──► SAT-5 ──┐
User B ──► PAT-7 ──► SAT-5 ──┤
User C ──► PAT-7 ──► SAT-5 ──┼──► URP ──► Network
Attacker ──► PAT-7 ──► SAT-5 ─┘    ↑
                                     │
                              SAT validates everything
                              before it reaches network.
                              Constitutional gates.
                              No direct access.
```

**Key insight**: The attacker CANNOT reach the network directly. They must go through:
1. Their own PAT (which serves them but cannot bypass system rules)
2. The SAT layer (which serves the system and blocks malicious actions)
3. The URP (which validates resource pledges and consensus)

Three independent trust boundaries vs. blockchain's zero.

## 5. Network Effect: More Users = Stronger System

```pseudocode
# Every new user adds:
#   +7 PAT agents (serve that user)
#   +5 SAT agents (serve the WHOLE system)
#   +N resources (compute, storage, bandwidth pledged to URP)

WHEN user_count = 1:
    system_sat_count = 5      # Minimal self-governance
    urp_capacity = 1 node     # Single node resources

WHEN user_count = 100:
    system_sat_count = 500    # 500 system agents self-organizing
    urp_capacity = 100 nodes  # 100x resource pool

WHEN user_count = 1_000_000:
    system_sat_count = 5_000_000  # 5M system agents
    urp_capacity = 1M nodes       # Massive distributed compute

    # At this scale, SAT teams form specialized departments:
    #   - Security Department (1M+ guardians)
    #   - Resource Department (1M+ allocators)
    #   - Healing Department (1M+ self-repair agents)
    #   - Audit Department (1M+ compliance validators)
    #   - Herald Department (1M+ network coordinators)
```

The more users join, the more SAT agents exist, the stronger the system becomes
at self-governance, self-healing, security, and resource optimization. This is
**antifragile** — the system gets BETTER under load, not worse.

## 6. Connection Flow

```
User Device
    │
    ▼
PAT-7 (user's personal agents)
    │
    │ PAT needs resources / network access
    │
    ▼
SAT-5 (system validation layer)
    │
    │ SAT validates: constitutional? resource-available? safe?
    │
    ▼
Universal Resource Pool (URP)
    │
    │ URP: consensus, routing, resource allocation
    │
    ▼
BIZRA Network (other nodes' URPs)
```

No node ever touches the network directly. Every request flows through:
**User → PAT → SAT → URP → Network**

## 7. Spec Modules (This Phase)

| Module | File | Focus |
|--------|------|-------|
| 54.00 | `phase_54_00_overview.md` | This document — architecture overview |
| 54.01 | `phase_54_01_pat_lifecycle.md` | PAT-7 minting, personalization, growth |
| 54.02 | `phase_54_02_sat_lifecycle.md` | SAT-5 minting, URP registration, daemon mode |
| 54.03 | `phase_54_03_pat_sat_interface.md` | How PAT talks to SAT (protocol contract) |
| 54.04 | `phase_54_04_urp_architecture.md` | Universal Resource Pool design |
| 54.05 | `phase_54_05_security_model.md` | Three-boundary security, attack surface analysis |
| 54.06 | `phase_54_06_scaling_topology.md` | SAT department formation at scale |
| 54.07 | `phase_54_07_tdd_anchors.md` | Test plan for all modules |

## 8. Constitutional Constraints

Both PAT and SAT operate under the BIZRA constitution, but with different gates:

| Gate | PAT | SAT |
|------|-----|-----|
| Ihsan (excellence) | >= 0.95 | >= 0.99 (stricter) |
| Daughter Test | Yes | Yes |
| ADL Gini (fairness) | <= 0.35 | <= 0.35 |
| SNR (signal quality) | >= 0.85 | >= 0.98 (stricter) |
| Resource budget | User's allocation | System pool |
| Mode | User-selected | Always `proactive_partner` |
| Human approval | Required (except auto mode) | Never (autonomous) |

SAT has STRICTER thresholds because it operates autonomously without human oversight.
The constitution IS the oversight.

## 9. Relation to Existing Specs

| Existing Spec | Relation |
|--------------|----------|
| Phase 25 (Genesis Bootstrap) | Genesis ceremony mints both PAT and SAT |
| Phase 29 (Primordial Activation) | PAT-7 / SAT-5 first mentioned in activation flow |
| Phase 30 (DDAGI OS Definition) | `pat_agents: Dict[str, PATAgent]` + `sat_agents: Dict[str, SATAgent]` |
| Phase 37 (DDAGI v4 Genesis) | SAT-5 → SAT-49 scaling roadmap |
| Phase 52.3 (PAT-7 Pipeline) | Detailed PAT-7 chain of reasoning |
| SAPE Adapter Spec | `wrap_sat_action()` constitutional gate |

This spec (Phase 54) is the **canonical, unified** PAT/SAT architecture document.
All prior references are consistent but fragmentary. This consolidates them.
