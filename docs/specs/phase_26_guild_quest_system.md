# Phase 26: Guild & Quest System

> Collaborative communities + gamified impact missions, constitutionally gated by Ihsan.

## Context

Guilds and Quests form the social-incentive layer of BIZRA. Guilds are thematic communities where nodes collaborate on shared missions. Quests are gamified impact missions within guilds. Quest completion is gated by Ihsan threshold (>= 0.95) — nodes must meet constitutional excellence standards to claim rewards.

Both systems are in-memory for v1 with optional JSON persistence. Future versions migrate to `core.memory.unified_store` (AgentDB V3).

Standing on Giants: Ostrom (1990) — polycentric governance, McGonigal (2011) — gameful design, Szabo (1997) — smart contracts, Nakamoto (2008) — decentralized membership, Al-Ghazali — Ihsan as community excellence.

## Package Structure

```
core/guild/                          core/quest/
  __init__.py     (40 lines)           __init__.py     (39 lines)
  types.py        (121 lines)          types.py        (117 lines)
  registry.py     (189 lines)          engine.py       (265 lines)
```

Total: ~771 lines across both packages.

---

## Guild System

### Data Types

```
ENUM GuildStatus: PENDING | ACTIVE | SUSPENDED
ENUM GuildRole:   MEMBER | ELDER | STEWARD

DATACLASS GuildMember (frozen=True):
  node_id: str
  guild_id: str
  joined_at: str          # ISO 8601 UTC
  role: GuildRole = MEMBER
  ihsan_score: float = 0.0

DATACLASS Guild:
  guild_id: str
  name: str
  description: str
  members: List[GuildMember]
  status: GuildStatus = ACTIVE
  created_at: str         # ISO 8601 UTC

  PROPERTY member_count -> len(members)
  PROPERTY online_count -> max(1, len(members))  # Simulated; production uses heartbeat
  PROPERTY mean_ihsan -> sum(m.ihsan_score) / len(members)
  METHOD has_member(node_id) -> bool

DATACLASS GuildJoinResult:
  success: bool
  guild: Optional[Guild]
  member: Optional[GuildMember]
  message: str
```

### Default Guilds (Pre-seeded)

| ID | Name | Description |
|----|------|-------------|
| `agriculture` | Agriculture & Food Security | Sustainable farming, water management, food sovereignty |
| `healthcare` | Healthcare & Wellbeing | Community health, telemedicine, preventive care |
| `education` | Education & Knowledge | Open learning, skill development, mentorship |
| `energy` | Energy & Environment | Renewable energy, conservation, climate action |
| `finance` | Finance & Economic Justice | Microfinance, cooperative economics, fair trade |

### GuildRegistry API

```
CLASS GuildRegistry:
  INIT(persist_path: Optional[Path]):
    _guilds: Dict[str, Guild] = {}
    _seed_default_guilds()

  METHOD register_guild(guild_id, name, description) -> Guild:
    IF guild_id already exists: RETURN existing guild (idempotent)
    CREATE guild, store in _guilds, persist
    RETURN guild

  METHOD join_guild(guild_id, node_id, ihsan_score=0.0, role=MEMBER) -> GuildJoinResult:
    IF guild not found: RETURN failure result
    IF already a member: RETURN success with existing member
    CREATE GuildMember, append to guild.members
    persist()
    RETURN success result

  METHOD leave_guild(guild_id, node_id) -> bool:
    Filter out member from guild.members
    RETURN True if removed

  METHOD get_guild(guild_id) -> Optional[Guild]
  METHOD list_guilds() -> List[Guild]
  METHOD get_online_count(guild_id) -> int

  PRIVATE _persist():
    IF persist_path IS None: RETURN (in-memory mode)
    Write JSON to persist_path
```

### Persistence Strategy

```
v1 (current):  In-memory Dict + optional JSON file
v2 (planned):  core.memory.unified_store (SQLite v2 + FTS5)
v3 (future):   Federated sync across nodes via core.federation
```

---

## Quest System

### Data Types

```
ENUM QuestDifficulty: SEED | SPROUT | BLOOM | FOREST
ENUM QuestStatus:     AVAILABLE | ACCEPTED | IN_PROGRESS | COMPLETED

DATACLASS QuestReward:
  seed_amount: float = 0.0
  bloom_amount: float = 0.0
  impt_amount: float = 0.0
  description: str = ""

DATACLASS Quest:
  quest_id: str
  title: str
  description: str
  guild_id: str                       # Owning guild
  difficulty: QuestDifficulty
  reward: QuestReward
  status: QuestStatus = AVAILABLE
  accepted_by: Optional[str] = None   # node_id
  accepted_at: Optional[str] = None   # ISO 8601
  completed_at: Optional[str] = None  # ISO 8601

DATACLASS QuestAcceptResult:
  success: bool
  quest: Optional[Quest]
  message: str
```

### Default Quests (Pre-seeded)

| ID | Title | Guild | Difficulty | Reward |
|----|-------|-------|------------|--------|
| `001-sustainable-water` | Sustainable Water Management | agriculture | BLOOM | 50 IMPT + 25 SEED |
| `002-open-curriculum` | Open Curriculum Builder | education | SPROUT | 30 IMPT + 15 SEED |
| `003-health-data-sovereignty` | Health Data Sovereignty Framework | healthcare | FOREST | 100 IMPT + 50 SEED + 5 BLOOM |
| `004-solar-microgrid` | Community Solar Microgrid | energy | BLOOM | 60 IMPT + 30 SEED |
| `005-cooperative-lending` | Cooperative Lending Circle | finance | SPROUT | 40 IMPT + 20 SEED |

### QuestEngine API

```
CLASS QuestEngine:
  INIT():
    _quests: Dict[str, Quest] = {}
    _seed_default_quests()

  METHOD register_quest(quest) -> Quest:
    Store quest in _quests
    RETURN quest

  METHOD accept_quest(quest_id, node_id) -> QuestAcceptResult:
    IF quest not found: RETURN failure
    IF quest.status != AVAILABLE: RETURN failure (not available)
    SET quest.accepted_by = node_id
    SET quest.accepted_at = now()
    SET quest.status = ACCEPTED
    RETURN success with reward description

  METHOD complete_quest(quest_id, node_id, ihsan_score) -> Optional[QuestReward]:
    IF quest not found: RETURN None
    IF quest.accepted_by != node_id: RETURN None (not assigned)
    # CONSTITUTIONAL GATE
    IF ihsan_score < UNIFIED_IHSAN_THRESHOLD (0.95):
      log.warning("Ihsan gate failed: %.4f < 0.95")
      RETURN None
    SET quest.status = COMPLETED
    SET quest.completed_at = now()
    RETURN quest.reward

  METHOD list_available(guild_id=None) -> List[Quest]:
    Filter by AVAILABLE status, optionally by guild_id

  METHOD get_accepted(node_id) -> List[Quest]:
    Filter by node_id + (ACCEPTED | IN_PROGRESS) status

  METHOD list_all() -> List[Quest]
```

### Ihsan Gate — The Constitutional Constraint

Quest completion is the only constitutionally-gated operation:

```
complete_quest(quest_id, node_id, ihsan_score):
  # This import makes the threshold centralized (SSOT)
  FROM core.integration.constants IMPORT UNIFIED_IHSAN_THRESHOLD  # 0.95

  IF ihsan_score < UNIFIED_IHSAN_THRESHOLD:
    REJECT — node has not earned the right to claim rewards
    RETURN None

  # Only excellence unlocks reward distribution
  RETURN quest.reward
```

This is "ethics compiled into architecture" — the node cannot claim rewards without meeting the constitutional excellence standard. No bypass, no exception.

---

## Cross-Module Integration

```
core.genesis.orchestrator
  ├── Step 9: GuildRegistry.join_guild(guild_id, node_id, ihsan_score)
  └── Step 10: QuestEngine.accept_quest(quest_id, node_id)

core.quest.engine
  └── complete_quest() imports UNIFIED_IHSAN_THRESHOLD from core.integration.constants

core.guild.types
  └── GuildMember.ihsan_score tracks per-member Ihsan for fitness landscape
```

---

## TDD Anchors

### test_guild_registry.py

```
TEST default_guilds_seeded:
  registry = GuildRegistry()
  guilds = registry.list_guilds()
  ASSERT len(guilds) == 5
  ASSERT any(g.guild_id == "agriculture" for g in guilds)

TEST join_guild_success:
  registry = GuildRegistry()
  result = registry.join_guild("agriculture", "NODE-001", ihsan_score=0.97)
  ASSERT result.success
  ASSERT result.guild.member_count == 1
  ASSERT result.member.node_id == "NODE-001"

TEST join_guild_not_found:
  registry = GuildRegistry()
  result = registry.join_guild("nonexistent", "NODE-001")
  ASSERT NOT result.success
  ASSERT "not found" IN result.message

TEST join_guild_idempotent:
  registry = GuildRegistry()
  registry.join_guild("agriculture", "NODE-001")
  result = registry.join_guild("agriculture", "NODE-001")
  ASSERT result.success
  ASSERT "Already a member" IN result.message

TEST leave_guild:
  registry = GuildRegistry()
  registry.join_guild("agriculture", "NODE-001")
  removed = registry.leave_guild("agriculture", "NODE-001")
  ASSERT removed
  ASSERT registry.get_guild("agriculture").member_count == 0

TEST mean_ihsan_tracks_fitness:
  registry = GuildRegistry()
  registry.join_guild("agriculture", "NODE-001", ihsan_score=0.98)
  registry.join_guild("agriculture", "NODE-002", ihsan_score=0.96)
  guild = registry.get_guild("agriculture")
  ASSERT guild.mean_ihsan == pytest.approx(0.97)

TEST register_custom_guild:
  registry = GuildRegistry()
  guild = registry.register_guild("custom", "Custom Guild", "test")
  ASSERT guild.guild_id == "custom"
  ASSERT len(registry.list_guilds()) == 6
```

### test_quest_engine.py

```
TEST default_quests_seeded:
  engine = QuestEngine()
  quests = engine.list_all()
  ASSERT len(quests) == 5
  ASSERT quests[0].status == QuestStatus.AVAILABLE

TEST accept_quest_success:
  engine = QuestEngine()
  result = engine.accept_quest("001-sustainable-water", "NODE-001")
  ASSERT result.success
  ASSERT result.quest.status == QuestStatus.ACCEPTED
  ASSERT result.quest.accepted_by == "NODE-001"

TEST accept_quest_not_found:
  engine = QuestEngine()
  result = engine.accept_quest("nonexistent", "NODE-001")
  ASSERT NOT result.success

TEST accept_quest_already_accepted:
  engine = QuestEngine()
  engine.accept_quest("001-sustainable-water", "NODE-001")
  result = engine.accept_quest("001-sustainable-water", "NODE-002")
  ASSERT NOT result.success
  ASSERT "not available" IN result.message

TEST complete_quest_ihsan_gate_pass:
  engine = QuestEngine()
  engine.accept_quest("001-sustainable-water", "NODE-001")
  reward = engine.complete_quest("001-sustainable-water", "NODE-001", ihsan_score=0.97)
  ASSERT reward IS NOT None
  ASSERT reward.impt_amount == 50.0

TEST complete_quest_ihsan_gate_fail:
  engine = QuestEngine()
  engine.accept_quest("001-sustainable-water", "NODE-001")
  reward = engine.complete_quest("001-sustainable-water", "NODE-001", ihsan_score=0.80)
  ASSERT reward IS None  # Constitutional gate blocks

TEST complete_quest_wrong_node:
  engine = QuestEngine()
  engine.accept_quest("001-sustainable-water", "NODE-001")
  reward = engine.complete_quest("001-sustainable-water", "NODE-002", ihsan_score=0.99)
  ASSERT reward IS None  # Not the accepting node

TEST list_available_filtered_by_guild:
  engine = QuestEngine()
  ag_quests = engine.list_available("agriculture")
  ASSERT all(q.guild_id == "agriculture" for q in ag_quests)
```

## Edge Cases

- Guild membership is idempotent — joining again returns existing membership
- GuildMember is frozen (immutable) — Ihsan score captured at join time
- Quest can only be accepted once (first-come-first-served)
- Token amounts in QuestReward are floats for fractional economics
- Both systems use in-memory storage — restart clears all state (by design for v1)
