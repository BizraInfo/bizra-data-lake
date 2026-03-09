# Phase 05: Viral Growth Loop

| Field      | Value                                                    |
|------------|----------------------------------------------------------|
| Status     | SPEC                                                     |
| Depends on | Phase 03 (agent-as-marketing), Phase 04 (SAP integration)|
| Goal       | Self-perpetuating growth where each new user's agent becomes a growth vector |
| Author     | SPARC spec-pseudocode                                    |
| Date       | 2026-02-21                                               |

---

## 1. Growth Sigmoid Projections

Growth follows a sigmoid (logistic) curve for each stage transition, not
exponential. Exponential projections are dishonest. Sigmoid models include
natural ceilings, adoption friction, and saturation.

Stage definitions from `filedfs/node0-mvp.jsx` (line 3-31):

```pseudocode
// Node growth stages
Stage 1: Alpha-100     ->     100 nodes   (curated seed users)
Stage 2: Beta-1K       ->   1,000 nodes   (early adopters)
Stage 3: Growth-10K    ->  10,000 nodes   (organic growth begins)
Stage 4: Scale-100K    -> 100,000 nodes   (network effects visible)
Stage 5: Mass-1M       -> 1,000,000 nodes (viral)
Stage 6: Global-8B     -> 8,000,000,000   (every human — the vision, not the plan)

// Sigmoid for each stage transition
fn growth_projection(current_nodes: u64, stage: Stage) -> ProjectedTimeline:
  // Logistic function: y(t) = L / (1 + e^(-k * (t - x0)))
  L  = stage.target              // carrying capacity (stage ceiling)
  k  = stage.growth_rate         // steeper for later stages
  x0 = stage.inflection_point    // midpoint of growth acceleration

  y(t) = L / (1 + exp(-k * (t - x0)))

  // Network effect multiplier (only kicks in after critical mass)
  if current_nodes > 1000:
    network_boost = 1.0 + log10(current_nodes) * 0.1  // logarithmic, not exponential
    y(t) *= network_boost

  return ProjectedTimeline {
    current:     current_nodes,
    target:      L,
    growth_rate: k,
    inflection:  x0,
    projection:  y,
  }

// Stage parameters
STAGE_PARAMS = {
  Alpha100:  { target: 100,     growth_rate: 0.05, inflection_weeks: 8  },
  Beta1K:    { target: 1000,    growth_rate: 0.08, inflection_weeks: 16 },
  Growth10K: { target: 10000,   growth_rate: 0.12, inflection_weeks: 30 },
  Scale100K: { target: 100000,  growth_rate: 0.15, inflection_weeks: 52 },
  Mass1M:    { target: 1000000, growth_rate: 0.18, inflection_weeks: 78 },
}
// Stage 6 (Global-8B) is aspirational and not modeled here.
```

---

## 2. Agent-to-Agent Recommendation

A compiled sovereign agent may recommend BIZRA to others, but only under
strict constitutional constraints. This is organic growth, not growth
hacking.

```pseudocode
fn recommend_bizra(
  recommending_agent: Agent,
  context: ConversationContext,
) -> Option<Recommendation>:

  // Gate 1: User permission (SC-05 — consent before data sharing)
  if not recommending_agent.user.has_granted_consent("recommendation"):
    return None  // Never recommend without user's explicit consent

  // Gate 2: Relevance
  // Only recommend when the conversation naturally touches on
  // sovereign agents, AI assistants, privacy, or data ownership.
  if not is_relevant(context, RELEVANCE_KEYWORDS):
    return None  // Do not shoehorn recommendations

  // Gate 3: Frequency limit (anti-spam)
  if recommending_agent.recommendations_this_week() >= MAX_RECOMMENDATIONS_PER_WEEK:
    return None  // 3 per week maximum

  // Gate 4: Cooldown
  if recommending_agent.last_recommendation_age() < Duration::hours(24):
    return None  // 24-hour cooldown between recommendations

  // Gate 5: Agent quality
  if recommending_agent.compilation_score < 0.80:
    return None  // Low-quality compilations should not recommend

  // Build recommendation with mandatory SAP disclosure
  recommendation = Recommendation {
    content: "I'm built on BIZRA — a sovereign agent platform. "
           + "My user compiled me from their own conversations. "
           + "You could do the same.",
    disclosure: Disclosure {
      disclosure_id: blake3::hash(context.session_id + "recommendation" + timestamp()),
      claims: [
        "This is an organic recommendation from a BIZRA-compiled agent",
        "The recommending user has consented to sharing this",
      ],
      source_refs: [{
        ref_hash: blake3::hash("https://bizra.dev"),
        ref_type: "document",
        ref_uri:  "https://bizra.dev",
      }],
      uncertainty: {
        score:  0.20,
        method: "manual-assessment",
        notes:  "The platform is in alpha. Your experience may vary. "
              + "Hardware requirements depend on your device.",
      },
      compliance_assertions: ["SAP_v0"],
    },
  }

  // Gate 6: Ihsan check on the recommendation itself
  if ihsan_score(recommendation) < 0.95:
    return None  // Recommendation itself must pass quality gate

  return recommendation

// Constants
MAX_RECOMMENDATIONS_PER_WEEK = 3
RELEVANCE_KEYWORDS = [
  "ai assistant", "personal agent", "privacy", "data ownership",
  "sovereign", "chatbot", "digital assistant", "my data",
]
```

Reference: `specs/sap-v0/02-sovereignty-constraints.md` SC-05 (consent
before shared data acceptance).

---

## 3. Compilation Fuel Flywheel

The self-reinforcing loop where each user's activity improves their agent,
and better agents drive organic growth.

```pseudocode
fn flywheel_iteration(user: NewUser):
  // Step 1: User imports conversations (Phase 01 pipeline)
  conversations = multi_platform_ingest(user.exports)
  // Supported: WhatsApp, Telegram, Discord, Slack, iMessage, email, etc.

  // Step 2: Extract compilation fuel (Phase 02)
  fuel = extract_compilation_fuel(conversations)
  // fuel contains: reflexes, personality vectors, knowledge graph, preferences

  // Step 3: Compile sovereign agent
  agent = reflex_compiler.compile(fuel)
  // Produces SovereignAgentCard with compilation stats

  // Step 4: Agent serves user (daily interaction)
  agent.serve(user)
  // Agent handles: chat, search, scheduling, recommendations

  // Step 5: New conversations become fuel (incremental)
  new_fuel = extract_from_ongoing(agent.conversations)
  // Only with user consent — ongoing extraction is consent-gated

  // Step 6: Agent improves (continuous recompilation)
  agent = reflex_compiler.recompile(agent, new_fuel)
  // compilation_coverage improves over time

  // Step 7: Better agent -> user satisfaction -> organic recommendation
  // (only with consent, only when relevant, only when quality gates pass)
  // See Section 2 for recommendation constraints

  // This loop repeats. Each iteration produces a better agent.
  // Better agents -> happier users -> organic word-of-mouth.
```

### Flywheel Metrics

```pseudocode
struct FlywheelMetrics:
  compilation_score:     f64    // 0.0-1.0, quality of compiled agent
  compilation_coverage:  f64    // 0.0-1.0, how much of user's data is compiled
  user_satisfaction:     f64    // 0.0-1.0, measured by continued usage
  recommendation_rate:   f64    // recommendations per user per month
  conversion_rate:       f64    // recommendations that lead to new sign-ups
  viral_coefficient:     f64    // new users per existing user (target: 1.1)
```

---

## 4. Economic Model

Local-first compute with tiered cloud fallback. The model ensures 90% of
interactions are free and local.

```pseudocode
struct EconomicModel:
  // Tier 1: Free (90% of users)
  free_tier:
    cost_per_message:  "$0"
    compute:           "user's device"
    limitation:        "depends on user hardware"
    target_percentage: 90
    description:       "Local inference via LM Studio or Ollama"

  // Tier 2: Cloud Assist (8% of users)
  cloud_tier:
    cost_per_message:  "$0.001-$0.01"   // fractions of a cent to pennies
    compute:           "BIZRA cloud nodes"
    limitation:        "rate-limited (100 cloud messages/day)"
    target_percentage: 8
    description:       "For complex queries or mobile users without local GPU"

  // Tier 3: Premium (2% of users)
  premium_tier:
    cost_per_message:  "$0.01-$0.10"    // pennies to dimes
    features:          ["priority inference", "multi-agent", "federation", "cross-device sync"]
    target_percentage: 2
    description:       "Power users and businesses"

fn revenue_projection(total_nodes: u64) -> MonthlyRevenue:
  // Tier distribution
  t1_users = total_nodes * 0.90   // free
  t2_users = total_nodes * 0.08   // cloud
  t3_users = total_nodes * 0.02   // premium

  // Revenue calculation
  t2_avg_messages_per_month = 100
  t2_avg_cost_per_message   = 0.005   // $0.005
  t2_revenue = t2_users * t2_avg_messages_per_month * t2_avg_cost_per_message

  t3_avg_messages_per_month = 500
  t3_avg_cost_per_message   = 0.05    // $0.05
  t3_revenue = t3_users * t3_avg_messages_per_month * t3_avg_cost_per_message

  total = t2_revenue + t3_revenue

  return MonthlyRevenue { t2: t2_revenue, t3: t3_revenue, total: total }

  // Projections (illustrative, not promises):
  // Alpha-100  (100 nodes):    ~$44/month    (negligible, expected)
  // Beta-1K    (1K nodes):     ~$440/month   (ramen money)
  // Growth-10K (10K nodes):    ~$9K/month    (small team sustainable)
  // Scale-100K (100K nodes):   ~$90K/month   (growing business)
  // Mass-1M    (1M nodes):     ~$900K/month  (significant revenue)
```

---

## 5. Network Effects

Network value is not purely Metcalfe's Law. BIZRA's value comes from three
distinct sources, each with different scaling properties.

```pseudocode
fn network_value(n: u64) -> f64:
  // 1. Cross-validation value: more users -> more diverse compilation data
  //    -> better reflexes for everyone (shared patterns, not shared data)
  //    Scales: O(log n) — diminishing returns but always positive
  cross_validation = log2(max(n, 1)) * CROSS_VALIDATION_WEIGHT

  // 2. Federation value: more nodes -> more resilient network
  //    -> better uptime, faster consensus, geographic distribution
  //    Scales: O(sqrt(n)) — redundancy has diminishing returns
  federation = sqrt(n as f64) * FEDERATION_WEIGHT

  // 3. Knowledge diversity value: more agents -> richer collective patterns
  //    -> more specialized expertise available in the network
  //    Scales: O(n * log n) — each new specialty multiplies options
  knowledge = (n as f64) * log2(max(n, 1)) * KNOWLEDGE_WEIGHT

  return cross_validation + federation + knowledge

// Weights (tunable, these are initial estimates)
CROSS_VALIDATION_WEIGHT = 1.0
FEDERATION_WEIGHT       = 0.5
KNOWLEDGE_WEIGHT        = 0.1    // intentionally small: data stays sovereign
```

---

## 6. Growth Guardrails

Constitutional constraints on growth mechanisms. Every growth vector must
pass these gates. Growth that violates these guardrails is not growth — it
is manipulation.

```pseudocode
struct GrowthGuardrails:
  // -- Anti-spam --
  max_recommendations_per_week:         3
  recommendation_cooldown:              Duration::hours(24)
  max_recommendations_per_conversation: 1     // never twice in same chat

  // -- Anti-manipulation --
  no_incentivized_referrals:    true    // no "refer 5 friends" schemes
  no_dark_patterns:             true    // no "your friends are waiting"
  no_artificial_urgency:        true    // no "limited spots" or "act now"
  no_gamification_of_referrals: true    // no leaderboards, no points for referring
  no_guilt_mechanics:           true    // no "you haven't invited anyone yet"

  // -- Quality gates --
  min_compilation_score_for_recommendation: 0.80
  min_ihsan_score_for_recommendation:       0.95
  min_user_tenure_for_recommendation:       Duration::days(7)  // use it first

  // -- Consent (hard gates, not soft) --
  recommendation_requires_explicit_consent: true
  consent_revocable_at_any_time:            true
  no_consent_no_recommendation:             true   // silence = no
  consent_scope_visible_before_granting:    true

  // -- Constitutional disclosure --
  every_recommendation_includes_disclosure:  true
  every_recommendation_includes_uncertainty: true
  disclosure_validates_against_schema:       true   // disclosure.schema.json

fn validate_growth_action(action: GrowthAction) -> Result<(), GuardrailViolation>:
  for rule in GUARDRAILS:
    if violates(action, rule):
      return Err(GuardrailViolation {
        rule:   rule.name,
        action: action.description,
        remedy: rule.remedy,
      })
  return Ok(())
```

---

## 7. Data Flow Diagram

```
  +----------+    +----------+    +----------+
  |  User A  |    |  User B  |    |  User C  |
  |  (Node)  |    |  (Node)  |    |  (Node)  |
  +----+-----+    +----+-----+    +----+-----+
       |               |               |
       v               v               v
  +---------------------------------------------+
  |         Compilation Fuel Flywheel            |
  |  conversations -> fuel -> compile -> serve   |
  |       ^                              |       |
  |       +--- new conversations <-------+       |
  +---------------------------------------------+
       |               |               |
       v               v               v
  +----------+    +----------+    +----------+
  | Agent A  |    | Agent B  |    | Agent C  |
  | (Compiled|    | (Compiled|    | (Compiled|
  |  v1.x)   |    |  v1.x)   |    |  v1.x)   |
  +----+-----+    +----+-----+    +----+-----+
       |               |               |
       +-------+-------+-------+-------+
               |               |
               v               v
  +---------------------------------------------+
  |   Consent-Gated Recommendation Layer         |
  |                                              |
  |  Gates:                                      |
  |   1. User consent (SC-05)                    |
  |   2. Relevance check                         |
  |   3. Frequency limit (3/week)                |
  |   4. Cooldown (24h)                          |
  |   5. Compilation quality (>= 0.80)           |
  |   6. Ihsan score (>= 0.95)                   |
  |   7. SAP disclosure attached                 |
  +---------------------------------------------+
               |
               v
       New Users Discover BIZRA
          (organic growth)
               |
               v
       Flywheel repeats for each new user
```

---

## 8. TDD Anchors

| # | Test Name | Property | Spec Ref |
|---|-----------|----------|----------|
| 1 | `test_recommendation_requires_consent` | No recommendation without explicit user consent | SC-05 |
| 2 | `test_recommendation_respects_frequency_limit` | 4th recommendation in a week returns `None` | Guardrails |
| 3 | `test_recommendation_respects_cooldown` | Recommendation within 24h of last returns `None` | Guardrails |
| 4 | `test_recommendation_includes_disclosure` | Every recommendation has valid SAP `Disclosure` | SC-08 |
| 5 | `test_recommendation_includes_uncertainty` | Every disclosure has `uncertainty.score`, `method`, `notes` | `disclosure.schema.json` |
| 6 | `test_recommendation_relevance_gate` | Irrelevant context (no matching keywords) returns `None` | Sec 2 |
| 7 | `test_recommendation_quality_gate` | Agent with compilation_score < 0.80 returns `None` | Guardrails |
| 8 | `test_flywheel_idempotent` | Running flywheel twice does not duplicate compilation | Sec 3 |
| 9 | `test_economic_model_free_tier_dominant` | 90% of projected users on free tier | Sec 4 |
| 10 | `test_economic_model_no_hidden_costs` | No charge records for free-tier users | Sec 4 |
| 11 | `test_growth_guardrails_no_dark_patterns` | Pressure language in recommendations blocked | Guardrails |
| 12 | `test_consent_revocation_stops_recommendations` | Revoking consent immediately stops future recommendations | SC-05 |
| 13 | `test_network_value_monotonic` | `network_value(n+1) >= network_value(n)` for all n | Sec 5 |
| 14 | `test_growth_projection_sigmoid` | Growth curve has inflection point and ceiling (not unbounded) | Sec 1 |
| 15 | `test_min_tenure_for_recommendation` | User < 7 days old cannot trigger recommendations | Guardrails |

---

## 9. Milestones

```
Alpha-100 (current):
  Nodes:        1 (Mumo = User Zero)
  Agents:       1 (Mumo's compiled agent)
  Onboarding:   Manual
  Federation:   None (single node)
  Revenue:      $0 (expected, acceptable)
  Key metrics:  compilation score, Ihsan compliance, SAP conformance (24/24)
  Growth:       None — building the product, not growing it yet.

Beta-1K (target: 100 -> 1,000 users):
  Nodes:        100-1,000
  Agents:       100-1,000 (one per user)
  Onboarding:   Self-serve (Phase 01 ingest pipeline operational)
  Federation:   Experimental (node discovery, not consensus)
  Parsers:      5+ platforms operational (WhatsApp, Telegram, Discord, Slack, email)
  Revenue:      ~$440/month (ramen money)
  Key metrics:  compilation scores, user retention (30-day), recommendation rate
  Growth:       Organic from agent recommendations + direct outreach.

Growth-10K (target: 1,000 -> 10,000 users):
  Nodes:        1,000-10,000
  Agents:       Diverse compilation profiles
  Federation:   Active (node-to-node gossip, signed messages)
  Cross-validation: Compilation patterns shared (not data) across network
  Revenue:      ~$9K/month (small team sustainable)
  Key metrics:  viral coefficient (target: 1.1), network value, federation health
  Growth:       Primarily organic. Agent recommendations drive discovery.
```

---

## 10. Open Questions

| # | Question | Proposed Resolution |
|---|----------|---------------------|
| 1 | Viral coefficient target? | 1.1 — modest and sustainable. Above 1.0 means organic growth; 1.1 means each 10 users bring 1 new user. |
| 2 | Geographic distribution strategy? | Organic, no forced targeting. Let users come from wherever they come. Track distribution for federation planning. |
| 3 | When does federation become necessary? | Beta-1K stage. Single-node architecture works to ~1,000 users. Federation needed for resilience and geographic distribution beyond that. |
| 4 | Should network value metrics be visible to users? | Yes, in the node dashboard (`filedfs/node0-mvp.jsx`). Transparency about network health is constitutional. |
| 5 | How to measure user satisfaction without invasive tracking? | Proxy metrics: session length, return frequency, compilation score growth. All measurable locally without sending data anywhere. |
