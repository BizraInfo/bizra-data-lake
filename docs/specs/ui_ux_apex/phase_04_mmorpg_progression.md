# UI/UX APEX — Phase 04: MMORPG Progression — Levelling the Node

> Atlas Tier UI evolution + Sovereign Quests gamification system.
> Sprint priority: 4 (builds on existing `core/guild/` + `core/quest/`).

> Standing on Giants: McGonigal (gameful design, 2011) · Ostrom (polycentric governance, 1990) ·
> Szabo (smart contracts, 1997) · Al-Ghazali (Ihsān as growth, 1095)
> Repo anchors: `core/guild/`, `core/quest/`, `docs/specs/phase_26_guild_quest_system.md`

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| MP-01 | UI adapts its complexity based on the user's Atlas Tier (Seed/Sprout/Rooted/Elder) |
| MP-02 | Seed tier: minimalist — core chat + status only |
| MP-03 | Sprout tier: adds Cognitive Helix preview + Iḥsān mini-gauge |
| MP-04 | Rooted tier: full dashboard — A2A handshakes, earnings, Guild panels |
| MP-05 | Elder tier: adds federation topology map + Guardian Council access |
| MP-06 | "Sovereign Quests" replace traditional onboarding |
| MP-07 | Each quest has: title, narrative description, acceptance criteria, XP reward, SEED reward |
| MP-08 | Quest completion gated by Ihsān ≥ 0.95 (constitutional requirement from phase_26) |
| MP-09 | XP bar and Atlas Tier badge visible in persistent header |
| MP-10 | Tier-up ceremony: full-screen Sacred Geometry bloom animation, Playfair Display fanfare |
| MP-11 | Quest log accessible from persistent footer (≤ 3 active quests shown) |

---

## 2. Atlas Tier Definitions

```
TIER       XP RANGE    UI COMPLEXITY         UNLOCK
──────────────────────────────────────────────────────────────────
Seed       0–499       Minimal               Default at first launch
Sprout     500–2499    Moderate              Helix preview, Guild discovery
Rooted     2500–9999   Full                  A2A, Marketplace, Guild panels
Elder      10000+      Expert                Federation map, Guardian Council
```

---

## 3. Sovereign Quests Catalog (initial set)

```
QUEST: "First Reflex"
  Description: "Compile your first AHK Reflex. Let your Node learn a new gesture."
  Acceptance:  Ghost Overlay dispatches 1 AHK action successfully, Ihsān ≥ 0.95
  XP Reward:   150
  SEED Reward: 10
  Tier Unlock: Seed → counts toward Sprout

QUEST: "First Coin"
  Description: "Earn your first SEED via AaaS. Let the market come to you."
  Acceptance:  1 A2A task_result with status=COMPLETED, earning > 0 SEED
  XP Reward:   300
  SEED Reward: 0 (market earns SEED, quest rewards XP only)
  Tier Unlock: Sprout → counts toward Rooted

QUEST: "The Philosopher's Node"
  Description: "Build a GoT graph with ≥ 50 nodes across 3+ domains."
  Acceptance:  GoT graph node_count ≥ 50, distinct domain_clusters ≥ 3
  XP Reward:   500
  SEED Reward: 25
  Tier Unlock: Rooted quest

QUEST: "Guardian's Trust"
  Description: "Achieve a session Iḥsān average ≥ 0.97 over 20+ actions."
  Acceptance:  rolling_20_avg ≥ 0.97, action_count ≥ 20
  XP Reward:   800
  SEED Reward: 50

QUEST: "Guild Founder"
  Description: "Create or join a Guild with ≥ 3 active members."
  Acceptance:  guild.member_count ≥ 3, user.guild_role != null
  XP Reward:   600
  SEED Reward: 30
```

---

## 4. Data Model

```typescript
// MMORPG Progression — client types
// Python source of truth: core/guild/types.py, core/quest/types.py

type AtlasTier = "Seed" | "Sprout" | "Rooted" | "Elder";

interface QuestState {
  id:              string;
  title:           string;
  narrative:       string;
  status:          "locked" | "available" | "active" | "completed" | "failed";
  xp_reward:       number;
  seed_reward:     number;
  acceptance:      string;    // human-readable criteria
  progress:        QuestProgress;
  ihsan_gate:      number;    // must be >= this (0.95 from constants)
}

interface QuestProgress {
  steps_complete:  number;
  steps_total:     number;
  last_checkpoint: string;    // ISO 8601
}

interface NodeProgression {
  xp_total:       number;
  tier:           AtlasTier;
  tier_xp_start:  number;     // XP at start of current tier
  tier_xp_end:    number;     // XP to reach next tier
  quests_active:  QuestState[];   // max 3
  quests_log:     QuestState[];   // all completed
  guild_name:     string | null;
  guild_role:     string | null;
}
```

---

## 5. Pseudocode

### 5.1 AtlasTierController

```
MODULE AtlasTierController:

  FUNCTION compute_tier(xp_total) -> AtlasTier:
    IF xp_total < 500:   RETURN "Seed"
    IF xp_total < 2500:  RETURN "Sprout"
    IF xp_total < 10000: RETURN "Rooted"
    RETURN "Elder"

  FUNCTION on_xp_gained(delta_xp):
    prev_tier = compute_tier(state.xp_total)
    state.xp_total += delta_xp
    new_tier  = compute_tier(state.xp_total)

    update_xp_bar_animation(state.xp_total)

    IF new_tier != prev_tier:
      trigger_tier_up_ceremony(new_tier)
      unlock_ui_features(new_tier)

  FUNCTION unlock_ui_features(tier):
    MATCH tier:
      "Seed":
        show_features(["core_chat", "status_indicator", "ihsan_mini"])
      "Sprout":
        show_features(["core_chat", "status_indicator", "ihsan_mini",
                       "helix_preview", "ghost_overlay", "guild_discovery"])
      "Rooted":
        show_features(["all_sprout", "full_dashboard",
                       "a2a_handshakes", "marketplace", "guild_panels",
                       "resonance_feed"])
      "Elder":
        show_features(["all_rooted", "federation_topology",
                       "guardian_council", "elder_analytics"])

  FUNCTION trigger_tier_up_ceremony(new_tier):
    // Full-screen Sacred Geometry bloom
    render_fullscreen_overlay(
      animation = "genesis_seal_bloom",
      duration  = 3000,
      color     = #C9A962,
    )
    draw_text(
      f"Sovereign Ascension",
      font = "Playfair Display 48px",
      color = #C9A962,
      animation = "fade_in_up"
    )
    draw_text(
      f"You have reached the {new_tier} Tier",
      font = "Playfair Display italic 24px",
      color = #F8F4EC,
    )
    draw_tier_badge(new_tier)
    play_sound("tier_up.ogg")   // optional, user-toggleable
    schedule(3000, dismiss_overlay)
```

### 5.2 SovereignQuestEngine (UI layer)

```
MODULE SovereignQuestEngine:

  FUNCTION init():
    progression = await fetch("/api/v1/progression")
    render_xp_bar(progression)
    render_quest_footer(progression.quests_active)

    ws = open_kernel_ws("/api/v1/quests/stream")
    ws.on("quest_progress", on_quest_progress)
    ws.on("quest_completed", on_quest_completed)
    ws.on("xp_gained", FUNCTION(event): AtlasTierController.on_xp_gained(event.delta))

  FUNCTION render_xp_bar(progression):
    pct = (progression.xp_total - progression.tier_xp_start) /
          (progression.tier_xp_end - progression.tier_xp_start)
    bar = XPBar(
      percent   = pct * 100,
      tier      = progression.tier,
      tier_badge = tier_badge_svg(progression.tier),
      color     = #C9A962,
    )
    header.render(bar)

  FUNCTION render_quest_footer(quests_active):
    footer_panel = create_collapsible_footer("Sovereign Quests")
    FOR quest in quests_active[:3]:
      draw_quest_card(quest)
    footer_panel.render()

  FUNCTION draw_quest_card(quest):
    progress_pct = quest.progress.steps_complete / quest.progress.steps_total * 100
    draw_card(
      title       = quest.title,
      narrative   = truncate(quest.narrative, 120),
      progress    = ProgressBar(progress_pct, color=#2e56c9),
      xp_label    = f"+{quest.xp_reward} XP",
      seed_label  = f"+{quest.seed_reward} SEED" IF quest.seed_reward > 0,
    )

  FUNCTION on_quest_completed(event):
    quest = find_quest(event.quest_id)
    show_quest_completion_toast(quest)
    emit_xp_gained(quest.xp_reward)

  FUNCTION show_quest_completion_toast(quest):
    toast_notification(
      title   = f"Quest Complete: {quest.title}",
      message = f"+{quest.xp_reward} XP  +{quest.seed_reward} SEED",
      style   = { border: #C9A962, icon: quest_seal_svg },
      duration = 8000
    )
    play_sound("quest_complete.ogg")  // optional
```

### 5.3 Quest Validation Backend (Python — Node0)

```
MODULE QuestValidator:
  // Called by core/quest/engine.py when acceptance criteria may be met.

  FUNCTION validate_quest(quest_id, user_id) -> ValidationResult:
    quest = quest_registry.get(quest_id)
    criteria = quest.acceptance_criteria

    // Constitutional gate — Ihsān required
    ihsan = get_session_ihsan_average(user_id)
    IF ihsan < UNIFIED_IHSAN_THRESHOLD:    // from constants.py
      RETURN ValidationResult(
        passed  = False,
        reason  = f"Iḥsān {ihsan:.4f} < required {UNIFIED_IHSAN_THRESHOLD}",
      )

    // Domain-specific checks
    MATCH quest_id:
      "first_reflex":
        ahk_count = count_successful_ahk_dispatches(user_id)
        RETURN ValidationResult(passed = ahk_count >= 1)

      "first_coin":
        earnings = get_a2a_earnings(user_id)
        RETURN ValidationResult(passed = earnings > 0)

      "philosophers_node":
        graph = load_got_graph(user_id)
        clusters = cluster_by_domain(graph)
        distinct_clusters = len(set(clusters.values()))
        RETURN ValidationResult(
          passed = len(graph.nodes) >= 50 AND distinct_clusters >= 3
        )

      "guardians_trust":
        avg = get_rolling_ihsan_avg(user_id, window=20)
        RETURN ValidationResult(passed = avg >= 0.97)

      "guild_founder":
        membership = get_guild_membership(user_id)
        IF NOT membership: RETURN ValidationResult(passed=False)
        guild = get_guild(membership.guild_id)
        RETURN ValidationResult(passed = guild.member_count >= 3)

    RETURN ValidationResult(passed=False, reason="Unknown quest")
```

---

## 6. TDD Anchors

```python
# tests/ui_ux_apex/test_mmorpg_progression.py

class TestAtlasTierController:
    @pytest.mark.parametrize("xp,expected_tier", [
        (0, "Seed"), (499, "Seed"),
        (500, "Sprout"), (2499, "Sprout"),
        (2500, "Rooted"), (9999, "Rooted"),
        (10000, "Elder"),
    ])
    def test_tier_boundaries(self, controller, xp, expected_tier):
        assert controller.compute_tier(xp) == expected_tier

    def test_tier_up_ceremony_fires_on_crossing(self, controller, mock_ceremony):
        controller.state.xp_total = 490
        controller.on_xp_gained(15)  # crosses 500 → Sprout
        assert mock_ceremony.called_with("Sprout")

    def test_no_ceremony_within_tier(self, controller, mock_ceremony):
        controller.state.xp_total = 100
        controller.on_xp_gained(50)  # stays in Seed
        assert not mock_ceremony.called

class TestQuestValidator:
    def test_ihsan_gate_enforced(self, validator):
        """Quest validation fails if session Ihsān below threshold."""
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
        with mock_ihsan_average(UNIFIED_IHSAN_THRESHOLD - 0.01):
            result = validator.validate_quest("first_reflex", "u_test")
        assert not result.passed
        assert "Iḥsān" in result.reason

    def test_first_reflex_requires_ahk_dispatch(self, validator):
        with mock_ahk_dispatch_count(0):
            result = validator.validate_quest("first_reflex", "u_test")
        assert not result.passed

        with mock_ahk_dispatch_count(1):
            result = validator.validate_quest("first_reflex", "u_test")
        assert result.passed

    def test_philosophers_node_domain_check(self, validator):
        """50 nodes but only 1 domain → still fails."""
        with mock_got_graph(nodes=50, domains=1):
            result = validator.validate_quest("philosophers_node", "u_test")
        assert not result.passed

        with mock_got_graph(nodes=50, domains=3):
            result = validator.validate_quest("philosophers_node", "u_test")
        assert result.passed
```
