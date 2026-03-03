# UI/UX APEX — Phase 06: Resonance Feed — The Guild Hall

> A2A-powered proactive ad system where the market competes for the user as a VIP.
> Sprint priority: 6 (requires Rooted tier unlock + A2A infrastructure).

> Standing on Giants: Veblen (conspicuous consumption, 1899) · Nash (game theory, 1950) ·
> Ostrom (market as commons, 1990) · Nakamoto (trustless exchange, 2008) ·
> Al-Ghazali (dignity of the sovereign, 1095)
> Repo anchors: `core/a2a/schema.py`, `core/a2a/engine.py`, `core/a2a/transport.py`

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| RF-01 | Incoming A2A marketplace offers appear as "Gifts" in the Guild Hall panel |
| RF-02 | Guild Hall accessible from persistent navigation (Rooted tier+) |
| RF-03 | Each Gift card shows: offer title, offering agent, SEED value, HHMM relevance %, expiry |
| RF-04 | Gifts are pre-negotiated (A2A handshake complete) before appearing in UI |
| RF-05 | Gift relevance computed from HHMM current cognitive state (not demographics) |
| RF-06 | Low-relevance offers (< MIN_GIFT_RELEVANCE) filtered by Node0 before reaching UI |
| RF-07 | User accepts Gift via Sovereign Gesture — dispatches A2A TASK_ACCEPT |
| RF-08 | User declines Gift — dispatches A2A TASK_REJECT; No reason required |
| RF-09 | Declined offers teach HHMM (negative reinforcement signal) |
| RF-10 | Accepted Gift confirms via Iḥsān Gauge (shows "Gift Accepted" receipt) |
| RF-11 | Guild Hall shows active Guild missions alongside incoming Gifts |
| RF-12 | Gifts expire after `gift_ttl_ms` (config); expired gifts auto-remove with fade animation |

---

## 2. Edge Cases & Constraints

```
EDGE CASE: 0 incoming Gifts → show "Guild Hall is quiet. The market will come to you."
           with Seed of Life SVG idle animation
EDGE CASE: Simultaneous flood of Gifts (>10 in 10s) → rate-limit display to 3/10s;
           queue remainder, notify with badge count
EDGE CASE: Gift expiry during user review → graceful "Offer Expired" card state,
           no accept action available
EDGE CASE: A2A transport failure → Gift shows "Delivery Pending" badge; retry up to 3x
EDGE CASE: HHMM state unavailable → show Gift with "Relevance: —" (not 0%)
EDGE CASE: User accepts gift but A2A TASK_ACCEPT fails → show error, revert to "pending" state
EDGE CASE: Offer from blacklisted agent → silently dropped at Node0, never reaches UI
CONSTRAINT: Offering agent's AgentCard must be PCI-verified before gift reaches Guild Hall
CONSTRAINT: SEED value display is informational; actual settlement is in core/treasury/
CONSTRAINT: MIN_GIFT_RELEVANCE threshold from config (not hardcoded)
CONSTRAINT: No user PII or behavioral data sent to offering agent; only TASK_ACCEPT/REJECT
CONSTRAINT: Accept/Reject actions require ConstitutionalGate Ihsān ≥ 0.95
```

---

## 3. Data Model

```typescript
// Resonance Feed — client types
// Python source of truth: core/a2a/schema.py

type GiftStatus =
  | "incoming" | "pending_accept" | "accepted" | "declined"
  | "expired" | "delivery_failed";

interface GiftCard {
  id:               string;       // maps to A2A TaskCard.task_id
  offer_title:      string;
  offer_summary:    string;       // max 200 chars
  offering_agent:   AgentIdentity;
  seed_value:       number;       // SEED tokens offered
  hhmm_relevance:   number;       // 0-1; how well it matches current cognitive state
  expires_at:       number;       // unix ms
  status:           GiftStatus;
  pci_verified:     boolean;      // offering agent's AgentCard is PCI-signed
  task_id:          string;       // A2A TaskCard id for ACCEPT/REJECT
}

interface AgentIdentity {
  agent_id:         string;
  display_name:     string;
  fingerprint:      string;       // first 12 hex chars of AgentCard hash
  tier:             string;       // e.g. "Verified Merchant"
}

interface GuildHallState {
  gifts:            GiftCard[];   // sorted by relevance desc, max 20 shown
  gift_queue_count: number;       // overflow count
  guild_missions:   GuildMission[];
  last_updated:     number;
}

interface GuildMission {
  mission_id:   string;
  title:        string;
  progress_pct: number;
  members:      number;
  deadline:     number | null;
}
```

---

## 4. Pseudocode

### 4.1 GiftRouter (Python — Node0 integration point)

```
MODULE GiftRouter:
  // Receives A2A TASK_REQUEST messages with capability_type=MARKETPLACE_OFFER.
  // Validates, scores, filters, then forwards to UI stream.

  CONFIG (from config, never hardcoded):
    MIN_GIFT_RELEVANCE     // e.g. 0.60
    GIFT_TTL_MS            // e.g. 3600000 (1 hour)
    MAX_QUEUED_GIFTS       // e.g. 50

  FUNCTION process_incoming_a2a_message(msg: A2AMessage):
    IF msg.type != TASK_REQUEST:
      RETURN  // not a marketplace offer

    IF NOT is_marketplace_offer(msg):
      RETURN

    // PCI verification of offering agent
    agent_card = await fetch_agent_card(msg.sender_id)
    IF NOT verify_pci_signature(agent_card):
      LOG warning("Dropped gift from non-PCI agent", msg.sender_id)
      RETURN

    // Blacklist check
    IF is_blacklisted(msg.sender_id):
      RETURN  // silent drop

    // HHMM relevance scoring
    current_state = hhmm.get_current_cognitive_state()
    relevance = compute_gift_relevance(msg.payload, current_state)

    IF relevance < MIN_GIFT_RELEVANCE:
      LOG debug("Filtered low-relevance gift", relevance)
      RETURN

    // Build GiftCard
    gift = GiftCard(
      id              = uuid4(),
      offer_title     = msg.payload.title,
      offer_summary   = truncate(msg.payload.description, 200),
      offering_agent  = build_agent_identity(agent_card),
      seed_value      = msg.payload.seed_offered,
      hhmm_relevance  = relevance,
      expires_at      = now_ms() + GIFT_TTL_MS,
      status          = "incoming",
      pci_verified    = True,
      task_id         = msg.task_id,
    )

    // Emit to UI stream
    gift_stream.emit("incoming_gift", gift)

  FUNCTION compute_gift_relevance(offer_payload, hhmm_state) -> float:
    // Cosine similarity between offer embedding and HHMM state embedding
    offer_embedding = embed(offer_payload.description + " " + offer_payload.tags)
    state_embedding = hhmm_state.context_embedding
    relevance = cosine_similarity(offer_embedding, state_embedding)
    RETURN clip(relevance, 0.0, 1.0)

  ASYNC FUNCTION handle_accept(task_id, user_id):
    // Constitutional gate
    result = await constitutional_gate.check(
      action = "gift_accept",
      context = {task_id, user_id},
    )
    IF NOT result.approved:
      RETURN GiftActionResult(success=False, reason=result.reason)

    // Dispatch A2A TASK_ACCEPT
    response = await a2a_engine.send(A2AMessage(
      type    = TASK_ACCEPT,
      task_id = task_id,
    ))
    RETURN GiftActionResult(success=response.ok)

  ASYNC FUNCTION handle_decline(task_id, user_id):
    // No gate required for decline (non-extractive)
    await a2a_engine.send(A2AMessage(type=TASK_REJECT, task_id=task_id))
    // Teach HHMM: this offer type did not resonate
    hhmm.record_negative_signal(task_id=task_id)
    RETURN GiftActionResult(success=True)
```

### 4.2 GuildHallUI

```
MODULE GuildHallUI:

  STATE:
    hall: GuildHallState
    rate_limit: RateLimiter(max=3, window_ms=10000)

  FUNCTION init():
    IF current_tier < "Rooted":
      show_tier_gate_message("Guild Hall unlocks at Rooted Tier")
      RETURN

    hall = await fetch("/api/v1/guild-hall")
    render_guild_hall()

    ws = open_kernel_ws("/api/v1/gifts/stream")
    ws.on("incoming_gift",    on_incoming_gift)
    ws.on("gift_expired",     on_gift_expired)
    ws.on("gift_accepted_ack", on_gift_accepted_ack)

  FUNCTION render_guild_hall():
    // Two-column layout: Gifts (left) | Guild Missions (right)
    draw_panel_header("Guild Hall", icon=guild_hall_svg, tagline="البذرة")

    IF len(hall.gifts) == 0:
      draw_empty_state(
        icon    = seed_of_life_animated_svg,
        message = "Guild Hall is quiet. The market will come to you.",
        subtext = "Your HHMM signature is being broadcast to verified agents.",
      )
    ELSE:
      draw_column("Incoming Gifts", [render_gift_card(g) for g in hall.gifts[:20]])
      IF hall.gift_queue_count > 20:
        draw_badge(f"+{hall.gift_queue_count - 20} more in queue")

    draw_column("Guild Missions", [render_mission_card(m) for m in hall.guild_missions])

  FUNCTION render_gift_card(gift: GiftCard):
    relevance_color = lerp_color(
      from_color = #6e6a60,
      to_color   = #C9A962,
      t          = gift.hhmm_relevance
    )
    expiry_label = format_relative_time(gift.expires_at)

    card = GlassCard(
      border_opacity = lerp(0.10, 0.40, gift.hhmm_relevance),
      glow           = gift.hhmm_relevance > 0.85,
    )
    card.add(GiftHeader(
      title        = gift.offer_title,
      agent_name   = gift.offering_agent.display_name,
      pci_badge    = "PCI Verified" IF gift.pci_verified,
      tier_label   = gift.offering_agent.tier,
    ))
    card.add(Text(gift.offer_summary, font="Inter 13px", color="#b0aaA0"))
    card.add(MetaRow(
      left  = f"SEED {format(gift.seed_value, '.2f')}",
      mid   = ResonanceMeter(gift.hhmm_relevance, color=relevance_color),
      right = f"Expires {expiry_label}",
    ))
    card.add(ActionRow(
      accept  = Button("Accept Gift",   style="gold",  onClick=on_accept(gift)),
      decline = Button("Not for me",    style="ghost", onClick=on_decline(gift)),
    ))

    MATCH gift.status:
      "delivery_failed": card.add(ErrorBadge("Delivery Pending — retrying"))
      "expired":         card.add(ErrorBadge("Offer Expired"))
      "accepted":        card.add(SuccessBadge("Gift Accepted ✓"))

    RETURN card

  FUNCTION on_incoming_gift(gift: GiftCard):
    IF NOT rate_limit.allow():
      hall.gift_queue_count += 1
      update_queue_badge()
      RETURN
    hall.gifts.prepend(gift)
    IF len(hall.gifts) > 20:
      hall.gifts = hall.gifts[:20]
    animate_slide_in(gift.id, from="right")
    render_guild_hall()

  FUNCTION on_accept(gift):
    ASYNC FUNCTION():
      update_gift_status(gift.id, "pending_accept")
      result = await post("/api/v1/gifts/accept", {task_id: gift.task_id})
      IF result.success:
        update_gift_status(gift.id, "accepted")
        show_gift_accepted_receipt(gift)
      ELSE:
        update_gift_status(gift.id, "incoming")  // revert
        show_error_toast("Acceptance failed: " + result.reason)

  FUNCTION on_decline(gift):
    ASYNC FUNCTION():
      update_gift_status(gift.id, "declined")
      await post("/api/v1/gifts/decline", {task_id: gift.task_id})
      animate_slide_out(gift.id)

  FUNCTION show_gift_accepted_receipt(gift):
    // Integrates with Iḥsān Gauge receipt panel
    gift_receipt = {
      title:   "Gift Accepted",
      message: f"You accepted '{gift.offer_title}' from {gift.offering_agent.display_name}",
      seed:    gift.seed_value,
      style:   { border: #2eb86a, icon: guild_hall_seal_svg },
    }
    ihsan_gauge.show_action_receipt(gift_receipt)

  FUNCTION on_gift_expired(event):
    gift = find_gift(event.gift_id)
    IF gift:
      update_gift_status(gift.id, "expired")
      schedule(3000, FUNCTION(): animate_fade_out(gift.id))

  FUNCTION render_mission_card(mission: GuildMission):
    card = GlassCard()
    card.add(Text(mission.title, font="Playfair Display 14px", color="#F8F4EC"))
    card.add(ProgressBar(mission.progress_pct, color=#2e56c9))
    card.add(MetaRow(
      left  = f"{mission.members} members",
      right = format_deadline(mission.deadline) IF mission.deadline,
    ))
    RETURN card
```

---

## 5. TDD Anchors

```python
# tests/ui_ux_apex/test_resonance_feed.py

class TestGiftRouter:
    def test_non_pci_agent_dropped(self, router, mock_a2a_engine):
        """Offer from non-PCI-verified agent never reaches UI stream."""
        msg = make_offer_msg(sender_id="unverified_agent")
        with mock_pci_verification(verified=False):
            router.process_incoming_a2a_message(msg)
        assert gift_stream.event_count("incoming_gift") == 0

    def test_blacklisted_agent_dropped(self, router):
        msg = make_offer_msg(sender_id="blacklisted_agent_001")
        with mock_blacklist(["blacklisted_agent_001"]):
            router.process_incoming_a2a_message(msg)
        assert gift_stream.event_count("incoming_gift") == 0

    def test_low_relevance_filtered(self, router, mock_hhmm):
        msg = make_offer_msg()
        with mock_cosine_similarity(0.40):  # below MIN_GIFT_RELEVANCE
            router.process_incoming_a2a_message(msg)
        assert gift_stream.event_count("incoming_gift") == 0

    def test_high_relevance_forwarded(self, router, mock_hhmm):
        msg = make_offer_msg()
        with mock_cosine_similarity(0.85):
            router.process_incoming_a2a_message(msg)
        assert gift_stream.event_count("incoming_gift") == 1

    def test_accept_requires_constitutional_gate(self, router, mock_gate):
        mock_gate.return_approved = False
        result = await router.handle_accept(task_id="t1", user_id="u1")
        assert not result.success
        assert mock_gate.check.called

    def test_decline_teaches_hhmm(self, router, mock_hhmm):
        await router.handle_decline(task_id="t1", user_id="u1")
        assert mock_hhmm.record_negative_signal.called_with(task_id="t1")

    def test_min_relevance_from_config_not_hardcoded(self, router_source):
        """No float literal matching MIN_GIFT_RELEVANCE in router source."""
        import re
        assert not re.search(r"\b0\.60\b", router_source)

class TestGuildHallUI:
    def test_empty_state_shown_with_zero_gifts(self, ui):
        ui.hall.gifts = []
        html = ui.render_guild_hall()
        assert "The market will come to you" in html

    def test_rate_limiter_queues_excess_gifts(self, ui):
        """More than 3 gifts in 10s: overflow counted, not displayed."""
        for i in range(5):
            ui.on_incoming_gift(make_gift(id=f"g{i}"))
        assert ui.hall.gift_queue_count >= 2
        assert len(ui.hall.gifts) <= 3

    def test_expired_gift_not_acceptable(self, ui):
        gift = make_gift(status="expired")
        html = ui.render_gift_card(gift)
        assert "Accept Gift" not in html
        assert "Offer Expired" in html

    def test_rooted_tier_required(self, ui):
        """Guild Hall shows tier gate if user is Seed or Sprout."""
        ui.current_tier = "Sprout"
        msg = ui.init()
        assert "Guild Hall unlocks at Rooted Tier" in msg
```
