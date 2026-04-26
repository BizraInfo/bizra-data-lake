# Ads Readiness Checklist — BIZRA Launch v0.1

**Policy:** Paid ads are the highest-liability public surface. This checklist is **all-or-nothing** — every gate must be green before any paid placement (Meta, X, LinkedIn, YouTube, Google, TikTok, programmatic).

**Current status (2026-04-24):** ❌ NOT READY.

---

## Gate 1 — Claim discipline (blocking)

- [ ] All ad copy drafts use only A-class or B-class-with-citation claims from `PUBLIC_CLAIMS_REGISTER.md`
- [ ] No C-class claim (SNR 0.974, $0.10→$0.008, 100% pass, 73/100 nodes) appears in any paid creative — text, image, or video
- [ ] No D-class claim (AGI, financial returns, first-in-world, benchmark superiority, security cert) appears in any paid creative
- [ ] Every numeric claim in any ad links to a public receipt/methodology accessible within 1 click from the ad
- [ ] English ↔ Arabic parity verified: if EN is ambition, AR is ambition; if EN cites a receipt, AR cites the same receipt
- [ ] No "guaranteed," "proven," "best," "only," "100%," "no risk," "save you money" language

**Owner:** operator. **Sign-off required:** yes.

## Gate 2 — Visual QA of creatives (blocking)

- [ ] Every creative used in a paid placement is either (a) an SVG template from `assets/editable_svg/` with known text, or (b) a raster whose text has been visually verified word-for-word against the approved copy in `CLAIM_SAFE_LAUNCH_COPY.md`
- [ ] All 12 `rendered_concepts/` boards that will be reused have been visually inspected for: malformed letters, Arabic ligature/shape errors, color drift from brand tokens, mis-scaled logo, generic "sacred geometry" risk
- [ ] Small-text legibility verified at the smallest rendered placement size (Stories/Reels sidebar thumbnail)
- [ ] No placeholder / lorem ipsum / draft watermark in any creative
- [ ] Color accuracy verified against `data/bizra_visual_tokens.json` (Genesis Gold `#C9A962`, Celestial Navy `#0A1628`, Origin Black `#050B14`, Ivory `#F6F2E9`)

**Owner:** creative lead + operator. **Sign-off required:** yes.

## Gate 3 — Landing page coherence (blocking)

- [ ] bizra.ai hero copy matches or is consistent with the ad copy that will drive to it
- [ ] Current bizra.ai claims C4/C5/C7/C9 are either (a) removed, (b) linked to a public receipt, or (c) moved off the hero to a technical/under-the-hood page
- [ ] bizra.ai SPA renders correctly for social link previews (OG tags set, OG image set to an approved `ready_to_post` asset)
- [ ] bizra.ai privacy policy published if any ad claims "no telemetry," "local-only," or similar C1/C2 claims
- [ ] Any CTA in the ad has a working destination (page + form + confirmation flow)

**Owner:** operator + web lead. **Sign-off required:** yes.

## Gate 4 — Platform policy compliance (blocking per platform)

- [ ] **Meta (Facebook/IG):** no financial returns, no misleading claims, no "before/after" without substantiation, no health/medical claims, religious/political content flagged
- [ ] **X (Twitter):** content policy OK, account verified preferred
- [ ] **LinkedIn:** professional tone, no consumer-finance misrepresentation
- [ ] **YouTube / Google Ads:** claim substantiation policy (destination page has the data backing any quantitative ad claim)
- [ ] **Arabic ad targeting:** no content that triggers MENA religious/political ad restrictions
- [ ] Age-gating / geo-gating set correctly per platform
- [ ] Ad account 2FA enabled; billing method verified

**Owner:** operator. **Sign-off required per platform.**

## Gate 5 — Tracking & measurement (advisory, not blocking)

- [ ] UTM parameters set consistently (utm_source, utm_medium, utm_campaign, utm_content)
- [ ] Landing page has click-tracking / conversion event wired
- [ ] Privacy-respecting analytics chosen (self-hosted / no-cookie preferred — consistent with sovereignty brand stance)
- [ ] Daily spend cap set before any campaign activates
- [ ] Kill-switch procedure documented (who can pause all ads within 10 minutes?)

**Owner:** operator. **Sign-off recommended, not strictly blocking.**

## Gate 6 — Post-launch monitoring (advisory)

- [ ] Comments / replies monitored on first 24h of each ad (flag inappropriate questions, critical replies, impersonation)
- [ ] Claim-drift check: if a commenter asks "is X claim true?", there must be a public receipt to link
- [ ] Any account / ad rejection logged with the exact policy cited + mitigation

**Owner:** operator. **Sign-off recommended.**

---

## Blocking issues in current state

1. **Gate 3 blocked:** bizra.ai currently carries C4/C5/C7/C9 in public-facing copy without linked receipts. Any paid ad that drives to this site inherits that liability.
2. **Gate 2 blocked:** the 12 concept boards have not been visually QA'd in this session (text-only environment).
3. **Gate 1 drafts exist** in `CLAIM_SAFE_LAUNCH_COPY.md §6` but have not been human-signed-off.

## Recommendation

**Do not run paid ads until:**
1. Gate 3 is cleared by either removing / rewriting C4/C5/C7/C9 on bizra.ai **or** publishing receipts and linking from each claim.
2. Gate 2 is cleared by a human visual review of creatives (low effort).
3. Gate 1 is signed off by the operator on the specific `CLAIM_SAFE_LAUNCH_COPY.md §6` copy blocks.

Until then: organic launch only. Organic doesn't trigger platform-policy review, lets you learn audience response, and lets real reactions inform the ad copy. This is also more consistent with the "meaning before speed, proof before claim" brand posture.
