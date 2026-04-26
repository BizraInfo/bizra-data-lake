# Profile Rollout Checklist — BIZRA Launch v0.1

**Scope:** Organic rollout to owned profile surfaces. Safer than paid ads; lower liability; still needs discipline.

**Current status (2026-04-24):** ✅ READY FOR ORGANIC once visual spot-check is done.

---

## Platform-by-platform rollout

### X (Twitter)

**Assets to use:**
- Header: `assets/editable_svg/BIZRA_x_header_1500x500.svg` (render to PNG 1500×500)
- Avatar: `assets/editable_svg/BIZRA_avatar_dark.svg` → PNG 400×400 min
- Pinned post: `assets/ready_to_post/BIZRA_04_launch_announcement_1080x1080.png`

**Copy:**
- Bio: see `CLAIM_SAFE_LAUNCH_COPY.md §3`
- Pinned post copy: `CLAIM_SAFE_LAUNCH_COPY.md §1` (short form)

**Pre-flight:**
- [ ] Visual QA of X_header SVG render (no cut-off text, Arabic renders correctly)
- [ ] Avatar circular-crop test (no content lost in the circle mask)
- [ ] Bio character count ≤ 160
- [ ] @bizra_ai handle claimed and consistent with `data/bizra_visual_tokens.json`

### LinkedIn

**Assets to use:**
- Cover: `assets/editable_svg/BIZRA_linkedin_cover_1584x396.svg` → PNG
- Profile photo / Page logo: `assets/editable_svg/BIZRA_avatar_dark.svg` → PNG 400×400
- Launch post image: `assets/ready_to_post/BIZRA_04_launch_announcement_1080x1080.png`

**Copy:**
- Headline: `CLAIM_SAFE_LAUNCH_COPY.md §3`
- About first-paragraph: `CLAIM_SAFE_LAUNCH_COPY.md §3`
- Launch post: `CLAIM_SAFE_LAUNCH_COPY.md §1` (medium form)

**Pre-flight:**
- [ ] LinkedIn Company Page exists and is verified (if using a brand page)
- [ ] Cover SVG renders without cut at LinkedIn's actual safe-area (the visible area is smaller than 1584×396)
- [ ] Launch post includes a clear CTA to bizra.ai
- [ ] Arabic version of About is present as a second paragraph OR as a second post

### YouTube

**Assets to use:**
- Banner: `assets/editable_svg/BIZRA_youtube_banner_2560x1440.svg` → PNG
- Channel icon: `assets/editable_svg/BIZRA_avatar_dark.svg` → PNG 800×800 min
- Video thumbnail (future): `assets/editable_svg/BIZRA_launch_post_1080.svg` variants

**Pre-flight:**
- [ ] Banner designed with YouTube's safe-area constraint (mobile-safe 1546×423 center)
- [ ] Channel description uses `CLAIM_SAFE_LAUNCH_COPY.md §3` medium
- [ ] First video / trailer plan exists OR channel launches with just branding + link to bizra.ai

### Instagram

**Assets to use:**
- Profile photo: `assets/editable_svg/BIZRA_avatar_dark.svg` → PNG 1024×1024
- Launch carousel: `assets/ready_to_post/BIZRA_04_launch_announcement_1080x1080.png` + `BIZRA_05_manifesto_poster_1080x1350.png` + `BIZRA_06_product_value_infographic_1080x1350.png`
- Story templates: can be derived from 1080×1350 by extending canvas to 1080×1920

**Pre-flight:**
- [ ] Handle `@bizra_ai` claimed and consistent
- [ ] Bio matches `CLAIM_SAFE_LAUNCH_COPY.md §3`
- [ ] Arabic alt-text on every post image for accessibility + MENA discoverability
- [ ] No auto-translation reliance — post bilingual captions explicitly

### Facebook Page (if applicable)

- [ ] Skip unless operator has a clear use case for FB. MENA audience often has presence but engagement is weak for dev-adjacent brands.

### Threads (Meta)

- [ ] Auto-claimed from IG. Cross-post launch-day content. Same copy as X.

### Other

- [ ] **Mastodon / Bluesky / Farcaster:** consider if sovereignty framing has traction in those communities. Use X copy as baseline.
- [ ] **GitHub org page:** update README + org avatar + org bio to the claim-safe launch copy.
- [ ] **Product Hunt:** defer until there's a publicly usable artifact to launch. PH without a working link is weak.

---

## Shared pre-flight (cross-platform)

- [ ] Every SVG template rendered to PNG at the exact target platform size (don't trust uploader resizing)
- [ ] Every raster passed a 5-second visual spot-check for small-text / Arabic-shaping / color-drift issues
- [ ] Every profile uses the same avatar artwork (consistency is trust)
- [ ] Every profile URL appears in the bio / description exactly as `bizra.ai` (no trailing slash inconsistency)
- [ ] Arabic copy reviewed by a human Arabic speaker for ligature / spacing / dialect register
- [ ] `hello@bizra.ai` email is live and routed before publishing "contact" in any bio

---

## Rollout sequence (recommended)

**Phase 1 — foundation (day 0, silent):**
1. Claim all handles. Lock them.
2. Upload avatar + cover/header to every platform. Set bio. Do NOT publish any post yet.
3. Publish bizra.ai claim-safe hero copy (separate from this lane — requires authorized website deploy).

**Phase 2 — launch moment (day 1):**
4. Publish pinned / primary launch post simultaneously across X, LinkedIn, IG, Threads.
5. Use short form copy from `CLAIM_SAFE_LAUNCH_COPY.md §1` / §2.
6. Attach `BIZRA_04_launch_announcement_1080x1080.png`.
7. Post Arabic version on the same profiles (either as a second post or bilingual caption).
8. Founder personal-profile quote-post referencing the brand post.

**Phase 3 — first-week support (days 2–7):**
9. Day 2: manifesto poster (`BIZRA_05`) with §5 manifesto caption.
10. Day 3: product-value infographic (`BIZRA_06`) — but only if the numbers on the infographic pass claim-register review.
11. Day 4: brand-identity board or vision board with movement line "Every human is a node. Every node is a seed."
12. Day 5: founder letter (long form `CLAIM_SAFE_LAUNCH_COPY.md §1`).
13. Day 6–7: respond to engagement; collect questions into a FAQ draft.

**Phase 4 — iteration (week 2+):**
14. Any paid-ad decision requires `ADS_READINESS_CHECKLIST.md` to clear first.
15. Press / investor outreach (separate lane).

---

## Blocking / hard stops

- Stop rollout if: any platform flags an account for policy review, any C-class claim is found on a live post, any Arabic copy is reported to have a translation error, or the operator signals pause.
- Do NOT auto-respond to critical comments; route to operator for at-human response.

---

## Approval status

- **Phase 1 (silent foundation):** ✅ can proceed with operator authorization; no publishing risk.
- **Phase 2 (launch moment):** ✅ can proceed with operator authorization AFTER visual spot-check of the 3 launch-day images.
- **Phase 3 (first-week):** ✅ can proceed; requires operator sign-off on each day's specific image/caption pair.
- **Phase 4 (paid ads):** ❌ blocked by `ADS_READINESS_CHECKLIST.md`.
