# Claim-Safe Rewrite Pack — bizra.ai / bizra.info

**Purpose:** Drop-in replacement copy for every claim that must change. Every replacement is A-class (SAFE_NOW) or B-class with clear citation stub. English + Arabic parity maintained.

**Policy floor:** If it hasn't been measured, phrase as a direction. If it has been measured, cite the receipt. If it is uncertain, mark it uncertain. (Brand canon §5 Law of Assumption.)

---

## §C1 — "local agents / no cloud dependency"

### Why replace

Architecturally local-first, but cloud-*optional* (Postgres, URP reconciliation, federation gateway). Absolute "no cloud" is a false statement. Paid-ad reviewers flag it under "misleading capability claim."

### Replacement — English

> **Your machine. Your keys. Your node.**
> BIZRA is designed to run locally. Your data stays on your node unless you choose to share it.

### Replacement — Arabic

> **جهازك. مفاتيحك. عقدتك.**
> بذرة صُمّمت لتعمل محلياً. بياناتك تبقى على عقدتك، ما لم تختر أنت مشاركتها.

---

## §C2 — "no telemetry"

### Why replace

Architecturally defensible, but requires a published privacy policy + a way for users to verify. Absolute "no telemetry" without a statement is a liability.

### Replacement (conditional) — English

**If a privacy policy is published:**
> **No hidden telemetry.**
> We don't ship analytics, crash reports, or usage data off your node. See our [privacy statement](/privacy).

**If no privacy policy yet (default until published):**
> **Your actions stay on your node unless you choose to share.**
> We're building the privacy posture we want — see [our approach](/privacy).

### Replacement — Arabic

**مع سياسة خصوصية منشورة:**
> **لا تتبّع خفي.**
> لا نرسل تحليلات أو تقارير أخطاء أو بيانات استخدام من عقدتك. [سياسة الخصوصية](/privacy)

**قبل النشر (الافتراضي):**
> **أفعالك تبقى على عقدتك ما لم تختر أنت مشاركتها.**
> نحن نبني سياسة الخصوصية التي نؤمن بها. [طريقتنا](/privacy)

---

## §C3 — "Ed25519 receipt signatures" on consumer hero

### Why replace (consumer hero)

Technical detail that speaks to a developer audience, not a consumer audience. Architecturally true; keep in dev docs and investor deck.

### Replacement (consumer hero) — English

> **Every action leaves a receipt.**
> Every visible effect on your node is signed and chained — so nothing is claimed without proof.
> [See the chain →](/under-the-hood/receipts)

### Replacement (consumer hero) — Arabic

> **كل فعل يُثبت بإيصال.**
> كل أثر مرئي على عقدتك موقّع ومربوط — فلا ادّعاء بلا برهان.
> [شاهد السلسلة ←](/under-the-hood/receipts)

### Where the technical detail belongs

- `/under-the-hood/receipts` sub-page → full Ed25519 + BLAKE3-chained-receipt explanation, link to `bizra-omega/bizra-core/src/canonical_receipt.rs`.
- Investor deck.
- GitHub README.
- Press pitch to technical journalists.

---

## §C4 — "cost per action $0.10 → $0.008"

### Why remove

Precise dollar figures in marketing without a published methodology trip ad-platform "unsupported quantifiable claim" policies. There is no receipt chain backing these numbers at any public URL.

### Replacement — English (directional)

> **Designed to make verified action radically cheaper than cloud AI.**
> Local execution, tiered inference, and a receipt-native protocol mean most work never leaves your machine.

### Replacement — Arabic

> **مصمَّمة لجعل الفعل الموثَّق أرخص كثيراً من الذكاء السحابي.**
> تنفيذ محلي، واستدلال متدرّج، وبروتوكول يعتمد الإيصال — فمعظم العمل لا يغادر جهازك.

### If receipts become available later

Only then may exact-number framing return (see `RECEIPTIFICATION_REQUIREMENTS.md §C4`). Until then: no $.

---

## §C5 — "SNR 0.974"

### Why remove

Exact benchmark number. No published benchmark protocol, no receipt. Regulator / skeptic reads this and asks: "Baseline? Test set? Verifier?" No answer → liability.

### Replacement — English (directional)

> **Signal-vs-noise discipline built into every output.**
> Claims stay tied to evidence. Noise stays marked as noise.

### Replacement — Arabic

> **انضباط الإشارة مقابل الضجيج في كل مخرج.**
> الادّعاءات مربوطة بالبرهان. الضجيج يُصرَّح به.

### If receipts become available later

See `RECEIPTIFICATION_REQUIREMENTS.md §C5`. Until a published benchmark report + methodology exists at a public URL, keep directional wording.

---

## §C6 — "8,072 verified tests"

### Why replace (unless receipt)

Exact number in marketing requires a timestamped link to a CI artifact + the commit hash the count was measured on. Otherwise brittle (number changes constantly).

### Replacement — English (directional, always-safe)

> **Thousands of verified tests across the sovereign core.**
> Every merge must pass CI — the same discipline we apply to our claims.
> [See latest CI →](https://github.com/…/actions)

### Replacement — Arabic

> **آلاف الاختبارات الموثّقة في نواة السيادة.**
> كل دمج يجب أن يجتاز CI — نفس الانضباط الذي نطبّقه على ادّعاءاتنا.
> [شاهد آخر تشغيل ←](https://github.com/…/actions)

### If a receipted variant is desired

Add a timestamped JSON receipt at `/receipts/test-count-YYYY-MM-DD.json` containing `{commit_hash, pytest_count, cargo_test_count, timestamp_utc, ci_run_url}`. Update hero wording accordingly. See `RECEIPTIFICATION_REQUIREMENTS.md §C6`.

---

## §C7 — "100% pass rate"

### Why replace

Brittle — any future CI red falsifies. Compliance-adjacent (regulator: "100% of what, audited how?"). Replace with a *policy* claim instead of a metric claim.

### Replacement — English

> **CI must pass before merge — the same discipline we apply to our claims.**

### Replacement — Arabic

> **CI يجب أن ينجح قبل أي دمج — نفس الانضباط الذي نطبّقه على ادّعاءاتنا.**

---

## §C8 — "Ihsan Gate >= 0.95"

### Why contextualize (not remove)

Accurate to `core/integration/constants.py`. Safe to keep *with framing*: it's an internal conscience threshold, not a user-visible metric.

### Replacement — English (contextualized)

> **We hold our outputs to a high conscience threshold (Ihsan ≥ 0.95) before we ship.**
> [What Ihsan means here →](/ihsan)

### Replacement — Arabic

> **نلتزم بعتبة إحسان عالية (إحسان ≥ 0.95) قبل أي إصدار.**
> [معنى الإحسان هنا ←](/ihsan)

### Required sub-page

`/ihsan` — short explainer: what Ihsan is, how the 0.95 threshold applies internally, why it's a policy not a guarantee.

---

## §C9 — "73 of 100 nodes remaining"

### Why remove (or rewire)

Manufactured scarcity without a live counter backed by source-of-truth. Regulator: "Is this real or manipulated?"

### Option A — Remove entirely — English

> **Early-access cohort forming.**
> Join the waitlist for updates — no counter, no pressure.

### Option A — Arabic

> **تتشكّل مجموعة الوصول المبكر.**
> انضمّ إلى قائمة الانتظار لتصلك التحديثات — بلا عدّاد، بلا ضغط.

### Option B — Wire live counter (if Genesis-100 is a real cap)

Requires (see `RECEIPTIFICATION_REQUIREMENTS.md §C9`):

1. Source-of-truth database / Airtable with current active-Genesis-100-cohort count.
2. API endpoint returning `{active_nodes: N, cap: 100, updated_at: "<UTC>"}`.
3. Hero JS reads and renders: "Early-access cohort: N / 100. Updated daily."
4. Display explicit `updated_at` so the claim is verifiable.

Until option B is wired: use option A.

---

## §K1 — "BIZRA is live." (inside kit's own launch copy)

### Why replace

"Live" implies production readiness beyond evidence. Node0 is Tier A/B/C ✅ but Tier D ❌ (see audit). Softening to `"The Seed is public."` preserves launch energy while staying honest about stage.

### Replacement — English

> **The Seed is public.**

### Replacement — Arabic

> **بذرة الآن علنيّة.**

---

## Full replacement hero pack (integrated)

### English

```
BIZRA
The Seed of Sovereign Intelligence.

A human-first AI ecosystem built on meaning, proof, and Ihsan.
Not another chatbot. Not another platform that owns you.
One human. One node. One sovereign path.

Your machine. Your keys. Your node.
Every action leaves a receipt.
We hold our outputs to a high conscience threshold before we ship.

Build with meaning. Act with proof. Grow with Ihsan.

[CTA: The Seed is public →]   [Secondary: Under the hood →]
```

### Arabic

```
بذرة
بذرة الذكاء السيادي.

منظومة ذكاء إنساني أولاً، مبنية على المعنى والبرهان والإحسان.
ليست أداة محادثة أخرى. وليست منصة تملكك.
إنسان واحد. عقدة واحدة. طريق سيادي واحد.

جهازك. مفاتيحك. عقدتك.
كل فعل يُثبت بإيصال.
نلتزم بعتبة إحسان عالية قبل أي إصدار.

ابنِ بالمعنى. اعمل بالبرهان. وانمُ بالإحسان.

[CTA: بذرة الآن علنيّة ←]   [ثانوي: تحت الغطاء ←]
```

**Numeric ladder moves off the hero** to `/under-the-hood/` — backed by receipts or clearly marked directional. No exact $ / SNR / test-count / scarcity-counter wording on the hero at launch.
