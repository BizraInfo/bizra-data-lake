# 📝 BIZRA Commit Style Guide

**Learning from:** Slack's "delight in details" + Conventional Commits + Storytelling

---

## 🎯 Philosophy

**Good commit messages are:**
- ✅ Clear (what changed?)
- ✅ Contextual (why did it change?)
- ✅ Memorable (will I remember this in 6 months?)
- ✅ Human (code is read by people, not just machines)

**Bad commit messages are:**
- ❌ "fix bug"
- ❌ "update stuff"
- ❌ "wip"
- ❌ "asdfasdf"

---

## 📐 Structure

```
<type>(<scope>): <emoji> <subject>

<body>

<footer>
```

### Type (Required)

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:     New feature
fix:      Bug fix
docs:     Documentation only
style:    Formatting (no code change)
refactor: Code restructuring (no feature change)
perf:     Performance improvement
test:     Adding/fixing tests
chore:    Build process, dependencies
ci:       CI/CD pipeline changes
```

### Scope (Optional but Recommended)

```
feat(backend):  Backend code
feat(frontend): Frontend code
feat(agents):   Multi-agent system
feat(ci):       CI/CD pipeline
feat(docs):     Documentation
```

### Emoji (Optional but Encouraged)

Add ONE emoji at the start of the subject to make it scannable:

```
✨ feat:     New feature (sparkles)
🐛 fix:      Bug fix (bug)
📚 docs:     Documentation (books)
🎨 style:    Code formatting (art palette)
♻️  refactor: Code restructuring (recycle)
⚡ perf:     Performance (lightning)
✅ test:     Tests (checkmark)
🔧 chore:    Build/dependencies (wrench)
🚀 ci:       CI/CD pipeline (rocket)
🔒 security: Security fix (lock)
```

### Subject (Required)

**50 characters or less** - This appears in GitHub's UI

**Good:**
```
✨ feat(agents): Add adaptive prior learning module
🐛 fix(ci): Prevent timeout in Python 3.12 tests
📚 docs(roadmap): Update Genesis 100 timeline
```

**Bad:**
```
add stuff
fix
update things that were broken before
```

### Body (Optional but Encouraged for Complex Changes)

**Explain:**
1. What problem does this solve?
2. Why did you choose this approach?
3. What alternatives did you consider?

**Wrap at 72 characters per line**

### Footer (Optional)

```
Fixes #123
Closes #456
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## ✨ Examples: Before & After

### Example 1: Feature Addition

**❌ Bad:**
```
add GEM modules
```

**✅ Good:**
```
✨ feat(benchmark): Add GEM modules for hidden architecture

New benchmark and spearpoint components:
- AdaptivePriorLearning: Bayesian category belief tracking (GEM #1)
- MIRASMemory: Retrieval-augmented memory with HNSW search (GEM #2)
- ZScorer: Statistical z-score routing with online mean/variance (GEM #4)
- TrueSpearpointLoop: v9 hidden architecture benchmark composer

This fixes CI ImportError and brings GEM #1, #2, #4 to life.
Your agents just got smarter. 🧠

Fixes #789
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Why it's better:**
- Explains WHAT (4 new modules)
- Explains WHY (fixes CI, enables GEMs)
- Adds personality ("Your agents just got smarter")
- Links to context (issue #789, co-author)

---

### Example 2: Bug Fix

**❌ Bad:**
```
fix bug in webhook
```

**✅ Good:**
```
🐛 fix(ci): Discord webhook failing on canceled runs

The webhook was throwing errors when CI runs were manually canceled
because it expected a 'conclusion' field that doesn't exist for
canceled jobs.

Solution: Check for 'status' first, fallback to 'conclusion' if present.

This prevents the notification channel from filling with error messages.

Fixes #234
```

**Why it's better:**
- Explains the problem
- Explains the solution
- Explains the impact
- Links to issue

---

### Example 3: Refactoring

**❌ Bad:**
```
refactor code
```

**✅ Good:**
```
♻️ refactor(agents): Extract Tank/Healer/DPS behavioral contracts

Pulled agent behaviors into explicit Rust traits to enforce
compile-time guarantees. This makes it impossible to create
a "Tank" agent that doesn't implement defense mechanisms.

Before: 500 lines of implicit behavior checks at runtime
After: 200 lines of trait implementations, verified at compile time

No user-facing changes, but the type system now catches bugs
that used to appear in production.

Performance: No change (same logic, better structure)
```

**Why it's better:**
- Explains the motivation
- Quantifies the change (500 → 200 lines)
- Clarifies no user impact
- Notes performance is unchanged

---

### Example 4: Documentation

**❌ Bad:**
```
update docs
```

**✅ Good:**
```
📚 docs(roadmap): Add Genesis 100 timeline with weekly breakdown

Created public ROADMAP.md showing:
- Phase 1: Foundation (COMPLETE) ✅
- Phase 2: Genesis Launch (IN PROGRESS) 🔄
- Phase 3: Ecosystem Growth (PLANNED) 🔲

Transparency matters. Users deserve to know where we're going.

This addresses feedback from Discord #general channel where users
asked "when is X feature coming?"

Now they can check ROADMAP.md anytime.
```

**Why it's better:**
- Explains what was added
- Explains why it matters ("Transparency matters")
- Links to community feedback
- Shows empathy for users

---

## 🎨 Personality Guidelines

### DO: Add Human Touch

**✅ Good:**
```
🎉 feat(frontend): Onboarding flow is now delightful

First-time users see a welcoming seed animation, clear
step-by-step guidance, and encouraging messages.

"Consulting your 7-agent constellation..." feels so much
better than "Loading..."

Small details matter. This is how trust starts.
```

**Why:** Explains the feeling, not just the function

---

### DON'T: Overdo It

**❌ Too Much:**
```
🎉🎊🎈 feat(frontend): OMG THE ONBOARDING IS SO AMAZING NOW!!!

THIS IS THE BEST FEATURE EVER CREATED IN THE HISTORY OF SOFTWARE!!!
USERS WILL CRY TEARS OF JOY!!! 🚀🚀🚀🌟🌟🌟✨✨✨

(also fixed a typo)
```

**Why:** Cringe. Be excited, but professional.

---

### DO: Tell Stories

**✅ Good:**
```
🐛 fix(agents): Healer no longer panic-restarts on corrupted state

The Healer agent would completely restart whenever it encountered
malformed JSON in the memory store. This meant users lost context
mid-conversation.

Now, the Healer:
1. Detects corruption
2. Isolates the bad data
3. Restores from the last known-good checkpoint
4. Logs the incident for debugging

Your conversations stay coherent, even when things break internally.

This fixes the "why did my agent forget everything?" bug reported
by 12 users in the past week.
```

**Why:** Explains the user impact, not just the technical fix

---

## 🚫 Anti-Patterns

### Anti-Pattern 1: "WIP" Commits

**❌ Don't:**
```
wip
wip 2
wip final
wip ACTUALLY final this time
```

**✅ Do:**
```
🚧 chore(agents): Work in progress - refactoring Tank trait

Not ready for review yet. Pushing to backup work-in-progress.

Current status:
- [x] Extract trait definition
- [x] Implement for SecurityAgent
- [ ] Implement for VerificationAgent (tomorrow)
- [ ] Add unit tests
```

---

### Anti-Pattern 2: Mixing Unrelated Changes

**❌ Don't:**
```
fix: fix bug and also add new feature and update docs and refactor
```

**✅ Do:**
Make 3 separate commits:
```
1. 🐛 fix(agents): Fix memory leak in Healer
2. ✨ feat(frontend): Add impact visualization
3. 📚 docs(readme): Update installation instructions
```

---

### Anti-Pattern 3: Passive Voice

**❌ Don't:**
```
Memory was improved
Bug was fixed
Feature was added
```

**✅ Do:**
```
Improve memory efficiency by 40%
Fix race condition in agent spawning
Add real-time impact tracking
```

---

## 📏 Length Guidelines

### Subject Line
- **Ideal:** 50 characters
- **Maximum:** 72 characters
- **Rule:** If you can't fit it, your commit is probably doing too much

### Body
- **Minimum:** 0 lines (simple changes)
- **Ideal:** 3-5 lines (most changes)
- **Maximum:** ~20 lines (complex changes)
- **Rule:** Wrap at 72 characters per line

---

## 🎯 Quick Checklist

Before committing, ask yourself:

- [ ] **Can I understand this in 6 months?**
- [ ] **Does it explain WHY, not just WHAT?**
- [ ] **Would this help a new contributor?**
- [ ] **Is the subject line under 50 chars?**
- [ ] **Did I link relevant issues?**
- [ ] **Is there personality without being unprofessional?**

---

## 💡 Inspiration Sources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- Slack's internal commit culture (based on public talks)
- Linux kernel commit guidelines

---

## 🤝 Examples from BIZRA History

### Real Commits from Our Repo

**Great Example #1:**
```
✨ feat(benchmark): Add GEM modules — AdaptivePrior, MIRAS Memory, ZScorer

New benchmark and spearpoint components required by __init__.py exports:
- AdaptivePriorLearning: Bayesian category belief tracking (GEM #1)
- MIRASMemory: retrieval-augmented memory with HNSW search (GEM #2)
- ZScorer: statistical z-score routing with online mean/variance (GEM #4)
- TrueSpearpointLoop: v9 hidden architecture benchmark composer

Fixes CI ImportError: 'No module named core.benchmark.adaptive_prior'

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Why it's great:**
- Clear type and scope
- Lists what was added
- Explains why (fixes import error)
- Credits co-author

---

**You can improve it by adding personality:**
```
✨ feat(benchmark): Unlock hidden architecture with 4 new GEM modules

Your agents are about to get significantly smarter.

New capabilities:
- AdaptivePriorLearning: Learns from experience (Bayesian belief tracking)
- MIRASMemory: Remembers like you remember (HNSW semantic search)
- ZScorer: Detects outliers automatically (statistical routing)
- TrueSpearpointLoop: The v9 composer orchestrating them all

This brings GEM #1, #2, #4 to life and fixes the CI import error
that was blocking deployment.

Your agents just leveled up. 🧠⚡

Closes #345
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## 🎓 Final Wisdom

> "Code is read 10x more than it's written. Commit messages are read 100x more."

Make them count. Make them clear. Make them memorable.

Happy committing! ✨
