---
name: smart-browser-management
description: >
  Browser Management via MCP — Page navigation, content reading, form interaction,
  tab management, screenshot capture, and element finding. Sovereign web interaction
  that keeps the human in control. Triggers on: browse, search, navigate, open page,
  fill form, read page, extract text, screenshot, find element, click button, web search.
license: MIT
metadata:
  author: m.beshr
  version: '1.0'
  category: productivity
  bizra_skill_tree: browser_skill_tree
  default_capability: true
  mcp_server: bizra-browser
---

# Smart Browser Management (AI Cowork)

Sovereign web interaction via MCP — the node browses on behalf of its human with constitutional gates at every step.

## BIZRA Skill Tree Mapping

Maps to `browser_skill_tree()` in `bizra-agent/src/skills/skill_tree.rs`:

| Skill Node | Mastery Required | Maps to Capability |
|---|---|---|
| `br_navigate` | Novice (boot) | URL navigation, SAT validates URLs |
| `br_history` | Competent navigate | Back/forward traversal |
| `br_tabs` | Competent navigate | Open, close, switch tabs |
| `br_read` | Novice (boot) | Page reading (accessibility tree) |
| `br_extract` | Competent read | Raw text extraction |
| `br_find` | Competent read | Element finding by description |
| `br_screenshot` | Competent read | Visual state capture |
| `br_fill` | Competent read, SAT | Form field interaction |
| `br_click` | Competent fill, SAT | Element clicking |
| `br_execute_js` | All prereqs, SAT + HITL | JavaScript execution (highest gate) |

## Constitutional Requirements

- SAT validates every URL before navigation (UrlValidator)
- No credential entry without HITL approval
- No downloads from untrusted sources without confirmation
- Cookie consent auto-declined (privacy-first)
- Receipt emitted for every page interaction
- Sensitive fields (password, credit card, token) always require HITL

## When to Use This Skill

Use when the user asks to:

- **Browse or search** the web for information
- **Open a page** or navigate to a URL
- **Read or extract** content from a webpage
- **Fill a form** or interact with web elements
- **Take a screenshot** for documentation or verification
- **Manage tabs** — open, close, switch between pages
- **Find elements** on a page by description

## Phase 1: Navigation (br_navigate — Novice from boot)

**Step 1 — SAT URL validation (automatic, every navigation)**

Before any navigation, UrlValidator checks:
- Domain not in constitutional blocklist
- No `javascript:` or `data:text/html` schemes
- No credential redirect patterns
- If allowlist-only mode: domain must be in allowlist

**Step 2 — Navigate**
```
Action: Navigate { url: "https://example.com" }
Receipt: { url, timestamp, status_code, title }
```

**Step 3 — History + Tabs (unlocks at Competent)**
```
Action: GoBack / GoForward
Action: NewTab / CloseTab { tab_id } / SwitchTab { tab_id }
```

## Phase 2: Reading (br_read — Novice from boot)

**Step 1 — Read page via accessibility tree**
```
Action: ReadPage → returns structured element tree with refs
```

**Step 2 — Extract text (unlocks at Competent)**
```
Action: ExtractText → returns raw text content, prioritizing article
```

**Step 3 — Find elements (unlocks at Competent)**
```
Action: FindElements { description: "submit button" } → returns matching refs
```

**Step 4 — Screenshot (unlocks at Competent)**
```
Action: Screenshot → returns visual state for verification
```


## Phase 3: Form Interaction (br_fill — requires br_read at Competent, SAT)

**Constitutional gate**: SAT must approve every field interaction.

**Step 1 — Identify form fields via ReadPage**
```
ReadPage → find all input, select, textarea elements → present to user
```

**Step 2 — Fill fields (SAT validates each)**
```
Action: FillField { element_ref: "search_input", value: "BIZRA" }
```

**Step 3 — Sensitive field detection (HITL required)**

If element_ref contains: `password`, `credit`, `ssn`, `secret`, `token`, `key`
→ HITL approval required before any interaction
→ System NEVER enters credentials automatically

**Step 4 — Click elements (requires br_fill at Competent, SAT)**
```
Action: Click { element_ref: "submit_btn" }
```

SAT validates: is this a destructive action? (purchase, delete, submit)
If yes → HITL confirmation required.

## Phase 4: JavaScript Execution (br_execute_js — highest gate)

**Prerequisites**: br_read + br_fill + br_click ALL at Competent
**Gate**: SAT approval + HITL confirmation (always)

This is the most dangerous browser capability. It can:
- Access page DOM arbitrarily
- Read cookies and storage
- Modify page content
- Send network requests

**Every JS execution must**:
1. Be reviewed by SAT for safety
2. Be presented to user for HITL approval
3. Produce a receipt with the code executed and result
4. Be logged in the mission manifest

```
Action: ExecuteJs { code: "document.title" }
→ SAT review: code has no side effects, approved
→ HITL: "Execute this JavaScript? 'document.title'"
→ User confirms → execute → receipt
```

## Phase 5: Safety (Constitutional — applies to ALL phases)

1. **URL validation on every navigation** — SAT runs UrlValidator automatically
2. **Privacy-first cookie handling** — auto-decline unless user configures otherwise
3. **No silent credential entry** — sensitive fields always need human confirmation
4. **Receipt chain** — every page interaction produces a BLAKE3-chained receipt
5. **Tab cleanup** — orphan tabs closed on session end, manifest records all

## Decision Flowchart

```
User request
├─ "search for" / "look up" / "find online"
│   → Navigate to search engine → Read results → Present
├─ "open" / "go to" / "navigate to" + URL
│   → SAT validate URL → Navigate → Read → Present
├─ "fill" / "submit" / "enter" + form
│   → Read page → Identify fields → SAT validate → Fill → HITL for sensitive
├─ "screenshot" / "show me" / "what does it look like"
│   → Navigate if needed → Screenshot → Present
├─ "extract" / "get the text" / "read the article"
│   → Navigate if needed → ExtractText → Present
└─ "click" / "press" / "select" + element
    → Read page → Find element → SAT validate → Click → Receipt
```
