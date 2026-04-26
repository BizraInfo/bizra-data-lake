# BIZRA Public Launch Media Kit v0.1 — Repo Workspace

**Purpose:** in-repo copy of the public launch media kit for inspection, QA, and gated production use. **Nothing here is published** until a separate, authorized rollout step.

## What this directory contains

```
public_launch_media_kit_v0_1/
├── README.md                                   ← this file
├── ASSET_INDEX.md                              ← inventory of every file with size + sha256
├── HANDOFF_NOTES.md                            ← session handoff for next reviewer
├── QA_NOTES.md                                 ← our QA of the kit (asset + claim quality)
├── bizra_public_launch_media_kit_v0_1.zip      ← verbatim copy of the Downloads package
└── extracted/bizra_public_launch_media_kit_v0_1/   ← unpacked contents (read-only for review)
    ├── README_HANDOFF.md                       ← kit's own handoff note
    ├── index.html                              ← local preview page (not for publishing)
    ├── assets/
    │   ├── ready_to_post/                      ← PNG + WebP exports at post sizes
    │   ├── editable_svg/                       ← SVG templates (avatar, lockup, social covers, hero, manifesto)
    │   └── rendered_concepts/                  ← concept boards (some AI-rendered — review small text!)
    ├── copy/BIZRA_LAUNCH_COPY.md               ← kit's recommended launch copy
    ├── data/
    │   ├── asset_manifest.json                 ← 44 entries with sha256 + dimensions
    │   └── bizra_visual_tokens.json            ← brand colors + typography + motto
    └── docs/
        ├── README_HANDOFF-like notes (kit-internal)
        ├── ASSET_USAGE_NOTES.md                ← kit's usage notes
        ├── CLAIM_DISCIPLINE.md                 ← kit's claim-discipline policy (aligned with brand canon v0.2)
        └── QA_REPORT.md                        ← kit's self-reported QA (44 tracked files)
```

## Provenance

- Source: `~/Downloads/bizra_public_launch_media_kit_v0_1.zip`
- Source size: 42,655,784 bytes (42.7 MB)
- Source sha256: `b98abed09e9809dd474ab061393a4d7c354b35fe0aa782cf674d6b9cb7e887b8`
- Copied into repo: 2026-04-24 09:59 GST
- Repo-side zip sha256: `b98abed09e9809dd474ab061393a4d7c354b35fe0aa782cf674d6b9cb7e887b8` (byte-identical)
- Downloads original preserved (not moved, not modified)

## What this directory is NOT

- Not BIZRA runtime canon.
- Not published content. Nothing here has hit `bizra.ai`, any social platform, or any paid ad placement.
- Not signed-off for production use. See `QA_NOTES.md` and the `public_launch_readiness/` sibling folder for required pre-publish work.
- Not part of the Cognitive Foundry review cycle or any canon pack. This is a separate lane.

## Where to go next

- For claim-safety review → `../public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md`
- For website audit → `../public_launch_readiness/WEBSITE_AUDIT_bizra_ai_bizra_info.md`
- For copy to actually use when posting → `../public_launch_readiness/CLAIM_SAFE_LAUNCH_COPY.md`
- For a concrete rollout path → `../public_launch_readiness/NEXT_IMPLEMENTATION_PLAN.md`

## Rules carried in from the authorizing lane

- Do NOT publish, upload, or post any asset from this directory without explicit authorization.
- Do NOT edit `bizra.ai` source, DNS, or any external surface.
- Do NOT mutate the Downloads original.
- Do NOT merge with Cognitive Foundry canon packs or MEMORY.md or runtime code.
- Review small text on AI-rendered concept boards before any paid campaign reuse (per kit's own warning).
