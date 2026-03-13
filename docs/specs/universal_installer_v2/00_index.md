# BIZRA Universal Sovereign Installer v2.0 — Spec Index

> LOCKED: v2.0 | 2026-03-08 | Dubai
> Constitutional Anchor: Rule 4 (Simplicity) + Mother Test
> Design Principle: 3 taps, 3 minutes, any device, any language

## Spec Modules

| # | File | Domain | LOC |
|---|------|--------|-----|
| 01 | `01_device_detection.md` | Hardware detection, DeviceProfile, adaptive model selection | 278 |
| 02 | `02_installer_flow.md` | 6-step install pipeline, 3-tap UX, health check, audit receipt | 362 |
| 03 | `03_i18n_language.md` | Language sovereignty, i18n architecture, RTL, PoT governance | 272 |
| 04 | `04_platform_adapters.md` | Windows/macOS/Linux/Android/iOS adapters, Tauri shell | 482 |
| 05 | `05_urp_economy.md` | Resource dedication, URP pool, PoI verification, SEED rewards | 493 |
| 06 | `06_update_lifecycle.md` | Self-update, delta patches, rollback, disk management, profiles | 457 |
| 07 | `07_tdd_anchors.md` | Test plan: unit, integration, E2E, Mother Test, 8 Billion Test | 502 |

## Design Laws (from source spec)

1. **Mother Test** — 3 minutes, 3 taps, your mother can do it
2. **Zero Prerequisites** — no runtimes, no admin, no internet, no tech knowledge
3. **Hardware Adaptation** — 1GB phone to 128GB workstation
4. **Language Sovereignty** — language is identity, not setting
5. **Progressive Capability** — degrade gracefully, never break
6. **Sovereign Economics** — users keep 100%, Zakat 2.5% only
7. **Rooted in Revelation** — Quran > Hadith > scholars > papers > code
8. **One Command** — `bizra` launches everything

## Authority Hierarchy

1. Quran (supreme)
2. Authenticated Hadith
3. Classical scholars (Al-Ghazali, Ibn Khaldun, Al-Khwarizmi)
4. Founding papers (Ramadan 2023)
5. constants.py + codebase (lowest — if code conflicts with revelation, code is wrong)

## Prerequisite Gate

Genesis-100 (68 checks, 5 SAT agents) must pass L1-L3 before installer ships publicly.

## Document Dependencies

- Constitutional Sources v1.0 (Quran/Hadith mapping)
- Definition of Done v1.0 (68 checks)
- Terminal Build Contract v1.0 (7 views, 49 criteria)
- Identity Canon v1.0 / Proof Canon v1.1
- CLI Reference (docs/CLI_REFERENCE.md)
