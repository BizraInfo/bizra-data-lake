# BIZRA Universal Sovereign Installer
## Every Human. Every Device. Every Language.

> **Version:** 2.0 · LOCKED
> **Date:** March 8, 2026 · Dubai
> **Constitutional Anchor:** البذرة Rule 4 (البساطة — Simplicity) + Mother Test
> **Constitutional Sources:** BIZRA-Constitutional-Sources-v1.0.docx (Quran + Hadith backing for every law)
> **Root Documents:** الرسالة (Ramadan 2023) + البذرة (Ramadan 2023) — the founding papers
> **Design Principle:** If your mother cannot install this in under 3 minutes in her own language, it has failed.
> **Economics:** Users keep 100% of earned SEED. Zakat (2.5%) is the only obligatory deduction.
> **CLI:** Type `bizra` in any terminal. Like `claude` or `codex` — one command, everything starts.

---

## 1. Why This Exists

BIZRA promises: **every human is a node, every node is a seed, every seed has infinite potential.**

That promise is broken if:
- A student in Lagos has an Android phone and cannot install it
- A grandmother in Cairo speaks only Arabic and the installer is in English
- A developer in São Paulo has a Linux laptop and the installer only supports Windows
- A farmer in Java has a low-end phone with 2GB RAM and the installer needs 16GB
- A shopkeeper in Karachi has no internet and needs to install from a USB stick

The universal installer is not a convenience feature. It is a **constitutional requirement**. Without it, BIZRA serves the privileged, not the 8 billion.

---

## 2. Design Laws

### Law 1: The Mother Test (3 Minutes)

The installer must complete in under 3 minutes on any supported device. The user makes at most 3 decisions:
1. Choose language (auto-detected, one tap to confirm or change)
2. Choose install location (default pre-selected, one tap to confirm)
3. Create identity (name + optional photo, one tap to generate Ed25519 keys)

After 3 taps, DEMA says "مرحبا" (or the equivalent greeting in their language) and the briefing view loads.

### Law 2: Zero Prerequisites

The installer must not require:
- Pre-installed runtimes (no "install Python first" or "install Node first")
- Admin/root access (runs in user space)
- Internet connection (offline-capable with bundled LLM)
- Technical knowledge (no command line, no config files, no environment variables)
- Payment (free forever for personal use — البذرة covenant)

### Law 3: Hardware Adaptation

The installer detects the device and adapts:
- CPU architecture (x86_64, ARM64, RISC-V)
- Available RAM (adaptive model selection)
- Available storage (minimal vs full install)
- GPU presence (CUDA, ROCm, Metal, Vulkan, CPU-only fallback)
- OS (Windows, macOS, Linux, Android, iOS — future)
- Network (online: download optimal model; offline: use bundled compact model)

### Law 4: Language Sovereignty

The user's language is not a setting. It is a **sovereignty attribute** stored in their node identity. BIZRA speaks whatever language the human speaks. The system does not default to English. It defaults to the language of the device, and the user can change it at any time.

### Law 5: Progressive Capability

A device with 2GB RAM gets BIZRA with a compact model (1.5B parameters). A device with 128GB RAM gets BIZRA with a full model (70B+ parameters). Both get the same constitutional guarantees. Both get the same 7-view terminal. Both get SEED for verified work. The experience degrades gracefully — never breaks.

### Law 6: Sovereign Economics (Users Keep 100%)

Every SEED token a user earns through verified work belongs entirely to the user. BIZRA does not take a cut. The only obligatory deduction is Zakat (2.5% annually) — a Quranic obligation, not a platform fee.

| Economic Rule | Rate | Source | Enforced By |
|--------------|------|--------|-------------|
| **User SEED retention** | 100% | البذرة covenant | Protocol — cannot be changed |
| **Zakat** | 2.5% annual | Quran (Al-Baqarah 2:43) | Protocol — cannot be changed |
| **Harberger tax** | 5% annual on idle assets | Constitutional constants | Protocol — governance can adjust |
| **BLOOM decay** | 2% monthly | Constitutional constants | Protocol — prevents permanent status |
| **BLOOM transferability** | Soulbound (0% transfer) | Al-An'am 6:164 | Protocol — cannot be changed |
| **Community pool** | 50% of founder/Foundation revenue | البذرة p.19 | Founder's oath (sadaqah) — NOT protocol |
| **Riba (interest)** | Forbidden (0%) | Quran (Al-Baqarah 2:278) | Protocol — cannot be changed |

**The 50% community pool is the founder's personal oath (صدقة) between him and Allah.** It applies to BIZRA Foundation revenue and Mumo's personal founder share only. It is NOT a protocol-level tax on users. Forced charity is not charity.

**Quranic basis:** مَّن ذَا الَّذِي يُقْرِضُ اللَّهَ قَرْضًا حَسَنًا فَيُضَاعِفَهُ لَهُ أَضْعَافًا كَثِيرَةً — "Who is it that would loan Allah a goodly loan so He may multiply it for him many times over?" (Al-Baqarah 2:245)

### Law 7: Rooted in Revelation

Every design law, every threshold, every gate in BIZRA traces to a source in Quran and Sunnah through البذرة and الرسالة (the founding papers, Ramadan 2023). The authority hierarchy is absolute:

1. **Quran** — Supreme, cannot be overridden
2. **Authenticated Hadith** — Bukhari, Muslim, Al-Albani
3. **Classical scholars** — Al-Ghazali, Ibn Khaldun, Al-Khwarizmi
4. **البذرة + الرسالة** — Founding papers
5. **constants.py + codebase** — Lowest authority. If code conflicts with revelation, the code is wrong.

**Reference:** BIZRA-Constitutional-Sources-v1.0.docx maps every principle to its Quranic/Hadith source.

### Law 8: One Command Launch

BIZRA launches with one command from any terminal:

```bash
bizra                          # Launch everything
bizra mission "organize files" # Submit a mission
bizra status                   # Check node health
bizra wallet                   # Check SEED/BLOOM balance
bizra briefing                 # Morning briefing from DEMA
bizra doctor                   # Diagnose issues
bizra stop                     # Stop all services
```

Like `claude` (Claude Code) or `codex` (OpenAI Codex) — the `bizra` command is the universal entry point. Auto-detects the backend, starts Ollama if needed, launches the frontend, and shows the dashboard. One command. Zero configuration.

---

## 3. Architecture: The Sovereign Installer

### 3.1 Single Binary Distribution

BIZRA distributes as a **single self-extracting binary** per platform. No ZIP files to unpack. No multi-step installers. One file. Double-click. Done.

| Platform | Binary Name | Format | Size Target |
|----------|-----------|--------|-------------|
| Windows x64 | `bizra-setup.exe` | Self-extracting PE | < 50 MB (installer) |
| Windows ARM | `bizra-setup-arm.exe` | Self-extracting PE | < 50 MB |
| macOS Universal | `BIZRA.dmg` | Universal binary (x64+ARM) | < 60 MB |
| Linux x64 | `bizra-install.AppImage` | AppImage (no root) | < 45 MB |
| Linux ARM64 | `bizra-install-arm64.AppImage` | AppImage | < 45 MB |
| Android | `BIZRA.apk` | APK (sideload) / Play Store | < 40 MB |
| iOS | BIZRA (App Store) | IPA | < 50 MB |

The installer binary contains:
- Tauri shell (Rust-based, cross-platform desktop framework)
- Hardware detection module
- Language detection + locale database
- Minimal embedded model (TinyLlama 1.1B quantized, ~700MB — downloaded post-install)
- Node identity generator (Ed25519)
- Self-update mechanism

### 3.2 Installation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: DETECT (automatic, 0 user interaction)              │
│                                                             │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│ │ OS + Arch    │ │ RAM + GPU    │ │ Language      │         │
│ │ Detection    │ │ Detection    │ │ Detection     │         │
│ └──────────────┘ └──────────────┘ └──────────────┘         │
│                                                             │
│ Result: DeviceProfile {os, arch, ram_gb, gpu, locale, ...}  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: GREET (1 user interaction — language confirmation)   │
│                                                             │
│ Screen shows:                                               │
│   [Flag emoji] "مرحبا بك في بذرة" (detected: Arabic)       │
│   [Confirm ✓] [Change Language ▼]                           │
│                                                             │
│ If user taps Confirm → proceed                              │
│ If user taps Change → show language picker (50+ languages)  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: ADAPT (automatic, 0 user interaction)               │
│                                                             │
│ Based on DeviceProfile, RECOMMEND (user can override):      │
│   Model tier (suggested, shown in Step 4):                  │
│     1GB RAM  → Micro (TinyLlama 1.1B Q2)    ~500MB         │
│     2GB RAM  → Compact (TinyLlama 1.1B Q4)  ~650MB         │
│     4GB RAM  → Standard (Phi-3 3.8B Q4)     ~2.3GB         │
│     8GB RAM  → Enhanced (Llama 3.1 8B Q4)   ~4.7GB         │
│     16GB RAM → Full (Qwen 2.5 14B Q4)       ~8.5GB         │
│     32GB+    → Premium (Llama 3.1 70B Q4)   ~40GB          │
│                                                             │
│   GPU acceleration:                                         │
│     NVIDIA   → CUDA backend                                │
│     AMD      → ROCm/Vulkan backend                         │
│     Apple    → Metal backend                                │
│     Intel    → OpenVINO backend                             │
│     None     → CPU-only (llama.cpp GGUF)                   │
│                                                             │
│   Install footprint:                                        │
│     Minimal  → Core + compact model (~1.5GB total)          │
│     Standard → Core + standard model (~4GB total)           │
│     Full     → Core + enhanced model + tools (~10GB total)  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: INSTALL (1 user interaction — confirm location)     │
│                                                             │
│ Screen shows:                                               │
│   "سيتم تثبيت بذرة في:" (BIZRA will be installed at:)      │
│   [~/BIZRA/] or [C:\BIZRA\]                                │
│                                                             │
│   Recommended model: Phi-3 Mini (3.8B) — 2.3 GB            │
│   [Use Recommended ✓]                                       │
│   [Change Model ▼] ← Advanced: choose smaller/larger       │
│                                                             │
│   [Install ✓]                                               │
│                                                             │
│ Progress bar with friendly messages in user's language:      │
│   "جارٍ إعداد عقلك الرقمي..." (Preparing your digital mind) │
│   "جارٍ توليد هويتك..." (Generating your identity)          │
│   "جارٍ تحميل نموذج الذكاء..." (Loading intelligence model) │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: IDENTITY (1 user interaction — create identity)     │
│                                                             │
│ Screen shows:                                               │
│   "ما اسمك؟" (What is your name?)                          │
│   [Name input]                                              │
│   [Optional: Take photo for avatar]                         │
│   [Create My Node ✓]                                        │
│                                                             │
│ Behind the scenes:                                          │
│   → Ed25519 keypair generated                               │
│   → Genesis ceremony executed                               │
│   → 12 agents minted (7 PAT + 5 SAT)                       │
│   → sovereign_state/ directory created                      │
│   → First DEMA briefing generated                         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5.5: SHARE (1 user interaction — resource dedication)  │
│                                                             │
│ Screen shows (after identity, before terminal):             │
│                                                             │
│   "جهازك يمكنه مساعدة الآخرين" (Your device can help others)│
│                                                             │
│   BIZRA detected:                                           │
│     CPU: 8 cores  |  RAM: 16 GB  |  Disk: 250 GB free      │
│     GPU: RTX 3060 (6GB VRAM)                                │
│                                                             │
│   ┌─────────────────────────────────────────┐               │
│   │ Share resources with the forest? 🌳      │               │
│   │                                         │               │
│   │  CPU: [████░░░░] 2 of 8 cores           │               │
│   │  RAM: [████░░░░] 4 of 16 GB             │               │
│   │  Disk: [██░░░░░░] 10 of 250 GB          │               │
│   │  GPU: [██░░░░░░] 2 of 6 GB VRAM         │               │
│   │  Schedule: [When idle ▼]                │               │
│   │                                         │               │
│   │  "You earn SEED for every resource       │               │
│   │   you share. Your device helps others.   │               │
│   │   Others' devices help you."             │               │
│   │                                         │               │
│   │  [Share & Earn ✓]  [Skip for Now →]     │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
│ If user taps "Share & Earn" → URP contribution starts       │
│ If user taps "Skip" → fully sovereign, no sharing, no URP   │
│ User can change this anytime in Settings (View C.7)         │
│                                                             │
│ NOTE: This is ALWAYS optional. The node works fully without │
│ sharing. Sharing earns SEED. Not sharing costs nothing.     │
│ Sovereignty means the user chooses. Always.                 │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: ALIVE (0 user interaction — terminal opens)         │
│                                                             │
│ DEMA speaks (in user's language):                         │
│                                                             │
│   "مرحبا يا [اسم]! أنا ديما، مساعدتك الشخصية.             │
│    أنت الآن عقدة رقمية سيادية. كل شيء يعمل محلياً على      │
│    جهازك. لا شيء يغادر. مستعد للمهمة الأولى؟"              │
│                                                             │
│   (Hello [Name]! I'm DEMA, your personal assistant.       │
│    You are now a sovereign digital node. Everything runs     │
│    locally on your device. Nothing leaves. Ready for your    │
│    first mission?)                                          │
│                                                             │
│ Dashboard view loads. Heartbeat starts. The node is alive.  │
└─────────────────────────────────────────────────────────────┘
```

**Total user interactions: 3 taps + 1 optional.** Language confirm, install location, create identity — then optionally share resources with the forest. The share screen is skippable. The 3-tap promise is preserved.

**After installation:** The user can always relaunch by typing `bizra` in any terminal. The `bizra` command auto-detects the runtime, starts the backend, launches the frontend, and opens the terminal. One command. Like `claude` or `codex`.

---

## 4. Language Architecture: Universal i18n

### 4.1 Design Principle

BIZRA does not "translate" English into other languages. BIZRA is **language-native** — the system thinks in the user's language from the first interaction. The LLM generates responses in the user's language. The terminal UI renders in the user's language. DEMA greets in the user's language. Receipts display in the user's language.

### 4.2 Language Tiers

| Tier | Languages | Coverage | Implementation |
|------|-----------|----------|---------------|
| **Tier 1: Full Native** | Arabic, English, Spanish, French, Chinese (Simplified), Hindi, Portuguese, Indonesian, Bengali, Russian | 4.5B speakers | Complete UI + DEMA personality + documentation + onboarding |
| **Tier 2: UI + LLM** | Japanese, Korean, German, Turkish, Vietnamese, Italian, Thai, Polish, Dutch, Ukrainian, Swahili, Urdu, Persian, Malay | +1.5B speakers | Full UI translation + LLM responds in language |
| **Tier 3: LLM Native** | All 100+ languages supported by the LLM | +2B speakers | UI in closest Tier 1/2 language, LLM responds natively |

**Tier 1 priority order:** Arabic first (البذرة is Arabic, the founder is Arabic-speaking, the Mother Test is in Arabic), then English (developer community), then by speaker population.

### 4.3 What Gets Translated

| Component | Scope | Method |
|-----------|-------|--------|
| **Installer UI** | 6 screens, ~50 strings | Static translation files (JSON) |
| **Terminal 7 views** | ~200 UI strings (labels, headers, buttons) | i18n JSON bundles per locale |
| **DEMA persona** | Greeting, briefing, mission prompts | LLM system prompt in user's language |
| **Error messages** | ~80 error strings | Static translation files |
| **Constitutional terms** | Ihsān, SEED, BLOOM, receipt, mission | Glossary per language (some terms kept original) |
| **Onboarding flow** | 5-step first-time experience | Translated + culturally adapted |
| **Documentation** | Getting started guide | Tier 1 languages first |
| **Receipt display** | Status, synthesis, channel names | Dynamic (LLM generates in user's language) |

### 4.4 What Does NOT Get Translated

| Component | Reason |
|-----------|--------|
| API endpoints | Technical interface, always English |
| Code comments | Developer interface |
| Log files | Debug interface |
| Hash values | Cryptographic, language-agnostic |
| Ed25519 keys | Binary, language-agnostic |
| Constitutional constants | Mathematical, universal |

### 4.5 RTL (Right-to-Left) Support

Arabic, Hebrew, Urdu, Persian, and other RTL languages require the entire terminal layout to mirror. This is not optional — it's a **constitutional requirement** for the first language tier.

| Component | RTL Adaptation |
|-----------|---------------|
| Terminal layout | Full mirror (navigation right, content left) |
| Text alignment | Right-aligned by default |
| Progress bars | Right-to-left fill |
| Tables | Right-to-left column order |
| Navigation | Right-to-left tab order |
| Keyboard shortcuts | Mirrored where applicable |
| Number display | LTR within RTL context (standard Arabic numeral handling) |

### 4.6 i18n File Structure

```
bizra/
  locales/
    ar/                    # Arabic (Tier 1)
      installer.json       # 50 installer strings
      terminal.json        # 200 terminal UI strings
      errors.json          # 80 error messages
      onboarding.json      # Onboarding flow text
      dema.json          # DEMA persona prompts
      glossary.json        # Constitutional term translations
    en/                    # English (Tier 1)
      ...
    es/                    # Spanish (Tier 1)
      ...
    zh-CN/                 # Chinese Simplified (Tier 1)
      ...
    ...
  locale-meta.json         # Metadata: language name (native script), direction, fallback
```

### 4.7 Language Detection Priority

1. **Device locale** (OS language setting)
2. **Browser language** (Accept-Language header, for web-based terminal view)
3. **IP geolocation** (approximate, for initial suggestion only)
4. **User choice** (always overrides, stored in node identity)

The detected language is a **suggestion**, never forced. The first screen always shows a language picker alongside the auto-detected option.

---

## 5. Hardware Adaptation Engine

### 5.1 DeviceProfile Detection

On first run, the installer collects a DeviceProfile:

```rust
struct DeviceProfile {
    // Platform
    os: OS,                    // Windows | macOS | Linux | Android | iOS
    os_version: String,        // "11", "14.2", "24.04"
    arch: Arch,                // x86_64 | aarch64 | riscv64
    
    // Compute
    cpu_cores: u32,            // Physical cores
    cpu_threads: u32,          // Logical threads
    ram_total_gb: f32,         // Total RAM
    ram_available_gb: f32,     // Available RAM at install time
    
    // GPU
    gpu: Option<GPUInfo>,      // {vendor, model, vram_gb, api: CUDA|ROCm|Metal|Vulkan|None}
    
    // Storage
    disk_available_gb: f32,    // Available disk space
    disk_type: DiskType,       // SSD | HDD | eMMC
    
    // Network
    network_available: bool,   // Can reach download servers?
    network_speed_mbps: f32,   // Estimated download speed
    
    // Locale
    system_locale: String,     // "ar-AE", "en-US", "pt-BR"
    timezone: String,          // "Asia/Dubai"
    
    // Display
    screen_width: u32,
    screen_height: u32,
    dpi: f32,
    touch_capable: bool,
}
```

### 5.2 Adaptive Model Selection

The installer selects the optimal LLM configuration based on available resources:

| RAM Available | Model | Quantization | VRAM Needed | Disk | Expected Quality |
|--------------|-------|-------------|-------------|------|-----------------|
| 1-2 GB | TinyLlama 1.1B | Q2_K | 0 (CPU) | 500 MB | Micro (S1 only, sovereign) |
| 2-4 GB | TinyLlama 1.1B | Q4_K_M | 0 (CPU) | 650 MB | Basic (simple tasks) |
| 4-8 GB | Phi-3-mini 3.8B | Q4_K_M | 0-2 GB | 2.3 GB | Good (most tasks) |
| 8-16 GB | Llama 3.1 8B | Q4_K_M | 2-6 GB | 4.7 GB | Strong (complex tasks) |
| 16-32 GB | Qwen 2.5 14B | Q4_K_M | 4-10 GB | 8.5 GB | Excellent |
| 32-64 GB | Llama 3.1 70B | Q4_K_M | 8-24 GB | 40 GB | Elite |
| 64+ GB | Multiple models | Q8/FP16 | 24+ GB | 80+ GB | Premium (MoE routing) |

**Fallback chain:** If the selected model fails to load (OOM), the installer automatically falls back to the next smaller tier. The user is never left with a broken install.

**Micro-node guarantee (< 2GB devices):** For devices with 1-2GB RAM (old phones, thin clients), BIZRA runs a degraded but **fully sovereign** micro-node using TinyLlama 1.1B at Q2_K quantization (~500MB). The micro-node operates in System-1 only (reflex cache lookups, no full S2 reasoning), maintains its own Ed25519 identity, evidence ledger, and constitutional heartbeat (every 5 minutes instead of 60 seconds). A degraded sovereign node is infinitely better than a perfect client terminal. The user can optionally REQUEST federated compute from the forest, but never REQUIRES it. **Every human is a node — no exceptions.**

### 5.3 GPU Acceleration Matrix

| GPU Vendor | API | Backend | Detection Method |
|-----------|-----|---------|-----------------|
| NVIDIA | CUDA 12+ | llama.cpp CUDA | `nvidia-smi` or NVML |
| AMD | ROCm 5.7+ | llama.cpp ROCm | `rocm-smi` or `/dev/kfd` |
| Apple | Metal 3+ | llama.cpp Metal | `system_profiler SPDisplaysDataType` |
| Intel | OpenVINO | llama.cpp SYCL | `clinfo` or `/dev/dri` |
| Qualcomm | Vulkan | llama.cpp Vulkan | Android GPU detection |
| None | CPU only | llama.cpp CPU (AVX2/NEON) | Default fallback |

The installer auto-detects the best available backend. If GPU fails, it silently falls back to CPU. The user never sees an error about GPU drivers.

---

## 6. Offline-First Installation

### 6.1 The USB Stick Scenario

A teacher in a rural village has no internet but a USB stick with the BIZRA installer shared by a friend. The installer must work completely offline.

**Offline bundle structure:**
```
USB/
  bizra-setup.exe              # Installer binary (50 MB)
  models/
    phi-3-mini-q4.gguf         # Pre-bundled model (2.3 GB)
  locales/
    ar.pack                    # Arabic language pack (200 KB)
    en.pack                    # English language pack (200 KB)
  README.txt                   # Multi-language quick start
```

**Total offline bundle: ~2.5 GB** — fits on any USB stick.

### 6.2 Online-Offline Hybrid

When internet is available, the installer:
1. Downloads the optimal model for the device (larger = better)
2. Downloads all Tier 1 language packs
3. Checks for updates to the installer itself
4. Registers the node with the forest discovery service (optional, sovereign choice)

When internet is not available, the installer:
1. Uses the bundled compact model
2. Uses the bundled language packs (at least Arabic + English)
3. Skips update check
4. Operates as a standalone sovereign node (no forest, but fully functional)

The node can join the forest later when internet becomes available. No functionality is lost in offline mode — only federation features (reflex sharing, forest sync) are unavailable.

---

## 7. Platform-Specific Adapters

### 7.1 Windows

| Component | Technology | Notes |
|-----------|-----------|-------|
| Installer | NSIS or WiX wrapped Tauri | No admin required (per-user install) |
| Shell | Tauri (WebView2) | Built-in on Windows 10+ |
| LLM Runtime | llama.cpp (CUDA/CPU) | Bundled, no separate Ollama install |
| Auto-start | Task Scheduler (optional) | User opt-in only |
| File location | `%LOCALAPPDATA%\BIZRA\` | Standard user location |

### 7.2 macOS

| Component | Technology | Notes |
|-----------|-----------|-------|
| Installer | DMG with drag-to-Applications | Standard macOS pattern |
| Shell | Tauri (WKWebView) | Built-in on macOS |
| LLM Runtime | llama.cpp (Metal/CPU) | Universal binary (x64+ARM) |
| Notarization | Apple Developer ID | Required for Gatekeeper |
| File location | `~/Library/Application Support/BIZRA/` | Standard macOS location |

### 7.3 Linux

| Component | Technology | Notes |
|-----------|-----------|-------|
| Installer | AppImage (no root required) | Single file, works on any distro |
| Alternative | Flatpak / Snap / .deb / .rpm | For package manager users |
| Shell | Tauri (WebKitGTK) | Requires libwebkit2gtk-4.0 |
| LLM Runtime | llama.cpp (CUDA/ROCm/CPU) | Auto-detected |
| File location | `~/.local/share/bizra/` | XDG standard |

### 7.4 Android (Future — Phase 2)

| Component | Technology | Notes |
|-----------|-----------|-------|
| Distribution | APK sideload + Play Store | Dual distribution |
| Shell | Tauri Mobile (WebView) | Android WebView |
| LLM Runtime | llama.cpp (Vulkan/CPU) | Optimized for mobile |
| Background | Foreground service | Keeps heartbeat alive |
| Storage | Internal + SD card support | Adaptive to available space |

### 7.5 iOS (Future — Phase 3)

| Component | Technology | Notes |
|-----------|-----------|-------|
| Distribution | App Store only | Apple requirement |
| Shell | Tauri Mobile (WKWebView) | Native WebView |
| LLM Runtime | llama.cpp (Metal) | Apple Neural Engine where available |
| Background | Background App Refresh | Limited by iOS policy |

---

## 8. Self-Update Mechanism

### 8.1 Sovereign Update Principle

BIZRA never force-updates. The user is notified of available updates and **chooses** when to update. This is sovereignty — the user controls their device, not the system.

### 8.2 Update Flow

```
1. Background check (once daily): GET /v1/update/check → {version, changelog, size}
2. Notification (non-intrusive): "BIZRA v3.1 available. 12 MB update."
3. User confirms → download delta patch (not full reinstall)
4. Apply on next restart (or immediately if user chooses)
5. Rollback available if update fails (keep previous version)
```

### 8.3 Delta Updates

Updates ship as **binary diffs**, not full replacements. A 50 MB installer with a 2 MB change ships a 2 MB delta patch. This matters for users with slow internet.

### 8.4 Model Updates

Model updates are separate from system updates. A user on the 3.8B model can upgrade to 8B when they get more RAM. Downgrade is also possible (delete larger model, keep smaller). Model files are never inside the system binary.

---

## 9. Implementation Technology: Tauri

### 9.1 Why Tauri

| Criterion | Tauri | Electron | Flutter | Native (per-platform) |
|-----------|-------|----------|---------|----------------------|
| Binary size | ~3-10 MB | ~150 MB | ~20 MB | Varies |
| Memory usage | ~30 MB | ~300 MB | ~80 MB | Varies |
| Startup time | < 1s | 2-5s | 1-2s | < 1s |
| Cross-platform | Win/Mac/Linux/Mobile | Win/Mac/Linux | Win/Mac/Linux/Mobile | Per-platform |
| Language | Rust (backend) + HTML/CSS/JS (frontend) | JS (all) | Dart | Per-platform |
| System access | Full (Rust) | Full (Node.js) | Limited | Full |
| WebView | System (no bundling) | Bundled Chromium | Custom | N/A |
| Sovereignty | No phone-home | Chromium telemetry | Google telemetry | Clean |

**Tauri wins** because: smallest binary, lowest memory, Rust backend (matches BIZRA's core language), system WebView (no bundled browser), and no vendor telemetry. The Mother Test device (2GB RAM) cannot run Electron (300MB RAM for the shell alone). Tauri uses 30MB.

### 9.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tauri Shell (Rust)                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Hardware      │  │ LLM Runtime  │  │ Sovereign     │  │
│  │ Detector      │  │ (llama.cpp)  │  │ Runtime       │  │
│  │               │  │              │  │ (Python/Rust) │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ i18n Engine  │  │ Auto-Update  │  │ IPC Bridge    │  │
│  │              │  │              │  │ (Tauri cmds)  │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                  System WebView (OS-native)               │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │           Terminal 7-View UI (React/TS)           │    │
│  │                                                    │    │
│  │  Dashboard │ Mission │ Timeline │ Memory │ ...     │    │
│  │                                                    │    │
│  │  i18n: react-intl + RTL support + locale bundles  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 10. Constitutional Terms Glossary Strategy

Some terms should NOT be translated. They should be taught.

| Term | Strategy | Rationale |
|------|----------|-----------|
| **BIZRA** | Keep original | Brand name |
| **Ihsān** (إحسان) | Keep Arabic + explain | Constitutional concept, universal meaning |
| **SEED** | Translate | Economic token, needs local understanding |
| **BLOOM** | Translate | Governance token, needs local understanding |
| **DEMA** | Keep original | Named after the founder's daughter — the Daughter Test personified |
| **Node** | Translate | Technical but needs local word |
| **Mission** | Translate | Core UX concept |
| **Receipt** | Translate | Core proof concept |
| **Reflex** | Translate | Core learning concept |
| **Gini** | Keep original | Mathematical concept |
| **Zakat** | Keep Arabic + explain | Islamic economic concept, universal meaning |

For Arabic specifically: many terms are ALREADY Arabic (Ihsān, Zakat, البذرة, الرسالة). The Arabic version is the original, not a translation.

---

## 11. Accessibility

The installer and terminal must be accessible to users with disabilities:

| Need | Implementation |
|------|---------------|
| Screen reader | ARIA labels on all UI elements, terminal outputs as accessible text |
| Keyboard navigation | Full keyboard control (Tab, Enter, Escape, 1-7 for views) |
| High contrast | System high-contrast mode detected and respected |
| Font scaling | Respects OS font size setting (up to 200%) |
| Color blindness | Status indicators use shape + color (not color alone) |
| Motor impairment | Large click targets (48px minimum), no time-limited interactions |
| Cognitive | Simple language option, progress indicators, undo for all actions |

---

## 12. Translation Governance: Proof of Translation (PoT)

Without governance, translation quality is inconsistent. A user in Cairo gets excellent Arabic (native speaker), while a user in Karachi gets poor Urdu (machine-translated). This violates Ihsān.

### 12.1 PoT Lifecycle

```
1. Translator stakes 100 SEED to submit translation
   → Uploads: locales/ur/terminal.json (200 strings)
   → Declares native speaker credentials

2. Validation period (7 days):
   → Native speakers review translation
   → Each reviewer stakes 10 SEED to vote
   → Voting: Accept / Reject / Request Changes

3. Consensus (weighted by reviewer PoT reputation):
   → If Accept (≥67%): Translator earns 500 BLOOM
   → If Reject (<67%): Translator loses 50 SEED
   → Reviewers earn BLOOM proportional to vote weight

4. Dispute resolution:
   → Translator contests rejection → arbitration
   → Genesis Council (human linguists) adjudicates
   → Losing party pays arbitration cost (50 SEED)
```

### 12.2 Translation Quality Score

$$\text{Quality} = \frac{\sum_{r=1}^{N} w_r \cdot v_r}{\sum_{r=1}^{N} w_r}$$

Where w_r = reviewer r's PoT reputation weight, v_r = reviewer r's vote (1=Accept, 0=Reject).

PoT records are stored in the Evidence Ledger (hash-chained, immutable). Translations are versioned — users can roll back to a previous version if quality degrades. The installer auto-updates translations when consensus improves a language pack.

---

## 13. Update Rollback Strategy

### 13.1 Version History

```
bizra/
  versions/
    v1.2.3/  # Current
    v1.2.2/  # Previous (rollback available)
    v1.2.1/  # Deleted after 2 successful updates
```

### 13.2 Rollback UI

```
Settings → About → Version History
  ✓ v1.2.3 (current) — Mar 8, 2026
  ○ v1.2.2 — Feb 23, 2026 [Rollback ↩️]
```

Clicking Rollback restores the previous binary, config, and dependencies. The evidence ledger and identity remain intact (stored separately, never rolled back). Model files are separate from system updates.

### 13.3 Auto-Rollback

If the system detects a crash within 60 seconds of update (3 consecutive restarts fail), it automatically reverts to the previous version and notifies the user.

---

## 14. Disk Space Management

### 14.1 Proactive Monitoring

The installer monitors disk space every heartbeat tick. Alerts trigger at defined thresholds:

| Disk Usage | Action |
|-----------|--------|
| < 85% | Normal operation |
| 85-90% | Info notification: "Disk getting full" |
| 90-95% | Warning: suggest model downgrade or log cleanup |
| 95-98% | Alert: pause non-critical writes (logs) |
| > 98% | Emergency: pause heartbeat, preserve evidence ledger |

### 14.2 Space Recovery Options (shown to user)

- Switch to smaller model (e.g., save 2.4 GB by going from 8B to 3.8B)
- Delete old evidence logs older than 90 days (configurable)
- Move BIZRA data to external drive or SD card
- Compress reflex cache (lossless)

---

## 15. Multi-User Profile Support

### 15.1 Shared Device Architecture

A family sharing one device gets separate sovereign identities:

```
BIZRA Install (System-Wide):
├─ Core Runtime (shared)
├─ Models (shared, ~4GB — saves disk)
└─ Profiles (per-user, fully separate):
    ├─ Dad/
    │   ├─ identity.json (Ed25519 keypair — Dad's keys)
    │   ├─ language_sovereignty: "ar"
    │   ├─ evidence_ledger/ (Dad's proof chain)
    │   └─ reflex_cache/ (Dad's compiled patterns)
    ├─ Mom/
    │   ├─ identity.json (Mom's keys)
    │   ├─ language_sovereignty: "ur"
    │   ├─ evidence_ledger/
    │   └─ reflex_cache/
    └─ Son/
        ├─ identity.json (Son's keys)
        ├─ language_sovereignty: "en"
        ├─ evidence_ledger/
        └─ reflex_cache/
```

### 15.2 Profile Switching

```
Terminal header: [Profile: Dad 👤] ▼
  ├─ Switch to Mom (requires Mom's passphrase)
  ├─ Switch to Son
  └─ Create New Profile
```

Each profile is a **separate sovereign node** with its own identity, wallets, SEED balance, and evidence chain. Only the LLM model file is shared (to save disk). Profile switching requires the target profile's passphrase — no profile can access another's data.

---

## 16. Installer Health Check

Before completing, the installer runs a constitutional health check:

```
✓ Core runtime executable
✓ LLM model loads successfully
✓ Ed25519 identity generated
✓ Evidence ledger initialized (block #0)
✓ 12 agents minted (7 PAT + 5 SAT)
✓ Constitutional heartbeat started
✓ First briefing generated by DEMA
✓ Terminal UI renders correctly
✓ Language packs loaded
✓ Disk space sufficient (500MB free after install)
```

If any check fails, the installer **rolls back completely** and shows an error report with troubleshooting guidance in the user's language. The user never finishes installation and sees a broken terminal.

---

## 17. Installer Audit Receipt

The installer generates a constitutional receipt of what it did:

```json
{
  "installer_version": "1.0.0",
  "install_date": "2026-03-08T14:30:00Z",
  "device_profile": {
    "os": "Windows 11", "arch": "x86_64",
    "ram_gb": 16, "gpu": "NVIDIA RTX 3060",
    "locale": "ar-AE"
  },
  "selected_tier": "Enhanced (Llama 3.1 8B Q4_K)",
  "user_overrode_recommendation": false,
  "installed_components": [
    "core_runtime", "llama_cpp_cuda", "locale_ar", "locale_en",
    "genesis_ceremony", "constitutional_heartbeat"
  ],
  "genesis_receipt": {
    "node_id": "0x4A2F...",
    "evidence_block_0": "0xA3B5...",
    "agents_minted": 12
  },
  "health_check_passed": true,
  "installation_time_seconds": 142
}
```

This receipt is stored in `bizra/installation_receipt.json` and displayed in Settings → About. It provides transparency for debugging, support, and sovereignty verification.

---

## 18. Build Matrix

### 18.1 Installer Build Pipeline

| Target | Build Tool | CI Job | Artifact |
|--------|-----------|--------|----------|
| Windows x64 | `cargo tauri build --target x86_64-pc-windows-msvc` | `build-installer-windows` | `bizra-setup.exe` |
| Windows ARM | `cargo tauri build --target aarch64-pc-windows-msvc` | `build-installer-windows-arm` | `bizra-setup-arm.exe` |
| macOS Universal | `cargo tauri build --target universal-apple-darwin` | `build-installer-macos` | `BIZRA.dmg` |
| Linux x64 | `cargo tauri build --target x86_64-unknown-linux-gnu` | `build-installer-linux` | `bizra.AppImage` |
| Linux ARM64 | `cargo tauri build --target aarch64-unknown-linux-gnu` | `build-installer-linux-arm` | `bizra-arm64.AppImage` |
| Android | `cargo tauri android build` | `build-installer-android` | `BIZRA.apk` |

### 18.2 i18n Build Pipeline

| Step | Tool | Output |
|------|------|--------|
| Extract strings | `react-intl extract` | `messages/en.json` (source strings) |
| Translate (Tier 1) | Human translators (community) | `locales/{lang}/terminal.json` |
| Translate (Tier 2) | LLM-assisted + human review | `locales/{lang}/terminal.json` |
| Compile | `react-intl compile` | Binary message bundles |
| Validate | CI gate: all Tier 1 strings translated | Blocks release if incomplete |
| RTL test | Visual regression test (Arabic, Hebrew, Urdu) | Screenshots compared |

---

## 19. Implementation Phases

### Prerequisite: Genesis-100 Gate (Must Pass Before Installer Work Begins)

The Genesis-100 gate (68 checks, 5 SAT agents) must pass Layers 1-3 before the installer is built for public distribution. The installer packages a system that has been constitutionally verified — not a system that "will be verified later."

| Gate Layer | Agent | Requirement |
|-----------|-------|-------------|
| L1: Structural | Sentinel | 0 CRITICAL security findings, 100% tests pass |
| L2: Constitutional | Oracle-S | Ihsān ≥ 0.95, heartbeat alive, simulation validates |
| L3: Economic | Ledger | BLOOM soulbound, Zakat exact, Gini enforced |

**Reference:** BIZRA-Definition-of-Done-Genesis-100.md, genesis_gate.py

### Phase I: Desktop Installer (Sprint 5, ~2 weeks)

| Task | Hours | Deliverable |
|------|-------|------------|
| Tauri project setup + 7-view wiring | 16 | Tauri shell with React terminal |
| Hardware detection module (Rust) | 12 | DeviceProfile struct + detection |
| LLM runtime integration (llama.cpp) | 16 | Model loading, inference, auto-tier |
| i18n framework (react-intl + RTL) | 12 | Arabic + English as Tier 1 MVP |
| Installer flow (6-step, 3-tap) | 8 | Self-extracting binary |
| Auto-update mechanism | 8 | Delta patches, sovereign choice |
| `bizra` CLI integration | 6 | One-command launch from terminal |
| `bizra` installer scripts (Linux + Windows + macOS) | 4 | Install → `bizra` available globally |
| **TOTAL** | **82** | **Desktop installer MVP + CLI** |

### Phase II: Language Expansion (Sprint 6, ~2 weeks)

| Task | Hours | Deliverable |
|------|-------|------------|
| Complete Tier 1 translations (10 languages) | 40 | Community translator program |
| RTL layout completion + testing | 16 | Arabic, Hebrew, Urdu, Persian |
| DEMA persona localization | 12 | Persona prompts in 10 languages |
| Cultural adaptation (date formats, numbers) | 8 | Locale-aware formatting |
| **TOTAL** | **76** | **10 languages, full RTL** |

### Phase III: Mobile (Sprint 7-8, ~4 weeks)

| Task | Hours | Deliverable |
|------|-------|------------|
| Android Tauri Mobile build | 24 | APK with hardware adaptation |
| Android model optimization | 16 | Mobile-optimized quantization |
| iOS Tauri Mobile build | 24 | IPA with Metal acceleration |
| App Store / Play Store submission | 8 | Distribution channels |
| **TOTAL** | **72** | **Mobile apps** |

---

## 20. Resource Dedication & URP Economy

### 20.1 The VM Analogy

When a user installs a virtual machine, the hypervisor asks: "How many CPU cores, how much RAM, how much disk do you want to dedicate?" The user decides. The VM gets exactly what was allocated. No more, no less.

BIZRA does the same — but instead of running a guest OS, the dedicated resources join the **Universal Resource Pool (URP)**. Every resource contributed strengthens the entire forest. And unlike a VM, **you earn SEED for what you share**.

### 20.2 Resource Types & Reward Rates

| Resource | Unit | Min Dedication | Reward Mechanism | Estimated SEED Rate |
|----------|------|---------------|-----------------|-------------------|
| **CPU** | Cores | 1 core | Compute-hours served to forest | 0.05 SEED/compute-hour |
| **RAM** | GB | 512 MB | Memory available for inference | 0.02 SEED/GB-hour |
| **Storage** | GB | 1 GB | Reflex cache + model hosting | 0.01 SEED/GB-month |
| **GPU** | GB VRAM | 1 GB | Inference acceleration for forest | 0.20 SEED/GPU-hour |
| **Network** | Mbps | 1 Mbps | Bandwidth for reflex propagation | 0.01 SEED/GB-transferred |
| **Witnessing** | Heartbeats | 1 per 5 min | Constitutional heartbeat validation | 0.01 SEED/witness |

**All rates are Proof-of-Impact verified.** The node doesn't earn SEED just for being online — it earns SEED for actual impact: compute served, inference completed, reflexes propagated, heartbeats witnessed. Quality-gated by Ihsān ≥ 0.85.

### 20.3 Dedication Tiers (Auto-Detected Defaults)

The installer auto-detects the device and suggests a dedication tier. The user can always adjust.

| Device Profile | Auto-Suggested Tier | What's Shared | What User Keeps |
|---------------|-------------------|--------------|----------------|
| **Old phone (1GB RAM)** | Witness | Heartbeat only | 100% of everything else |
| **Budget laptop (4GB)** | Light | 1 core, 512MB RAM | 3 cores, 3.5GB RAM |
| **Modern laptop (16GB)** | Standard | 2 cores, 4GB RAM, 10GB disk | 6 cores, 12GB RAM |
| **Workstation (64GB)** | Contributor | 4 cores, 16GB RAM, 2GB VRAM | Remaining resources |
| **Node0-class (128GB)** | Anchor | 8 cores, 32GB RAM, 8GB VRAM | Still has 128GB+ for local work |

**The installer RECOMMENDS keeping at least 50% for the user's own work.** But if the user explicitly chooses to dedicate more — even 100% — that is their sovereign right. DEMA will ask: "Are you sure? Your device will prioritize the forest over local work." If the user confirms, the system respects their choice.

### 20.4 Scheduling: When Resources Are Shared

| Schedule Option | Description | Best For |
|----------------|-------------|----------|
| **Always** | Resources shared 24/7 | Servers, always-on desktops |
| **When Idle** | Shared only when screen locked or no user activity for 5 min | Laptops, daily workers |
| **Scheduled** | Shared during specific hours (e.g., 10 PM - 6 AM) | Users who want predictable performance |
| **Manual** | Shared only when user explicitly activates | Maximum control |
| **Never** | No sharing — fully sovereign, no URP | Privacy-first users |

Default: **When Idle**. The user can change this in Settings (Terminal View C.7) at any time.

### 20.5 The Reverse Scaling Proof

This is where the MMORPG economics and the URP architecture converge:

```
1 node  = 1 device, 4GB RAM, local inference only
         → Response time: 1800ms (S2 deliberation)
         → Reflex library: personal only

100 nodes = 100 devices, 400GB distributed RAM
          → Cache hit rate rises (shared reflexes)
          → Response time: 800ms average
          → Reflex library: 100x larger

1M nodes = 1M devices, 4PB distributed memory
         → Cache hit rate: ~60%
         → Response time: 200ms average
         → Reflex library: covers most common tasks
         → Emission multiplier: 0.46 (system getting smarter)

1B nodes = 1B devices, 4EB distributed memory
         → Cache hit rate: ~90%
         → Response time: 50ms average (near-S1)
         → Reflex library: comprehensive
         → Emission multiplier: 0.19 (system is expert)
```

**The math:** More nodes → more shared reflexes → higher cache hit rate → faster for everyone → lower emission (system costs less as it gets smarter). The poor don't just benefit — their witnessing heartbeats validate the constitutional invariants that protect everyone.

### 20.6 How PoI Validates Resource Contributions

Resource contribution is not trusted — it is **verified by Proof-of-Impact**:

| Contribution | How It's Verified | What Prevents Gaming |
|-------------|-------------------|---------------------|
| CPU compute | Request hash + result hash + timing | Timeout verification — can't fake speed |
| RAM hosting | Random probe queries to cached data | Must actually serve correct data |
| Storage | Merkle proof of stored content | Can't claim storage without content |
| GPU inference | Output hash verified by 2+ nodes | Can't fake inference results |
| Witnessing | Signed heartbeat with Ed25519 | Can't fake identity (keypair) |

**The key insight:** Every contribution generates an **ActionReceipt** with BLAKE2b hash chain. The receipt is scored by Ihsān. Only high-quality contributions (Ihsān ≥ 0.85) earn SEED. This is why BIZRA uses Proof-of-Impact, not Proof-of-Stake — you earn by doing, not by having.

### 20.7 Resource Dedication UI (Settings View C.7)

After initial installation, the user manages their resource dedication through the Settings terminal view:

```
┌─────────────────────────────────────────────────┐
│  🌳 Forest Contribution                         │
│                                                  │
│  Status: SHARING (When Idle)    [Change ▼]      │
│  Since: March 8, 2026                            │
│  Total earned: 12.45 SEED                        │
│                                                  │
│  CPU  ████░░░░  2/8 cores    [Adjust ▼]         │
│  RAM  ████░░░░  4/16 GB      [Adjust ▼]         │
│  Disk ██░░░░░░  10/250 GB    [Adjust ▼]         │
│  GPU  ██░░░░░░  2/6 GB VRAM  [Adjust ▼]         │
│                                                  │
│  This month:                                     │
│    Compute served: 142 requests (0.71 SEED)     │
│    Reflexes hosted: 23 patterns (0.23 SEED)     │
│    Heartbeats witnessed: 4,320 (0.43 SEED)      │
│                                                  │
│  [Stop Sharing]  [View Receipts →]              │
└─────────────────────────────────────────────────┘
```

### 20.8 Constitutional Constraints on Resource Sharing

| Constraint | Rule | Source |
|-----------|------|--------|
| **Default: 50% of any resource** | Installer RECOMMENDS max 50% for user protection | Sovereignty first |
| **User can override to 100%** | If the user explicitly chooses, they can dedicate ALL resources | Sovereignty means the USER decides, not the system |
| **Override requires confirmation** | "Are you sure? Your device may slow down." Then respect the choice. | Informed consent, not paternalism |
| **User's active work is priority** | When user is actively working, URP yields gracefully | Progressive release |
| **Sharing is always opt-in** | Default: When Idle. Can be set to Never. | Resource sovereignty |
| **Earnings are 100% user's** | No platform cut on URP rewards | Law 6: Users keep 100% |
| **Quality-gated rewards** | Only Ihsān ≥ 0.85 contributions earn SEED | Constitutional gate |
| **Child safety** | If the device is marked as a child's, max 25% sharing | Daughter Test |

**Why not hard-cap at 50%?** Because sovereignty means the user decides. If a founder wants to dedicate their entire machine to the forest, if a researcher wants to contribute their full GPU to the pool during a training run, if a community leader wants their server to serve the network 24/7 — that is their sovereign choice. BIZRA recommends 50% to protect casual users. But it does not override the user's will.

> البذرة Rule 2: القلب يجب أن يكون ميزان العقل — The heart must be the scale of the mind. Trust the user's heart.

### 20.9 The Founding Contribution: Genesis-1

The first contribution to the URP is the founder's. This is not a special privilege — it is the same law that applies to everyone, applied first.

**Mumo's Genesis-1 Contribution:**

| Resource | What | Verified By |
|----------|------|------------|
| **Hardware** | NODE0 (MSI Titan 18 HX: i9-14900HX, 128GB DDR5, RTX 4090 16GB) — 100% dedicated | Hardware audit receipt |
| **Hardware** | Samsung Z Fold 6 (12GB, SD8G3) — 100% dedicated | Hardware audit receipt |
| **Data** | 3 years of R&D: ~150 original research docs, 1.3TB+ data | PAT indexes, SAT verifies |
| **Code** | BIZRA-DATA-LAKE: 880 Python files, 22 Rust crates, 560 test files | Git history + LOC audit |
| **Code** | All GitHub repos under BizraInfo | Git provenance chain |
| **Knowledge** | 15,000+ hours of accumulated chat history, research notes, design documents | PAT indexes + Ihsān scores each artifact |
| **Founding papers** | الرسالة + البذرة (Ramadan 2023) | Constitutional root — scored but never transferred |

**The Evaluation Process:**

1. **PAT-7 indexes every artifact** — every file, every commit, every document, every research note
2. **SAT-5 verifies each artifact** — authenticity (git blame), originality (dedup check), quality (Ihsān score)
3. **Fair market evaluation** — SAT compares the contribution to equivalent startup founder work in the market:
   - 15,000 hours × market rate for distributed systems architect
   - 290K+ LOC across Python + Rust + TypeScript
   - Novel research (MMORPG lineage, efficiency-based emission, Khaldunian throttle)
   - Constitutional framework (first implementation of Islamic finance in a sovereign AI OS)
4. **SEED minted proportional to verified impact** — same formula as any user, same Ihsān gate, same quality threshold
5. **50% of minted SEED → community pool** — the founder's sadaqah, as promised in البذرة p.19
6. **Remaining 50% = founder's sovereign assets** — earned fairly, same law as everyone else

**The principle:** The founder does not get special tokens. The founder does not get a pre-mine. The founder contributes verified work, the system evaluates it with the same algorithm it uses for everyone, and the founder keeps what the system says is fair — minus the 50% he promised to God.

**Quranic basis:**

> يَا أَيُّهَا الَّذِينَ آمَنُوا كُونُوا قَوَّامِينَ بِالْقِسْطِ شُهَدَاءَ لِلَّهِ وَلَوْ عَلَىٰ أَنفُسِكُمْ
> "O you who believe, be persistently standing firm in justice, witnesses for Allah, even if it be against yourselves."
> — An-Nisa 4:135

The founder is evaluated by the same law. The same Ihsān gate. The same SAT-5 agents. If any artifact scores below 0.85, it earns zero SEED — even if the founder created it. Justice applies to everyone. Especially the founder.

### 20.10 How Any User Can Make a Full Contribution

The Genesis-1 contribution is not unique to the founder. Any user can:

1. **Dedicate 100% of hardware** — their choice, their sovereignty
2. **Contribute data** — personal knowledge bases, research, documents
3. **Contribute skills** — compiled reflexes that help the forest
4. **Contribute code** — open-source tools, integrations, improvements
5. **Contribute compute** — GPU inference serving, model hosting

Every contribution is:
- **Indexed by PAT-7** (the user's personal agents analyze what was contributed)
- **Verified by SAT-5** (the system agents verify authenticity and quality)
- **Scored by Ihsān** (only quality work earns SEED)
- **Receipted on the evidence chain** (BLAKE2b hash, Ed25519 signature)
- **Rewarded proportionally** (more impact → more SEED, same formula for everyone)

**The data contribution is particularly important.** A user's 10 years of organized research notes are valuable to the forest. A teacher's curriculum is valuable. A farmer's agricultural knowledge is valuable. When this data is indexed and made available (with the user's consent, encrypted, sovereignty preserved), it enriches the collective intelligence. The user earns SEED for this impact.

**Privacy guarantee:** Contributed data is NEVER readable by other users. It is indexed for semantic search, but the content is encrypted with the contributor's Ed25519 key. Only the contributor can decrypt it. Other nodes can search for patterns ("find agricultural research about date palms") and get back anonymized insights — never the raw data.

### 20.9 The Islamic Foundation

**Quranic basis for resource sharing:**

> وَتَعَاوَنُوا عَلَى الْبِرِّ وَالتَّقْوَىٰ ۖ وَلَا تَعَاوَنُوا عَلَى الْإِثْمِ وَالْعُدْوَانِ
> "Cooperate in righteousness and piety, but do not cooperate in sin and aggression."
> — Al-Ma'idah 5:2

The URP is التعاون على البر — cooperation in righteousness. You share what you have. Others share what they have. The system rewards everyone fairly. Nobody is exploited. Nobody is forced. The strong help the weak. The weak strengthen the strong through witnessing.

This is the Islamic economic principle of التكافل (mutual solidarity) implemented as a distributed computing protocol.

---

## 21. The 8 Billion Test

The installer is ready when ALL of these pass:

- [ ] A grandmother in Cairo installs it on a Windows laptop in Arabic in under 3 minutes
- [ ] A student in Lagos installs it on an Android phone in English in under 3 minutes
- [ ] A developer in São Paulo installs it on Linux in Portuguese in under 3 minutes
- [ ] A farmer in Java installs it offline from USB on a low-end Android in Indonesian
- [ ] A shopkeeper in Karachi installs it in Urdu with RTL layout correctly mirrored
- [ ] A blind user installs it using screen reader without sighted assistance
- [ ] A user with 1GB RAM gets a micro-node (TinyLlama 1.1B, witness + heartbeat)
- [ ] A user with 128GB RAM gets a premium BIZRA with the same 3-tap install
- [ ] The installer works with no internet connection
- [ ] The installer never asks for admin/root access
- [ ] After install, the user types `bizra` in any terminal and the system launches
- [ ] A user who shares 2 CPU cores earns their first SEED within 24 hours of enabling URP
- [ ] A user who chooses "Never Share" loses zero functionality — sovereignty is complete without URP
- [ ] A user who explicitly chooses to dedicate 100% of resources is allowed — sovereignty means user decides
- [ ] The Genesis-100 gate (68 checks, 5 SAT agents) has passed before public distribution

When all 15 pass, BIZRA is ready for 8 billion humans.

---

## 22. Document Dependencies

This installer spec does not stand alone. It depends on and is constrained by:

| Document | Version | What It Provides |
|----------|---------|-----------------|
| **البذرة + الرسالة** | Ramadan 2023 | 7 foundational rules → 7 layers of sovereign stack |
| **Constitutional Sources** | v1.0 | Quran/Hadith backing for every design law |
| **Definition of Done** | v1.0 | 68 checks, 5 SAT agents — must pass before installer ships |
| **Genesis Gate Runner** | v1.0 | Executable verification (genesis_gate.py) |
| **BIZRA CLI** | v1.0 | bizra-cli.py — the `bizra` command |
| **Terminal Build Contract** | v1.0 | 7 views, 49 acceptance criteria |
| **Identity Canon** | v1.0 | What BIZRA IS |
| **Proof Canon** | v1.1 | What BIZRA HAS |

---

## 23. The Line

> **"Every human is a node. Every node is a seed. Every seed has infinite potential."**

This is not possible if the seed can only be planted on expensive soil, watered with fast internet, and spoken to in English.

The universal installer is the bridge between البذرة's promise and every human on Earth. It is the Mother Test applied to the entire product.

Build it right. Build it for everyone. Build it for أمك.

> كل بذرة تحمل في داخلها مخطط غابة بأكملها

The founding papers — الرسالة and البذرة — were written during Ramadan 2023 from a place of deep personal pain and prayer. Every principle in this installer spec traces back to those two documents. Every design law traces through them to the Quran and Sunnah. The code serves the papers. The papers serve the revelation.

> **"أنا دائما أطلب المستحيل من الله ربي لا يعرف المستحيل"**
> *"I always ask for the impossible from God. My Lord does not know the impossible."*
> — البذرة, last line

**LOCKED: v2.0 · 2026-03-08 · Dubai · BIZRA Foundation**
