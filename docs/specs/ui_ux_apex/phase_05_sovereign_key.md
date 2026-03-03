# UI/UX APEX — Phase 05: Sovereign Key — Biometric Ed25519 Haptic/Visual UX

> Physical hardware key presence triggers cryptographically-unique UI awakening.
> Sprint priority: 5 (hardware dependency — requires physical Sovereign Key device).

> Standing on Giants: Diffie & Hellman (public-key cryptography, 1976) ·
> Bernstein (Ed25519, 2011) · Norman (mapping + feedback, 1988) ·
> Al-Ghazali (personal covenant, 1095)
> Repo anchors: `bizra-omega/fate-binding/` (Ed25519 + Dilithium),
>               `core/pci/` (Proof-Carrying Inference)

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| SK-01 | UI detects Sovereign Key insertion via OS USB HID event |
| SK-02 | On detection: play "awakening" animation — Genesis Seal bloom from key icon |
| SK-03 | Genesis Seal is cryptographically derived from the key's Ed25519 public key (visual hash) |
| SK-04 | Each key produces a unique, deterministic Genesis Seal (no two seals are identical) |
| SK-05 | BIZRA header displays "Key Active" badge with truncated key fingerprint (first 12 hex chars) |
| SK-06 | On key removal: UI shows "Sovereign Standby" — gentle fade, reduced capability mode |
| SK-07 | Hardware-signed actions show a "Key-Verified" badge in Iḥsān Gauge receipts |
| SK-08 | Key presence required for Elder-tier actions (Guardian Council, federation mutations) |
| SK-09 | If key not present and Elder action attempted: UI shows "Sovereign Key Required" blocker |
| SK-10 | Key detection gracefully degrades if no hardware key is registered (soft-key fallback) |

---

## 2. Edge Cases & Constraints

```
EDGE CASE: Key inserted while BIZRA not focused → detect on next focus gain
EDGE CASE: Multiple USB HIDs present → only respond to devices matching BIZRA vendor ID
           (vendor ID from config, never hardcoded)
EDGE CASE: Key fingerprint collision (astronomically unlikely with Ed25519) →
           log warning, treat as separate key
EDGE CASE: Key removed mid-action → action proceeds if already past Ihsān gate;
           future actions enter soft-key mode
EDGE CASE: Key present but OS driver error → show "Key Error" badge, log diagnostics
EDGE CASE: No Sovereign Key registered → show "Soft Key Mode" (passphrase-only);
           all features available except Elder-tier hardware-gated actions
CONSTRAINT: Ed25519 public key NEVER sent to external services; stays on-device
CONSTRAINT: Genesis Seal is a deterministic visual hash (SVG path generated from pubkey bytes)
            — NOT a cryptographic signature. Purely aesthetic, no security claim.
CONSTRAINT: Key fingerprint in UI is display-only (first 12 hex chars + ellipsis)
CONSTRAINT: Hardware key vendor ID and product ID come from config, not source code
```

---

## 3. Data Model

```typescript
// Sovereign Key — client types

interface SovereignKeyInfo {
  status:          "active" | "standby" | "error" | "soft_key" | "not_registered";
  fingerprint:     string;        // first 12 hex chars of SHA-256(pubkey)
  fingerprint_full: string;       // full hex, stored locally only
  genesis_seal:    string;        // SVG path data, derived from pubkey
  vendor_name:     string;        // e.g. "BIZRA Hardware"
  inserted_at:     number | null; // unix ms
}

interface GenesisSeal {
  svg_paths:       string[];   // deterministic from pubkey bytes
  primary_color:   string;     // Genesis Gold (#C9A962) always
  accent_color:    string;     // derived from pubkey byte 32 → hue rotation
  rotation_deg:    number;     // derived from pubkey byte 33
}
```

---

## 4. Pseudocode

### 4.1 SovereignKeyMonitor (background daemon)

```
MODULE SovereignKeyMonitor:
  // Platform-native HID monitoring (OS-level, not in browser).
  // On Windows: uses WMI USBDeviceAdded event or SetupDiGetClassDevs polling.
  // Emits events via IPC to BIZRA UI process.

  CONFIG (from config file, NOT hardcoded):
    BIZRA_VENDOR_ID   // HID vendor ID
    BIZRA_PRODUCT_ID  // HID product ID

  FUNCTION start():
    register_hid_listener(vendor=BIZRA_VENDOR_ID, product=BIZRA_PRODUCT_ID)

  ON hid_device_connected(device):
    IF device.vendor_id != BIZRA_VENDOR_ID:
      RETURN
    pubkey_bytes = read_pubkey_from_device(device)
    IF pubkey_bytes == null:
      ipc_emit("sovereign_key_event", {status:"error", reason:"pubkey_read_failed"})
      RETURN
    fingerprint = sha256_hex(pubkey_bytes)
    genesis_seal = derive_genesis_seal(pubkey_bytes)
    ipc_emit("sovereign_key_event", {
      status:       "active",
      fingerprint:  fingerprint,
      genesis_seal: genesis_seal,
      vendor_name:  device.product_string,
      inserted_at:  now_ms(),
    })

  ON hid_device_disconnected(device):
    IF device.vendor_id != BIZRA_VENDOR_ID:
      RETURN
    ipc_emit("sovereign_key_event", {status:"standby"})

  FUNCTION derive_genesis_seal(pubkey_bytes: bytes[32]) -> GenesisSeal:
    // Deterministic SVG path generation — purely aesthetic, no crypto claim
    // Map pubkey bytes to SVG Sacred Geometry path parameters:

    // Seed of Life: center + 6 petals; params derived from pubkey
    petal_radii   = [lerp(30, 55, pubkey_bytes[i] / 255) for i in 0..5]
    petal_angles  = [i * 60 + (pubkey_bytes[6+i] / 255 * 12 - 6) for i in 0..5]
    accent_hue    = int(pubkey_bytes[32-1] / 255 * 360)  // hue rotation
    rotation_deg  = (pubkey_bytes[32-2] / 255) * 45

    svg_paths = generate_seed_of_life_svg(petal_radii, petal_angles)
    accent_color = hsl_to_hex(accent_hue, saturation=0.6, lightness=0.55)

    RETURN GenesisSeal(
      svg_paths    = svg_paths,
      primary_color = "#C9A962",
      accent_color  = accent_color,
      rotation_deg  = rotation_deg,
    )
```

### 4.2 SovereignKeyUI (BIZRA UI process)

```
MODULE SovereignKeyUI:

  STATE:
    key_info: SovereignKeyInfo = {status: "soft_key"}

  FUNCTION init():
    ipc.on("sovereign_key_event", on_key_event)
    render_key_badge(key_info)

  FUNCTION on_key_event(event):
    prev_status = key_info.status
    key_info = build_key_info(event)

    MATCH event.status:
      "active":
        IF prev_status != "active":
          play_awakening_animation(key_info.genesis_seal)
        render_key_badge_active(key_info)

      "standby":
        play_standby_animation()
        render_key_badge_standby()

      "error":
        render_key_badge_error(event.reason)

  FUNCTION play_awakening_animation(seal: GenesisSeal):
    // Full header region animation — 2 second bloom
    animation_sequence = [
      // Step 1 (0–300ms): Key icon pulses gold
      { target: key_icon, effect: "pulse", color: "#C9A962", duration: 300 },
      // Step 2 (300–1200ms): Genesis Seal blooms from center
      { target: genesis_seal_svg, effect: "scale_bloom",
        from_scale: 0, to_scale: 1, easing: "ease-out-elastic", duration: 900 },
      // Step 3 (1200–1800ms): Seal rotates to final orientation
      { target: genesis_seal_svg, effect: "rotate",
        deg: seal.rotation_deg, duration: 600, easing: "ease-out" },
      // Step 4 (1800–2000ms): Fingerprint text fades in
      { target: fingerprint_badge, effect: "fade_in", duration: 200 },
    ]
    execute_animation_sequence(animation_sequence)
    IF NOT reduced_motion:
      play_ambient_chime()  // optional, user-configurable

  FUNCTION render_key_badge_active(key_info):
    badge = KeyBadge(
      icon          = genesis_seal_svg(key_info.genesis_seal),
      label         = "Key Active",
      sublabel      = key_info.fingerprint[:12] + "…",
      border_color  = "#C9A962",
      glow          = True,
      tooltip       = "Sovereign Key verified · " + key_info.vendor_name,
    )
    header.render_badge("sovereign_key", badge)

  FUNCTION render_key_badge_standby():
    badge = KeyBadge(
      icon         = key_icon_svg(muted=True),
      label        = "Sovereign Standby",
      border_color = "#3a3730",
      glow         = False,
    )
    header.render_badge("sovereign_key", badge)
    fade_to_soft_key_mode()

  FUNCTION gate_elder_action(action_name) -> bool:
    IF key_info.status == "active":
      RETURN True
    show_key_required_blocker(action_name)
    RETURN False

  FUNCTION show_key_required_blocker(action_name):
    modal = Modal(
      title   = "Sovereign Key Required",
      message = f"'{action_name}' requires your physical Sovereign Key.\n"
                "Insert your key to proceed.",
      icon    = key_icon_svg(alert=True),
      style   = { border: "#C9A962", background: "#0a1220" },
    )
    modal.render()
```

### 4.3 Key-Verified Badge in Veto Receipts

```
// Extension to phase_03_ihsan_gauge.md VetoReceiptPanel:

// When VetoReceipt.was_key_signed == True, add to receipt:
  draw_badge(
    "KEY-VERIFIED",
    icon  = genesis_seal_mini_svg(key_info.genesis_seal),
    color = "#C9A962",
    tooltip = "This action was signed by your Sovereign Key",
  )
```

---

## 5. TDD Anchors

```python
# tests/ui_ux_apex/test_sovereign_key.py

class TestSovereignKeyMonitor:
    def test_non_bizra_hid_ignored(self, monitor, mock_hid):
        """HID devices with wrong vendor ID do not trigger key events."""
        mock_hid.connect(vendor_id=0x9999, product_id=0x0001)
        assert monitor.ipc_events == []

    def test_pubkey_read_failure_emits_error(self, monitor, mock_hid):
        mock_hid.connect(vendor_id=BIZRA_VENDOR_ID, pubkey=None)
        event = monitor.ipc_events[-1]
        assert event["status"] == "error"
        assert "pubkey_read_failed" in event["reason"]

    def test_genesis_seal_is_deterministic(self, monitor):
        """Same pubkey always produces same Genesis Seal."""
        pubkey = bytes(range(32))
        seal1 = monitor.derive_genesis_seal(pubkey)
        seal2 = monitor.derive_genesis_seal(pubkey)
        assert seal1.svg_paths == seal2.svg_paths
        assert seal1.accent_color == seal2.accent_color

    def test_genesis_seal_differs_between_keys(self, monitor):
        """Different pubkeys produce different seals."""
        seal1 = monitor.derive_genesis_seal(bytes(range(32)))
        seal2 = monitor.derive_genesis_seal(bytes(reversed(range(32))))
        assert seal1.svg_paths != seal2.svg_paths

    def test_vendor_id_from_config_not_hardcoded(self, monitor_source):
        """Source code contains no literal HID vendor ID."""
        import re
        assert not re.search(r"0x[0-9A-Fa-f]{4}", monitor_source)

class TestSovereignKeyUI:
    def test_elder_action_gated_without_key(self, ui):
        """Elder action blocked when key status is 'soft_key'."""
        ui.key_info.status = "soft_key"
        result = ui.gate_elder_action("guardian_council_vote")
        assert result is False
        assert ui.last_modal_title == "Sovereign Key Required"

    def test_elder_action_passes_with_key(self, ui):
        ui.key_info.status = "active"
        result = ui.gate_elder_action("guardian_council_vote")
        assert result is True

    def test_awakening_animation_not_repeated_on_same_key(self, ui, mock_animator):
        """If key was already active, no repeated bloom animation."""
        ui.key_info.status = "active"
        ui.on_key_event({"status": "active", "fingerprint": "abc123"})
        assert mock_animator.play_awakening_animation.call_count == 0
```
