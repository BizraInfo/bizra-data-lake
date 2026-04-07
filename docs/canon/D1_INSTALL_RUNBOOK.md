# D1 Install Runbook — Native Ubuntu 24.04 LTS on MSI Titan
**Spearpoint:** BIZRA-STS-001 | **Frozen:** 2026-04-07 | **Status:** PAYLOAD VERIFIED

---

## Hardware Target

| Component | Value |
|-----------|-------|
| Machine | MSI Titan (i9-14900HX, 32 cores, 128 GB RAM) |
| GPU | NVIDIA RTX 4090 (16 GB VRAM) |
| Storage Controller | Intel VMD (Volume Management Device) |
| Existing Disks | 2x NVMe in RAID 0 (Windows 11 — DO NOT TOUCH) |
| Target Disk | Samsung 990 PRO 2TB — **NVMe slot 3 (empty)** |
| Installer | BIZRA Sovereign USB (Ubuntu 24.04 LTS + autoinstall) |

---

## VMD Preflight (MANDATORY — Execute Before ANY Install)

Intel VMD remaps NVMe devices behind a virtual controller. Ubuntu kernels before 6.8
may not see VMD-managed drives. Ubuntu 24.04 LTS ships kernel 6.8 (HWE 6.11+), which
includes the `vmd` driver. However, BIOS configuration determines visibility.

### Step 0: BIOS Preparation

1. Power on → Press **DEL** to enter BIOS (MSI Click BIOS)
2. Navigate: **Advanced** → **Integrated Peripherals** → **VMD Configuration**
3. **Document current settings** (photograph the screen)
4. **CRITICAL FINDING (researched 2026-04-07):** MSI Titan BIOS does **NOT** expose
   per-slot VMD control. VMD is **all-or-nothing** on this board. Options:
   - **Option A (Recommended for dual-boot):** **Leave VMD enabled**. Ubuntu 24.04
     kernel 6.8+ has `vmd` driver (`CONFIG_VMD=m`). Add `vmd.enable=1` to kernel
     params if drives aren't visible. The **installer ISO initramfs** may lack
     `vmd.ko` — if drives don't appear, press `e` at GRUB and append `vmd.enable=1`.
   - **Option B (Simplest for Linux-only):** **Disable VMD entirely**. NVMe drives
     appear as standard devices. **WARNING:** This will break the Windows RAID 0
     array if Windows was installed with VMD/RST enabled.
   - **Navigation:** DEL at POST → Advanced → Intel Advanced Menu → VMD Setup Menu →
     Enable VMD Controller: [Disabled]. Alt path: Advanced → Integrated Peripherals
     → VMD Configuration. Some firmware revisions: Settings → Advanced → PCI
     Subsystem Settings.
5. Ensure **Secure Boot** is enabled (Ubuntu 24.04 supports it)
6. Save and exit BIOS

### Step 1: Boot Live USB (NO DISK WRITES)

1. Insert BIZRA Sovereign USB
2. Boot from USB (F11 boot menu on MSI)
3. Select **"Try Ubuntu"** — NOT "Install Ubuntu"
4. Open terminal in live session

### Step 2: VMD Verification Commands

Run ALL of these. Screenshot or log the output.

```bash
# 1. Check if VMD driver is loaded
lspci | grep -i vmd
dmesg | grep -i vmd

# 2. List all block devices — confirm NVMe appears
lsblk
fdisk -l

# 3. Check NVMe specifically
ls /dev/nvme*
nvme list    # (if nvme-cli available)

# 4. Confirm drive size matches Samsung 990 PRO 2TB (~1.86 TiB)
# Look for a ~2TB device that is NOT the RAID array
```

### Step 3: Decision Gate

| Outcome | Action |
|---------|--------|
| NVMe visible at `/dev/nvmeXnY` with correct size | Proceed to Step 4 |
| NVMe NOT visible, VMD enabled | Return to BIOS, try Option A or B |
| NVMe visible but wrong size | STOP — you may be looking at the RAID array |
| No NVMe devices at all | VMD blocking everything — try disabling VMD for slot 3 |

**STOP if uncertain. Never proceed with install if the target disk is ambiguous.**

### Step 4: Fix Autoinstall Storage Matcher

The shipped autoinstall (`deploy/autoinstall/user-data`) uses `match: size: smallest`.
This is DANGEROUS on a multi-disk system. Replace with the exact device path.

```bash
# In the autoinstall user-data, replace:
#   match:
#     size: smallest
# With:
#   match:
#     path: /dev/nvmeXnY    # <-- exact path from Step 2
```

The corrected storage block should look like:

```yaml
storage:
  layout:
    name: lvm
    match:
      path: /dev/nvme0n1    # REPLACE with actual path from lsblk
```

### Step 5: Proceed with Install

Only after ALL of the following are TRUE:

- [ ] NVMe physically installed in slot 3
- [ ] Live USB booted successfully
- [ ] `lsblk` shows the NVMe with correct ~2TB size
- [ ] Device path confirmed and noted (e.g., `/dev/nvme0n1`)
- [ ] autoinstall `user-data` patched with exact device path
- [ ] Windows RAID array (`/dev/mdX` or RAID NVMe) is **not** the target
- [ ] BIOS screenshot saved as evidence

Then proceed:

```bash
# Reboot into installer mode
sudo reboot
# Select "Install Ubuntu" from GRUB
# Autoinstall will use the patched user-data
```

### Step 6: Post-Install Verification

```bash
# After first boot into native Ubuntu:
neofetch                           # System summary
nvidia-smi                         # GPU detection
lscpu                              # CPU topology (32 cores)
free -h                            # RAM (128 GB)
df -h                              # Disk layout
cat /proc/cmdline                  # Kernel boot params
uname -r                           # Kernel version (should be 6.8+)
systemctl status docker            # Docker autostart
```

Save all output to `evidence/d1_install/post_install_verification.txt`.

---

## Known Issues

### VMD + Ubuntu
- Ubuntu 24.04 kernel 6.8 includes the `vmd` module — loads automatically when VMD
  hardware is detected
- If the kernel does NOT detect VMD drives, try adding `vmd.enable=1` to kernel params
  via GRUB at boot (press `e` at GRUB menu, append to `linux` line)
- HWE kernel 6.11 (available via `linux-generic-hwe-24.04`) has improved VMD support

### MSI BIOS Specifics
- MSI Click BIOS 5: VMD settings under Advanced → Integrated Peripherals
- Some MSI firmwares hide VMD under Advanced → Intel Advanced Menu → VMD Setup Menu
- If "VMD for Direct Assign" appears, set to **Disabled** for the target slot
- RAID mode (RST/IRST) and VMD are coupled — changing VMD may affect RAID visibility

### Dual-Boot GRUB
- Ubuntu installer should detect Windows Boot Manager automatically
- If not: `sudo update-grub` after install
- GRUB timeout: 5 seconds (configurable in `/etc/default/grub`)
- Default boot: set to Ubuntu (change via `GRUB_DEFAULT=`)

---

## Rollback Plan

If install fails or corrupts the system:

1. The Windows RAID 0 array on slots 1-2 is **untouched** (separate physical disks)
2. Remove the Samsung 990 PRO from slot 3
3. Boot normally — Windows RAID 0 works as before
4. The USB installer is reusable — no state was modified on existing disks

---

## Evidence Requirements

After successful install, commit the following:

```
evidence/d1_install/
├── bios_vmd_screenshot.png       # BIOS VMD settings (before)
├── lsblk_live_session.txt        # Live USB disk detection
├── post_install_verification.txt # neofetch + nvidia-smi + lscpu + df
├── dmesg_vmd.txt                 # VMD driver messages
└── install_receipt.json          # Timestamp, device path, kernel version
```

---

## Decision: Path A vs Path B

| | Path A: Dedicated NVMe | Path B: Full Wipe |
|---|---|---|
| Risk | Zero (separate disk) | High (RAID rebuild) |
| Cost | ~AED 650 (Samsung 990 PRO 2TB) | Zero hardware cost |
| Timeline | Same day (noon delivery) | Multi-day backup + migrate |
| Windows | Preserved (dual-boot) | Gone |
| Recommendation | **DO THIS FIRST** | Revisit after Path A proves Linux works |

**Decision recorded:** Path A. Order NVMe. Path B deferred to Day 10+.

---

*Frozen by: Claude Opus 4.6 | Chain: BIZRA-STS-001 Day 4 | Authority: §7 Closure-First*
