# BIZRA Migration Cheatsheet — Paste into Claude Mobile

Copy everything below into a new Claude chat on your phone:

---

I am installing Ubuntu 24.04 to replace Windows on my dev machine. Help me verify my partition layout. Here is my context:

## Machine
- MSI laptop, Intel i9-14900HX, 128GB RAM, RTX 4090
- Two NVMe drives: nvme0n1 (2.05 TB) + nvme1n1 (2.05 TB) — NO RAID, separate drives
- RAID was not assembled by Ubuntu installer — using drives independently (safer)

## What I'm doing
- Full replacement: manual Ubuntu install to nvme0n1 only
- nvme1n1 left untouched — will be /data after first boot
- Booted from USB (Ubuntu 24.04 live)
- Using "Something Else" (manual partitioning)

## Target partition layout on /dev/nvme0n1 ONLY
1. 1 GB — FAT32 — /boot/efi (EFI System Partition)
2. 2 GB — ext4 — /boot
3. 32 GB — swap
4. 250 GB — ext4 — / (root)
5. Rest (~1.76 TB) — ext4 — /home

## /dev/nvme1n1: DO NOT TOUCH during install. Format as /data after first boot.

## Bootloader target: /dev/nvme0n1

## Safety checks (all must be true)
1. Target disk is /dev/nvme0n1
2. /dev/nvme1n1 is NOT being partitioned or formatted
3. sda (USB, 123.99 GB VFAT) is NOT selected
4. Bootloader points to /dev/nvme0n1

## Backups (all safe, off internal disk)
- Code: 3 repos on GitHub (HEAD c8993393)
- Secrets: D: USB drive
- 04_GOLD data: D: USB drive (2.4GB)
- Bootstrap script: github.com/BizraInfo/bizra-data-lake/blob/main/deploy/post-install-bootstrap.sh

## After install, first login run:
```
curl -fsSL https://raw.githubusercontent.com/BizraInfo/bizra-data-lake/main/deploy/post-install-bootstrap.sh | bash
```

I will send you photos of the installer partition screen. Tell me GO or STOP based on the safety checks above.

---
