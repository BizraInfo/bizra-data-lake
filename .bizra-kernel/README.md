# BIZRA Context Kernel v1.1

Persistent context configuration for Claude sessions working with the BIZRA ecosystem.

**Location:** `C:\award-winner-design\.bizra-kernel\`

## Quick Start (High-SNR)

```bash
# Option 1: Single command load (Claude)
@load-kernel

# Option 2: CLI (local)
python load-kernel.py --mode snr

# Option 3: Windows shortcut
init-kernel.bat  # copies high-SNR prompt to clipboard
```

## File Structure

```
.bizra-kernel/
|- README.md            # This file
|- core-context.json    # Main kernel configuration (load first)
|- hooks.yaml           # Pre-configured workflow hooks
|- memory.json          # Session-persistent memory
|- tools.yaml           # Tool configurations & protocols
|- standards.yaml       # Quality gates & Ihsan thresholds
```

## Update Protocol

- Frequency: After major milestones or architectural changes
- Method: Manual edit or via `/kernel-update` command
- Backup: Git-tracked, versioned

## Version History

- v1.1 (2025-12-02): High-SNR loader scripts; corrected path to `.bizra-kernel`
- v1.0 (2025-11-18): Initial kernel creation
