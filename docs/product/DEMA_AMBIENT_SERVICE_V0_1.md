# Dema Ambient Service v0.1

**Date:** 2026-04-27 GST
**Status:** PLANNED → first implementation slice
**Scope:** Phase A0.5 of the Dema GTM Masterplan v0.1.
**Truth label:** MEASURED at the file level (every artifact emitted is a real
  receipt over real local state); DERIVED at the product-promise level until
  the operator runs the systemd unit on their own hardware.

---

## §1 Purpose

Phase A0 (Ambient Kernel) ships local presence on Node0. Phase A0.5 turns
that presence into the **operator's always-on service**: Dema wakes with the
device, ticks safely, holds a single-instance lock, and shuts down cleanly
on signal.

This phase ships **no continuous network listener, no desktop control, no
autonomous social posting, no MEMORY.md edits.** The service is a local
heartbeat with operator-supplied install hooks (systemd / Task Scheduler).

---

## §2 What's added

### `scripts/dema/dema_daemon.py` — extended

| New flag | Purpose |
|---|---|
| `--loop` | Continuous tick mode (mutually exclusive with `--once`). |
| `--interval-seconds N` | Seconds between ticks (default 60). |
| `--max-ticks N` | Stop after N ticks (default: unbounded). |
| `--max-seconds N` | Hard ceiling on a single loop run (default 24h). |

Plus:

- **PID file + POSIX advisory lock** at
  `sovereign_state/dema/runtime/dema_daemon.pid`. `_acquire_lock()`
  atomically refuses to start if another daemon is running.
- **Graceful shutdown** on `SIGINT` / `SIGTERM` via small-step
  interruptible sleep.
- **Clean lock release** in `finally` for normal shutdown; stale/corrupt
  locks are reclaimed on the next start only when their PID is not alive.

### `scripts/dema/dema_service.py` — new

```
status                       JSON: running?, pid, profile, last tick.
start-once                   Run one tick (delegates to daemon.tick).
print-systemd-user-unit      Emits a Linux systemd --user unit file.
print-windows-task-command   Emits a Task Scheduler command (placeholder).
doctor                       Sanity check; non-zero exit on findings.
```

### `tests/scripts/test_dema_ambient_service.py` — new

13 contract tests covering: `--loop --max-ticks 2` exits safely, lock-file
lifecycle (acquire / refuse-second / release), status JSON shape, systemd
unit text contains no public network exposure marker, tick receipts created
under sandbox root, no committed state files produced, doctor verdict on
healthy + degraded states.

---

## §3 Storage layout (additions)

```
sovereign_state/dema/
  └── runtime/
       └── dema_daemon.pid       single-instance lock; cleaned on shutdown
                               or reclaimed on next start if stale
```

`runtime/` lives under `sovereign_state/` which is gitignored; nothing the
service writes ends up in committed artifacts.

---

## §4 Linux installation (systemd --user)

```bash
# 1. Generate the unit text.
.venv/bin/python scripts/dema/dema_service.py print-systemd-user-unit > dema.service

# 2. Install + enable + start.
mkdir -p ~/.config/systemd/user
cp dema.service ~/.config/systemd/user/dema.service
systemctl --user daemon-reload
systemctl --user enable --now dema.service

# 3. Watch it tick.
journalctl --user -u dema.service -f
```

The unit declaration includes `PrivateNetwork=true` (no public network),
`NoNewPrivileges=true`, `ProtectSystem=strict`, `ProtectHome=read-only`,
and a single `ReadWritePaths={root}` entry that confines the daemon's
writes to the operator's Dema state root.

### WSL caveat

`systemd --user` works on WSL2 with `systemd=true` in `/etc/wsl.conf` and
`wsl --shutdown` after editing. If `systemctl --user` fails, fall back to
running the daemon under `tmux`/`nohup` until WSL systemd is active. The
WSL fallback path:

```bash
nohup .venv/bin/python scripts/dema/dema_daemon.py --loop --interval-seconds 60 \
  > sovereign_state/dema/runtime/dema_daemon.out 2>&1 &
```

---

## §5 Windows Task Scheduler (placeholder)

```bash
.venv/bin/python scripts/dema/dema_service.py print-windows-task-command
```

Emits a `schtasks /Create /SC ONLOGON ...` command. **Native Windows
support is a placeholder in v0.1.** The full Windows path (signed
binary, headless service, AHK action mesh) lands in Phase A7.

---

## §6 Disable / uninstall

```bash
systemctl --user disable --now dema.service
rm ~/.config/systemd/user/dema.service
systemctl --user daemon-reload

# Clear any stale lock file (only if nothing is running):
rm -f sovereign_state/dema/runtime/dema_daemon.pid
```

---

## §7 Logs and receipts

Every tick still writes to:

- `sovereign_state/dema/logs/<YYYY-MM-DD>.jsonl` (DailyLog)
- `sovereign_state/dema/receipts/<YYYY-MM-DD>/<rid>.json` (DemaReceipt)

The service does not introduce a new log surface; it just keeps the kernel
ticking on a schedule.

---

## §8 Safety contract (asserted by tests)

The Ambient Service inherits the kernel's non-claims and adds:

- **Single instance only** — `_acquire_lock` refuses to start a second
  daemon while a live PID is in `dema_daemon.pid`.
- **Graceful shutdown** — `SIGINT` / `SIGTERM` flips a stop flag; the
  current tick finishes, then the lock is released.
- **No public network listener** — the systemd unit declares
  `PrivateNetwork=true`. The daemon does not bind any port.
- **No desktop control** — no key/mouse simulation, no app launching.
- **No autonomous social** — `social` remains in `not_touched_paths`.
- **Sandbox-bound writes** — `ReadWritePaths={root}` in the systemd
  unit and a `lock_path.is_relative_to(root)` check in the doctor.

---

## §9 What's NOT shipped in v0.1

- Native Windows daemon (placeholder only)
- Continuous browser/desktop control (Phase A7)
- Auto-restart heuristics beyond systemd `Restart=on-failure`
- Mother-language UI in service output (status/doctor are JSON-only)
- Long-term memory promotion from dream phase (Phase A6)
- Network-exposed control endpoints (off the roadmap by design)

---

## §10 Bounds

This service carries **no AGI guarantee**, **no token-value claim**, and
**no public claim** of any kind. Every tick is local-only,
truth-labeled, receipt-linked, and bound by the kernel's non-claims plus
the additions in §8.

If any clause here conflicts with the BIZRA Topology Canon (2026-03-25),
the Origin Manifest, or the Brand Canon v0.2, those canonical sources win
and this doc must be amended.
