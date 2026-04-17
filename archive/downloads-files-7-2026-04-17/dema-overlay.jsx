import { useState, useEffect, useCallback, useRef } from "react";

// ─── DEMA v0 — Constitutional Desktop Overlay ───
// بسم الله الرحمن الرحيم
// "If it can't serve one, it has no right to claim eight billion."

const GOLD = "#C9A962";
const GOLD_DIM = "rgba(201,169,98,0.15)";
const GOLD_GLOW = "rgba(201,169,98,0.4)";
const VOID = "#020408";
const VOID_PANEL = "rgba(2,4,8,0.92)";
const VOID_GLASS = "rgba(2,4,8,0.78)";
const SURFACE = "rgba(255,255,255,0.04)";
const TEXT_PRIMARY = "rgba(255,255,255,0.92)";
const TEXT_SECONDARY = "rgba(255,255,255,0.5)";
const TEXT_DIM = "rgba(255,255,255,0.25)";
const GREEN = "#2ECC71";
const RED = "#E74C3C";
const AMBER = "#F39C12";

// Mock data — in production, these come from Rust governance layer via Tauri commands
const MOCK_RECEIPTS = [
  { id: "r-001", action: "Organized 47 files from Downloads → Projects", ts: "09:14", status: "verified", hash: "a3f8e2...", governance: "PERMITTED", reflexTime: "1.21ms" },
  { id: "r-002", action: "Triaged 12 emails across 3 accounts", ts: "09:22", status: "verified", hash: "7b2c91...", governance: "PERMITTED", reflexTime: "0.89ms" },
  { id: "r-003", action: "Generated morning brief from calendar + tasks", ts: "09:30", status: "verified", hash: "e5d1f4...", governance: "PERMITTED", reflexTime: "2.14ms" },
  { id: "r-004", action: "Blocked: attempted access to /etc/shadow", ts: "09:31", status: "denied", hash: "—", governance: "DENIED (membrane)", reflexTime: "0.003ms" },
  { id: "r-005", action: "Compiled reflex: daily backup → B:\\SOVEREIGN", ts: "09:45", status: "verified", hash: "c8a3b7...", governance: "PERMITTED", reflexTime: "153ms→1.21ms" },
];

const MOCK_REFLEXES = [
  { name: "Morning Brief", pattern: "daily @ 09:00", speed: "1.21ms", hits: 47, status: "active" },
  { name: "Download Organizer", pattern: "on file_added → classify", speed: "0.89ms", hits: 312, status: "active" },
  { name: "Email Triage", pattern: "on inbox_update → prioritize", speed: "2.14ms", hits: 89, status: "active" },
  { name: "Backup Sentinel", pattern: "daily @ 23:00", speed: "1.54ms", hits: 23, status: "active" },
  { name: "Git Auto-Commit", pattern: "on file_saved in BIZRA-DATA-LAKE", speed: "3.21ms", hits: 577, status: "learning" },
];

const MOCK_COMMANDS = [
  { cmd: "organize downloads", desc: "Sort files by type → project folders", icon: "📂" },
  { cmd: "morning brief", desc: "Calendar + emails + tasks → one view", icon: "☀️" },
  { cmd: "triage emails", desc: "Priority sort across all 6 accounts", icon: "📧" },
  { cmd: "backup now", desc: "Snapshot → B:\\BIZRA-SOVEREIGN", icon: "🛡️" },
  { cmd: "show receipts", desc: "Full proof trail for today", icon: "📜" },
  { cmd: "system health", desc: "Heartbeat + membrane + governance status", icon: "💚" },
  { cmd: "find in data lake", desc: "Search across C:\\BIZRA-DATA-LAKE", icon: "🔍" },
  { cmd: "run proof cycle", desc: "Execute autopoietic loop Phase 1→7", icon: "🔄" },
];

const HeartbeatPulse = ({ bpm = 72 }) => {
  const [beat, setBeat] = useState(false);
  useEffect(() => {
    const interval = setInterval(() => {
      setBeat(true);
      setTimeout(() => setBeat(false), 200);
    }, (60 / bpm) * 1000);
    return () => clearInterval(interval);
  }, [bpm]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%",
        background: beat ? GREEN : "rgba(46,204,113,0.4)",
        boxShadow: beat ? `0 0 12px ${GREEN}` : "none",
        transition: "all 0.15s ease",
      }} />
      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: TEXT_SECONDARY, letterSpacing: 1 }}>
        HHMM S1 • {bpm} BPM
      </span>
    </div>
  );
};

const MembraneIndicator = ({ status = "SEALED" }) => (
  <div style={{
    display: "flex", alignItems: "center", gap: 6,
    padding: "3px 10px", borderRadius: 4,
    background: status === "SEALED" ? "rgba(46,204,113,0.1)" : "rgba(231,76,60,0.1)",
    border: `1px solid ${status === "SEALED" ? "rgba(46,204,113,0.3)" : "rgba(231,76,60,0.3)"}`,
  }}>
    <div style={{
      width: 5, height: 5, borderRadius: "50%",
      background: status === "SEALED" ? GREEN : RED,
    }} />
    <span style={{
      fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
      color: status === "SEALED" ? GREEN : RED, letterSpacing: 1.5,
    }}>
      MEMBRANE {status}
    </span>
  </div>
);

const GoldLine = ({ width = "100%" }) => (
  <div style={{
    width, height: 1,
    background: `linear-gradient(90deg, transparent, ${GOLD}, transparent)`,
    opacity: 0.3,
  }} />
);

const Receipt = ({ receipt, index }) => {
  const isDenied = receipt.status === "denied";
  return (
    <div style={{
      padding: "12px 16px",
      background: isDenied ? "rgba(231,76,60,0.06)" : SURFACE,
      borderLeft: `2px solid ${isDenied ? RED : GOLD}`,
      borderRadius: "0 6px 6px 0",
      marginBottom: 6,
      animation: `fadeSlideIn 0.4s ease ${index * 0.08}s both`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5, color: TEXT_PRIMARY, lineHeight: 1.5, flex: 1 }}>
          {receipt.action}
        </span>
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
          color: isDenied ? RED : GREEN,
          padding: "2px 6px", borderRadius: 3,
          background: isDenied ? "rgba(231,76,60,0.15)" : "rgba(46,204,113,0.1)",
          marginLeft: 8, whiteSpace: "nowrap",
        }}>
          {receipt.governance}
        </span>
      </div>
      <div style={{ display: "flex", gap: 16, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM }}>
        <span>{receipt.ts}</span>
        <span>hash: {receipt.hash}</span>
        <span>⚡ {receipt.reflexTime}</span>
      </div>
    </div>
  );
};

const CommandPalette = ({ isOpen, onClose, onExecute }) => {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const inputRef = useRef(null);

  const filtered = MOCK_COMMANDS.filter(c =>
    c.cmd.toLowerCase().includes(query.toLowerCase()) ||
    c.desc.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    if (isOpen && inputRef.current) inputRef.current.focus();
    setQuery("");
    setSelected(0);
  }, [isOpen]);

  const handleKey = useCallback((e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setSelected(s => Math.min(s + 1, filtered.length - 1)); }
    if (e.key === "ArrowUp") { e.preventDefault(); setSelected(s => Math.max(s - 1, 0)); }
    if (e.key === "Enter" && filtered[selected]) { onExecute(filtered[selected]); onClose(); }
    if (e.key === "Escape") onClose();
  }, [filtered, selected, onClose, onExecute]);

  if (!isOpen) return null;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.6)",
      backdropFilter: "blur(8px)",
      display: "flex", alignItems: "flex-start", justifyContent: "center",
      paddingTop: 120,
      animation: "fadeIn 0.2s ease",
    }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        width: 560, maxHeight: 440,
        background: VOID_PANEL,
        border: `1px solid rgba(201,169,98,0.2)`,
        borderRadius: 12,
        boxShadow: `0 24px 80px rgba(0,0,0,0.8), 0 0 1px ${GOLD_DIM}`,
        overflow: "hidden",
      }}>
        <div style={{ padding: "16px 20px", borderBottom: `1px solid rgba(255,255,255,0.06)` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ color: GOLD, fontSize: 14 }}>❯</span>
            <input
              ref={inputRef}
              value={query}
              onChange={e => { setQuery(e.target.value); setSelected(0); }}
              onKeyDown={handleKey}
              placeholder="What do you need, Mumo?"
              style={{
                flex: 1, background: "transparent", border: "none", outline: "none",
                color: TEXT_PRIMARY, fontSize: 15,
                fontFamily: "'IBM Plex Mono', monospace",
              }}
            />
            <kbd style={{
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
              color: TEXT_DIM, padding: "2px 6px",
              border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 4,
            }}>ESC</kbd>
          </div>
        </div>
        <div style={{ maxHeight: 340, overflowY: "auto" }}>
          {filtered.map((cmd, i) => (
            <div key={cmd.cmd} onClick={() => { onExecute(cmd); onClose(); }}
              style={{
                padding: "12px 20px",
                background: i === selected ? GOLD_DIM : "transparent",
                cursor: "pointer",
                display: "flex", alignItems: "center", gap: 12,
                borderLeft: i === selected ? `2px solid ${GOLD}` : "2px solid transparent",
                transition: "all 0.15s ease",
              }}
              onMouseEnter={() => setSelected(i)}>
              <span style={{ fontSize: 16, width: 24, textAlign: "center" }}>{cmd.icon}</span>
              <div>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, color: TEXT_PRIMARY }}>{cmd.cmd}</div>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: TEXT_DIM, marginTop: 2 }}>{cmd.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const SeedChainViz = () => {
  const links = [
    { ar: "نيّة", en: "NIYYAH", status: "complete" },
    { ar: "بيّنة", en: "BAYYINAH", status: "complete" },
    { ar: "حدّ", en: "HADD", status: "complete" },
    { ar: "أمانة", en: "AMANAH", status: "active" },
    { ar: "ثمرة", en: "THAMARA", status: "pending" },
    { ar: "إيصال", en: "IISAL", status: "pending" },
  ];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "8px 0" }}>
      {links.map((link, i) => (
        <div key={link.en} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
            padding: "6px 8px", borderRadius: 6,
            background: link.status === "active" ? GOLD_DIM : link.status === "complete" ? "rgba(46,204,113,0.08)" : SURFACE,
            border: `1px solid ${link.status === "active" ? GOLD : link.status === "complete" ? "rgba(46,204,113,0.2)" : "rgba(255,255,255,0.05)"}`,
            minWidth: 52,
          }}>
            <span style={{
              fontFamily: "'Amiri', serif", fontSize: 14,
              color: link.status === "active" ? GOLD : link.status === "complete" ? GREEN : TEXT_DIM,
            }}>{link.ar}</span>
            <span style={{
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, letterSpacing: 1,
              color: link.status === "active" ? GOLD : link.status === "complete" ? "rgba(46,204,113,0.6)" : TEXT_DIM,
            }}>{link.en}</span>
          </div>
          {i < links.length - 1 && (
            <div style={{
              width: 12, height: 1,
              background: link.status === "complete" ? "rgba(46,204,113,0.3)" : "rgba(255,255,255,0.08)",
            }} />
          )}
        </div>
      ))}
    </div>
  );
};

const MetricCard = ({ label, value, sub, color = GOLD }) => (
  <div style={{
    padding: "12px 14px", borderRadius: 8,
    background: SURFACE,
    border: `1px solid rgba(255,255,255,0.04)`,
    minWidth: 100,
  }}>
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM, letterSpacing: 1.5, marginBottom: 6 }}>
      {label}
    </div>
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 22, color, fontWeight: 600, lineHeight: 1 }}>
      {value}
    </div>
    {sub && <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM, marginTop: 4 }}>{sub}</div>}
  </div>
);

export default function DEMAOverlay() {
  const [view, setView] = useState("dashboard");
  const [cmdOpen, setCmdOpen] = useState(false);
  const [time, setTime] = useState(new Date());
  const [lastAction, setLastAction] = useState(null);
  const [showNotif, setShowNotif] = useState(false);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") { e.preventDefault(); setCmdOpen(true); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleExecute = (cmd) => {
    setLastAction(cmd);
    setShowNotif(true);
    setTimeout(() => setShowNotif(false), 3000);
  };

  const dubaiTime = time.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const dubaiDate = time.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

  const navItems = [
    { id: "dashboard", label: "DASHBOARD", icon: "◈" },
    { id: "receipts", label: "RECEIPTS", icon: "◉" },
    { id: "reflexes", label: "REFLEXES", icon: "⚡" },
    { id: "governance", label: "GOVERNANCE", icon: "🛡" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: `radial-gradient(ellipse at 20% 0%, rgba(201,169,98,0.03) 0%, transparent 50%), ${VOID}`,
      color: TEXT_PRIMARY,
      fontFamily: "'IBM Plex Mono', monospace",
      position: "relative",
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Amiri:wght@400;700&family=Newsreader:ital,wght@0,300;0,400;0,600;1,300&display=swap');
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulseGold { 0%, 100% { box-shadow: 0 0 0 0 rgba(201,169,98,0.3); } 50% { box-shadow: 0 0 16px 4px rgba(201,169,98,0.15); } }
        @keyframes breathe { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(201,169,98,0.2); border-radius: 2px; }
      `}</style>

      {/* ─── Top Bar ─── */}
      <div style={{
        position: "sticky", top: 0, zIndex: 100,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px",
        background: VOID_PANEL,
        borderBottom: `1px solid rgba(201,169,98,0.1)`,
        backdropFilter: "blur(20px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {/* DEMA Logotype */}
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{
              fontFamily: "'Amiri', serif", fontSize: 22, color: GOLD,
              fontWeight: 700, letterSpacing: 2,
            }}>ديما</span>
            <span style={{
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 11,
              color: TEXT_DIM, letterSpacing: 3, fontWeight: 300,
            }}>DEMA</span>
            <span style={{
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 9,
              color: TEXT_DIM, opacity: 0.5,
            }}>v0.1.0</span>
          </div>
          <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.06)" }} />
          <HeartbeatPulse />
          <MembraneIndicator />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            onClick={() => setCmdOpen(true)}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "6px 14px", borderRadius: 6,
              background: SURFACE,
              border: `1px solid rgba(255,255,255,0.06)`,
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}>
            <span style={{ fontSize: 12, color: TEXT_DIM }}>⌘K</span>
            <span style={{ fontSize: 11, color: TEXT_DIM }}>Command</span>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, color: TEXT_PRIMARY, fontWeight: 500 }}>
              {dubaiTime}
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM }}>
              Dubai GMT+4
            </div>
          </div>
        </div>
      </div>

      {/* ─── Navigation ─── */}
      <div style={{
        display: "flex", gap: 2, padding: "0 24px",
        borderBottom: `1px solid rgba(255,255,255,0.04)`,
        background: "rgba(2,4,8,0.5)",
      }}>
        {navItems.map(item => (
          <button key={item.id} onClick={() => setView(item.id)} style={{
            padding: "10px 16px",
            background: "transparent",
            border: "none",
            borderBottom: view === item.id ? `2px solid ${GOLD}` : "2px solid transparent",
            color: view === item.id ? GOLD : TEXT_DIM,
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11,
            letterSpacing: 1.5,
            cursor: "pointer",
            display: "flex", alignItems: "center", gap: 6,
            transition: "all 0.2s ease",
          }}>
            <span style={{ fontSize: 12 }}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </div>

      {/* ─── Content ─── */}
      <div style={{ padding: 24, maxWidth: 1000, margin: "0 auto", animation: "fadeIn 0.3s ease" }}>

        {view === "dashboard" && (
          <div>
            {/* Greeting */}
            <div style={{ marginBottom: 28 }}>
              <h1 style={{
                fontFamily: "'Newsreader', serif", fontSize: 28, fontWeight: 300,
                color: TEXT_PRIMARY, marginBottom: 4,
              }}>
                Good morning, Mumo.
              </h1>
              <p style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: TEXT_DIM }}>
                {dubaiDate} • NODE0 online • 5 receipts today • 3 reflexes fired
              </p>
            </div>

            {/* Seed Chain */}
            <div style={{
              padding: 16, borderRadius: 10,
              background: SURFACE,
              border: `1px solid rgba(201,169,98,0.08)`,
              marginBottom: 20,
            }}>
              <div style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2, marginBottom: 8 }}>
                سلسلة البذرة — SEED CHAIN
              </div>
              <SeedChainViz />
            </div>

            {/* Metrics Row */}
            <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
              <MetricCard label="TESTS" value="12,662" sub="11,216 Py + 1,446 Rs" color={GREEN} />
              <MetricCard label="COMMITS" value="577" sub="bizra-omega v2.0.0" />
              <MetricCard label="CRATES" value="26" sub="Rust workspace" />
              <MetricCard label="SNR" value="0.97" sub="5 P0 gaps remain" color={AMBER} />
              <MetricCard label="REFLEXES" value="126×" sub="153ms → 1.21ms" color={GOLD} />
              <MetricCard label="UPTIME" value="6.5h" sub="zero errors" color={GREEN} />
            </div>

            <GoldLine />

            {/* Morning Brief */}
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2, marginBottom: 12 }}>MORNING BRIEF</div>
              <div style={{
                padding: 20, borderRadius: 10,
                background: `linear-gradient(135deg, rgba(201,169,98,0.04), transparent)`,
                border: `1px solid rgba(201,169,98,0.1)`,
              }}>
                <div style={{
                  fontFamily: "'Newsreader', serif", fontSize: 16, color: TEXT_PRIMARY,
                  lineHeight: 1.7, fontWeight: 300,
                }}>
                  <span style={{ color: GOLD, fontFamily: "'Amiri', serif" }}>◈</span>{" "}
                  3 unread across mumo@bizra.org + 2 personal accounts.{" "}
                  1 flagged: potential investor inquiry from Dubai Future Foundation.{" "}
                  Calendar clear until Dhuhr. Recommended: close METRICS_CANONICAL.md before noon sprint.{" "}
                  Backup due in 14 hours. BIZRA-DATA-LAKE last commit: 2h ago.
                </div>
              </div>
            </div>

            {/* Recent Receipts Preview */}
            <div style={{ marginTop: 24 }}>
              <div style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                marginBottom: 12,
              }}>
                <span style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2 }}>LATEST RECEIPTS</span>
                <button onClick={() => setView("receipts")} style={{
                  background: "transparent", border: `1px solid rgba(201,169,98,0.2)`,
                  color: GOLD, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
                  padding: "4px 10px", borderRadius: 4, cursor: "pointer",
                  letterSpacing: 1,
                }}>VIEW ALL →</button>
              </div>
              {MOCK_RECEIPTS.slice(0, 3).map((r, i) => <Receipt key={r.id} receipt={r} index={i} />)}
            </div>
          </div>
        )}

        {view === "receipts" && (
          <div>
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "baseline",
              marginBottom: 20,
            }}>
              <h2 style={{ fontFamily: "'Newsreader', serif", fontSize: 22, fontWeight: 300 }}>
                Proof Trail
              </h2>
              <span style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2 }}>
                {MOCK_RECEIPTS.length} RECEIPTS TODAY • BLAKE3 CHAINED
              </span>
            </div>
            <div style={{
              padding: "8px 16px", marginBottom: 16, borderRadius: 6,
              background: "rgba(201,169,98,0.05)",
              border: `1px solid rgba(201,169,98,0.1)`,
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM,
            }}>
              Every action receipted. Every receipt hashed. Every hash chained. Immutable proof.
            </div>
            {MOCK_RECEIPTS.map((r, i) => <Receipt key={r.id} receipt={r} index={i} />)}
            <div style={{
              marginTop: 16, padding: 12, borderRadius: 6,
              background: SURFACE, textAlign: "center",
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM,
            }}>
              Chain root: 350d642099bde68b → ... → latest
            </div>
          </div>
        )}

        {view === "reflexes" && (
          <div>
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "baseline",
              marginBottom: 20,
            }}>
              <h2 style={{ fontFamily: "'Newsreader', serif", fontSize: 22, fontWeight: 300 }}>
                Reflex Network
              </h2>
              <span style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2 }}>
                S2→S1 MYELINATION • SKILL → REFLEX
              </span>
            </div>
            <div style={{
              padding: "8px 16px", marginBottom: 16, borderRadius: 6,
              background: "rgba(201,169,98,0.05)",
              border: `1px solid rgba(201,169,98,0.1)`,
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM,
            }}>
              Repeated patterns auto-compile into reflexes. 153ms deliberate → 1.21ms automatic. You don't think about breathing.
            </div>
            {MOCK_REFLEXES.map((reflex, i) => (
              <div key={reflex.name} style={{
                padding: "14px 16px", marginBottom: 8,
                background: SURFACE,
                borderLeft: `2px solid ${reflex.status === "active" ? GOLD : AMBER}`,
                borderRadius: "0 8px 8px 0",
                animation: `fadeSlideIn 0.4s ease ${i * 0.08}s both`,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: 13, color: TEXT_PRIMARY, fontWeight: 500 }}>
                    {reflex.name}
                  </span>
                  <span style={{
                    fontSize: 10, padding: "2px 8px", borderRadius: 3,
                    background: reflex.status === "active" ? "rgba(201,169,98,0.1)" : "rgba(243,156,18,0.1)",
                    color: reflex.status === "active" ? GOLD : AMBER,
                    letterSpacing: 1,
                  }}>
                    {reflex.status.toUpperCase()}
                  </span>
                </div>
                <div style={{ display: "flex", gap: 20, fontSize: 11, color: TEXT_DIM }}>
                  <span>⏱ {reflex.pattern}</span>
                  <span>⚡ {reflex.speed}</span>
                  <span>× {reflex.hits} hits</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {view === "governance" && (
          <div>
            <h2 style={{
              fontFamily: "'Newsreader', serif", fontSize: 22, fontWeight: 300,
              marginBottom: 20,
            }}>
              Constitutional Governance
            </h2>

            {/* Authority Hierarchy */}
            <div style={{
              padding: 20, borderRadius: 10,
              background: SURFACE,
              border: `1px solid rgba(201,169,98,0.08)`,
              marginBottom: 20,
            }}>
              <div style={{ fontSize: 10, color: TEXT_DIM, letterSpacing: 2, marginBottom: 14 }}>
                AUTHORITY HIERARCHY
              </div>
              {[
                { level: "Quran", status: "FROZEN", ar: "القرآن" },
                { level: "Hadith", status: "FROZEN", ar: "الحديث" },
                { level: "البذرة (The Seed)", status: "FROZEN", ar: "البذرة" },
                { level: "الرسالة (The Letter)", status: "FROZEN", ar: "الرسالة" },
                { level: "Enforceable Spine v1.1", status: "RATIFIED", ar: "العمود الفقري" },
                { level: "Root Invariants", status: "ACTIVE", ar: "" },
                { level: "Specifications", status: "MUTABLE", ar: "" },
                { level: "Code", status: "MUTABLE", ar: "" },
              ].map((item, i) => (
                <div key={item.level} style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "8px 12px", marginBottom: 4,
                  borderLeft: `2px solid ${i < 4 ? GOLD : i < 6 ? GREEN : TEXT_DIM}`,
                  background: i < 4 ? "rgba(201,169,98,0.03)" : "transparent",
                  borderRadius: "0 4px 4px 0",
                }}>
                  <span style={{ fontFamily: "'Amiri', serif", fontSize: 13, color: GOLD, minWidth: 50 }}>
                    {item.ar}
                  </span>
                  <span style={{ fontSize: 12, color: TEXT_PRIMARY, flex: 1 }}>{item.level}</span>
                  <span style={{
                    fontSize: 9, letterSpacing: 1.5,
                    color: item.status === "FROZEN" ? RED : item.status === "RATIFIED" ? GOLD : TEXT_DIM,
                    padding: "2px 6px", borderRadius: 3,
                    border: `1px solid ${item.status === "FROZEN" ? "rgba(231,76,60,0.3)" : item.status === "RATIFIED" ? "rgba(201,169,98,0.3)" : "rgba(255,255,255,0.06)"}`,
                  }}>
                    {item.status}
                  </span>
                </div>
              ))}
            </div>

            {/* Frozen Anchors */}
            <div style={{
              padding: 20, borderRadius: 10,
              background: "rgba(231,76,60,0.03)",
              border: `1px solid rgba(231,76,60,0.1)`,
              marginBottom: 20,
            }}>
              <div style={{ fontSize: 10, color: RED, letterSpacing: 2, marginBottom: 12 }}>
                FROZEN ANCHORS — NON-NEGOTIABLE
              </div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {[
                  { name: "ZANN_ZERO", desc: "No claims without evidence" },
                  { name: "RIBA_ZERO", desc: "Zero usurious extraction" },
                  { name: "Gini ≤ 0.35", desc: "Hard-cap inequality" },
                  { name: "Ihsan ≥ 0.95", desc: "Excellence floor" },
                ].map(anchor => (
                  <div key={anchor.name} style={{
                    padding: "8px 12px", borderRadius: 6,
                    background: "rgba(231,76,60,0.06)",
                    border: `1px solid rgba(231,76,60,0.15)`,
                  }}>
                    <div style={{ fontSize: 12, color: TEXT_PRIMARY, fontWeight: 500 }}>{anchor.name}</div>
                    <div style={{ fontSize: 10, color: TEXT_DIM, marginTop: 2 }}>{anchor.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Membrane Status */}
            <div style={{
              padding: 20, borderRadius: 10,
              background: "rgba(46,204,113,0.03)",
              border: `1px solid rgba(46,204,113,0.1)`,
            }}>
              <div style={{ fontSize: 10, color: GREEN, letterSpacing: 2, marginBottom: 12 }}>
                MEMBRANE — FAIL-CLOSED
              </div>
              <div style={{ fontSize: 12, color: TEXT_SECONDARY, lineHeight: 1.8 }}>
                Tax: 0.007ms measured. Default: DENY. Every action must be explicitly permitted by constitutional gate.
                Today: 4 PERMITTED, 1 DENIED. The denied action (filesystem boundary violation) proves the membrane works.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ─── Notification Toast ─── */}
      {showNotif && lastAction && (
        <div style={{
          position: "fixed", bottom: 24, right: 24, zIndex: 999,
          padding: "14px 20px", borderRadius: 10,
          background: VOID_PANEL,
          border: `1px solid rgba(201,169,98,0.2)`,
          boxShadow: `0 12px 40px rgba(0,0,0,0.6)`,
          animation: "slideDown 0.3s ease",
          display: "flex", alignItems: "center", gap: 12,
        }}>
          <span style={{ fontSize: 18 }}>{lastAction.icon}</span>
          <div>
            <div style={{ fontSize: 12, color: TEXT_PRIMARY }}>{lastAction.cmd}</div>
            <div style={{ fontSize: 10, color: GREEN }}>Executing... ⚡</div>
          </div>
        </div>
      )}

      {/* ─── Command Palette ─── */}
      <CommandPalette isOpen={cmdOpen} onClose={() => setCmdOpen(false)} onExecute={handleExecute} />

      {/* ─── Bottom Status Bar ─── */}
      <div style={{
        position: "fixed", bottom: 0, left: 0, right: 0,
        padding: "6px 24px",
        background: VOID_PANEL,
        borderTop: `1px solid rgba(255,255,255,0.04)`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
        fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TEXT_DIM,
        backdropFilter: "blur(12px)",
      }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <span>NODE0: MSI Titan 18 HX</span>
          <span>•</span>
          <span>bizra-omega v2.0.0</span>
          <span>•</span>
          <span>26 crates</span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ color: GOLD, fontFamily: "'Amiri', serif", fontSize: 12 }}>
            بسم الله الرحمن الرحيم
          </span>
        </div>
      </div>
    </div>
  );
}
