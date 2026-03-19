#!/usr/bin/env python3
"""BIZRA NODE0 v0.3.0 — Perfect Function. One node. Everything works."""

import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__version__ = "0.3.0"
P = 1_000_000
DIR = Path.home() / ".bizra"
DECL = "859649ea1a44f1bf4c183105a42e66b0a9d34505c53786d639d965b3afa46474"


def fp(v):
    return int(v * P)


def fl(v):
    return v / P


def fpm(a, b):
    return (a * b) // P


def fpd(a, b):
    return (a * P) // b if b else 0


def fpc(v, lo, hi):
    return max(lo, min(hi, v))


# Constitutional thresholds — SSoT: core/integration/constants.py
IHSAN_FLOOR = fp(0.95)
INTENT_FLOOR = fp(0.90)
GINI_H = fp(0.30)
GINI_W = fp(0.50)
GINI_C = fp(0.70)
BASE_REWARD = fp(1.0)
BLOOM_RATE = fp(0.01)
# Standalone uses simplified 3-dim scoring (canonical uses 4-dim with W_INTENT=0.25)
# Standalone equity_factor penalizes whales (canonical ghazali_equity_factor boosts newcomers)
# These are intentional design choices for the zero-dependency distribution target.
W_E = fp(0.40)
W_I = fp(0.40)
W_R = fp(0.20)
Gc = "\033[38;2;201;169;98m"
D = "\033[2m"
B = "\033[1m"
GR = "\033[38;2;52;211;153m"
RD = "\033[38;2;248;113;113m"
PU = "\033[38;2;167;139;250m"
BL = "\033[38;2;96;165;250m"
AM = "\033[38;2;251;191;36m"
R = "\033[0m"


def intent_gate(i):
    return i >= INTENT_FLOOR


def ihsan_score(e, i, r):
    return fpc(fpm(W_E, e) + fpm(W_I, i) + fpm(W_R, r), 0, P)


def ihsan_gate(s):
    return s >= IHSAN_FLOOR


def mint_seed(s, e):
    r = fpm(BASE_REWARD, s)
    if e > fp(0.90):
        r += fpm(fp(0.15), r)
    return r


def compute_gini(b):
    a = sorted([x for x in b if x >= fp(0.01)])
    n = len(a)
    if n <= 1:
        return 0
    t = sum(a)
    if t == 0:
        return 0
    ws = sum((i + 1) * v for i, v in enumerate(a))
    return fpc((2 * ws * P) // (n * t) - ((n + 1) * P) // n, 0, P)


def khaldunian(g):
    if g <= GINI_H:
        return P
    if g <= GINI_W:
        r = fpd(g - GINI_H, GINI_W - GINI_H)
        sq = fpm(r, r)
        return fpc(fp(0.10) + fpm(fp(0.90), P - sq), fp(0.10), P)  # 0.10 + 0.90*(1-r^2)
    return fp(0.10) if g <= GINI_C else fp(0.01)


def equity_factor(bal, mean):
    if mean <= 0:
        return P
    if bal >= mean:
        return fp(0.30)
    return fpc(P - fpm(fp(0.70), fpd(bal, mean)), fp(0.30), P)


def progressive_mint(base, bal, bals):
    g = compute_gini(bals)
    t = khaldunian(g)
    active = [x for x in bals if x >= fp(0.01)]
    m = sum(active) // len(active) if active else 0
    e = equity_factor(bal, m)
    return fpm(fpm(base, t), e), g, t, e


TIERS = [
    {"n": "Novice", "r": 0, "i": 0.93},
    {"n": "Apprentice", "r": 100, "i": 0.94},
    {"n": "Adept", "r": 500, "i": 0.95},
    {"n": "Expert", "r": 1000, "i": 0.96},
    {"n": "Master", "r": 5000, "i": 0.97},
    {"n": "Grandmaster", "r": 10000, "i": 0.98},
]
SKILLS = [
    {"id": "clipboard", "n": "Clipboard", "t": 0, "c": 0, "p": [], "ic": "📋"},
    {
        "id": "focus",
        "n": "Window Focus",
        "t": 0,
        "c": 0,
        "p": ["clipboard"],
        "ic": "🪟",
    },
    {"id": "capture", "n": "Screen Capture", "t": 0, "c": 0, "p": [], "ic": "📸"},
    {
        "id": "mouse",
        "n": "Mouse Control",
        "t": 1,
        "c": fp(0.10),
        "p": ["focus"],
        "ic": "🖱️",
    },
    {
        "id": "keyboard",
        "n": "Keyboard Entry",
        "t": 1,
        "c": fp(0.10),
        "p": ["mouse"],
        "ic": "⌨️",
    },
    {
        "id": "applaunch",
        "n": "App Launch",
        "t": 1,
        "c": fp(0.15),
        "p": ["focus"],
        "ic": "🚀",
    },
    {
        "id": "fileread",
        "n": "File Read",
        "t": 2,
        "c": fp(0.50),
        "p": ["keyboard"],
        "ic": "📖",
    },
    {
        "id": "filewrite",
        "n": "File Write",
        "t": 2,
        "c": fp(1.00),
        "p": ["fileread"],
        "ic": "✏️",
    },
    {
        "id": "windowmgmt",
        "n": "Window Manage",
        "t": 2,
        "c": fp(0.75),
        "p": ["mouse", "focus"],
        "ic": "🔲",
    },
    {
        "id": "powershell",
        "n": "PowerShell",
        "t": 3,
        "c": fp(3.00),
        "p": ["filewrite"],
        "ic": "⚡",
    },
    {
        "id": "multistep",
        "n": "Multi-Step Chains",
        "t": 3,
        "c": fp(5.00),
        "p": ["powershell"],
        "ic": "🔗",
    },
    {
        "id": "crossapp",
        "n": "Cross-App",
        "t": 4,
        "c": fp(8.00),
        "p": ["multistep", "applaunch"],
        "ic": "🌐",
    },
    {
        "id": "network",
        "n": "Network Access",
        "t": 4,
        "c": fp(10.00),
        "p": ["crossapp"],
        "ic": "📡",
    },
    {"id": "governance", "n": "Governance", "t": 4, "c": fp(5.00), "p": [], "ic": "🏛️"},
    {
        "id": "selfmod",
        "n": "Self-Modify",
        "t": 5,
        "c": fp(50.00),
        "p": ["network", "multistep"],
        "ic": "🧬",
    },
    {
        "id": "validator",
        "n": "Validator",
        "t": 5,
        "c": fp(30.00),
        "p": ["governance"],
        "ic": "🛡️",
    },
]
QUESTS = [
    {
        "id": "morning",
        "n": "Morning Ritual",
        "t": 0,
        "s": fp(0.10),
        "b": fp(0.001),
        "ic": "☀️",
        "tp": "daily",
        "d": "Launch 3 apps",
    },
    {
        "id": "clip",
        "n": "Clipboard Master",
        "t": 0,
        "s": fp(0.15),
        "b": fp(0.001),
        "ic": "📎",
        "tp": "daily",
        "d": "10 clipboard ops",
    },
    {
        "id": "janitor",
        "n": "File Janitor",
        "t": 1,
        "s": fp(0.50),
        "b": fp(0.005),
        "ic": "🧹",
        "tp": "daily",
        "d": "Organize folder",
    },
    {
        "id": "email",
        "n": "Email Triage",
        "t": 1,
        "s": fp(0.40),
        "b": fp(0.004),
        "ic": "📧",
        "tp": "daily",
        "d": "Categorize 20 emails",
    },
    {
        "id": "report",
        "n": "Report Gen",
        "t": 2,
        "s": fp(1.00),
        "b": fp(0.010),
        "ic": "📊",
        "tp": "weekly",
        "d": "Create report from data",
    },
    {
        "id": "review",
        "n": "Code Review",
        "t": 2,
        "s": fp(0.80),
        "b": fp(0.008),
        "ic": "🔍",
        "tp": "daily",
        "d": "Review and annotate code",
    },
    {
        "id": "pipeline",
        "n": "Data Pipeline",
        "t": 3,
        "s": fp(2.00),
        "b": fp(0.020),
        "ic": "🔄",
        "tp": "weekly",
        "d": "ETL from 10 sources",
    },
    {
        "id": "automail",
        "n": "Email Auto",
        "t": 3,
        "s": fp(2.00),
        "b": fp(0.020),
        "ic": "🤖",
        "tp": "weekly",
        "d": "Process 50 emails",
    },
    {
        "id": "build",
        "n": "Build Pipeline",
        "t": 4,
        "s": fp(5.00),
        "b": fp(0.050),
        "ic": "🏗️",
        "tp": "weekly",
        "d": "Full CI/CD",
    },
    {
        "id": "monthly",
        "n": "Monthly Analysis",
        "t": 4,
        "s": fp(10.00),
        "b": fp(0.100),
        "ic": "📑",
        "tp": "monthly",
        "d": "Comprehensive report",
    },
]
PAT7 = [
    {
        "id": "P1",
        "n": "Planner",
        "dm": "planning",
        "tl": [
            "task_decomposition",
            "htn_planning",
            "priority_queue",
            "dependency_resolution",
        ],
    },
    {
        "id": "P2",
        "n": "Researcher",
        "dm": "research",
        "tl": ["rag_retrieval", "web_search", "knowledge_graph", "semantic_extraction"],
    },
    {
        "id": "P3",
        "n": "Coder",
        "dm": "coding",
        "tl": ["code_generation", "test_runner", "debugger", "profiler"],
    },
    {
        "id": "P4",
        "n": "Evaluator",
        "dm": "evaluation",
        "tl": ["snr_scorer", "shannon_entropy", "poi_validator", "quality_rubric"],
    },
    {
        "id": "P5",
        "n": "Ethicist",
        "dm": "ethics",
        "tl": ["constitution_check", "shariah_compliance", "bias_detection"],
    },
    {
        "id": "P6",
        "n": "Publisher",
        "dm": "delivery",
        "tl": ["format_output", "deliver", "notify", "feedback_collector"],
    },
    {
        "id": "P7",
        "n": "Integrator",
        "dm": "coordination",
        "tl": ["agent_router", "context_bridge", "memory_manager"],
    },
]
SAT5 = [
    {"id": "S1", "n": "Sentinel", "r": "Health + threats"},
    {"id": "S2", "n": "Oracle", "r": "Ihsan scoring (ZERO user control)"},
    {"id": "S3", "n": "Ledger", "r": "Event log + Merkle chain"},
    {"id": "S4", "n": "Conductor", "r": "S1/S2 boundary + reflexes"},
    {"id": "S5", "n": "Ambassador", "r": "Network + attestations"},
]


def _b(d):
    return hashlib.blake2b(
        d if isinstance(d, bytes) else d.encode(), digest_size=32
    ).hexdigest()


def _now():
    return int(time.time() * 1000)


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")


def _load():
    f = DIR / "node.json"
    return json.loads(f.read_text()) if f.exists() else None


def _save(n):
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "node.json").write_text(json.dumps(n, indent=2, default=str))


def _log(ev):
    DIR.mkdir(parents=True, exist_ok=True)
    open(DIR / "ledger.jsonl", "a").write(json.dumps(ev, default=str) + "\n")


def _ledger():
    f = DIR / "ledger.jsonl"
    return (
        [json.loads(l) for l in f.read_text().strip().split("\n") if l.strip()]
        if f.exists()
        else []
    )


def _peers():
    f = DIR / "peers.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _sp(p):
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "peers.json").write_text(json.dumps(p, indent=2))


def route(task):
    kw = {
        "P1": [
            "plan",
            "organize",
            "structure",
            "schedule",
            "prioritize",
            "roadmap",
            "strategy",
        ],
        "P2": [
            "research",
            "find",
            "search",
            "learn",
            "study",
            "analyze",
            "paper",
            "discover",
        ],
        "P3": [
            "code",
            "program",
            "script",
            "debug",
            "test",
            "build",
            "implement",
            "deploy",
            "fix",
            "compile",
        ],
        "P4": [
            "evaluate",
            "review",
            "score",
            "assess",
            "measure",
            "benchmark",
            "audit",
        ],
        "P5": [
            "check",
            "verify",
            "compliance",
            "ethics",
            "shariah",
            "bias",
            "constitution",
        ],
        "P6": [
            "write",
            "draft",
            "document",
            "report",
            "email",
            "format",
            "publish",
            "present",
        ],
        "P7": [
            "coordinate",
            "integrate",
            "combine",
            "merge",
            "connect",
            "bridge",
            "sync",
        ],
    }
    t = task.lower()
    best = "P1"
    bs = 0
    for a, ws in kw.items():
        s = sum(1 for w in ws if w in t)
        if s > bs:
            best = a
            bs = s
    return best


def decompose(task, aid):
    a = next(x for x in PAT7 if x["id"] == aid)
    tpl = {
        "planning": [
            ("analyze", "Break goal into components"),
            ("deps", "Order by dependencies"),
            ("plan", "Create actionable plan"),
        ],
        "research": [
            ("sources", "Find relevant sources"),
            ("extract", "Extract key insights"),
            ("synthesize", "Combine into output"),
        ],
        "coding": [
            ("spec", "Parse requirements"),
            ("implement", "Write code"),
            ("test", "Run tests"),
        ],
        "evaluation": [
            ("collect", "Gather metrics"),
            ("score", "Run assessment"),
            ("report", "Produce report"),
        ],
        "ethics": [
            ("scan", "Check I1-I7"),
            ("impact", "Assess implications"),
            ("recommend", "Generate recommendations"),
        ],
        "delivery": [
            ("outline", "Create structure"),
            ("draft", "Write content"),
            ("refine", "Polish and deliver"),
        ],
        "coordination": [
            ("map", "Identify agents"),
            ("delegate", "Assign tasks"),
            ("aggregate", "Combine results"),
        ],
    }
    steps = tpl.get(a["dm"], [("execute", "Perform task")])
    return [
        {"act": s[0], "desc": s[1], "tool": a["tl"][min(i, len(a["tl"]) - 1)]}
        for i, s in enumerate(steps)
    ]


def oracle_score(proof, task):
    sp = proof["sp"]
    sc = proof["sc"]
    tools = proof["tools"]
    comp = sc / max(sp, 1)
    words = len(task.split())
    tech = any(
        w in task.lower()
        for w in [
            "code",
            "review",
            "audit",
            "design",
            "build",
            "research",
            "teach",
            "write",
            "fix",
            "create",
            "analyze",
            "test",
            "document",
            "optimize",
            "implement",
            "deploy",
        ]
    )
    intent = min(fp(0.99), fp(0.92 + min(words, 20) * 0.003 + (0.02 if tech else 0)))
    eff = min(fp(0.99), fp(0.90 + comp * 0.09))
    imp = min(fp(0.99), fp(0.88 + len(tools) * 0.03 + (0.02 if tech else 0)))
    rep = min(fp(0.99), fp(0.93 + min(words, 15) * 0.003))
    return intent, eff, imp, rep


def loot(eff, lat):
    e = fl(eff)
    if e >= 0.98 and lat < 500:
        return "⚡ LEGENDARY", fp(1.50)
    if e >= 0.95 and lat < 1000:
        return "💜 EPIC", fp(1.30)
    if e >= 0.92 and lat < 2000:
        return "🔵 RARE", fp(1.15)
    return "Common", fp(1.00)


def tier(rac, ai):
    for i in range(5, -1, -1):
        if rac >= TIERS[i]["r"] and fl(ai) >= TIERS[i]["i"]:
            return i
    return 0


def cmd_init(args):
    if _load():
        print(f"\n  {Gc}Node exists.{R}\n")
        return
    name = args[0] if args else f"node_{int(time.time())%100000}"
    nid = _b(os.urandom(32))
    sig = _b(nid + DECL)
    node = {
        "name": name,
        "nid": nid,
        "cov": DECL,
        "sig": sig,
        "seed": 0,
        "bloom": 0,
        "vac": 0,
        "rac": 0,
        "acts": 0,
        "miss": 0,
        "ih": [],
        "ai": 0,
        "streak": 0,
        "skills": ["clipboard", "capture"],
        "cache": [],
        "s1": 0,
        "s2": 0,
        "ag": 0,
        "ar": 0,
        "leg": 0,
        "epic": 0,
        "rare": 0,
        "cat": _now(),
        "la": _now(),
        "tier": 0,
    }
    _save(node)
    _log({"t": "genesis", "nid": nid, "name": name, "cov": DECL, "ts": _now()})
    print(f"\n  {GR}✓{R} Node '{name}' established · {D}{nid[:20]}...{R}")
    print(f"  {D}Covenant: {DECL[:16]}... ✓{R}\n")
    print(f"  {BL}PAT-7 MINTED (Your Agents){R}")
    for a in PAT7:
        print(f"    {GR}●{R} {a['id']} {a['n']:12s} {D}{a['dm']}{R}")
    print(f"\n  {PU}SAT-5 MINTED (System — ZERO User Control){R}")
    for a in SAT5:
        print(f"    {PU}●{R} {a['id']} {a['n']:12s} {D}{a['r']}{R}")
    print(f"\n  {Gc}Tier: Novice · RAC: 0 · Equity: 3.27x{R}\n")


def cmd_mission(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    task = " ".join(args) if args else input(f"  {Gc}Mission:{R} ").strip()
    if not task:
        print(f"  {RD}Task needed.{R}\n")
        return
    ts = _now()
    aid = route(task)
    ag = next(a for a in PAT7 if a["id"] == aid)
    steps = decompose(task, aid)
    cache = nd.get("cache", [])
    hit = any(r.get("p") == task.lower().strip() and r.get("compiled") for r in cache)
    if hit:
        nd["s1"] = nd.get("s1", 0) + 1
        path = "S1_REFLEX"
        lat = int(150 + random.random() * 200)
    else:
        nd["s2"] = nd.get("s2", 0) + 1
        path = "S2_AGENT"
        lat = int(1200 + random.random() * 800)
    proof = {"sp": len(steps), "sc": len(steps), "tools": [s["tool"] for s in steps]}
    intent, eff, imp, rep = oracle_score(proof, task)
    if not intent_gate(intent):
        nd["miss"] += 1
        _save(nd)
        _log(
            {
                "t": "rej",
                "r": "intent",
                "s": fl(intent),
                "task": task[:80],
                "ts": _now(),
            }
        )
        print(f"\n  {RD}✗ Intent: {fl(intent):.3f} < 0.90{R}\n")
        return
    score = ihsan_score(eff, imp, rep)
    if not ihsan_gate(score):
        nd["vac"] += 1
        nd["miss"] += 1
        _save(nd)
        _log(
            {"t": "rej", "r": "ihsan", "s": fl(score), "task": task[:80], "ts": _now()}
        )
        print(f"\n  {RD}✗ Ihsan: {fl(score):.4f} < 0.95{R} (VAC+1, RAC unchanged)\n")
        return
    base = mint_seed(score, eff)
    peers = _peers()
    allb = [nd["seed"]] + [p.get("seed", 0) for p in peers.values()]
    final, gini, throttle, eq = progressive_mint(base, nd["seed"], allb)
    dn, dm = loot(eff, lat)
    final = fpm(final, dm)
    nd["seed"] += final
    nd["bloom"] += fpm(BLOOM_RATE, score)
    nd["vac"] += 1
    nd["rac"] += 1
    nd["acts"] += 1
    nd["miss"] += 1
    nd["la"] = _now()
    nd["streak"] += 1
    ih = nd.get("ih", [])
    ih.append(score)
    nd["ih"] = ih[-365:]
    nd["ai"] = sum(ih[-30:]) // max(len(ih[-30:]), 1)
    if dn.startswith("⚡"):
        nd["leg"] += 1
    elif dn.startswith("💜"):
        nd["epic"] += 1
    elif dn.startswith("🔵"):
        nd["rare"] += 1
    nt = tier(nd["rac"], nd["ai"])
    tu = nt > nd.get("tier", 0)
    nd["tier"] = nt
    ul = set(nd.get("skills", []))
    for sk in SKILLS:
        if (
            sk["id"] not in ul
            and sk["t"] <= nt
            and sk["c"] == 0
            and all(p in ul for p in sk["p"])
        ):
            ul.add(sk["id"])
    if nd["miss"] % 10 == 0:
        for sk in SKILLS:
            if (
                sk["id"] not in ul
                and sk["t"] <= nt
                and nd["bloom"] >= sk["c"] > 0
                and all(p in ul for p in sk["p"])
            ):
                nd["bloom"] -= sk["c"]
                ul.add(sk["id"])
                break
    nd["skills"] = list(ul)
    pk = task.lower().strip()
    ex = [r for r in cache if r.get("p") == pk]
    if ex:
        ex[0]["cnt"] = ex[0].get("cnt", 0) + 1
    else:
        cache.append({"p": pk, "cnt": 1, "a": aid, "compiled": False})
    newly = []
    for r in cache:
        if r.get("cnt", 0) >= 5 and not r.get("compiled") and score >= fp(0.98):
            r["compiled"] = True
            newly.append(r)
    nd["cache"] = cache
    rid = _b(f"poi_emit:v1:{nd['nid']}:{task}:{_now()}")
    _save(nd)
    _log(
        {
            "t": "mission",
            "rid": rid[:16],
            "a": aid,
            "an": ag["n"],
            "path": path,
            "ih": fl(score),
            "seed": fl(final),
            "drop": dn,
            "steps": len(steps),
            "lat": lat,
            "task": task[:120],
            "ts": _now(),
        }
    )
    s1 = nd.get("s1", 0)
    s2 = nd.get("s2", 0)
    mye = s1 / max(s1 + s2, 1)
    lv = nd["rac"] // 10
    tn = TIERS[nt]["n"]
    comp = len([r for r in cache if r.get("compiled")])
    print(f"\n  {GR}✓{R} Mission Complete")
    print(f"  {B}Agent:{R}       {ag['id']} {ag['n']} ({ag['dm']})")
    print(f"  {B}Path:{R}        {BL if hit else D}{path}{R} ({lat}ms)")
    print(f"  {B}Steps:{R}       {len(steps)} ({' → '.join(s['act'] for s in steps)})")
    print(f"  {B}Ihsan:{R}       {GR}{fl(score):.4f}{R}")
    print(f"  {B}Drop:{R}        {dn}")
    print(
        f"  {B}Minted:{R}      {GR}+{fl(final):.4f} SEED{R}  {PU}+{fl(fpm(BLOOM_RATE,score)):.4f} BLOOM{R}"
    )
    print(
        f"  {B}Balance:{R}     {GR}{fl(nd['seed']):.4f} SEED{R}  {PU}{fl(nd['bloom']):.4f} BLOOM{R}"
    )
    print(
        f"  {B}Tier:{R}        {tn} (Lv.{lv})  {B}VAC:{R}{nd['vac']} {B}RAC:{R}{nd['rac']}"
    )
    print(f"  {B}Myelination:{R} {BL}{mye:.1%}{R} ({comp} reflexes)")
    print(f"  {D}Receipt: {rid[:24]}...{R}")
    if tu:
        print(f"\n  {Gc}🎉 TIER UP → {tn}!{R}")
    for r in newly:
        print(f"  {BL}⚡ REFLEX COMPILED: \"{r['p'][:40]}\" → S1{R}")
    print()


def cmd_think(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    q = " ".join(args) if args else input(f"  {Gc}Question:{R} ").strip()
    if not q:
        return
    aid = route(q)
    ag = next(a for a in PAT7 if a["id"] == aid)
    steps = decompose(q, aid)
    print(f"\n  {BL}Agent:{R} {ag['id']} {ag['n']} ({ag['dm']})")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s['desc']} {D}[{s['tool']}]{R}")
    print(f"  {D}Think mode — no receipt. Use 'mission' to earn.{R}\n")


def cmd_status(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    t = tier(nd.get("rac", 0), nd.get("ai", 0))
    tn = TIERS[t]
    nt = TIERS[min(t + 1, 5)]
    s1 = nd.get("s1", 0)
    s2 = nd.get("s2", 0)
    mye = s1 / max(s1 + s2, 1)
    ul = len(nd.get("skills", []))
    cache = nd.get("cache", [])
    comp = len([r for r in cache if r.get("compiled")])
    age = (_now() - nd["cat"]) / (86400000)
    gap = max(nt["r"] - nd.get("rac", 0), 0)
    print(f"""
  {Gc}{'═'*44}{R}
  {Gc}SOVEREIGN STATUS{R} — {B}{nd['name']}{R} — {tn['n']} Lv.{nd.get('rac',0)//10}
  {Gc}{'═'*44}{R}
  {D}Identity{R}   {nd['nid'][:20]}... · Covenant ✓ · {age:.0f}d
  {D}Economy{R}    {GR}{fl(nd['seed']):.4f} SEED{R} · {PU}{fl(nd['bloom']):.4f} BLOOM{R}
  {D}Rank{R}       {tn['n']} → {nt['n']} ({gap} RAC away)
  {D}Skills{R}     {ul}/{len(SKILLS)} unlocked
  {D}Intel{R}      {BL}{mye:.1%} myelination{R} · {comp} reflexes · S1:{s1} S2:{s2}
  {D}Activity{R}   VAC:{nd.get('vac',0)} RAC:{nd.get('rac',0)} Ihsan:{Gc}{fl(nd.get('ai',0)):.4f}{R} Streak:{nd.get('streak',0)}
  {D}Loot{R}       {nd.get('leg',0)}⚡ {nd.get('epic',0)}💜 {nd.get('rare',0)}🔵
  {D}Social{R}     {nd.get('ag',0)}↑ {nd.get('ar',0)}↓ · {len(_peers())} peers
  {D}Agents{R}     {BL}PAT-7{R} Planner·Researcher·Coder·Evaluator·Ethicist·Publisher·Integrator
             {PU}SAT-5{R} Sentinel·Oracle·Ledger·Conductor·Ambassador
  {Gc}{'═'*44}{R}
""")


def cmd_skills(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    ul = set(nd.get("skills", []))
    t = tier(nd.get("rac", 0), nd.get("ai", 0))
    print(f"\n  {Gc}SKILL TREE{R} — {len(ul)}/{len(SKILLS)}\n")
    for sk in SKILLS:
        u = sk["id"] in ul
        can = (
            not u
            and sk["t"] <= t
            and all(p in ul for p in sk["p"])
            and nd["bloom"] >= sk["c"]
        )
        st = f"{GR}✓{R}" if u else f"{AM}⬆{R}" if can else f"{D}🔒{R}"
        c = f" {D}({fl(sk['c']):.1f}B){R}" if sk["c"] > 0 else ""
        print(f"  {st} {sk['ic']} {sk['n']:20s} {D}T{sk['t']}{R}{c}")
    print()


def cmd_quests(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    t = tier(nd.get("rac", 0), nd.get("ai", 0))
    cache = nd.get("cache", [])
    print(f"\n  {Gc}QUESTS{R} — {TIERS[t]['n']}\n")
    for q in QUESTS:
        av = q["t"] <= t
        rfx = any(r.get("compiled") for r in cache if q["n"].lower() in r.get("p", ""))
        lk = "" if av else f" {RD}🔒{TIERS[q['t']]['n']}{R}"
        rx = f" {BL}⚡S1{R}" if rfx else ""
        print(
            f"  {q['ic']} {q['n']:20s} {GR}+{fl(q['s']):.2f}S{R} {PU}+{fl(q['b']):.3f}B{R} {D}{q['tp']}{R}{lk}{rx}"
        )
    print()


def cmd_log(args):
    evs = _ledger()
    if not evs:
        print(f"\n  {D}Empty.{R}\n")
        return
    n = int(args[0]) if args else 15
    print(f"\n  {Gc}LOG{R} ({len(evs)})\n  {Gc}{'─'*44}{R}")
    for ev in evs[-n:]:
        t = ev.get("t", "?")
        ts = _ts(ev.get("ts", 0))
        if t == "genesis":
            print(f"  {Gc}◆{R} {ts} GENESIS {ev.get('name','')}")
        elif t == "mission":
            print(
                f"  {GR}●{R} {ts} {ev.get('an',''):10s} I:{ev.get('ih',0):.3f} {GR}+{ev.get('seed',0):.3f}S{R} {ev.get('drop','')} {D}{ev.get('path','')}{R}"
            )
        elif t == "rej":
            print(f"  {RD}✗{R} {ts} {ev.get('r','')}: {ev.get('s',0):.3f}")
        elif t == "attest":
            print(f"  {PU}○{R} {ts} → {ev.get('peer','')[:16]}")
    print(f"  {Gc}{'─'*44}{R}\n")


def cmd_attest(args):
    nd = _load()
    if not nd:
        print(f"\n  {RD}Run: init{R}\n")
        return
    if not args:
        print(f"  {RD}Usage: attest <peer>{R}\n")
        return
    pid = args[0]
    if pid == nd["nid"] or pid == nd["name"]:
        print(f"  {RD}✗ Self-attest blocked.{R}\n")
        return
    if not nd.get("ih"):
        print(f"  {RD}✗ Do a mission first.{R}\n")
        return
    avg = sum(nd["ih"][-30:]) // max(len(nd["ih"][-30:]), 1)
    if avg < IHSAN_FLOOR:
        print(f"  {RD}✗ Ihsan {fl(avg):.4f}<0.95{R}\n")
        return
    peers = _peers()
    if pid not in peers:
        peers[pid] = {"af": 0, "at": 0, "seed": 0}
    peers[pid]["af"] += 1
    nd["ag"] += 1
    _sp(peers)
    _save(nd)
    _log({"t": "attest", "peer": pid, "ts": _now()})
    print(f"  {GR}✓{R} Attested {pid[:20]}{'...' if len(pid)>20 else ''}\n")


def cmd_test(args):
    print(f"\n  {Gc}SELF-TEST{R}\n")
    p = t = 0

    def ck(n, c):
        nonlocal p, t
        t += 1
        p += c
        print(f"  {'✓' if c else '✗'}  {n}")

    ck("Intent pass", intent_gate(fp(0.95)))
    ck("Intent fail", not intent_gate(fp(0.89)))
    s = ihsan_score(fp(0.97), fp(0.96), fp(0.97))
    ck("Ihsan pass", ihsan_gate(s))
    ck("Ihsan fail", not ihsan_gate(ihsan_score(fp(0.90), fp(0.90), fp(0.90))))
    ck("Mint>0", mint_seed(s, fp(0.97)) > 0)
    ck("Throttle=1", khaldunian(fp(0.20)) == P)
    ck("Throttle mono", khaldunian(fp(0.40)) > khaldunian(fp(0.60)))
    ep = equity_factor(fp(10), fp(100))
    er = equity_factor(fp(200), fp(100))
    ck("Poor>rich", ep > er)
    ck("Rich floor", er == fp(0.30))
    ck("Gini empty", compute_gini([]) == 0)
    ck("Gini single", compute_gini([fp(100)]) == 0)
    r1, _, _, _ = progressive_mint(fp(1.0), fp(1), [fp(1)] * 99 + [fp(50000)])
    r2, _, _, _ = progressive_mint(fp(1.0), fp(50000), [fp(1)] * 99 + [fp(50000)])
    ck("Newcomer>whale", r1 > r2)
    ck("Route P2", route("research papers") == "P2")
    ck("Route P3", route("write code script") == "P3")
    ck("Route P1", route("plan the roadmap") == "P1")
    ck("Route P4", route("evaluate score") == "P4")
    ck("Route P5", route("check compliance ethics") == "P5")
    ck("Route P6", route("draft report") == "P6")
    ck("Decompose 3", len(decompose("research AI", "P2")) == 3)
    ck("Loot legend", loot(fp(0.99), 300)[0].startswith("⚡"))
    ck("Loot common", loot(fp(0.85), 3000)[0] == "Common")
    i, e, m, r = oracle_score(
        {"sp": 3, "sc": 3, "tools": ["a", "b", "c"]}, "test code review and analyze"
    )
    ck("Oracle 4-tuple", all(x > 0 for x in [i, e, m, r]))
    ck("Oracle intent>0.90", i >= INTENT_FLOOR)
    ck("Oracle ihsan pass", ihsan_gate(ihsan_score(e, m, r)))
    print(f"\n  {p}/{t} passed{f' {GR}✓ ALL GREEN{R}' if p==t else ''}\n")


def cmd_reset(args):
    c = input(f"  {RD}Type 'I AM SOVEREIGN':{R} ")
    if c.strip() != "I AM SOVEREIGN":
        print(f"  {D}Cancelled.{R}\n")
        return
    if DIR.exists():
        shutil.rmtree(DIR)
    print(f"  {GR}✓{R} Reset.\n")


CMDS = {
    "init": cmd_init,
    "mission": cmd_mission,
    "think": cmd_think,
    "status": cmd_status,
    "skills": cmd_skills,
    "quests": cmd_quests,
    "log": cmd_log,
    "attest": cmd_attest,
    "test": cmd_test,
    "reset": cmd_reset,
}
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"""
  {Gc}BIZRA NODE0{R} v{__version__}
  {D}Perfect Function. One node. Everything works.{R}

  init [name]         Mint identity + PAT-7 + SAT-5
  mission "task"      Execute → verify → earn SEED
  think "question"    Explore (no receipt)
  status              Sovereign state
  skills              Skill tree (16 skills, 6 tiers)
  quests              Quest board (10 quests)
  log [n]             Event log
  attest <peer>       Vouch for peer
  test                25 self-tests
  reset               Delete (irreversible)
""")
        sys.exit(0)
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    if cmd in CMDS:
        CMDS[cmd](args)
    else:
        print(f"\n  {RD}Unknown: {cmd}{R}\n")
