"""NODE0 Asset Inventory Scanner v1.1 - Lightweight targeted scan"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# Only scan key directories, limit depth
TARGETS = [
    (r"C:\BIZRA-DATA-LAKE", 3),
    (r"C:\BIZRA-DATA-LAKE\00_GENESIS", 5),
    (r"C:\BIZRA-DATA-LAKE\04_GOLD", 2),
    (r"C:\BIZRA-DATA-LAKE\docs", 4),
    (r"C:\BIZRA-DATA-LAKE\research_archive", 2),
    (r"C:\BIZRA-DATA-LAKE\golden_gems", 2),
    (r"C:\BIZRA-DATA-LAKE\core", 4),
    (r"C:\BIZRA-DATA-LAKE\specs", 3),
    (r"C:\BIZRA-DATA-LAKE\missions", 2),
    (r"C:\BIZRA-DATA-LAKE\sovereign_state", 3),
    (r"C:\BIZRA-NODE0\research-papers", 2),
    (r"C:\BIZRA-NODE0\docs", 3),
    (r"C:\BIZRA-NODE0\crates", 3),
    (r"C:\BIZRA-NODE0\genesis", 3),
    (r"C:\BIZRA-NODE0\blockchain", 3),
    (r"C:\BIZRA-Dual-Agentic-system--main\docs", 4),
    (r"C:\BIZRA-Dual-Agentic-system--main\constitution", 2),
    (r"C:\BIZRA-Dual-Agentic-system--main\HyperGraphRAG", 3),
    (r"C:\BIZRA-Dual-Agentic-system--main\core", 4),
    (r"C:\BIZRA-PROJECTS\03-RESEARCH", 3),
    (r"C:\BIZRA-PROJECTS\04-DOCUMENTATION", 3),
    (r"C:\BIZRA-PROJECTS\chat data samples", 2),
]
SKIP = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".cache",
    "dist",
    ".benchmarks",
    ".fastembed_cache",
}

RULES = {
    "sacred": {
        "pat": ["al_risalah", "al_bazrah", "الرسالة", "البذرة", "the massage"],
        "pth": ["01_ARABIC_FOUNDING"],
    },
    "research_paper": {
        "ext": [".tex", ".bib"],
        "pat": ["arxiv", "latex", "hierarchical-coordination", "recursive language"],
        "pth": ["research-papers", "research_archive"],
    },
    "architecture_spec": {
        "pat": ["architecture", "blueprint", "unified", "ddagi"],
        "pth": ["docs/architecture", "docs/adr"],
    },
    "phase_spec": {"pat": ["phase_"], "pth": ["docs/specs"]},
    "constitution": {
        "pat": [
            "constitution",
            "ihsan",
            "soul.md",
            "identity.md",
            "covenant",
            "genesis_covenant",
        ],
        "pth": ["constitution"],
    },
    "evidence": {
        "pat": ["evidence", "proof", "receipt", "ledger", "attestation"],
        "pth": ["sovereign_state", "evidence-pack", "04_GOLD"],
    },
    "token_econ": {"pat": ["token", "seed", "bloom", "zakat", "tokenomics"]},
    "investor": {
        "pat": [
            "investor",
            "pitch",
            "strategy_deck",
            "revenue",
            "competitive",
            "moneyshot",
            "technical_brief",
        ]
    },
    "sape": {"pat": ["sape", "snr", "elite_assessment", "elite_analysis"]},
    "golden_gem": {"pth": ["golden_gems"]},
    "knowledge_data": {"ext": [".parquet", ".jsonl", ".index", ".npy", ".db"]},
    "chat_history": {"pat": ["conversations", "chat data", "data-202"]},
    "pollution": {"pat": ["magicmock"]},
}


def classify(fp, fn):
    cats = []
    fpl = fp.lower().replace("\\", "/")
    fnl = fn.lower()
    _, ext = os.path.splitext(fn)
    for cat, r in RULES.items():
        hit = False
        for p in r.get("pth", []):
            if p.lower() in fpl:
                hit = True
                break
        if not hit:
            for p in r.get("pat", []):
                if p.lower() in fnl or p.lower() in fpl:
                    hit = True
                    break
        if not hit:
            for e in r.get("ext", []):
                if ext.lower() == e:
                    hit = True
                    break
        if hit:
            cats.append(cat)
    return cats or ["unclassified"]


def fmt(sz):
    for u in ["B", "KB", "MB", "GB"]:
        if sz < 1024:
            return f"{sz:.1f} {u}"
        sz /= 1024
    return f"{sz:.1f} TB"


def scan(root, max_depth):
    assets = []
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dp, dns, fns in os.walk(root):
        cur_depth = dp.count(os.sep) - root_depth
        if cur_depth >= max_depth:
            dns.clear()
            continue
        dns[:] = [d for d in dns if d not in SKIP]
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                st = os.stat(fp)
                cats = classify(fp, fn)
                _, ext = os.path.splitext(fn)
                assets.append(
                    {
                        "filename": fn,
                        "ext": ext.lower(),
                        "abs": fp,
                        "root": root,
                        "bytes": st.st_size,
                        "size": fmt(st.st_size),
                        "modified": datetime.fromtimestamp(st.st_mtime).strftime(
                            "%Y-%m-%d"
                        ),
                        "categories": cats,
                        "primary": cats[0],
                    }
                )
            except:
                pass
    return assets


print("NODE0 ASSET INVENTORY v1.1 - Targeted Scan")
print("=" * 60)

all_a = []
for root, depth in TARGETS:
    if os.path.exists(root):
        a = scan(root, depth)
        if a:
            print(f"  {root}: {len(a):,} files")
            all_a.extend(a)

# Deduplicate by absolute path
seen = set()
unique = []
for a in all_a:
    p = a.get("abs", "")
    if p not in seen:
        seen.add(p)
        unique.append(a)
all_a = unique

# Stats
cats = defaultdict(int)
exts = defaultdict(int)
total_sz = 0
pollution = 0
for a in all_a:
    total_sz += a.get("bytes", 0)
    for c in a.get("categories", []):
        cats[c] += 1
    exts[a.get("ext", "")] += 1
    if "pollution" in a.get("categories", []):
        pollution += 1

print(f"\n{'='*60}")
print(f"TOTAL: {len(all_a):,} unique files | {fmt(total_sz)}")
print(
    f"Pollution: {pollution} | Sacred: {cats.get('sacred',0)} | IP docs: {cats.get('architecture_spec',0)+cats.get('phase_spec',0)+cats.get('research_paper',0)+cats.get('constitution',0)}"
)
print(f"{'='*60}")

print("\nCATEGORIES:")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:,}")

print("\nEXTENSIONS (top 15):")
for k, v in sorted(exts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k or '(none)'}: {v:,}")

# Sacred
sacred = [
    a
    for a in all_a
    if "sacred" in a.get("categories", []) or "constitution" in a.get("categories", [])
]
print(f"\nSACRED/CONSTITUTIONAL ({len(sacred)}):")
for s in sorted(sacred, key=lambda x: x["filename"]):
    print(f"  {s['filename']} ({s['size']}) [{s['modified']}]")

# IP assets
ip_cats = {
    "research_paper",
    "architecture_spec",
    "phase_spec",
    "constitution",
    "golden_gem",
    "token_econ",
    "investor",
}
ip = [a for a in all_a if set(a.get("categories", [])) & ip_cats]
print(f"\nHIGH-VALUE IP ({len(ip)}):")
for a in sorted(ip, key=lambda x: (x["primary"], x["filename"])):
    print(f"  [{a['primary']}] {a['filename']} ({a['size']})")

# Pollution
poll = [a for a in all_a if "pollution" in a.get("categories", [])]
if poll:
    print(f"\nPOLLUTION ({len(poll)} files to quarantine):")
    print(f"  Pattern: MagicMock test artifacts in BIZRA-DATA-LAKE root")

# Filename duplicates
nm = defaultdict(list)
for a in all_a:
    nm[a["filename"]].append(a)
dupes = {
    k: v
    for k, v in nm.items()
    if len(v) > 1
    and any(e in k.lower() for e in [".md", ".pdf", ".yaml", ".json", ".py"])
}
if dupes:
    print(f"\nKEY DUPLICATES ({len(dupes)}):")
    for k, locs in sorted(dupes.items(), key=lambda x: -len(x[1]))[:20]:
        paths = [os.path.basename(l["root"]) for l in locs]
        print(f"  {k} -> {paths}")

# Save
out = r"C:\BIZRA-DATA-LAKE\scripts\ops"
os.makedirs(out, exist_ok=True)

manifest = {
    "version": "1.1",
    "mission": "NODE0-MISSION-001",
    "timestamp": datetime.now().isoformat(),
    "totals": {
        "files": len(all_a),
        "size": fmt(total_sz),
        "bytes": total_sz,
        "pollution": pollution,
        "sacred": len(sacred),
        "ip": len(ip),
        "duplicates": len(dupes),
    },
    "categories": dict(sorted(cats.items(), key=lambda x: -x[1])),
    "extensions": dict(sorted(exts.items(), key=lambda x: -x[1])),
}
with open(os.path.join(out, "node0_asset_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

with open(os.path.join(out, "node0_ip_assets.json"), "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "name": a["filename"],
                "cats": a["categories"],
                "size": a["size"],
                "path": a["abs"],
                "modified": a["modified"],
            }
            for a in ip
        ],
        f,
        indent=2,
        ensure_ascii=False,
    )

with open(os.path.join(out, "node0_full_inventory.jsonl"), "w", encoding="utf-8") as f:
    for a in all_a:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")

with open(os.path.join(out, "node0_sacred_assets.json"), "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "name": a["filename"],
                "cats": a["categories"],
                "size": a["size"],
                "path": a["abs"],
                "modified": a["modified"],
            }
            for a in sacred
        ],
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nOutputs -> {out}")
print("  node0_asset_manifest.json | node0_ip_assets.json")
print("  node0_full_inventory.jsonl | node0_sacred_assets.json")
print(f"\n{'='*60}")
print("MISSION #001 COMPLETE")
print(f"{'='*60}")
