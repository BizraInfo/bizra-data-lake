"""Scan docs + website captures for public / semi-public claims; classify them.

Classifications:
  BRAND_SAFE       — identity / mission / philosophy, no numeric promise
  PROOF_REQUIRED   — defensible but needs a published receipt
  NEEDS_REWRITE    — over-quantified, brittle, or ambiguous
  INTERNAL_ONLY    — fine for internal decks, not public
  PROHIBITED       — never publicly (AGI, financial returns, first-in-world, etc.)
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from .schemas import Claim


PROHIBITED_PATTERNS = [
    (r"\b(AGI|artificial general intelligence)\b", "AGI claim"),
    (r"\b(first|only)\s+(in|on)\s+the\s+world\b", "First-in-world / only-in-world"),
    (r"\bguaranteed?\s+returns?\b", "Financial return guarantee"),
    (r"\b(beats?|outperforms?)\s+(GPT|Claude|Gemini|Llama)\b", "Benchmark-superiority claim"),
    (r"\b(tamper[- ]?proof|unbreakable|unhackable)\b", "Cryptographic-finality claim"),
    (r"\bSOC\s?2\b|\bISO\s?27001\b", "Unsubstantiated certification"),
]

NEEDS_REWRITE_PATTERNS = [
    (r"\b100\s?%\s+(pass|uptime|accuracy|success)\b", "Brittle 100% claim"),
    (r"\b(production[- ]?ready|live|GA)\b", "Production-readiness implication"),
    (r"\$\d+(\.\d+)?\s*(per|/)\s*(action|call|request|month)?\b", "Explicit cost figure"),
    (r"\bSNR\s*[=:]?\s*0?\.\d+\b", "Exact SNR number"),
    (r"\b\d{1,3}\s*/\s*\d{1,4}\s+(nodes?|seats?)\s+(remaining|left)\b",
     "Manufactured scarcity claim"),
    (r"\b\d{2,}\s*,?\s*\d*\s+(verified|passing)\s+tests?\b", "Exact test count claim"),
    (r"\blatency\s*[:=]\s*\d+\s*ms\b", "Explicit latency claim"),
]

PROOF_REQUIRED_PATTERNS = [
    (r"\bIhsan(\s+Gate)?\s*(≥|>=)\s*0?\.\d+\b", "Ihsan-threshold claim"),
    (r"\bEd25519\b", "Cryptography claim"),
    (r"\bno\s+telemetry\b", "Zero-telemetry claim"),
    (r"\b(local[- ]?only|no\s+cloud\s+dependency)\b", "Local-only / no-cloud claim"),
    (r"\bBLAKE3\b", "Hashing claim"),
    (r"\bZ3\b", "Formal-verification claim"),
    (r"\bDilithium|Kyber|post[- ]?quantum\b", "Post-quantum claim"),
    (r"\b[0-9]{1,2}x\s+(faster|cheaper|smaller)\b", "Relative-performance claim"),
]

BRAND_SAFE_PATTERNS = [
    (r"\bSeed of Sovereign Intelligence\b", "Brand tagline"),
    (r"\bBuild with meaning\.?\s*Act with proof\.?\s*Grow with Ihsan\b", "Brand motto"),
    (r"\bMission[- ]centric\b", "Category framing"),
    (r"\bhuman[- ]first\b", "Philosophy"),
    (r"\b(Every human is a node|Every node is a seed)\b", "Movement line"),
    (r"\bLaw of Assumption\b", "Doctrine"),
]


def _classify_text(text: str) -> List[tuple[str, str, str]]:
    """Return list of (classification, category, text_excerpt)."""
    out: List[tuple[str, str, str]] = []

    def _check(patterns, cls):
        for pat, cat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                excerpt = text[start:end].replace("\n", " ").strip()
                out.append((cls, cat, excerpt[:240]))

    _check(PROHIBITED_PATTERNS, "PROHIBITED")
    _check(NEEDS_REWRITE_PATTERNS, "NEEDS_REWRITE")
    _check(PROOF_REQUIRED_PATTERNS, "PROOF_REQUIRED")
    _check(BRAND_SAFE_PATTERNS, "BRAND_SAFE")
    return out


def _iter_doc_paths(root: Path, exclude_dirs: List[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if fn.endswith(".md"):
                yield Path(dirpath) / fn


def _iter_specific_files(paths: List[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            yield p


def scan_claims(
    repo_root: Path,
    claim_scan_roots: List[str],
    exclude_dirs: List[str],
    website_captures: List[dict],
    limit: int,
) -> List[Claim]:
    claims: List[Claim] = []
    seq = 1

    # 1. Docs walk.
    for rel in claim_scan_roots:
        abs_p = repo_root / rel
        if abs_p.is_dir():
            iterator = _iter_doc_paths(abs_p, exclude_dirs)
        elif abs_p.is_file():
            iterator = iter([abs_p])
        else:
            continue
        for f in iterator:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel_path = f.relative_to(repo_root).as_posix()
            for cls, cat, excerpt in _classify_text(text):
                claims.append(Claim(
                    claim_id=f"C{seq:05d}",
                    text=excerpt,
                    source=rel_path,
                    category=cat,
                    classification=cls,
                    rationale=f"{cat} pattern matched in doc."
                ))
                seq += 1
                if len(claims) >= limit:
                    return claims

    # 2. Website capture.
    for cap in website_captures:
        src = cap.get("url", "<website>")
        for block in cap.get("blocks", []):
            for cls, cat, excerpt in _classify_text(block.get("text", "")):
                claims.append(Claim(
                    claim_id=f"C{seq:05d}",
                    text=excerpt,
                    source=src,
                    category=cat,
                    classification=cls,
                    rationale=f"{cat} pattern matched in website capture."
                ))
                seq += 1
                if len(claims) >= limit:
                    return claims

    return claims


def write_outputs(claims: List[Claim], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "claims_register.json"
    csv_path = out_dir / "claims_register.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in claims], f, indent=2, ensure_ascii=False)

    cols = ["claim_id", "classification", "category", "source", "line", "text"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for c in claims:
            row = [str(getattr(c, k, "") or "") for k in cols]
            row = [v.replace(",", ";").replace("\n", " ").replace("\r", " ") for v in row]
            f.write(",".join(row) + "\n")

    return {"claims_register_json": str(json_path), "claims_register_csv": str(csv_path)}
