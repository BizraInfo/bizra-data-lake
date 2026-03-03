# Phase 53.4: AI-Powered File Classifier

**Status:** SPEC DRAFT | **Script:** `scripts/migration/file_classifier.py`
**Giants:** Shannon (entropy distinguishes text from binary), Bayes (posterior probability for category given features)

---

## Purpose

Classify files in `05_IMPORTS/` into categories and BIZRA sub-projects, producing
a JSONL manifest for the import pipeline (Phase 53.5). Tiered: fast rule-based for
clear cases, local LLM (Ollama) for ambiguous ones.

## Classification Taxonomy

**Categories:** code, document, image, audio, video, model, config, data, archive, unknown
**BIZRA Projects:** data-lake, node0, dual-agentic, genesis, projects, hermes, voice, design, marketing, unaffiliated

## Data Flow

```
  05_IMPORTS/*  -->  Tier 1: Extension match (fast, conf >= 0.8)
                -->  Tier 2: Content sampling (magic bytes, shebangs, structure)
                -->  Tier 3: LLM inference via Ollama (ambiguous files only)
                -->  score_bizra_relevance() + extract_date()
                -->  06_INDEX/classification.jsonl
```

## Pseudocode

```python
"""scripts/migration/file_classifier.py -- AI-powered file classifier."""
from __future__ import annotations
import argparse, json, os, re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BIZRA_SOVEREIGN_ROOT = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

@dataclass
class FileClassification:
    path: str; category: str; subcategory: str; bizra_project: str
    relevance_score: float; date: Optional[str]; suggested_destination: str
    confidence: float; classification_tier: str; size_bytes: int; extension: str

# --- Tier 1: Extension ---
EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".py": ("code","python"), ".rs": ("code","rust"), ".ts": ("code","typescript"),
    ".tsx": ("code","react-tsx"), ".js": ("code","javascript"), ".jsx": ("code","react-jsx"),
    ".go": ("code","go"), ".java": ("code","java"), ".sh": ("code","shell"),
    ".bat": ("code","batch"), ".ps1": ("code","powershell"), ".sql": ("code","sql"),
    ".md": ("document","markdown"), ".txt": ("document","text"), ".pdf": ("document","pdf"),
    ".docx": ("document","word"), ".xlsx": ("document","excel"), ".pptx": ("document","ppt"),
    ".html": ("document","html"),
    ".png": ("image","png"), ".jpg": ("image","jpeg"), ".jpeg": ("image","jpeg"),
    ".gif": ("image","gif"), ".svg": ("image","svg"), ".webp": ("image","webp"),
    ".mp3": ("audio","mp3"), ".wav": ("audio","wav"), ".ogg": ("audio","ogg"),
    ".flac": ("audio","flac"), ".m4a": ("audio","m4a"),
    ".mp4": ("video","mp4"), ".webm": ("video","webm"), ".mkv": ("video","mkv"),
    ".mov": ("video","mov"),
    ".onnx": ("model","onnx"), ".pt": ("model","pytorch"),
    ".safetensors": ("model","safetensors"), ".gguf": ("model","gguf"),
    ".yaml": ("config","yaml"), ".yml": ("config","yaml"), ".toml": ("config","toml"),
    ".env": ("config","env"), ".ini": ("config","ini"),
    ".parquet": ("data","parquet"), ".csv": ("data","csv"), ".jsonl": ("data","jsonl"),
    ".arrow": ("data","arrow"), ".sqlite": ("data","sqlite"),
    ".zip": ("archive","zip"), ".tar": ("archive","tar"), ".gz": ("archive","gzip"),
    ".7z": ("archive","7zip"), ".rar": ("archive","rar"),
}

def classify_by_extension(path: str) -> Optional[tuple[str, str, float]]:
    ext = Path(path).suffix.lower()
    if ext in EXTENSION_MAP:
        cat, sub = EXTENSION_MAP[ext]
        return (cat, sub, 0.9)
    if ext == ".json": return ("config", "json", 0.6)  # Needs Tier 2
    if ext == ".bin":
        try:
            if os.path.getsize(path) > 100_000_000: return ("model", "binary-weights", 0.7)
        except OSError: pass
    return None

# --- Tier 2: Content Sampling ---
def analyze_content(path: str) -> Optional[tuple[str, str, float]]:
    """Read first 4096 bytes: detect binary via null ratio, code via patterns."""
    try:
        with open(path, "rb") as f: sample = f.read(4096)
    except (OSError, PermissionError): return None
    if not sample: return ("unknown", "empty", 0.5)
    null_ratio = sample.count(b"\x00") / len(sample)
    if null_ratio > 0.1:
        if sample[:4] == b"\x89PNG": return ("image", "png", 0.95)
        if sample[:3] == b"\xff\xd8\xff": return ("image", "jpeg", 0.95)
        if sample[:4] == b"GIF8": return ("image", "gif", 0.95)
        return ("binary", "unknown-binary", 0.6)
    try: text = sample.decode("utf-8", errors="replace")
    except Exception: return ("binary", "decode-failed", 0.5)
    if text.startswith("#!"): return ("code", "script-shebang", 0.9)
    if re.search(r"^(import |from .+ import |def |class )", text, re.M):
        return ("code", "python-detected", 0.85)
    if re.search(r"^(fn |pub fn |struct |impl |use |mod )", text, re.M):
        return ("code", "rust-detected", 0.85)
    if re.search(r"^(function |const |let |import .+ from )", text, re.M):
        return ("code", "js-detected", 0.85)
    if text.strip().startswith("{") or text.strip().startswith("["):
        return ("data", "json-detected", 0.7)
    if text.startswith("# ") or re.search(r"^#{1,6} ", text, re.M):
        return ("document", "markdown-detected", 0.8)
    return None

# --- BIZRA Relevance ---
PROJECT_SIGNALS: dict[str, list[str]] = {
    "data-lake": ["bizra-data-lake","data_lake","corpus","vector_engine","parquet","pipeline"],
    "node0": ["bizra-node0","node0","kernel","systemd","proactive"],
    "dual-agentic": ["dual-agentic","multi-agent","consensus","graph-of-thoughts"],
    "genesis": ["genesis-node","genesis_node","blockchain","api_server"],
    "hermes": ["hermes","messaging","integration"],
    "voice": ["bizra-voice","voice","tts","asr","speech"],
    "design": ["design","mockup","wireframe","figma","ui_ux"],
    "marketing": ["marketing","landing","campaign","seo"],
}

def score_bizra_relevance(path: str) -> tuple[str, float]:
    path_lower = path.lower()
    base = 0.5 if "bizra" in path_lower else 0.0
    best_project, best_score = "unaffiliated", 0.0
    for project, signals in PROJECT_SIGNALS.items():
        sig_score = sum(0.3 for s in signals if s in path_lower)
        total = min(base + sig_score, 1.0)
        if total > best_score: best_score, best_project = total, project
    return (best_project, best_score) if best_score > 0 else ("unaffiliated", base)

# --- Date Extraction ---
DATE_PATTERNS = [re.compile(r"(\d{4}-\d{2}-\d{2})"), re.compile(r"(\d{8})(?!\d)")]

def extract_date(path: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.search(Path(path).name)
        if m:
            d = m.group(1)
            try:
                if "-" in d and len(d) == 10: datetime.strptime(d, "%Y-%m-%d"); return d
                if len(d) == 8: return datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError: continue
    try: return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError: return None

# --- Destination Mapping ---
DEST_MAP = {"code": "01_CORE", "document": "03_ASSETS/documents", "image": "03_ASSETS/design",
            "audio": "03_ASSETS/voice", "video": "03_ASSETS/media", "model": "03_ASSETS/models",
            "data": "02_DATA_PIPELINE/00_INTAKE", "archive": "04_ARCHIVE", "unknown": "05_IMPORTS"}

def suggest_destination(fc: FileClassification) -> str:
    base = DEST_MAP.get(fc.category, "05_IMPORTS")
    if fc.category in ("code","config") and fc.bizra_project != "unaffiliated":
        return f"01_CORE/{fc.bizra_project}"
    return base

# --- Tier 3: LLM (optional) ---
def llm_classify(path: str, sample_text: str) -> Optional[tuple[str, str, float]]:
    try: import httpx
    except ImportError: return None
    prompt = (f"Classify this file. Respond ONLY JSON. Filename: {Path(path).name}\n"
              f"Content sample:\n{sample_text[:500]}\n"
              f"Categories: code,document,image,audio,video,model,config,data,archive\n"
              f'Respond: {{"category":"...","subcategory":"...","confidence":0.0-1.0}}')
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate",
                       json={"model":"llama3.2","prompt":prompt,"stream":False}, timeout=30.0)
        if r.status_code == 200:
            m = re.search(r"\{.*\}", r.json().get("response",""), re.DOTALL)
            if m:
                d = json.loads(m.group())
                return (d.get("category","unknown"), d.get("subcategory","llm"), d.get("confidence",0.5))
    except Exception: pass
    return None

# --- Main Pipeline ---
def classify_file(path: str, use_llm: bool = True) -> FileClassification:
    ext = Path(path).suffix.lower()
    try: size = os.path.getsize(path)
    except OSError: size = 0
    # Try tiers in order
    for tier_name, tier_fn in [("extension", lambda: classify_by_extension(path)),
                                ("content", lambda: analyze_content(path))]:
        result = tier_fn()
        if result and result[2] >= 0.7:
            cat, sub, conf = result
            proj, rel = score_bizra_relevance(path)
            fc = FileClassification(path, cat, sub, proj, rel, extract_date(path),
                                    "", conf, tier_name, size, ext)
            fc.suggested_destination = suggest_destination(fc); return fc
    if use_llm:
        try:
            with open(path, "r", errors="replace") as f: sample = f.read(500)
        except Exception: sample = ""
        t3 = llm_classify(path, sample)
        if t3:
            cat, sub, conf = t3; proj, rel = score_bizra_relevance(path)
            fc = FileClassification(path, cat, sub, proj, rel, extract_date(path),
                                    "", conf, "llm", size, ext)
            fc.suggested_destination = suggest_destination(fc); return fc
    proj, rel = score_bizra_relevance(path)
    fc = FileClassification(path, "unknown", "unclassified", proj, rel, extract_date(path),
                            "", 0.1, "fallback", size, ext)
    fc.suggested_destination = suggest_destination(fc); return fc

def classify_directory(input_dir: str, use_llm: bool = True) -> list[FileClassification]:
    results = []
    for dp, _, fns in os.walk(input_dir, followlinks=False):
        for f in fns: results.append(classify_file(os.path.join(dp, f), use_llm))
    return results

def write_classifications(results: list[FileClassification], output_path: str) -> None:
    with open(output_path, "w") as f:
        for fc in results: f.write(json.dumps(asdict(fc), default=str) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.4: File Classifier")
    parser.add_argument("--input", default=os.path.join(BIZRA_SOVEREIGN_ROOT, "05_IMPORTS"))
    parser.add_argument("--output", default=os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX", "classification.jsonl"))
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    results = classify_directory(args.input, use_llm=not args.no_llm)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    write_classifications(results, args.output)
    from collections import Counter
    cats = Counter(r.category for r in results)
    print(f"Classified {len(results)} files: {dict(cats.most_common())}")

if __name__ == "__main__":
    main()
```

## TDD Anchors

```python
"""tests/migration/test_file_classifier.py"""
import json, os
from pathlib import Path
import pytest

class TestExtensionClassification:
    def test_python(self) -> None:
        r = classify_by_extension("/src/main.py")
        assert r and r[0] == "code" and r[1] == "python" and r[2] >= 0.8
    def test_parquet(self) -> None:
        assert classify_by_extension("/data/out.parquet")[0] == "data"
    def test_png(self) -> None:
        assert classify_by_extension("/img/logo.png")[0] == "image"
    def test_unknown(self) -> None:
        assert classify_by_extension("/file.xyz123") is None

class TestBizraRelevance:
    def test_data_lake(self) -> None:
        proj, score = score_bizra_relevance("/mnt/c/BIZRA-DATA-LAKE/core/main.py")
        assert proj == "data-lake" and score > 0.5
    def test_non_bizra(self) -> None:
        proj, score = score_bizra_relevance("/tmp/random.txt")
        assert proj == "unaffiliated" and score == 0.0
    def test_genesis(self) -> None:
        proj, _ = score_bizra_relevance("/mnt/c/bizra-genesis-node/src/main.rs")
        assert proj == "genesis"

class TestDateExtraction:
    def test_iso_date(self) -> None:
        assert extract_date("/docs/report_2024-03-15.pdf") == "2024-03-15"
    def test_compact_date(self) -> None:
        assert extract_date("/backup/snap_20240315.tar.gz") == "2024-03-15"
    def test_mtime_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "nodate.txt"; f.write_text("x")
        assert extract_date(str(f)) is not None

class TestJsonlOutput:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        results = [FileClassification("/a.py","code","python","data-lake",0.8,
                   "2024-03-15","01_CORE/data-lake",0.9,"extension",1024,".py")]
        out = str(tmp_path / "test.jsonl"); write_classifications(results, out)
        record = json.loads(open(out).readline())
        assert record["category"] == "code" and record["bizra_project"] == "data-lake"

class TestFallbackRules:
    def test_unknown_file(self, tmp_path: Path) -> None:
        f = tmp_path / "mystery.xyz"; f.write_bytes(os.urandom(100))
        r = classify_file(str(f), use_llm=False)
        assert r.confidence <= 0.5

class TestContentAnalysis:
    def test_python_detection(self, tmp_path: Path) -> None:
        f = tmp_path / "noext"; f.write_text("import os\ndef main():\n    pass\n")
        r = analyze_content(str(f))
        assert r and r[0] == "code"
```
