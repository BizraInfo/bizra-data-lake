import hashlib
import json
from pathlib import Path

def iter_ledger(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            yield json.loads(line)

def compute(genesis_hash, ledger_path):
    h = hashlib.sha256()
    h.update(genesis_hash.encode("utf-8"))
    h.update(b"\0")
    count = 0
    for rec in iter_ledger(ledger_path):
        fh = rec.get("hash")
        if isinstance(fh, str) and fh:
             h.update(fh.encode("utf-8"))
             h.update(b"\0")
             count += 1
    return h.hexdigest(), count

manifest = json.loads(Path("BIZRA_KNOWLEDGE_MANIFEST.json").read_text(encoding="utf-8"))
genesis = manifest.get("genesis_link", "").strip()
print(f"Genesis: {genesis}")

h, c = compute(genesis, "BIZRA_KNOWLEDGE_LEDGER.jsonl")
print(f"Computed Hash: {h}")
print(f"Count: {c}")
