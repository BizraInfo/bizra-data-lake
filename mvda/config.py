"""MVDA Configuration — self-contained, no imports from core/."""

import os
from pathlib import Path

# Models
PAT_MODEL = os.getenv("BIZRA_PAT_MODEL", "gemma4:e4b")
SAT_MODEL = os.getenv("BIZRA_SAT_MODEL", "gemma4:26b-bizra-16k")
OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")

# Thresholds
IHSAN_THRESHOLD = 0.95
EVIDENCE_MIN_COUNT = 1

# Paths
DATA_LAKE_ROOT = Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "/data/bizra/repos/bizra-data-lake"))
LEDGER_PATH = Path(os.getenv("BIZRA_LEDGER_PATH", "/data/bizra/logs/mvda-dev-ledger.jsonl"))
LOGS_DIR = Path(os.getenv("BIZRA_LOGS_DIR", "/data/bizra/logs"))

# Constitutional anchors (values only, no imports)
ZANN_ZERO = True
RIBA_ZERO = True
CLAIM_MUST_BIND = True
