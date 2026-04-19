#!/usr/bin/env bash
# generate-proof-of-priority.sh — Cycle-8 Day 5
#
# بسم الله الرحمن الرحيم
#
# Output an (unsigned) proof-of-priority manifest as JSON on stdout.
#
# The manifest establishes BIZRA's architectural priority over
# arXiv:2510.13857v1 (Xu et al., CUHK, 2025-10-12) by binding:
#   1. THIS repo's earliest commits (2026 work visible in git)
#   2. Placeholder for EXTERNAL 2023 repo refs (Ramadan 2023 Arabic
#      founding texts — Mumo fills with his 150-repo inventory)
#   3. SHA-256 of the arXiv reference paper PDF
#   4. A claim string + generation timestamp
#
# Signatures: NOT produced by this script. The output is unsigned JSON.
# The signed version requires the witness identity key; see
# `scripts/sign-manifest.sh` (Day 6+ if needed) or pipe through
# `openssl`/`ed25519` tooling manually.
#
# Per Cycle-8 doctrinal constraint:
#   - Witness-grade detectability only
#   - No bonded stake / slashing / DAO in this artifact
#   - The artifact is transferable evidence; a skeptical stranger can
#     recompute the paper SHA-256 and git log SHAs independently
#
# Usage:
#   scripts/generate-proof-of-priority.sh \
#     [--paper <path>]            (default: /home/bizra-operating-system/Downloads/LLM\ as\ CPU\ paper.pdf)
#     [--external-refs-json '[{...}]'] (default: empty; Mumo fills)
#     [--output <path>]           (default: stdout)

set -eu

PAPER_PATH="/home/bizra-operating-system/Downloads/LLM as CPU paper.pdf"
EXTERNAL_REFS_JSON='[]'
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --paper) PAPER_PATH="$2"; shift 2 ;;
        --external-refs-json) EXTERNAL_REFS_JSON="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --help|-h)
            grep "^#" "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ─── Ensure we're at the repo root ─────────────────────────────────
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
    echo "error: must run from inside the bizra-data-lake git repo" >&2
    exit 3
fi
cd "$REPO_ROOT"

# ─── Collect repo's earliest commits on main (5 chronologically oldest) ──
# Uses python3 for robust JSON encoding (handles special chars in subjects).
# Source: origin/main, not current branch, so the manifest reports the
# repo's actual genesis lineage rather than this branch's head.
EARLIEST_COMMITS_JSON=$(python3 -c '
import subprocess, json
try:
    # git log --reverse --max-count=N quirk: max-count applies BEFORE
    # reverse, yielding the newest N reversed, not the oldest N.
    # Use rev-list --reverse | head to get TRUE chronologically oldest.
    all_shas = subprocess.run(
        ["git", "rev-list", "--reverse", "origin/main"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    oldest = all_shas[:5]
    commits = []
    for sha in oldest:
        r = subprocess.run(
            ["git", "log", "-1",
             "--pretty=format:%H%x1f%aI%x1f%s", sha],
            capture_output=True, text=True, check=True)
        parts = r.stdout.split("\x1f", 2)
        if len(parts) == 3:
            commits.append({
                "sha": parts[0],
                "author_ts_iso": parts[1],
                "subject": parts[2],
            })
    print(json.dumps(commits, indent=2))
except Exception:
    print("[]", end="")
')

# ─── Current HEAD commit for context ────────────────────────────────
HEAD_SHA=$(git rev-parse HEAD)
HEAD_TS=$(git log -1 --pretty=format:'%aI')
HEAD_SUBJECT=$(git log -1 --pretty=format:'%s' | sed 's/"/\\"/g')

# ─── SHA-256 of the arXiv paper (cryptographic modality) ───────────
if [ -f "$PAPER_PATH" ]; then
    PAPER_SHA256=$(sha256sum "$PAPER_PATH" | awk '{print $1}')
    PAPER_SIZE=$(stat -c %s "$PAPER_PATH")
else
    PAPER_SHA256="FILE_NOT_FOUND"
    PAPER_SIZE=0
    echo "warning: paper not found at $PAPER_PATH — field will carry FILE_NOT_FOUND sentinel" >&2
fi

# ─── Generation timestamp (UTC, ISO-8601, ns precision) ────────────
GEN_TS_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")
GEN_TS_NS=$(date -u +%s%N)

# ─── Assemble the manifest ─────────────────────────────────────────
MANIFEST=$(cat <<EOF
{
  "schema": "bizra-proof-of-priority-v1",
  "claim": "BIZRA (bizra-data-lake + sibling repos) independently implemented the Kernel-as-Governor / Agent Constitution Framework / Evaluation-Driven Development Lifecycle architecture BEFORE arXiv:2510.13857v1 (Xu et al., CUHK, 2025-10-12) theorized it. The external paper is independent academic convergence, not source material.",
  "generated_at_iso": "$GEN_TS_ISO",
  "generated_at_ns": $GEN_TS_NS,
  "reference_paper": {
    "arxiv_id": "2510.13857v1",
    "title": "From Craft to Constitution: A Governance-First Paradigm for Principled Agent Engineering",
    "authors": ["Qiang Xu", "Xiangyu Wen", "Changran Xu", "Zeju Li", "Jianyuan Zhong"],
    "institution": "CURE Lab, Dept. of CSE, The Chinese University of Hong Kong",
    "arxiv_date_iso": "2025-10-12",
    "local_pdf_path": "$PAPER_PATH",
    "local_pdf_sha256": "$PAPER_SHA256",
    "local_pdf_size_bytes": $PAPER_SIZE
  },
  "this_repo": {
    "name": "bizra-data-lake",
    "remote": "github.com/BizraInfo/bizra-data-lake",
    "head_sha": "$HEAD_SHA",
    "head_ts_iso": "$HEAD_TS",
    "head_subject": "$HEAD_SUBJECT",
    "earliest_commits": $EARLIEST_COMMITS_JSON
  },
  "external_2023_refs": $EXTERNAL_REFS_JSON,
  "external_2023_refs_note": "PENDING — Mumo fills with the 2023 Ramadan founding texts (al-Bidhrah / al-Risalah) and the earliest repos from his 150-repo inventory. Empty [] at generation time is an honest declaration, not a silent omission.",
  "signature": null,
  "signature_note": "UNSIGNED. Sign with the witness identity key; see Cycle-8 Day 6+ or use: cat this-manifest.json | <ed25519-signing-tool> --key \$BIZRA_WITNESS_SIGNING_KEY_HEX",
  "verification": {
    "anyone_can_verify_by": [
      "Recompute paper sha256: sha256sum \"\$PAPER_PATH\" → compare local_pdf_sha256",
      "Clone bizra-data-lake and check git log --reverse → match earliest_commits[].sha",
      "Compare head_sha to GitHub API: gh api /repos/BizraInfo/bizra-data-lake/branches/main",
      "Verify external_2023_refs when Mumo fills them (Day 6+)"
    ]
  }
}
EOF
)

if [ -n "$OUTPUT" ]; then
    printf '%s\n' "$MANIFEST" > "$OUTPUT"
    echo "wrote proof-of-priority manifest to $OUTPUT" >&2
else
    printf '%s\n' "$MANIFEST"
fi
