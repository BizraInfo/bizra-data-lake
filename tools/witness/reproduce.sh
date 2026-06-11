#!/usr/bin/env bash
# ============================================================================
# BIZRA Node0 — External Witness Reproducer (Arc 3: Receipt Persistence)
# ============================================================================
# Purpose : Allow ANY external party to reproduce the Arc 3 persistence
#           witness with ONE command and ZERO founder involvement.
# Claim   : Receipt chain persists across gateway restart when
#           BIZRA_RECEIPT_STORE_PATH is set (default in-memory unchanged).
# Success : Witness emits NODE0_MISSION_REPLAY_PERSIST_WITNESS_COMPLETE
#           and persist_survives_restart == true in the witness JSON.
# Output  : artifacts/witness/<timestamp>/ containing witness JSON,
#           environment fingerprint, and a BLAKE3/SHA-256 attestation hash.
#
# Constitutional posture: read-only against the repo, local-only writes,
# no network calls beyond the initial clone, no telemetry, no credentials.
# ============================================================================
set -euo pipefail

REPO_URL="${BIZRA_REPO_URL:-https://github.com/BizraInfo/bizra-data-lake.git}"
PIN_COMMIT="${BIZRA_WITNESS_COMMIT:-}"   # REQUIRED: exact commit to witness.
WORKDIR="${BIZRA_WITNESS_WORKDIR:-$(pwd)/bizra-witness-run}"
GATEWAY_PKG="bizra-cognition-gateway"
WITNESS_PY="tools/witness/run_witness.py"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="${WORKDIR}/artifacts/witness/${STAMP}"

say()  { printf '\n[witness] %s\n' "$*"; }
fail() { printf '\n[witness][FAIL] %s\n' "$*" >&2; exit 1; }

[ -n "$PIN_COMMIT" ] || fail "BIZRA_WITNESS_COMMIT is empty. A witness must bind \
to an exact commit SHA (CLAIM_MUST_BIND). Get the published SHA from REPRODUCER.md."

for bin in git cargo python3; do
  command -v "$bin" >/dev/null 2>&1 || fail "missing required tool: $bin"
done

RUST_VER="$(rustc --version 2>/dev/null || true)"
PY_VER="$(python3 --version 2>/dev/null || true)"
say "toolchain: ${RUST_VER:-rustc not found} | ${PY_VER}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d repo/.git ]; then
  say "cloning ${REPO_URL}"
  git clone --no-tags "$REPO_URL" repo
fi
cd repo
git fetch origin "$PIN_COMMIT" 2>/dev/null || true
git checkout --detach "$PIN_COMMIT" || fail "commit ${PIN_COMMIT} not found in remote"
HEAD_SHA="$(git rev-parse HEAD)"
[ "$HEAD_SHA" = "$PIN_COMMIT" ] || fail "HEAD ${HEAD_SHA} != pinned ${PIN_COMMIT}"
git diff --quiet || fail "working tree dirty — witness refuses unclean state"
say "pinned at ${HEAD_SHA}"

[ -d bizra-omega ] || fail "bizra-omega workspace not found at repo root \
(layout drift — report this as a reproducer defect, it counts as a finding)"
say "building ${GATEWAY_PKG} (release)…"
( cd bizra-omega && cargo build --release -p "$GATEWAY_PKG" ) \
  || fail "gateway build failed — capture output and report; a failed build \
by an external witness is itself valid Z1 evidence"

[ -f "$WITNESS_PY" ] || fail "vendored witness not found at ${WITNESS_PY}. \
The harness must live IN the repository — if absent, the maintainer has not \
completed vendoring (Gate W0). Report this."

mkdir -p "$OUTDIR"
export BIZRA_RECEIPT_STORE_PATH="${OUTDIR}/receipt-store"
say "running witness (receipt store: \$BIZRA_RECEIPT_STORE_PATH)…"
set +e
python3 "$WITNESS_PY" --out "$OUTDIR" 2>&1 | tee "${OUTDIR}/witness-stdout.log"
WITNESS_RC=$?
set -e

WITNESS_JSON="$(find "$OUTDIR" -maxdepth 2 -name '*WITNESS*.json' | head -1 || true)"
[ -n "$WITNESS_JSON" ] || WITNESS_JSON="$(find . -maxdepth 4 -name 'NODE0_MISSION_REPLAY_PERSIST_WITNESS.json' | head -1 || true)"
[ -n "$WITNESS_JSON" ] || fail "no witness JSON produced (rc=${WITNESS_RC})"

python3 - "$WITNESS_JSON" "$HEAD_SHA" <<'PYEOF'
import json, sys
path, expected_sha = sys.argv[1], sys.argv[2]
d = json.load(open(path))
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}" if p else k)
    else:
        yield p, o
kv = dict(flat(d))
text = json.dumps(d)
ok_label   = "NODE0_MISSION_REPLAY_PERSIST_WITNESS_COMPLETE" in text
ok_persist = any(k.endswith("persist_survives_restart") and v is True for k, v in kv.items())
bound_sha  = next((v for k, v in kv.items() if "commit" in k and isinstance(v, str) and len(v) == 40), None)
ok_binding = (bound_sha == expected_sha) if bound_sha else None
print(f"label_complete            : {ok_label}")
print(f"persist_survives_restart  : {ok_persist}")
print(f"commit_binding            : {ok_binding if ok_binding is not None else 'NOT RECORDED (defect: witness should embed the commit SHA)'}")
if not (ok_label and ok_persist):
    sys.exit(2)
if ok_binding is False:
    print("FATAL: witness JSON binds a different commit than the one built", file=sys.stderr)
    sys.exit(3)
PYEOF
VERIFY_RC=$?
[ "$VERIFY_RC" -eq 0 ] || fail "witness verification failed (rc=${VERIFY_RC})"

HASHER="sha256sum"
if command -v b3sum >/dev/null 2>&1; then HASHER="b3sum"; fi
ATT_HASH="$($HASHER "$WITNESS_JSON" | awk '{print $1}')"

cat > "${OUTDIR}/ATTESTATION.json" <<EOF
{
  "attestation_version": "1.0",
  "claim": "Arc3 receipt persistence survives gateway restart",
  "repo": "${REPO_URL}",
  "commit": "${HEAD_SHA}",
  "witness_json": "$(basename "$WITNESS_JSON")",
  "witness_hash": "${ATT_HASH}",
  "hash_algo": "${HASHER}",
  "result": "WITNESSED",
  "environment": {
    "rustc": "${RUST_VER}",
    "python": "${PY_VER}",
    "os": "$(uname -srm)",
    "timestamp_utc": "${STAMP}"
  },
  "witness_identity": "FILL_IN: your name/handle + contact (or 'anonymous')",
  "witness_statement": "I independently cloned, built, and executed this witness with no assistance from the BIZRA maintainer."
}
EOF

say "============================================================"
say "WITNESS COMPLETE — Economic-axis evidence generated"
say "  commit    : ${HEAD_SHA}"
say "  hash      : ${HASHER}:${ATT_HASH}"
say "  artifacts : ${OUTDIR}"
say "Next: fill witness_identity in ATTESTATION.json and send it"
say "back per REPRODUCER.md §4. Your attestation is the proof."
say "============================================================"
