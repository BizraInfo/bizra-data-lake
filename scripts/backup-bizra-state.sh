#!/bin/bash
# ============================================================================
# BIZRA Node0 — State Backup (Pre-Migration)
# ============================================================================
# Archives all non-git BIZRA state to a portable tar.gz.
# Run this BEFORE installing Linux dual boot.
#
# Usage:
#   bash scripts/backup-bizra-state.sh [output_dir]
#
# Default output: ~/bizra-backup-YYYY-MM-DD.tar.gz
# ============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_DIR="${1:-$HOME}"
BACKUP_NAME="bizra-backup-${TIMESTAMP}"
STAGING_DIR="/tmp/${BACKUP_NAME}"

echo "========================================="
echo "  BIZRA State Backup — Pre-Migration"
echo "========================================="
echo ""

# Create staging directory
mkdir -p "${STAGING_DIR}"

# 1. ~/.bizra (URP state, invites, SEED, chain head, daemon state)
if [ -d "$HOME/.bizra" ]; then
    echo "[1/6] Backing up ~/.bizra ..."
    cp -r "$HOME/.bizra" "${STAGING_DIR}/dot-bizra"
    echo "  Done: $(find "${STAGING_DIR}/dot-bizra" -type f | wc -l) files"
else
    echo "[1/6] SKIP: ~/.bizra not found"
fi

# 2. .env (secrets — NOT in git)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    echo "[2/6] Backing up .env ..."
    cp "${REPO_ROOT}/.env" "${STAGING_DIR}/dot-env"
    echo "  Done"
else
    echo "[2/6] SKIP: .env not found"
fi

# 3. Config files
if [ -d "${REPO_ROOT}/config" ]; then
    echo "[3/6] Backing up config/ ..."
    cp -r "${REPO_ROOT}/config" "${STAGING_DIR}/config"
    echo "  Done: $(find "${STAGING_DIR}/config" -type f | wc -l) files"
else
    echo "[3/6] SKIP: config/ not found"
fi

# 4. 04_GOLD pipeline output (parquet, FAISS index, embeddings)
if [ -d "${REPO_ROOT}/04_GOLD" ]; then
    echo "[4/6] Backing up 04_GOLD/ ..."
    cp -r "${REPO_ROOT}/04_GOLD" "${STAGING_DIR}/04_GOLD"
    GOLD_SIZE=$(du -sh "${STAGING_DIR}/04_GOLD" | cut -f1)
    echo "  Done: ${GOLD_SIZE}"
else
    echo "[4/6] SKIP: 04_GOLD/ not found"
fi

# 5. Sovereign state (PID files, logs)
if [ -d "${REPO_ROOT}/sovereign_state" ]; then
    echo "[5/6] Backing up sovereign_state/ ..."
    cp -r "${REPO_ROOT}/sovereign_state" "${STAGING_DIR}/sovereign_state"
    echo "  Done"
else
    echo "[5/6] SKIP: sovereign_state/ not found"
fi

# 6. Record environment variables
echo "[6/6] Recording environment snapshot ..."
{
    echo "# BIZRA Environment Snapshot — ${TIMESTAMP}"
    echo "# Captured before migration to native Linux"
    echo ""
    echo "# System"
    echo "HOSTNAME=$(hostname)"
    echo "KERNEL=$(uname -r)"
    echo "PYTHON=$(python3 --version 2>&1)"
    echo "RUST=$(rustc --version 2>&1 || echo 'not installed')"
    echo ""
    echo "# BIZRA-specific env vars"
    env | grep -iE "^(BIZRA|LM_|OLLAMA|LMSTUDIO|GITHUB)" | sort || true
    echo ""
    echo "# Paths"
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "HOME=${HOME}"
    echo ""
    echo "# Git state"
    echo "BRANCH=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo 'unknown')"
    echo "COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "TAG=$(git -C "${REPO_ROOT}" describe --tags --abbrev=0 2>/dev/null || echo 'none')"
} > "${STAGING_DIR}/env-snapshot.txt"
echo "  Done"

# Create tar.gz
echo ""
echo "Creating archive ..."
ARCHIVE="${OUTPUT_DIR}/${BACKUP_NAME}.tar.gz"
tar -czf "${ARCHIVE}" -C /tmp "${BACKUP_NAME}"

# Cleanup staging
rm -rf "${STAGING_DIR}"

# Summary
ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
echo ""
echo "========================================="
echo "  Backup complete!"
echo "========================================="
echo "  Archive: ${ARCHIVE}"
echo "  Size:    ${ARCHIVE_SIZE}"
echo ""
echo "  Copy this file to a USB drive before"
echo "  installing Linux dual boot."
echo ""
echo "  To restore on native Linux:"
echo "    tar -xzf ${BACKUP_NAME}.tar.gz"
echo "    cp -r ${BACKUP_NAME}/dot-bizra ~/.bizra"
echo "    cp ${BACKUP_NAME}/dot-env ~/bizra-data-lake/.env"
echo "    cp -r ${BACKUP_NAME}/04_GOLD ~/bizra-data-lake/"
echo "    cp -r ${BACKUP_NAME}/config ~/bizra-data-lake/"
echo "========================================="
