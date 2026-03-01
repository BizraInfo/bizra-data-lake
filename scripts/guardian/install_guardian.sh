#!/bin/bash
# Install BIZRA Infrastructure Guardian as a systemd service
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/bizra-guardian.service"
GUARDIAN_SCRIPT="$SCRIPT_DIR/infra_guardian.py"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== BIZRA Infrastructure Guardian Installer ==="

# Verify files exist
if [ ! -f "$GUARDIAN_SCRIPT" ]; then
    echo "ERROR: infra_guardian.py not found at $GUARDIAN_SCRIPT"
    exit 1
fi

if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: bizra-guardian.service not found at $SERVICE_FILE"
    exit 1
fi

# Quick syntax check
python3 -c "import py_compile; py_compile.compile('$GUARDIAN_SCRIPT', doraise=True)" || {
    echo "ERROR: infra_guardian.py has syntax errors"
    exit 1
}

# Create log directory
mkdir -p /mnt/c/BIZRA-DATA-LAKE/logs/guardian
echo "[OK] Log directory created"

# Install service
cp "$SERVICE_FILE" "$SYSTEMD_DIR/bizra-guardian.service"
systemctl daemon-reload
echo "[OK] Service file installed"

# Enable and start
systemctl enable bizra-guardian.service
systemctl start bizra-guardian.service
echo "[OK] Guardian service enabled and started"

# Verify
sleep 2
if systemctl is-active --quiet bizra-guardian.service; then
    echo "[OK] Guardian is running"
    echo ""
    echo "Commands:"
    echo "  systemctl status bizra-guardian   # Check status"
    echo "  journalctl -u bizra-guardian -f   # Follow logs"
    echo "  python3 $GUARDIAN_SCRIPT --check  # Manual check"
    echo "  python3 $GUARDIAN_SCRIPT --report # JSON report"
else
    echo "[WARN] Guardian failed to start — check: journalctl -u bizra-guardian -e"
    exit 1
fi
