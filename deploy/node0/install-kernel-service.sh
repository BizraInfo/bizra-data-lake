#!/bin/bash
# BIZRA Sovereign Kernel — systemd service installer
# Standing on Giants: Lennart Poettering (systemd) · Deming (PDCA auto-start)
#
# Usage: sudo bash deploy/node0/install-kernel-service.sh

set -euo pipefail

SERVICE_NAME="bizra-kernel"
SERVICE_FILE="$(dirname "$0")/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)"
    exit 1
fi

if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: Service file not found: $SERVICE_FILE"
    exit 1
fi

echo "Installing ${SERVICE_NAME}.service..."
cp "$SERVICE_FILE" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
echo ""
echo "Installed. Commands:"
echo "  sudo systemctl start  ${SERVICE_NAME}   # Start now"
echo "  sudo systemctl status ${SERVICE_NAME}   # Check status"
echo "  sudo systemctl stop   ${SERVICE_NAME}   # Stop"
echo "  journalctl -u ${SERVICE_NAME} -f        # Follow logs"
echo ""
echo "The kernel will auto-start on next WSL boot."
echo "Dashboard: http://127.0.0.1:9740"
