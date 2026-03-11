#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 — Native Linux Production Installer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Installs Node0 onto a canonical Linux host layout for production operation.
# Designed for Debian/Ubuntu 22.04+ (systemd, Python 3.11+).
#
# Standing on Giants:
# - Deming (PDCA, 1950): Plan→Do→Check→Act — installer IS the "Do"
# - Burns (12-Factor App, 2011): strict separation of config, code, state
# - FHS 3.0 (2015): /opt for vendor packages, /etc for config, /var for state
# - Porat/TeleScript (agent identity, 1994): credential carrying at install
#
# Canonical Host Layout:
#   /opt/bizra-node0/                 — code (read-only after install)
#   /etc/bizra-node0/                 — config (operator-managed)
#   /var/lib/bizra-node0/             — mutable state (sovereign_state)
#   /var/log/bizra-node0/             — logs (logrotate-managed)
#   /data/bizra/                      — models/data (optional, operator-created)
#
# Usage:
#   sudo bash installers/install-node0-linux.sh [OPTIONS]
#
# Options:
#   --source DIR        Source directory (default: directory containing this script's parent)
#   --prefix DIR        Install prefix (default: /opt/bizra-node0)
#   --config-dir DIR    Config directory (default: /etc/bizra-node0)
#   --state-dir DIR     State directory (default: /var/lib/bizra-node0)
#   --log-dir DIR       Log directory (default: /var/log/bizra-node0)
#   --user USER         Service user (default: bizra)
#   --group GROUP       Service group (default: bizra)
#   --no-systemd        Skip systemd unit installation
#   --no-venv           Skip virtual environment creation
#   --uninstall         Remove installation (preserves state + config)
#   --dry-run           Show what would be done without executing
#   --help              Show this help
#
# Exit Codes:
#   0 — Installation successful
#   1 — Preflight check failed
#   2 — Missing dependency
#   3 — Permission denied (not root)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────
readonly VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REQUIRED_PYTHON_MAJOR=3
readonly REQUIRED_PYTHON_MINOR=11

# ─── Defaults ─────────────────────────────────────────────────────────────────
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
PREFIX="/opt/bizra-node0"
CONFIG_DIR="/etc/bizra-node0"
STATE_DIR="/var/lib/bizra-node0"
LOG_DIR="/var/log/bizra-node0"
SVC_USER="bizra"
SVC_GROUP="bizra"
INSTALL_SYSTEMD=true
CREATE_VENV=true
DRY_RUN=false
UNINSTALL=false

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── Logging ──────────────────────────────────────────────────────────────────
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ─── Argument Parsing ─────────────────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --source)      SOURCE_DIR="$2"; shift 2 ;;
            --prefix)      PREFIX="$2"; shift 2 ;;
            --config-dir)  CONFIG_DIR="$2"; shift 2 ;;
            --state-dir)   STATE_DIR="$2"; shift 2 ;;
            --log-dir)     LOG_DIR="$2"; shift 2 ;;
            --user)        SVC_USER="$2"; shift 2 ;;
            --group)       SVC_GROUP="$2"; shift 2 ;;
            --no-systemd)  INSTALL_SYSTEMD=false; shift ;;
            --no-venv)     CREATE_VENV=false; shift ;;
            --uninstall)   UNINSTALL=true; shift ;;
            --dry-run)     DRY_RUN=true; shift ;;
            --help)        show_help; exit 0 ;;
            *)             log_error "Unknown option: $1"; exit 1 ;;
        esac
    done
}

show_help() {
    head -39 "${BASH_SOURCE[0]}" | tail -35
}

# ─── Dry-run wrapper ─────────────────────────────────────────────────────────
run() {
    if $DRY_RUN; then
        log_info "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# ─── Preflight ────────────────────────────────────────────────────────────────
preflight() {
    log_step "Preflight checks"

    # Root check
    if [[ $EUID -ne 0 ]] && ! $DRY_RUN; then
        log_error "Must run as root (use sudo)"
        exit 3
    fi

    # Python version
    local py_cmd=""
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" &>/dev/null; then
            py_cmd="$candidate"
            break
        fi
    done

    if [[ -z "$py_cmd" ]]; then
        log_error "Python 3.11+ not found. Install: sudo apt install python3.11"
        exit 2
    fi

    local py_version
    py_version=$($py_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local py_major py_minor
    py_major=$(echo "$py_version" | cut -d. -f1)
    py_minor=$(echo "$py_version" | cut -d. -f2)

    if [[ $py_major -lt $REQUIRED_PYTHON_MAJOR ]] || \
       { [[ $py_major -eq $REQUIRED_PYTHON_MAJOR ]] && [[ $py_minor -lt $REQUIRED_PYTHON_MINOR ]]; }; then
        log_error "Python $py_version found, need >= $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"
        exit 2
    fi
    log_ok "Python $py_version ($py_cmd)"
    PYTHON_CMD="$py_cmd"

    # Source directory
    if [[ ! -f "$SOURCE_DIR/pyproject.toml" ]]; then
        log_error "Source directory missing pyproject.toml: $SOURCE_DIR"
        exit 1
    fi
    log_ok "Source: $SOURCE_DIR"

    # Required source files
    local required_files=("scripts/node0_standalone.py" "core/integration/constants.py" "core/sovereign/node0_authority.py")
    for f in "${required_files[@]}"; do
        if [[ ! -f "$SOURCE_DIR/$f" ]]; then
            log_error "Missing required file: $SOURCE_DIR/$f"
            exit 1
        fi
    done
    log_ok "Required source files present"

    # System dependencies
    for dep in pip3 bash jq; do
        if ! command -v "$dep" &>/dev/null; then
            log_warn "Optional dependency not found: $dep"
        fi
    done

    log_ok "Preflight PASSED"
}

# ─── User/Group Creation ─────────────────────────────────────────────────────
create_service_user() {
    log_step "Service user: $SVC_USER"

    if id "$SVC_USER" &>/dev/null; then
        log_ok "User $SVC_USER already exists"
    else
        run useradd --system --shell /usr/sbin/nologin \
            --home-dir "$STATE_DIR" --create-home \
            --comment "BIZRA Node0 service account" "$SVC_USER"
        log_ok "Created user: $SVC_USER"
    fi
}

# ─── Directory Structure ─────────────────────────────────────────────────────
create_directories() {
    log_step "Creating directory structure"

    # Code directory (root-owned, read-only for service)
    run mkdir -p "$PREFIX"
    run chmod 755 "$PREFIX"

    # Config directory (root-owned, readable by service)
    run mkdir -p "$CONFIG_DIR"
    run chmod 750 "$CONFIG_DIR"
    run chown "root:$SVC_GROUP" "$CONFIG_DIR"

    # State directory (service-owned, writable)
    run mkdir -p "$STATE_DIR"
    run mkdir -p "$STATE_DIR/sovereign_state"
    run mkdir -p "$STATE_DIR/checkpoints"
    run mkdir -p "$STATE_DIR/evidence"
    run chmod 750 "$STATE_DIR"
    run chown -R "$SVC_USER:$SVC_GROUP" "$STATE_DIR"

    # Log directory (service-writable, logrotate-managed)
    run mkdir -p "$LOG_DIR"
    run chmod 750 "$LOG_DIR"
    run chown "$SVC_USER:$SVC_GROUP" "$LOG_DIR"

    log_ok "Directories created"
    log_info "  Code:   $PREFIX"
    log_info "  Config: $CONFIG_DIR"
    log_info "  State:  $STATE_DIR"
    log_info "  Logs:   $LOG_DIR"
}

# ─── Code Installation ───────────────────────────────────────────────────────
install_code() {
    log_step "Installing code to $PREFIX"

    # Copy Python source
    run cp -r "$SOURCE_DIR/core" "$PREFIX/core"
    run cp -r "$SOURCE_DIR/scripts" "$PREFIX/scripts"
    run cp "$SOURCE_DIR/pyproject.toml" "$PREFIX/pyproject.toml"
    run cp "$SOURCE_DIR/README.md" "$PREFIX/README.md"
    run cp "$SOURCE_DIR/RELEASE.md" "$PREFIX/RELEASE.md"

    # Copy docs
    if [[ -d "$SOURCE_DIR/docs" ]]; then
        run cp -r "$SOURCE_DIR/docs" "$PREFIX/docs"
    fi

    # Copy deploy artifacts
    if [[ -d "$SOURCE_DIR/deploy" ]]; then
        run cp -r "$SOURCE_DIR/deploy" "$PREFIX/deploy"
    fi

    # Copy Rust workspace (optional, for building MVSA binary)
    if [[ -d "$SOURCE_DIR/bizra-omega" ]]; then
        run cp -r "$SOURCE_DIR/bizra-omega" "$PREFIX/bizra-omega"
        log_ok "Rust workspace copied"
    fi

    # Copy conftest and test infrastructure
    if [[ -d "$SOURCE_DIR/tests" ]]; then
        run cp -r "$SOURCE_DIR/tests" "$PREFIX/tests"
    fi
    if [[ -f "$SOURCE_DIR/conftest.py" ]]; then
        run cp "$SOURCE_DIR/conftest.py" "$PREFIX/conftest.py"
    fi

    # Set ownership: root owns code, read-only for service
    run chown -R "root:$SVC_GROUP" "$PREFIX"
    run chmod -R 755 "$PREFIX"
    run find "$PREFIX" -type f -exec chmod 644 {} +
    run chmod 755 "$PREFIX/scripts/"*.py "$PREFIX/scripts/"*.sh 2>/dev/null || true

    log_ok "Code installed: $(find "$PREFIX" -name '*.py' | wc -l) Python files"
}

# ─── Virtual Environment ─────────────────────────────────────────────────────
create_virtualenv() {
    if ! $CREATE_VENV; then
        log_info "Skipping virtualenv (--no-venv)"
        return
    fi

    log_step "Creating virtual environment"

    local venv_dir="$PREFIX/.venv"
    run $PYTHON_CMD -m venv "$venv_dir"
    run "$venv_dir/bin/pip" install --quiet --upgrade pip setuptools wheel
    run "$venv_dir/bin/pip" install --quiet -e "$PREFIX"
    run chown -R "root:$SVC_GROUP" "$venv_dir"

    log_ok "Virtual environment: $venv_dir"
    log_info "  Python: $("$venv_dir/bin/python" --version 2>&1 || echo 'unknown')"
}

# ─── Production Config ───────────────────────────────────────────────────────
install_config() {
    log_step "Installing production configuration"

    local env_file="$CONFIG_DIR/node0.env"
    if [[ ! -f "$env_file" ]] || $DRY_RUN; then
        if $DRY_RUN; then
            log_info "[DRY-RUN] Would write config to $env_file"
        else
            cat > "$env_file" <<'ENVEOF'
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 — Production Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Burns (12-Factor App, 2011) — config in environment

# ─── Runtime Mode ─────────────────────────────────────────────────────────────
BIZRA_ENV=production

# ─── Paths (FHS 3.0 compliant) ───────────────────────────────────────────────
BIZRA_DATA_LAKE_ROOT=/opt/bizra-node0
BIZRA_STATE_DIR=/var/lib/bizra-node0
BIZRA_LOG_DIR=/var/log/bizra-node0

# ─── Authentication (REQUIRED in production — Spine §4, L1) ──────────────────
# Generate: python3 -c "import secrets; print(secrets.token_urlsafe(64))"
# BIZRA_JWT_SECRET=CHANGE_ME_BEFORE_FIRST_BOOT
# BIZRA_API_KEYS=key1,key2

# ─── Inference (Spine §3, Helix 1-2) ─────────────────────────────────────────
# LMSTUDIO_HOST=http://127.0.0.1:1234
# OLLAMA_HOST=http://127.0.0.1:11434

# ─── API Server ──────────────────────────────────────────────────────────────
BIZRA_API_HOST=127.0.0.1
BIZRA_API_PORT=8091

# ─── Ghost Bridge (disabled by default in production — Wave 1 security) ──────
GHOST_WS_ENABLED=false

# ─── Constitutional Invariants (informational — cannot override code) ─────────
# IHSAN_THRESHOLD=0.95   (hardcoded in constants.py, not configurable)
# ADL_GINI_THRESHOLD=0.35 (hardcoded in constants.py, not configurable)
ENVEOF
            run chmod 640 "$env_file"
            run chown "root:$SVC_GROUP" "$env_file"
            log_ok "Config written: $env_file"
        fi
    else
        log_warn "Config exists, not overwriting: $env_file"
    fi
}

# ─── Systemd Unit ─────────────────────────────────────────────────────────────
install_systemd() {
    if ! $INSTALL_SYSTEMD; then
        log_info "Skipping systemd (--no-systemd)"
        return
    fi

    log_step "Installing systemd unit"

    local unit_file="/etc/systemd/system/bizra-node0.service"
    cat > "$unit_file" <<UNITEOF
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 — Sovereign Lifecycle Runtime
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Lennart Poettering (systemd, 2010) + Burns (12-Factor, 2011)
#
# Managed by: install-node0-linux.sh v$VERSION
# Layout: FHS 3.0 (/opt code, /etc config, /var state+logs)
[Unit]
Description=BIZRA Node0 Sovereign Lifecycle Runtime
Documentation=file://$PREFIX/docs/NODE0_STANDALONE_READINESS.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SVC_USER
Group=$SVC_GROUP

# Environment
EnvironmentFile=$CONFIG_DIR/node0.env
Environment=PYTHONPATH=$PREFIX
Environment=PATH=$PREFIX/.venv/bin:/usr/local/bin:/usr/bin:/bin

# Execution
ExecStartPre=$PREFIX/.venv/bin/python $PREFIX/scripts/node0_standalone.py health
ExecStart=$PREFIX/.venv/bin/python $PREFIX/scripts/node0_standalone.py serve --host 127.0.0.1 --port 8091
ExecReload=/bin/kill -HUP \$MAINPID

# Security hardening (Spine §4, L2: Constitutional Layer)
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service

# Filesystem access (principle of least privilege)
ReadOnlyPaths=$PREFIX
ReadWritePaths=$STATE_DIR $LOG_DIR

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=4G
CPUQuota=200%

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=300

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bizra-node0

# Watchdog (heartbeat integration — Spine §3, Helix 3)
WatchdogSec=120

[Install]
WantedBy=multi-user.target
UNITEOF

    run chmod 644 "$unit_file"

    # Logrotate
    local logrotate_file="/etc/logrotate.d/bizra-node0"
    cat > "$logrotate_file" <<LREOF
# BIZRA Node0 log rotation
# Standing on Giants: Deming (PDCA, 1950) — log lifecycle management
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 $SVC_USER $SVC_GROUP
    sharedscripts
    postrotate
        systemctl reload bizra-node0 2>/dev/null || true
    endscript
}
LREOF
    run chmod 644 "$logrotate_file"

    # Reload systemd
    run systemctl daemon-reload
    log_ok "Systemd unit installed: $unit_file"
    log_ok "Logrotate config installed: $logrotate_file"
    log_info "  Enable:  sudo systemctl enable bizra-node0"
    log_info "  Start:   sudo systemctl start bizra-node0"
    log_info "  Status:  sudo systemctl status bizra-node0"
    log_info "  Logs:    sudo journalctl -u bizra-node0 -f"
}

# ─── Verification ─────────────────────────────────────────────────────────────
verify_installation() {
    log_step "Verifying installation"

    local checks_passed=0
    local checks_total=0

    # Check directories
    for dir in "$PREFIX" "$CONFIG_DIR" "$STATE_DIR" "$LOG_DIR"; do
        checks_total=$((checks_total + 1))
        if [[ -d "$dir" ]]; then
            checks_passed=$((checks_passed + 1))
            log_ok "Directory exists: $dir"
        else
            log_error "Directory missing: $dir"
        fi
    done

    # Check key files
    for f in "$PREFIX/pyproject.toml" "$PREFIX/scripts/node0_standalone.py" "$PREFIX/core/integration/constants.py"; do
        checks_total=$((checks_total + 1))
        if [[ -f "$f" ]]; then
            checks_passed=$((checks_passed + 1))
            log_ok "File exists: $f"
        else
            log_error "File missing: $f"
        fi
    done

    # Check config
    checks_total=$((checks_total + 1))
    if [[ -f "$CONFIG_DIR/node0.env" ]]; then
        checks_passed=$((checks_passed + 1))
        log_ok "Config exists: $CONFIG_DIR/node0.env"
    else
        log_error "Config missing: $CONFIG_DIR/node0.env"
    fi

    # Check virtualenv
    if $CREATE_VENV; then
        checks_total=$((checks_total + 1))
        if [[ -f "$PREFIX/.venv/bin/python" ]]; then
            checks_passed=$((checks_passed + 1))
            log_ok "Virtual environment: $PREFIX/.venv/bin/python"
        else
            log_error "Virtual environment missing"
        fi
    fi

    # Check systemd
    if $INSTALL_SYSTEMD; then
        checks_total=$((checks_total + 1))
        if [[ -f "/etc/systemd/system/bizra-node0.service" ]]; then
            checks_passed=$((checks_passed + 1))
            log_ok "Systemd unit installed"
        else
            log_error "Systemd unit missing"
        fi
    fi

    echo ""
    if [[ $checks_passed -eq $checks_total ]]; then
        log_ok "Verification: $checks_passed/$checks_total PASSED"
    else
        log_error "Verification: $checks_passed/$checks_total ($(( checks_total - checks_passed )) failed)"
        return 1
    fi
}

# ─── Uninstall ────────────────────────────────────────────────────────────────
do_uninstall() {
    log_step "Uninstalling Node0"
    log_warn "This removes code and systemd unit. Config and state are PRESERVED."

    # Stop service
    if systemctl is-active bizra-node0 &>/dev/null; then
        run systemctl stop bizra-node0
        log_ok "Service stopped"
    fi
    if systemctl is-enabled bizra-node0 &>/dev/null; then
        run systemctl disable bizra-node0
        log_ok "Service disabled"
    fi

    # Remove systemd unit
    run rm -f /etc/systemd/system/bizra-node0.service
    run rm -f /etc/logrotate.d/bizra-node0
    run systemctl daemon-reload

    # Remove code (NOT config or state)
    run rm -rf "$PREFIX"

    log_ok "Uninstall complete"
    log_info "  Preserved config: $CONFIG_DIR"
    log_info "  Preserved state:  $STATE_DIR"
    log_info "  Preserved logs:   $LOG_DIR"
    log_info "  To fully remove:  sudo rm -rf $CONFIG_DIR $STATE_DIR $LOG_DIR"
}

# ─── Post-Install Summary ────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo " BIZRA Node0 — Installation Complete (v$VERSION)"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo " Layout:"
    echo "   Code:   $PREFIX"
    echo "   Config: $CONFIG_DIR/node0.env"
    echo "   State:  $STATE_DIR/sovereign_state"
    echo "   Logs:   $LOG_DIR"
    echo ""
    echo " REQUIRED before first boot:"
    echo "   1. Set JWT secret in $CONFIG_DIR/node0.env"
    echo "      python3 -c \"import secrets; print(secrets.token_urlsafe(64))\""
    echo "   2. Set API keys in $CONFIG_DIR/node0.env"
    echo ""
    echo " Quick start:"
    echo "   sudo systemctl enable --now bizra-node0"
    echo "   sudo journalctl -u bizra-node0 -f"
    echo ""
    echo " Manual commands:"
    echo "   $PREFIX/.venv/bin/python $PREFIX/scripts/node0_standalone.py health"
    echo "   $PREFIX/.venv/bin/python $PREFIX/scripts/node0_standalone.py activate --architect \"MoMo\""
    echo ""
    echo " Genesis ceremony:"
    echo "   sudo -u $SVC_USER bash $PREFIX/scripts/node0_genesis_ceremony.sh --full"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo "═══════════════════════════════════════════════════════════════════════"
    echo " BIZRA Node0 — Native Linux Installer v$VERSION"
    echo " Standing on Giants: Deming · Burns · FHS 3.0 · Porat/TeleScript"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    parse_args "$@"

    if $UNINSTALL; then
        do_uninstall
        exit 0
    fi

    preflight
    create_service_user
    create_directories
    install_code
    create_virtualenv
    install_config
    install_systemd
    verify_installation
    print_summary
}

main "$@"
