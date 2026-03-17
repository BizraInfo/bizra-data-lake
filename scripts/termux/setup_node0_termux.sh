#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════
# BIZRA Node0 — Termux Setup Script
# Every seed has infinite potential. ربي لا يعرف المستحيل
# ═══════════════════════════════════════════════════════════

set -e

BIZRA_HOME="$HOME/bizra"
NODE_USER="${1:-mumo}"
IHSAN_FLOOR="${2:-9500}"

echo "═══════════════════════════════════════════════════════"
echo "  BIZRA Node0 — Android Sovereign Setup"
echo "  User: $NODE_USER | Ihsan: $IHSAN_FLOOR"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: System packages ──
echo "[1/6] Installing system packages..."
pkg update -y
pkg upgrade -y
pkg install -y git rust openssl pkg-config

# ── Step 2: Clone repo ──
echo "[2/6] Cloning BIZRA repository..."
mkdir -p "$BIZRA_HOME"
if [ -d "$BIZRA_HOME/bizra-data-lake" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$BIZRA_HOME/bizra-data-lake"
    git pull --ff-only 2>/dev/null || echo "  Pull skipped (detached or conflict)"
else
    cd "$BIZRA_HOME"
    git clone --depth 1 https://github.com/BizraInfo/bizra-data-lake.git
fi

# ── Step 3: Build bizra-node ──
echo "[3/6] Building bizra-node (this takes 5-15 min on first run)..."
cd "$BIZRA_HOME/bizra-data-lake/bizra-omega"
cargo build --release -p bizra-node 2>&1 | tail -5

BINARY="$BIZRA_HOME/bizra-data-lake/bizra-omega/target/release/bizra-node"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Build failed — binary not found"
    exit 1
fi
echo "  Binary: $(ls -lh "$BINARY" | awk '{print $5}') at $BINARY"

# ── Step 4: Create state directory ──
echo "[4/6] Setting up state directory..."
STATE_DIR="$HOME/.bizra/node-$NODE_USER"
mkdir -p "$STATE_DIR"
echo "  State: $STATE_DIR"

# ── Step 5: Create boot script ──
echo "[5/6] Creating boot script..."
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/bizra-node.sh << BOOTEOF
#!/data/data/com.termux/files/usr/bin/bash
# BIZRA Node0 — Auto-start on boot
cd $BIZRA_HOME/bizra-data-lake/bizra-omega
nohup ./target/release/bizra-node \\
  --user $NODE_USER \\
  --ihsan $IHSAN_FLOOR \\
  --reflex-mode active \\
  --action-mode active \\
  --state-dir $STATE_DIR \\
  > $STATE_DIR/stdout.log 2>&1 &
echo \$! > $STATE_DIR/node.pid
BOOTEOF
chmod +x ~/.termux/boot/bizra-node.sh

# ── Step 6: Create helper scripts ──
echo "[6/6] Creating helper scripts..."

# Start script
cat > "$BIZRA_HOME/start.sh" << STARTEOF
#!/data/data/com.termux/files/usr/bin/bash
cd $BIZRA_HOME/bizra-data-lake/bizra-omega
./target/release/bizra-node \\
  --user $NODE_USER \\
  --ihsan $IHSAN_FLOOR \\
  --reflex-mode active \\
  --action-mode active \\
  --state-dir $STATE_DIR
STARTEOF
chmod +x "$BIZRA_HOME/start.sh"

# Start background script
cat > "$BIZRA_HOME/start-bg.sh" << BGEOF
#!/data/data/com.termux/files/usr/bin/bash
cd $BIZRA_HOME/bizra-data-lake/bizra-omega
nohup ./target/release/bizra-node \\
  --user $NODE_USER \\
  --ihsan $IHSAN_FLOOR \\
  --reflex-mode active \\
  --action-mode active \\
  --mcp-port 9600 \\
  --state-dir $STATE_DIR \\
  > $STATE_DIR/stdout.log 2>&1 &
PID=\$!
echo \$PID > $STATE_DIR/node.pid
echo "Node0 started (PID: \$PID, MCP: localhost:9600)"
echo "Logs: $STATE_DIR/stdout.log"
BGEOF
chmod +x "$BIZRA_HOME/start-bg.sh"

# Stop script
cat > "$BIZRA_HOME/stop.sh" << STOPEOF
#!/data/data/com.termux/files/usr/bin/bash
PID_FILE="$STATE_DIR/node.pid"
if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    kill "\$PID" 2>/dev/null && echo "Node0 stopped (PID: \$PID)" || echo "Process not running"
    rm -f "\$PID_FILE"
else
    echo "No PID file found"
    pkill -f bizra-node 2>/dev/null && echo "Killed bizra-node" || echo "Not running"
fi
STOPEOF
chmod +x "$BIZRA_HOME/stop.sh"

# Status script
cat > "$BIZRA_HOME/status.sh" << STATUSEOF
#!/data/data/com.termux/files/usr/bin/bash
PID_FILE="$STATE_DIR/node.pid"
echo "═══════════════════════════════════════════"
echo "  BIZRA Node0 — Android Status"
echo "═══════════════════════════════════════════"
if [ -f "\$PID_FILE" ] && kill -0 \$(cat "\$PID_FILE") 2>/dev/null; then
    echo "  Status:  RUNNING (PID: \$(cat "\$PID_FILE"))"
else
    echo "  Status:  STOPPED"
fi
echo "  User:    $NODE_USER"
echo "  Ihsan:   $IHSAN_FLOOR"
echo "  State:   $STATE_DIR"
if [ -f "$STATE_DIR/knowledge.seed" ]; then
    LINES=\$(wc -l < "$STATE_DIR/knowledge.seed")
    echo "  Knowledge: \$LINES entries"
else
    echo "  Knowledge: empty (teach me!)"
fi
if [ -f "$STATE_DIR/actions.log" ]; then
    ACTIONS=\$(wc -l < "$STATE_DIR/actions.log")
    echo "  Actions: \$ACTIONS receipts"
fi
echo "═══════════════════════════════════════════"
STATUSEOF
chmod +x "$BIZRA_HOME/status.sh"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Commands:"
echo "    ~/bizra/start.sh      — Interactive mode (stdin)"
echo "    ~/bizra/start-bg.sh   — Background + MCP on :9600"
echo "    ~/bizra/stop.sh       — Stop background node"
echo "    ~/bizra/status.sh     — Check status"
echo ""
echo "  Interactive commands (after start.sh):"
echo "    PING                  — Keepalive"
echo "    HEALTH                — System health"
echo "    RECEIVE<TAB>msg<TAB>ts — Process message"
echo "    TEACH<TAB>fact<TAB>content<TAB>0.99<TAB>ts"
echo "    KNOWS_ME              — AI knows me score"
echo "    SHUTDOWN              — Graceful stop"
echo ""
echo "  Auto-start: Install Termux:Boot from F-Droid"
echo "  State dir:  $STATE_DIR"
echo ""
echo "  Every seed has infinite potential."
echo "  ربي لا يعرف المستحيل"
echo "═══════════════════════════════════════════════════════"
