#!/bin/bash
# ============================================================================
# BIZRA Unified Node Installer - Linux/Mac One-Click Entry
# ============================================================================
# Run: curl -sSL https://bizra.ai/install | bash
# Or:  ./install.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                          ║${NC}"
echo -e "${BLUE}║     ██████╗ ██╗███████╗██████╗  █████╗                  ║${NC}"
echo -e "${BLUE}║     ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                 ║${NC}"
echo -e "${BLUE}║     ██████╔╝██║  ███╔╝ ██████╔╝███████║                 ║${NC}"
echo -e "${BLUE}║     ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                 ║${NC}"
echo -e "${BLUE}║     ██████╔╝██║███████╗██║  ██║██║  ██║                 ║${NC}"
echo -e "${BLUE}║     ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                 ║${NC}"
echo -e "${BLUE}║                                                          ║${NC}"
echo -e "${BLUE}║           UNIFIED NODE INSTALLER v0.1.0                  ║${NC}"
echo -e "${BLUE}║        Your Gateway to the Decentralized Future          ║${NC}"
echo -e "${BLUE}║                                                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    elif [ -f /etc/arch-release ]; then
        DISTRO="arch"
    else
        DISTRO="unknown"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo -e "${GREEN}[OK]${NC} Detected OS: $OS"

# Set BIZRA home
BIZRA_HOME="$HOME/.bizra"
mkdir -p "$BIZRA_HOME"
echo -e "${GREEN}[OK]${NC} BIZRA home: $BIZRA_HOME"

# Check for Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1)
        echo -e "${GREEN}[OK]${NC} Python found: $PYTHON_VERSION"
        return 0
    elif command -v python &> /dev/null; then
        # Check if it's Python 3
        PY_VER=$(python --version 2>&1)
        if [[ $PY_VER == *"Python 3"* ]]; then
            PYTHON_CMD="python"
            echo -e "${GREEN}[OK]${NC} Python found: $PY_VER"
            return 0
        fi
    fi
    return 1
}

# Install Python if needed
install_python() {
    echo -e "${YELLOW}[!]${NC} Python 3.8+ not found. Installing..."

    if [[ "$OS" == "macos" ]]; then
        # Check for Homebrew
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}[*]${NC} Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python@3.11
    elif [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif [[ "$DISTRO" == "redhat" ]]; then
            sudo dnf install -y python3 python3-pip
        elif [[ "$DISTRO" == "arch" ]]; then
            sudo pacman -Sy python python-pip
        else
            echo -e "${RED}[ERROR]${NC} Unable to install Python automatically."
            echo "Please install Python 3.8+ manually and run this script again."
            exit 1
        fi
    fi
}

# Main installation
main() {
    # Check/install Python
    if ! check_python; then
        install_python
        if ! check_python; then
            echo -e "${RED}[ERROR]${NC} Failed to install Python."
            exit 1
        fi
    fi

    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        echo -e "${YELLOW}[*]${NC} Installing pip..."
        $PYTHON_CMD -m ensurepip --upgrade
    fi
    echo -e "${GREEN}[OK]${NC} pip available"

    # Create virtual environment
    echo -e "${BLUE}[*]${NC} Creating virtual environment..."
    $PYTHON_CMD -m venv "$BIZRA_HOME/venv"
    source "$BIZRA_HOME/venv/bin/activate"
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"

    # Get the installer script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    INSTALLER_SCRIPT="$SCRIPT_DIR/../../bootstrap/install.py"

    if [ -f "$INSTALLER_SCRIPT" ]; then
        echo -e "${BLUE}[*]${NC} Running installer..."
        $PYTHON_CMD "$INSTALLER_SCRIPT"
    else
        # Download from network
        echo -e "${BLUE}[*]${NC} Downloading installer..."
        curl -sSL https://raw.githubusercontent.com/bizra-ai/unified-node/main/bootstrap/install.py -o "$BIZRA_HOME/install.py"
        $PYTHON_CMD "$BIZRA_HOME/install.py"
    fi

    # Create convenience scripts
    cat > "$BIZRA_HOME/start-bizra.sh" << 'EOFSTART'
#!/bin/bash
cd "$HOME/.bizra"
source venv/bin/activate
echo "Starting BIZRA Node..."
python -m bizra.main
EOFSTART
    chmod +x "$BIZRA_HOME/start-bizra.sh"

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Installation Complete!                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Your BIZRA node is ready."
    echo ""
    echo "  To start: ~/.bizra/start-bizra.sh"
    echo "  Dashboard: http://localhost:8888"
    echo ""
    echo "  Welcome to the decentralized AI civilization."
    echo ""
}

# Run main
main "$@"
