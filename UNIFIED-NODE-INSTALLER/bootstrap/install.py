#!/usr/bin/env python3
"""
BIZRA Unified Node Installer - Bootstrap Script
Cross-platform, zero-dependency installer entry point.

This script is designed to work on ANY system with Python 3.8+
It bootstraps the full installation process.

Usage:
    python install.py              # Interactive mode
    python install.py --headless   # Non-interactive with defaults
    python install.py --config profile.yaml  # Pre-configured install
"""

import os
import sys
import json
import platform
import subprocess
import urllib.request
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "0.1.0-alpha"
BIZRA_HOME = Path.home() / ".bizra"
CONFIG_FILE = BIZRA_HOME / "config.json"
DATA_DIR = BIZRA_HOME / "data"
MODELS_DIR = BIZRA_HOME / "models"
LOGS_DIR = BIZRA_HOME / "logs"

# Minimum requirements
MIN_RAM_GB = 4
MIN_DISK_GB = 10
RECOMMENDED_RAM_GB = 8
RECOMMENDED_DISK_GB = 50

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

class SystemTier(Enum):
    POTATO = 1      # <8GB RAM, no GPU
    NORMAL = 2      # 8-16GB RAM, optional GPU
    GAMING = 3      # 16-32GB RAM, dedicated GPU
    SERVER = 4      # 32GB+ RAM, high-end GPU

@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    arch: str
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    docker_available: bool
    tier: SystemTier

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['tier'] = self.tier.name
        return d


def detect_system() -> SystemInfo:
    """Detect system capabilities."""
    print("\n[1/6] Detecting system capabilities...")

    # OS info
    os_name = platform.system().lower()
    os_version = platform.release()
    arch = platform.machine()

    # CPU
    cpu_cores = os.cpu_count() or 1

    # RAM
    ram_gb = _get_ram_gb()

    # Disk
    disk_free_gb = _get_disk_free_gb()

    # GPU
    gpu_available, gpu_name, gpu_vram_gb = _detect_gpu()

    # Docker
    docker_available = _check_docker()

    # Determine tier
    tier = _determine_tier(ram_gb, gpu_available, gpu_vram_gb)

    return SystemInfo(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        disk_free_gb=disk_free_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        docker_available=docker_available,
        tier=tier,
    )


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong),
                ]
            memoryStatus = MEMORYSTATUS()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
            return memoryStatus.dwTotalPhys / (1024**3)
        else:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 8.0  # Default assumption


def _get_disk_free_gb() -> float:
    """Get free disk space in GB."""
    try:
        if platform.system() == "Windows":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(Path.home())),
                None, None,
                ctypes.pointer(free_bytes)
            )
            return free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(Path.home())
            return (stat.f_bavail * stat.f_frsize) / (1024**3)
    except Exception:
        pass
    return 50.0  # Default assumption


def _detect_gpu() -> tuple:
    """Detect GPU availability."""
    try:
        # Try nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            name = parts[0].strip()
            vram_str = parts[1].strip().replace(' MiB', '')
            vram_gb = float(vram_str) / 1024
            return True, name, vram_gb
    except Exception:
        pass
    return False, None, None


def _check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _determine_tier(ram_gb: float, gpu: bool, vram_gb: Optional[float]) -> SystemTier:
    """Determine system tier based on specs."""
    if ram_gb >= 32 and gpu and vram_gb and vram_gb >= 12:
        return SystemTier.SERVER
    elif ram_gb >= 16 and gpu and vram_gb and vram_gb >= 6:
        return SystemTier.GAMING
    elif ram_gb >= 8:
        return SystemTier.NORMAL
    else:
        return SystemTier.POTATO


# ============================================================================
# USER PROFILE
# ============================================================================

@dataclass
class UserProfile:
    username: str
    goals: list
    domains: list
    work_style: str
    privacy_level: str
    resource_allocation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_user_profile(system_info: SystemInfo, headless: bool = False) -> UserProfile:
    """Interactive or default profile creation."""
    print("\n[2/6] Creating your profile...")

    if headless:
        return UserProfile(
            username="bizra-user",
            goals=["productivity"],
            domains=["general"],
            work_style="balanced",
            privacy_level="balanced",
            resource_allocation=_default_resources(system_info),
        )

    # Interactive profile creation
    print("\n" + "="*60)
    print("     BIZRA Personal AI Think Tank - Profile Setup")
    print("="*60)

    # Username
    username = input("\nChoose a username (no email needed): ").strip()
    if not username:
        username = "bizra-user"

    # Goals
    print("\nWhat do you want your AI team to help you with?")
    print("  1. Build a business / Side hustle")
    print("  2. Learn new skills / Education")
    print("  3. Creative projects / Art & Writing")
    print("  4. Research & Analysis")
    print("  5. Personal productivity")
    print("  6. Trading & Finance")
    print("  7. Health & Wellness")
    print("  8. All of the above")

    goal_choice = input("\nEnter numbers separated by commas (e.g., 1,3,5): ").strip()
    goals = _parse_goals(goal_choice)

    # Domains
    print("\nWhat fields interest you most?")
    print("  1. Technology & Programming")
    print("  2. Business & Entrepreneurship")
    print("  3. Science & Research")
    print("  4. Arts & Creativity")
    print("  5. Finance & Investing")
    print("  6. Health & Fitness")
    print("  7. Philosophy & Self-Development")

    domain_choice = input("\nEnter numbers (e.g., 1,2,5): ").strip()
    domains = _parse_domains(domain_choice)

    # Work style
    print("\nHow do you prefer to work?")
    print("  1. Deep focus (long uninterrupted sessions)")
    print("  2. Quick bursts (frequent short tasks)")
    print("  3. Collaborative (back-and-forth dialogue)")
    print("  4. Autonomous (set goals, let AI execute)")

    style_choice = input("\nEnter choice [1-4, default=3]: ").strip()
    work_style = _parse_work_style(style_choice)

    # Privacy
    print("\nYour privacy preference?")
    print("  1. Maximum Privacy (all local, no cloud)")
    print("  2. Balanced (local + encrypted cloud backup)")
    print("  3. Connected (share insights for network benefits)")

    privacy_choice = input("\nEnter choice [1-3, default=2]: ").strip()
    privacy_level = _parse_privacy(privacy_choice)

    # Resources
    resource_allocation = _configure_resources(system_info)

    return UserProfile(
        username=username,
        goals=goals,
        domains=domains,
        work_style=work_style,
        privacy_level=privacy_level,
        resource_allocation=resource_allocation,
    )


def _parse_goals(choice: str) -> list:
    goal_map = {
        '1': 'business', '2': 'learning', '3': 'creative',
        '4': 'research', '5': 'productivity', '6': 'trading',
        '7': 'health', '8': 'all'
    }
    if '8' in choice:
        return list(goal_map.values())[:-1]
    return [goal_map.get(c.strip(), 'productivity') for c in choice.split(',') if c.strip() in goal_map]


def _parse_domains(choice: str) -> list:
    domain_map = {
        '1': 'technology', '2': 'business', '3': 'science',
        '4': 'arts', '5': 'finance', '6': 'health', '7': 'philosophy'
    }
    return [domain_map.get(c.strip(), 'general') for c in choice.split(',') if c.strip() in domain_map]


def _parse_work_style(choice: str) -> str:
    styles = {'1': 'deep_focus', '2': 'quick_bursts', '3': 'collaborative', '4': 'autonomous'}
    return styles.get(choice, 'collaborative')


def _parse_privacy(choice: str) -> str:
    levels = {'1': 'maximum', '2': 'balanced', '3': 'connected'}
    return levels.get(choice, 'balanced')


def _default_resources(system_info: SystemInfo) -> Dict[str, Any]:
    """Default resource allocation based on system tier."""
    configs = {
        SystemTier.POTATO: {'cpu_cores': 1, 'ram_gb': 2, 'storage_gb': 10, 'gpu_enabled': False},
        SystemTier.NORMAL: {'cpu_cores': 2, 'ram_gb': 4, 'storage_gb': 25, 'gpu_enabled': False},
        SystemTier.GAMING: {'cpu_cores': 4, 'ram_gb': 8, 'storage_gb': 50, 'gpu_enabled': True},
        SystemTier.SERVER: {'cpu_cores': 8, 'ram_gb': 16, 'storage_gb': 100, 'gpu_enabled': True},
    }
    return configs.get(system_info.tier, configs[SystemTier.NORMAL])


def _configure_resources(system_info: SystemInfo) -> Dict[str, Any]:
    """Interactive resource configuration."""
    default = _default_resources(system_info)

    print(f"\n[Resource Allocation] (System: {system_info.tier.name})")
    print(f"  Available: {system_info.cpu_cores} cores, {system_info.ram_gb:.1f}GB RAM, {system_info.disk_free_gb:.1f}GB disk")

    print(f"\nPress Enter to use defaults, or customize:")
    print(f"  CPU cores for BIZRA [{default['cpu_cores']}]: ", end="")
    cpu_input = input().strip()
    cpu_cores = int(cpu_input) if cpu_input else default['cpu_cores']

    print(f"  RAM in GB [{default['ram_gb']}]: ", end="")
    ram_input = input().strip()
    ram_gb = int(ram_input) if ram_input else default['ram_gb']

    print(f"  Storage in GB [{default['storage_gb']}]: ", end="")
    storage_input = input().strip()
    storage_gb = int(storage_input) if storage_input else default['storage_gb']

    gpu_enabled = False
    if system_info.gpu_available:
        print(f"  Enable GPU ({system_info.gpu_name})? [Y/n]: ", end="")
        gpu_input = input().strip().lower()
        gpu_enabled = gpu_input != 'n'

    return {
        'cpu_cores': min(cpu_cores, system_info.cpu_cores),
        'ram_gb': min(ram_gb, int(system_info.ram_gb * 0.75)),
        'storage_gb': min(storage_gb, int(system_info.disk_free_gb * 0.5)),
        'gpu_enabled': gpu_enabled,
    }


# ============================================================================
# INSTALLATION
# ============================================================================

def install_node(system_info: SystemInfo, profile: UserProfile):
    """Main installation process."""

    # Create directories
    print("\n[3/6] Creating BIZRA home directory...")
    for d in [BIZRA_HOME, DATA_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Save configuration
    print("[4/6] Saving configuration...")
    config = {
        'version': VERSION,
        'system': system_info.to_dict(),
        'profile': profile.to_dict(),
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    # Install core components
    print("[5/6] Installing BIZRA components...")
    _install_components(system_info, profile)

    # Finalize
    print("[6/6] Finalizing installation...")
    _create_shortcuts(system_info)
    _show_completion_message(profile)


def _install_components(system_info: SystemInfo, profile: UserProfile):
    """Install based on available runtime."""

    if system_info.docker_available:
        print("  -> Using Docker deployment...")
        _install_docker_deployment(system_info, profile)
    else:
        print("  -> Using native deployment...")
        _install_native_deployment(system_info, profile)


def _install_docker_deployment(system_info: SystemInfo, profile: UserProfile):
    """Install using Docker containers."""
    compose_content = f"""
version: '3.8'

services:
  bizra-core:
    image: bizra/node-core:latest
    container_name: bizra-core
    restart: unless-stopped
    volumes:
      - {DATA_DIR}:/bizra/data
      - {MODELS_DIR}:/bizra/models
    environment:
      - BIZRA_USER={profile.username}
      - BIZRA_TIER={system_info.tier.name}
    deploy:
      resources:
        limits:
          cpus: '{profile.resource_allocation["cpu_cores"]}'
          memory: {profile.resource_allocation["ram_gb"]}G

  bizra-pat:
    image: bizra/pat-engine:latest
    container_name: bizra-pat
    restart: unless-stopped
    depends_on:
      - bizra-core
    volumes:
      - {DATA_DIR}:/bizra/data
    environment:
      - GOALS={','.join(profile.goals)}
      - DOMAINS={','.join(profile.domains)}
      - WORK_STYLE={profile.work_style}
      - PRIVACY={profile.privacy_level}

  bizra-dashboard:
    image: bizra/dashboard:latest
    container_name: bizra-dashboard
    restart: unless-stopped
    ports:
      - "8888:8888"
    depends_on:
      - bizra-core
      - bizra-pat

  bizra-network:
    image: bizra/p2p-bridge:latest
    container_name: bizra-network
    restart: unless-stopped
    ports:
      - "9999:9999"
    depends_on:
      - bizra-core
"""
    compose_file = BIZRA_HOME / "docker-compose.yml"
    compose_file.write_text(compose_content)
    print(f"  -> Docker Compose written to {compose_file}")
    print("  -> NOTE: Run 'docker-compose up -d' in ~/.bizra to start")


def _install_native_deployment(system_info: SystemInfo, profile: UserProfile):
    """Install without Docker (native Python/Node)."""

    # Create startup script
    if system_info.os_name == "windows":
        script_content = f"""@echo off
REM BIZRA Node Startup Script
echo Starting BIZRA Node for {profile.username}...

cd /d "{BIZRA_HOME}"

REM Start core services
start /B python -m bizra.core
start /B python -m bizra.pat --goals "{','.join(profile.goals)}"
start /B python -m bizra.dashboard

echo BIZRA Node running at http://localhost:8888
pause
"""
        script_path = BIZRA_HOME / "start-bizra.bat"
    else:
        script_content = f"""#!/bin/bash
# BIZRA Node Startup Script
echo "Starting BIZRA Node for {profile.username}..."

cd "{BIZRA_HOME}"

# Start core services
python -m bizra.core &
python -m bizra.pat --goals "{','.join(profile.goals)}" &
python -m bizra.dashboard &

echo "BIZRA Node running at http://localhost:8888"
"""
        script_path = BIZRA_HOME / "start-bizra.sh"

    script_path.write_text(script_content)
    if system_info.os_name != "windows":
        os.chmod(script_path, 0o755)

    print(f"  -> Startup script created: {script_path}")

    # Create requirements
    requirements = """
# BIZRA Node Dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0
python-dotenv>=1.0.0
rich>=13.7.0
typer>=0.9.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
ollama>=0.1.0
libp2p>=0.1.0
"""
    (BIZRA_HOME / "requirements.txt").write_text(requirements)
    print("  -> Requirements file created")


def _create_shortcuts(system_info: SystemInfo):
    """Create desktop shortcuts."""
    if system_info.os_name == "windows":
        # Windows shortcut creation would go here
        print("  -> Desktop shortcut: Use start-bizra.bat")
    elif system_info.os_name == "darwin":
        print("  -> Desktop shortcut: Add start-bizra.sh to Applications")
    else:
        print("  -> Desktop shortcut: Add start-bizra.sh to your launcher")


def _show_completion_message(profile: UserProfile):
    """Show completion message."""
    print("\n" + "="*60)
    print("     BIZRA NODE INSTALLATION COMPLETE!")
    print("="*60)
    print(f"""
    Welcome, {profile.username}!

    Your Personal AI Think Tank is ready.

    NEXT STEPS:
    1. Start your node:
       - Windows: Run start-bizra.bat
       - Mac/Linux: Run ./start-bizra.sh
       - Docker: cd ~/.bizra && docker-compose up -d

    2. Open dashboard:
       http://localhost:8888

    3. Your PAT is configured for:
       - Goals: {', '.join(profile.goals)}
       - Domains: {', '.join(profile.domains)}
       - Style: {profile.work_style}

    Configuration saved to: {CONFIG_FILE}

    Join the network: https://bizra.ai/network
    Documentation: https://docs.bizra.ai
    Community: https://discord.bizra.ai

    Welcome to the decentralized AI civilization.
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main installer entry point."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██████╗ ██╗███████╗██████╗  █████╗                  ║
    ║     ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                 ║
    ║     ██████╔╝██║  ███╔╝ ██████╔╝███████║                 ║
    ║     ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                 ║
    ║     ██████╔╝██║███████╗██║  ██║██║  ██║                 ║
    ║     ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                 ║
    ║                                                          ║
    ║           UNIFIED NODE INSTALLER v{version}               ║
    ║        Your Gateway to the Decentralized Future          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(version=VERSION))

    # Parse arguments
    headless = "--headless" in sys.argv

    # Step 1: System detection
    system_info = detect_system()

    print(f"\n    System detected: {system_info.os_name.upper()} ({system_info.arch})")
    print(f"    Tier: {system_info.tier.name} ({system_info.ram_gb:.1f}GB RAM, {system_info.cpu_cores} cores)")
    if system_info.gpu_available:
        print(f"    GPU: {system_info.gpu_name} ({system_info.gpu_vram_gb:.1f}GB VRAM)")
    print(f"    Docker: {'Available' if system_info.docker_available else 'Not found'}")

    # Check minimum requirements
    if system_info.ram_gb < MIN_RAM_GB:
        print(f"\n    WARNING: {system_info.ram_gb:.1f}GB RAM detected.")
        print(f"    Minimum recommended: {MIN_RAM_GB}GB")
        print("    Installation may work but performance will be limited.")

    if system_info.disk_free_gb < MIN_DISK_GB:
        print(f"\n    ERROR: Only {system_info.disk_free_gb:.1f}GB disk space available.")
        print(f"    Minimum required: {MIN_DISK_GB}GB")
        sys.exit(1)

    # Step 2: User profile
    profile = create_user_profile(system_info, headless=headless)

    # Step 3: Confirm
    if not headless:
        print("\n" + "-"*60)
        print("Ready to install BIZRA Node with these settings:")
        print(f"  User: {profile.username}")
        print(f"  Goals: {', '.join(profile.goals)}")
        print(f"  Resources: {profile.resource_allocation['cpu_cores']} cores, {profile.resource_allocation['ram_gb']}GB RAM")
        print("-"*60)
        confirm = input("\nProceed with installation? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("Installation cancelled.")
            sys.exit(0)

    # Step 4: Install
    install_node(system_info, profile)


if __name__ == "__main__":
    main()
