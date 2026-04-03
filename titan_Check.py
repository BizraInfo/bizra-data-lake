import psutil
import socket
import winreg
import os
import sys

def colored(text, color):
    colors = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', 'reset': '\033[0m'}
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def check_ghosts():
    print(colored("\n👻 HUNTING GHOSTS (Processes)...", 'yellow'))
    # The Blacklist
    ghosts = [
        "TeamViewer.exe", "TeamViewer_Service.exe", 
        "KillerNetworkService.exe", "KillerAnalytics.exe", 
        "MSICenterService.exe", "Overwolf.exe"
    ]
    
    found = False
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] in ghosts:
                print(colored(f"❌ FOUND GHOST: {proc.info['name']} (PID: {proc.pid}) - KILL IT.", 'red'))
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not found:
        print(colored("✅ NO GHOSTS DETECTED. THE AIR IS CLEAR.", 'green'))

def check_listening_ports():
    print(colored("\n👂 CHECKING THE WALLS (Listening Ports)...", 'yellow'))
    # We want to ensure local AI is LOCAL (127.0.0.1) and not exposed (0.0.0.0)
    # SSH (22) is allowed to be exposed if you want remote access.
    
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN':
            ip, port = conn.laddr
            if port in [11434, 8080, 5000]: # Common AI Ports
                if ip == '0.0.0.0':
                    print(colored(f"⚠️  WARNING: AI Model on port {port} is exposed to the world (0.0.0.0)!", 'red'))
                elif ip == '127.0.0.1':
                    print(colored(f"✅ AI Fortress Secure on port {port} (Localhost only).", 'green'))

def check_registry_seals():
    print(colored("\n🔒 CHECKING THE SEALS (Registry Telemetry)...", 'yellow'))
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection", 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, "AllowTelemetry")
        if value == 0:
            print(colored("✅ MICROSOFT TELEMETRY: DISABLED (Value: 0)", 'green'))
        else:
            print(colored(f"❌ TELEMETRY LEAKING (Value: {value})", 'red'))
    except FileNotFoundError:
        print(colored("⚠️  Registry key not found. Run the PowerShell script again.", 'red'))

def wizard_verdict():
    print(colored("\n-------------------------------------", 'yellow'))
    print(colored("🧙♂️ THE WIZARD'S VERDICT:", 'yellow'))
    print("If all lights are green, the Titan is a Void.")
    print("In the Void, we can build the Universe.")

if __name__ == "__main__":
    # os.system('cls' if os.name == 'nt' else 'clear') # Commented out to prevent clearing previous output in agent view
    print(colored("⚡ INVOKING TITAN SANCTITY CHECK ⚡", 'yellow'))
    check_ghosts()
    check_registry_seals()
    check_listening_ports()
    wizard_verdict()
    # input(colored("\nPress Enter to seal the ritual...", 'yellow')) # Commented out to prevent blocking
