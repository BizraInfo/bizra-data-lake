import requests
import json
import time

def check_health():
    url = "http://127.0.0.1:8010/healthz"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n🔍 MEMORY SERVER (SYNAPSE) STATUS:")
            print("-----------------------------------")
            print(json.dumps(data, indent=2))
            
            # Check for Redis specifically
            if "checks" in data:
                checks = data["checks"]
                if isinstance(checks, dict):
                    if "synapse" in checks:
                         synapse_status = checks["synapse"]
                         print(f"\n[?] Synapse Status: {json.dumps(synapse_status)}")
                         if synapse_status.get("ok"):
                             print(f"✅ MEMORY SERVER: CONNECTED ({synapse_status.get('mode', 'unknown')})")
                         else:
                             print(f"❌ MEMORY SERVER: DISCONNECTED ({synapse_status.get('error', 'unknown')})")
                elif isinstance(checks, list):
                    # Legacy or fallback
                    for check in checks:
                        if check.get("name") == "redis":
                            status = check.get("status")
                            print(f"\n[?] Redis Connection Status: {status.upper()}")
                            return
            
            # Fallback if structure is different
            print("\n[?] Full Health Response printed above.")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    print("⏳ Waiting for server to stabilize...")
    time.sleep(2)
    check_health()
