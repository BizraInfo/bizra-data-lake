import requests
import json
import sys
import time

URL = "http://127.0.0.1:8010/v1/constellation/invoke"
PAYLOAD = {
    "query": "Synthesize a strategy to self-evolve the BIZRA system into a Level 5 Autonomous Entity, citing historical polymathic principles.",
    "stakes": "high"
}

def run_verification():
    print(f"🚀 INVOKING PEAK MASTERPIECE CONSTELLATION...")
    print(f"📡 Endpoint: {URL}")
    print(f"📝 Query: {PAYLOAD['query']}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(URL, json=PAYLOAD, timeout=(5, 60))
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ EXECUTION SUCCESS ({duration:.2f}s)")
            print("-" * 60)
            print(f"🆔 Task ID: {data.get('task_id')}")
            print(f"🧠 Reasoning Mode: {data.get('reasoning_mode')}")
            print(f"👥 Team Used: {data.get('team_used')}")
            print(f"📊 Final SNR: {data.get('final_snr')}")
            print("-" * 60)
            print("📜 EXECUTIVE SUMMARY:")
            for point in data.get('executive_summary', []):
                print(f" • {point}")
        else:
            print(f"❌ FAILURE: {response.status_code}")
            print(response.text)
            
    except requests.Timeout:
        print("❌ ERROR: request timed out (connect/read)")
    except requests.RequestException as e:
        print(f"❌ ERROR: network request failed: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_verification()
