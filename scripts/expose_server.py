from pyngrok import ngrok
import time
import sys

# Configure ngrok
# Note: If this fails due to missing token, user needs to run `ngrok config add-authtoken <token>`
try:
    public_url = ngrok.connect(8000).public_url
    print(f"✅ TUNNEL ESTABLISHED")
    print(f"---------------------------------------------------")
    print(f"YOUR MCP SERVER URL IS:")
    print(f"{public_url}/sse")
    print(f"---------------------------------------------------")
    print(f"Copy the above URL into ChatGPT configuration.")
    print(f"Press Ctrl+C to stop the tunnel.")
    
    # Keep alive
    while True:
        time.sleep(1)
except Exception as e:
    print(f"❌ Tunneling failed: {e}")
    print("You may need to sign up at ngrok.com and run: ngrok config add-authtoken <TOKEN>")
