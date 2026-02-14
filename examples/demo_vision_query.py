#!/usr/bin/env python3
"""
🔮 BIZRA Vision Model Test
Tests direct vision model inference using qwen/qwen3-vl-8b via LM Studio
"""

import httpx
import base64
import json
from pathlib import Path

# LM Studio endpoint
LM_STUDIO_URL = "http://192.168.56.1:1234/v1/chat/completions"
VISION_MODEL = "qwen/qwen3-vl-8b"

def test_vision_text_query():
    """Test vision model with text-only query (should work as fallback)"""
    print("\n" + "="*60)
    print("🔮 Test: Vision Model Text Query")
    print("="*60)
    
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert visual analyst for the BIZRA data lake system."
            },
            {
                "role": "user",
                "content": "Describe what a knowledge graph visualization would look like for a data lake with 56,000 nodes and 88,000 edges. Be concise."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to: {LM_STUDIO_URL}")
            print(f"📦 Model: {VISION_MODEL}")
            
            response = client.post(LM_STUDIO_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            print(f"\n✅ Response received!")
            print(f"📝 Answer:\n{content[:500]}...")
            
            # Extract metrics
            usage = result.get("usage", {})
            print(f"\n📊 Tokens: {usage.get('total_tokens', 'N/A')}")
            
            return True
            
    except httpx.TimeoutException:
        print("⏱️ Timeout - model may be loading")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_vision_with_placeholder_image():
    """Test vision model with a simple base64 image (1x1 pixel)"""
    print("\n" + "="*60)
    print("🖼️ Test: Vision Model with Image (1x1 placeholder)")
    print("="*60)
    
    # Minimal 1x1 red PNG (base64)
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this image? Respond with just the color name."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{red_pixel_png}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending image to vision model...")
            
            response = client.post(LM_STUDIO_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            print(f"\n✅ Vision model response:")
            print(f"📝 Color detected: {content}")
            
            return True
            
    except httpx.TimeoutException:
        print("⏱️ Timeout - vision inference may take longer")
        return False
    except httpx.HTTPStatusError as e:
        print(f"⚠️ HTTP Error: {e.response.status_code}")
        print(f"   This may mean vision API format differs from expected")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_available_models():
    """List all models and identify vision-capable ones"""
    print("\n" + "="*60)
    print("📋 Available Models in LM Studio")
    print("="*60)
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get("http://192.168.56.1:1234/v1/models")
            response.raise_for_status()
            
            models = response.json().get("data", [])
            
            vision_keywords = ["vl", "vision", "llava", "qwen-vl"]
            
            for m in models:
                model_id = m.get("id", "unknown")
                is_vision = any(kw in model_id.lower() for kw in vision_keywords)
                icon = "👁️" if is_vision else "📝"
                print(f"   {icon} {model_id}")
            
            print(f"\n   Total: {len(models)} models")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")


if __name__ == "__main__":
    print("🔮 BIZRA Vision Model Test Suite")
    print("=" * 60)
    
    # Check available models
    check_available_models()
    
    # Test 1: Text-only query to vision model
    text_ok = test_vision_text_query()
    
    # Test 2: Image query (if text works)
    if text_ok:
        test_vision_with_placeholder_image()
    
    print("\n" + "="*60)
    print("✅ Vision tests complete!")
    print("="*60)
