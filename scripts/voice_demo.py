"""
BIZRA Voice Demo Script
Run: python scripts/voice_demo.py
     python scripts/voice_demo.py --moshi   (Moshi full-duplex demo)
"""

import sys
import argparse
sys.path.insert(0, '.')

from core.voice import bizra_voice, list_voices, speak, get_moshi_voice


def demo_basic():
    """Basic TTS demo with pyttsx3."""
    print("=" * 60)
    print("         BIZRA VOICE SERVICE v1.0")
    print("=" * 60)
    
    # List voices
    voices = list_voices()
    print(f"\nAvailable voices: {len(voices)}")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v['name']}")
    
    print("\n" + "-" * 60)
    print("Running BIZRA System Announcements...")
    print("-" * 60)
    
    # 1. Startup
    print("\n[1/4] Startup announcement...")
    bizra_voice.announce('startup')
    
    # 2. Bridge online
    print("[2/4] Bridge connection...")
    bizra_voice.announce('bridge_online')
    
    # 3. Ihsan pass
    print("[3/4] Ihsan validation...")
    bizra_voice.announce('ihsan_pass')
    
    # 4. Status report
    print("[4/4] Full status report...")
    bizra_voice.status_report()
    
    print("\n" + "=" * 60)
    print("Voice demo complete!")
    print("=" * 60)


def demo_moshi():
    """
    Moshi full-duplex voice AI demo.
    
    This demonstrates the Kyutai Moshi model for real-time
    bidirectional voice dialogue.
    """
    print("=" * 60)
    print("     MOSHI FULL-DUPLEX VOICE AI DEMO")
    print("     Powered by Kyutai (transformers 4.57.1)")
    print("=" * 60)
    
    # Get Moshi instance
    moshi = get_moshi_voice()
    
    # Check status before loading
    print("\n[Step 1] Pre-load status check...")
    status = moshi.get_status()
    print(f"  Device: {status['device']}")
    if 'gpu_name' in status:
        print(f"  GPU: {status['gpu_name']}")
        print(f"  VRAM Total: {status.get('vram_total_gb', 'N/A')} GB")
        print(f"  VRAM Free: {status.get('vram_free_gb', 'N/A')} GB")
    print(f"  Model: {status['model_id']}")
    print(f"  Loaded: {status['is_loaded']}")
    
    # Confirm with user
    print("\n" + "-" * 60)
    print("Moshi model requires ~8.5GB VRAM.")
    print("Press ENTER to load model, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Load model
    print("\n[Step 2] Loading Moshi model...")
    print("  This may take 30-60 seconds on first run...")
    
    if moshi.load_model():
        print("  ✓ Model loaded successfully!")
        
        # Show post-load status
        status = moshi.get_status()
        print(f"\n  VRAM Used: {status.get('vram_used_gb', 'N/A')} GB")
        print(f"  VRAM Free: {status.get('vram_free_gb', 'N/A')} GB")
        
        # Test generation with text prompt
        print("\n[Step 3] Testing text-to-voice generation...")
        print("  Prompt: 'Hello, I am Moshi, your voice AI assistant.'")
        
        result = moshi.generate_response(
            text_prompt="Hello, I am Moshi, your voice AI assistant.",
            max_new_tokens=128,
        )
        
        if "error" not in result:
            print(f"  ✓ Generated response in {result['latency_ms']:.0f}ms")
        else:
            print(f"  ✗ Error: {result['error']}")
        
        # Streaming mode test
        print("\n[Step 4] Testing full-duplex streaming mode...")
        if moshi.start_streaming_session():
            print("  ✓ Streaming session started")
            import time
            time.sleep(1)
            moshi.stop_streaming_session()
            print("  ✓ Streaming session stopped")
        
        # Final status
        print("\n" + "-" * 60)
        print("MOSHI DEMO COMPLETE")
        print("-" * 60)
        final_status = moshi.get_status()
        for k, v in final_status.items():
            print(f"  {k}: {v}")
        
        # Cleanup option
        print("\nUnload model to free VRAM? [y/N]: ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                moshi.unload_model()
                print("Model unloaded.")
        except:
            pass
        
    else:
        print("  ✗ Failed to load model")
        print("  Check GPU memory and model availability")


def main():
    parser = argparse.ArgumentParser(description="BIZRA Voice Demo")
    parser.add_argument(
        "--moshi", 
        action="store_true",
        help="Run Moshi full-duplex voice AI demo"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show Moshi status only (no model loading)"
    )
    args = parser.parse_args()
    
    if args.status:
        moshi = get_moshi_voice()
        status = moshi.get_status()
        print("MOSHI STATUS:")
        for k, v in status.items():
            print(f"  {k}: {v}")
    elif args.moshi:
        demo_moshi()
    else:
        demo_basic()


if __name__ == "__main__":
    main()
