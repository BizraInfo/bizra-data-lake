#!/usr/bin/env python3
"""
BIZRA Moshi WebSocket Client Demo
=================================

Real-time voice dialogue client for the /voice/moshi WebSocket endpoint.

Features:
- Connect to FastAPI kernel WebSocket
- Stream microphone audio to Moshi
- Receive and play audio responses
- Full-duplex conversation

Usage:
    # Test connection and status
    python scripts/moshi_ws_client.py --status

    # Load model via WebSocket
    python scripts/moshi_ws_client.py --load

    # Start voice dialogue (requires pyaudio)
    python scripts/moshi_ws_client.py --dialogue

    # Send test audio file
    python scripts/moshi_ws_client.py --file audio.wav

Requirements:
    pip install websockets pyaudio numpy
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("moshi_client")

# Default endpoint
DEFAULT_WS_URL = "ws://127.0.0.1:8000/voice/moshi"
DEFAULT_HTTP_URL = "http://127.0.0.1:8000/voice/moshi/status"
DEFAULT_TOKEN_ENV = "BIZRA_API_TOKEN"


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    token = (explicit or "").strip()
    if token:
        return token
    env_token = os.getenv(DEFAULT_TOKEN_ENV, "").strip()
    return env_token or None


def _with_token_query(url: str, token: Optional[str]) -> str:
    if not token:
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["token"] = token
    return urlunparse(parsed._replace(query=urlencode(query)))


def _auth_headers(token: Optional[str]) -> dict:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def send_command(ws, action: str) -> dict:
    """Send a command and wait for response."""
    await ws.send(json.dumps({"type": "command", "action": action}))
    response = await ws.recv()
    return json.loads(response)


async def check_status(url: str = DEFAULT_HTTP_URL, token: Optional[str] = None):
    """Check Moshi status via HTTP endpoint."""
    try:
        import aiohttp
        token = _resolve_token(token)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_auth_headers(token)) as response:
                data = await response.json()
                
                print("\n" + "=" * 50)
                print("🎙️  MOSHI VOICE STATUS")
                print("=" * 50)
                
                # Handle both response formats
                if "moshi" in data:
                    status = data["moshi"]
                    print(f"  Model ID:    {status.get('model_id', 'N/A')}")
                    print(f"  Device:      {status.get('device', 'N/A')}")
                    print(f"  Loaded:      {'✅' if status.get('is_loaded') else '❌'}")
                    print(f"  Streaming:   {'🔴 LIVE' if status.get('streaming') else '⚪'}")
                    print(f"  CUDA:        {'✅' if status.get('cuda_available') else '❌ (CPU only)'}")
                    
                    if status.get('warning'):
                        print(f"\n  ⚠️  {status['warning']}")
                    
                    if "gpu_name" in status:
                        print(f"\n  GPU:         {status['gpu_name']}")
                        print(f"  VRAM Total:  {status.get('vram_total_gb', 0):.1f} GB")
                        print(f"  VRAM Used:   {status.get('vram_used_gb', 0):.1f} GB")
                        print(f"  VRAM Free:   {status.get('vram_free_gb', 0):.1f} GB")
                    
                    if "websocket_url" in data:
                        print(f"\n  WebSocket:   {data['websocket_url']}")
                elif isinstance(data.get("status"), dict):
                    status = data["status"]
                    print(f"  Model ID:    {status.get('model_id', 'N/A')}")
                    print(f"  Device:      {status.get('device', 'N/A')}")
                    print(f"  Loaded:      {'✅' if status.get('is_loaded') else '❌'}")
                    print(f"  Streaming:   {'🔴 LIVE' if status.get('streaming') else '⚪'}")
                else:
                    print(f"  Status: {data.get('status', 'Unknown')}")
                    print(f"  Message: {data.get('message', 'N/A')}")
                
                print("=" * 50 + "\n")
                return data
                
    except ImportError:
        logger.error("aiohttp not installed. Install with: pip install aiohttp")
        return None
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return None


async def load_model(url: str = DEFAULT_WS_URL, token: Optional[str] = None):
    """Load Moshi model via WebSocket."""
    try:
        import websockets
        token = _resolve_token(token)
        ws_url = _with_token_query(url, token)
        headers = _auth_headers(token)
        
        print("\n🔄 Connecting to WebSocket...")
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            # Skip welcome message
            welcome = await ws.recv()
            
            print("📡 Connected! Sending load command...")
            await ws.send(json.dumps({"type": "command", "action": "load"}))
            
            # First response: "Loading..." status
            result = json.loads(await ws.recv())
            print(f"⏳ {result.get('message', 'Loading...')}")
            
            # Second response: actual result (may take 30-60 seconds)
            result = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            
            if result.get("type") == "status":
                print(f"✅ {result.get('message', 'Model loaded')}")
                
                status = result.get("status", {})
                if status.get("is_loaded"):
                    print(f"   Device: {status.get('device')}")
                    if "vram_used_gb" in status:
                        print(f"   VRAM: {status['vram_used_gb']:.1f} GB")
            else:
                print(f"❌ Error: {result.get('message', 'Unknown error')}")
                
    except ImportError:
        logger.error("websockets not installed. Install with: pip install websockets")
    except asyncio.TimeoutError:
        logger.error("Model loading timed out (>120s)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


async def unload_model(url: str = DEFAULT_WS_URL, token: Optional[str] = None):
    """Unload Moshi model via WebSocket."""
    try:
        import websockets
        token = _resolve_token(token)
        ws_url = _with_token_query(url, token)
        headers = _auth_headers(token)
        
        print("\n🔄 Connecting to WebSocket...")
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            print("📡 Connected! Sending unload command...")
            
            result = await send_command(ws, "unload")
            print(f"✅ {result.get('message', 'Model unloaded')}")
                
    except ImportError:
        logger.error("websockets not installed. Install with: pip install websockets")
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")


async def send_audio_file(filepath: str, url: str = DEFAULT_WS_URL, token: Optional[str] = None):
    """Send an audio file to Moshi for processing."""
    try:
        import websockets
        import wave
        token = _resolve_token(token)
        ws_url = _with_token_query(url, token)
        headers = _auth_headers(token)
        
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return
        
        print(f"\n📂 Loading audio file: {filepath}")
        
        # Read audio file
        with wave.open(str(path), 'rb') as wf:
            params = wf.getparams()
            audio_data = wf.readframes(params.nframes)
            
            print(f"   Sample Rate: {params.framerate} Hz")
            print(f"   Channels:    {params.nchannels}")
            print(f"   Duration:    {params.nframes / params.framerate:.2f}s")
        
        print("\n🔄 Connecting to WebSocket...")
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            # Ensure model is loaded
            status = await send_command(ws, "status")
            if not status.get("status", {}).get("is_loaded"):
                print("📥 Loading model first...")
                await send_command(ws, "load")
            
            # Send audio
            print("📤 Sending audio...")
            b64_audio = base64.b64encode(audio_data).decode("utf-8")
            await ws.send(json.dumps({
                "type": "audio",
                "data": b64_audio
            }))
            
            # Wait for response
            response = await ws.recv()
            result = json.loads(response)
            
            if result.get("type") == "audio":
                response_audio = base64.b64decode(result["data"])
                print(f"🔊 Received audio response: {len(response_audio)} bytes")
                
                # Save response
                output_path = path.parent / f"{path.stem}_response.wav"
                with wave.open(str(output_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(response_audio)
                print(f"💾 Saved response to: {output_path}")
                
            elif result.get("type") == "audio_ack":
                print(f"✅ Audio received: {result.get('received_bytes')} bytes")
                print("   (Full processing not yet implemented)")
            else:
                print(f"Response: {result}")
                
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"Failed to send audio: {e}")


async def voice_dialogue(url: str = DEFAULT_WS_URL, duration: float = 30.0, token: Optional[str] = None):
    """Start real-time voice dialogue with Moshi."""
    try:
        import websockets
        import pyaudio
        import numpy as np
        token = _resolve_token(token)
        ws_url = _with_token_query(url, token)
        headers = _auth_headers(token)
        
        print("\n" + "=" * 50)
        print("🎙️  MOSHI VOICE DIALOGUE")
        print("=" * 50)
        print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop\n")
        
        # Audio settings
        SAMPLE_RATE = 24000
        CHANNELS = 1
        CHUNK = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        FORMAT = pyaudio.paInt16
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Input stream (microphone)
        input_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Output stream (speaker)
        output_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        
        print("🎤 Microphone ready")
        print("🔊 Speaker ready")
        print("=" * 50 + "\n")
        
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            # Ensure model is loaded
            status = await send_command(ws, "status")
            if not status.get("status", {}).get("is_loaded"):
                print("📥 Loading Moshi model...")
                result = await send_command(ws, "load")
                print(f"✅ {result.get('message', 'Loaded')}\n")
            
            # Start streaming session
            await send_command(ws, "start_stream")
            print("🔴 LIVE - Speak now!\n")
            
            start_time = time.time()
            chunks_sent = 0
            responses_received = 0
            
            async def send_audio():
                """Send microphone audio to WebSocket."""
                nonlocal chunks_sent
                while time.time() - start_time < duration:
                    try:
                        # Read microphone
                        audio_data = input_stream.read(CHUNK, exception_on_overflow=False)
                        
                        # Encode and send
                        b64_audio = base64.b64encode(audio_data).decode("utf-8")
                        await ws.send(json.dumps({
                            "type": "audio",
                            "data": b64_audio
                        }))
                        chunks_sent += 1
                        
                    except Exception as e:
                        logger.error(f"Send error: {e}")
                        break
                    
                    await asyncio.sleep(0)
            
            async def receive_audio():
                """Receive and play audio from WebSocket."""
                nonlocal responses_received
                while time.time() - start_time < duration:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        result = json.loads(response)
                        
                        if result.get("type") == "audio":
                            audio_data = base64.b64decode(result["data"])
                            output_stream.write(audio_data)
                            responses_received += 1
                            
                        elif result.get("type") == "audio_ack":
                            # Acknowledgment received
                            pass
                            
                    except asyncio.TimeoutError:
                        pass
                    except Exception as e:
                        if "closed" not in str(e).lower():
                            logger.error(f"Receive error: {e}")
                        break
            
            try:
                # Run send and receive concurrently
                await asyncio.gather(
                    send_audio(),
                    receive_audio()
                )
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping...")
            
            # Stop streaming
            await send_command(ws, "stop_stream")
            
            print(f"\n📊 Stats:")
            print(f"   Duration: {time.time() - start_time:.1f}s")
            print(f"   Chunks Sent: {chunks_sent}")
            print(f"   Responses: {responses_received}")
        
        # Cleanup
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()
        
        print("\n✅ Dialogue complete")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install websockets pyaudio numpy")
    except Exception as e:
        logger.error(f"Dialogue failed: {e}")


def list_audio_devices():
    """List available audio input devices."""
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        print("\n🎤 Available Input Devices:")
        print("-" * 40)
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                name = info.get("name", "Unknown")
                channels = info.get("maxInputChannels")
                rate = int(info.get("defaultSampleRate", 0))
                print(f"  [{i}] {name}")
                print(f"      Channels: {channels}, Rate: {rate} Hz")
        
        p.terminate()
        print()
        
    except ImportError:
        logger.error("pyaudio not installed. Install with: pip install pyaudio")


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Moshi WebSocket Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --status           Check Moshi status
  %(prog)s --load             Load Moshi model
  %(prog)s --unload           Unload Moshi model
  %(prog)s --dialogue         Start voice dialogue
  %(prog)s --file audio.wav   Process audio file
  %(prog)s --devices          List audio devices
        """
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check Moshi status via HTTP"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load Moshi model"
    )
    parser.add_argument(
        "--unload",
        action="store_true",
        help="Unload Moshi model"
    )
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="Start real-time voice dialogue"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Dialogue duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Audio file to send (WAV format)"
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="List audio input devices"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL (default: {DEFAULT_WS_URL})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=f"API token (default: ${DEFAULT_TOKEN_ENV} env var)"
    )
    
    args = parser.parse_args()
    
    # If no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Execute requested action
    if args.devices:
        list_audio_devices()
    elif args.status:
        asyncio.run(check_status(token=args.token))
    elif args.load:
        asyncio.run(load_model(args.url, token=args.token))
    elif args.unload:
        asyncio.run(unload_model(args.url, token=args.token))
    elif args.file:
        asyncio.run(send_audio_file(args.file, args.url, token=args.token))
    elif args.dialogue:
        asyncio.run(voice_dialogue(args.url, args.duration, token=args.token))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
