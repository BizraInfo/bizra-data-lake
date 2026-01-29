#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    BIZRA FLYWHEEL API SERVER
    
    FastAPI server exposing the flywheel's capabilities:
    - /health — Health check (unauthenticated)
    - /status — Full status (authenticated)
    - /infer — LLM inference (authenticated)
    - /embed — Text embeddings (authenticated)
    - /ws/audio — WebSocket for audio streaming (authenticated)
    
    All endpoints except /health are fail-closed authenticated.
    
    Created: 2026-01-29 | BIZRA Sovereignty
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the flywheel core
from flywheel import (
    Flywheel, 
    FlywheelState, 
    AuthResult,
    AUTH_TOKEN_ENV,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_PORT = int(os.getenv("FLYWHEEL_API_PORT", "8100"))
AUDIO_WS_PORT = int(os.getenv("FLYWHEEL_AUDIO_PORT", "8101"))

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class InferRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    system: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)


class InferResponse(BaseModel):
    response: str
    model: str
    tokens: int = 0


class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int


class HealthResponse(BaseModel):
    status: str
    state: str
    version: str


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════════

flywheel_instance: Optional[Flywheel] = None


async def require_auth(authorization: Optional[str] = Header(None)) -> str:
    """
    Fail-closed authentication dependency.
    
    Expects: Authorization: Bearer <token>
    """
    if not flywheel_instance:
        raise HTTPException(status_code=503, detail="Flywheel not initialized")
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = parts[1]
    result = flywheel_instance.auth.authenticate(token, "api")
    
    if result == AuthResult.ALLOWED:
        return token
    elif result == AuthResult.MISSING:
        raise HTTPException(status_code=401, detail="Token required")
    else:
        raise HTTPException(status_code=403, detail="Access denied")


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — Flywheel Activation/Deactivation
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage flywheel lifecycle with the FastAPI app."""
    global flywheel_instance
    
    print("═" * 80)
    print("    BIZRA FLYWHEEL API — Starting")
    print("═" * 80)
    
    # Initialize and activate flywheel
    flywheel_instance = Flywheel()
    await flywheel_instance.activate()
    
    yield
    
    # Deactivate on shutdown
    print("[FlywheelAPI] Shutting down...")
    await flywheel_instance.deactivate()
    print("[FlywheelAPI] Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="BIZRA Flywheel API",
    description="Autopoietic self-sustaining cognitive core",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint (unauthenticated).
    
    Used by load balancers and container orchestrators.
    """
    if not flywheel_instance:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "state": "UNINITIALIZED", "version": "1.0.0"}
        )
    
    status = await flywheel_instance.status()
    
    # Only return healthy if in READY state
    if status.state == FlywheelState.READY:
        return HealthResponse(
            status="healthy",
            state=status.state.value,
            version=status.version
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "state": status.state.value,
                "version": status.version
            }
        )


@app.get("/status")
async def status(token: str = Depends(require_auth)):
    """
    Full flywheel status (authenticated).
    
    Returns detailed component health, loaded models, etc.
    """
    status = await flywheel_instance.status()
    return asdict(status)


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest, token: str = Depends(require_auth)):
    """
    LLM inference endpoint (authenticated).
    
    Runs inference on the local Ollama instance.
    Fail-closed: Returns error if inference fails.
    """
    try:
        response = await flywheel_instance.inference.generate(
            prompt=request.prompt,
            model=request.model,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        return InferResponse(
            response=response,
            model=request.model or flywheel_instance.inference.default_model,
            tokens=len(response.split())  # Rough estimate
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, token: str = Depends(require_auth)):
    """
    Text embedding endpoint (authenticated).
    
    Generates embeddings using the local embedding model.
    """
    try:
        embedding = await flywheel_instance.inference.embed(
            text=request.text,
            model=request.model,
        )
        
        return EmbedResponse(
            embedding=embedding,
            dimensions=len(embedding)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Protocol:
    1. Client connects
    2. Client sends auth message: {"type": "auth", "token": "..."}
    3. Server responds: {"type": "auth_result", "success": true/false}
    4. If success, client can send audio chunks
    5. Server responds with audio chunks
    
    Audio format: 24kHz, mono, 16-bit PCM
    """
    await websocket.accept()
    
    try:
        # First message must be auth
        auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        
        if auth_message.get("type") != "auth":
            await websocket.send_json({"type": "error", "message": "First message must be auth"})
            await websocket.close(code=4001)
            return
        
        token = auth_message.get("token")
        result = flywheel_instance.auth.authenticate(token, "audio")
        
        if result != AuthResult.ALLOWED:
            await websocket.send_json({"type": "auth_result", "success": False, "reason": result.value})
            await websocket.close(code=4003)
            return
        
        await websocket.send_json({"type": "auth_result", "success": True})
        
        # Check if Moshi is available
        if not flywheel_instance.audio.available:
            await websocket.send_json({
                "type": "info",
                "message": "Moshi audio not available. Audio streaming is a placeholder."
            })
        
        # Audio streaming loop
        while True:
            try:
                data = await websocket.receive_bytes()
                
                # Process through Moshi (when available)
                response_audio = await flywheel_instance.audio.stream_audio(data)
                
                if response_audio:
                    await websocket.send_bytes(response_audio)
                    
            except WebSocketDisconnect:
                break
                
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Auth timeout"})
        await websocket.close(code=4002)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=4000)


@app.post("/warm")
async def warm_model(model: str, token: str = Depends(require_auth)):
    """
    Warm a specific model into memory.
    
    Useful for preloading models before they're needed.
    """
    success = await flywheel_instance.model_cache.warm_model(model)
    return {"model": model, "warmed": success}


@app.post("/transcribe")
async def transcribe_audio(
    audio: bytes = None,
    language: str = "en",
    token: str = Depends(require_auth)
):
    """
    Transcribe audio to text (Speech-to-Text).
    
    Uses faster-whisper for transcription.
    
    Args:
        audio: Audio bytes (WAV, MP3, etc.) - send as request body
        language: Language code (e.g., "en", "ar")
    
    Returns:
        Transcribed text
    """
    from fastapi import Request
    
    if not flywheel_instance.audio.stt_available:
        raise HTTPException(status_code=503, detail="STT not available")
    
    try:
        text = await flywheel_instance.audio.transcribe(audio, language=language)
        return {"text": text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/synthesize")
async def synthesize_speech(
    text: str,
    voice: str = "en-US-AriaNeural",
    token: str = Depends(require_auth)
):
    """
    Synthesize text to speech (Text-to-Speech).
    
    Uses edge-tts for synthesis.
    
    Args:
        text: Text to synthesize
        voice: Voice ID (e.g., "en-US-AriaNeural", "ar-SA-HamedNeural")
    
    Returns:
        Audio bytes (MP3)
    """
    from fastapi.responses import Response
    
    if not flywheel_instance.audio.tts_available:
        raise HTTPException(status_code=503, detail="TTS not available")
    
    try:
        audio_bytes = await flywheel_instance.audio.synthesize(text, voice=voice)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")


@app.post("/voice-chat")
async def voice_chat(
    audio: bytes,
    language: str = "en",
    voice: str = "en-US-AriaNeural",
    system: str = None,
    token: str = Depends(require_auth)
):
    """
    Complete voice chat: Audio → Transcribe → LLM → Synthesize → Audio
    
    End-to-end voice interaction with the flywheel.
    """
    from fastapi.responses import Response
    
    try:
        # 1. Transcribe audio to text
        user_text = await flywheel_instance.audio.transcribe(audio, language=language)
        
        # 2. Generate LLM response
        response_text = await flywheel_instance.inference.generate(
            prompt=user_text,
            system=system or "You are a helpful AI assistant. Respond concisely.",
            max_tokens=500,
        )
        
        # 3. Synthesize response to audio
        response_audio = await flywheel_instance.audio.synthesize(response_text, voice=voice)
        
        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={
                "X-Transcript-User": user_text,
                "X-Transcript-Assistant": response_text,
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import uvicorn
    
    print(f"[FlywheelAPI] Starting on port {API_PORT}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
