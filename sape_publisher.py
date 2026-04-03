#!/usr/bin/env python3
"""
sape_publisher.py - Enhanced SAPE Publisher Bridge
Connects Python Recursive Expander to Rust Sovereign Kernel via BIZRA Gateway

Phase Ω: Neural-Creative Interface Layer lowering barrier to Rust core logic
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerifiedFrame:
    """Mirrors Rust VerifiedFrame POD structure"""
    id: str
    did: str
    content: str
    proof: List[int]
    isnad_chain: List[str]
    got_branch: int
    got_depth: int
    timestamp: str

@dataclass
class FATEResponse:
    """Mirrors Rust FATEResponse"""
    verified: bool
    proof: List[int]
    confidence: float

class SAPEPublisher:
    """
    SAPE Publisher - Creative Interface to Sovereign Kernel

    Bridges the gap between Python's expressive power and Rust's guarantees.
    The "Creative Layer" that makes the "Logical Integrity" accessible.
    """

    def __init__(self, gateway_url: str = "http://localhost:8081"):
        self.gateway_url = gateway_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_url = gateway_url.replace("http", "ws") + "/ws"

        # Import recursive expander (assuming it's available)
        try:
            from bizra_kernel.recursive_node import RecursiveExpander
            self.expander = RecursiveExpander()
            logger.info("🔬 Recursive Expander initialized")
        except ImportError:
            logger.warning("⚠️  Recursive Expander not found, using mock")
            self.expander = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_and_verify(self, prompt: str, branch: int = 1, depth: int = 1) -> Optional[VerifiedFrame]:
        """
        Generate content via Recursive Expander, then verify via FATE Sidecar

        This is the neural-symbolic bridge in action:
        1. Neural generation (creative, unbounded)
        2. Symbolic verification (logical, constrained)
        """
        logger.info(f"🎨 Generating content for prompt: {prompt[:50]}...")

        # Step 1: Creative Generation (Python/Neural)
        if self.expander:
            try:
                expanded_content = await self.expander.apotheosis_expand(prompt)
                logger.info(f"✅ Content expanded to {len(expanded_content)} characters")
            except Exception as e:
                logger.error(f"❌ Expansion failed: {e}")
                expanded_content = prompt  # Fallback
        else:
            # Mock expansion for demo
            expanded_content = f"{prompt}\n\n[Apotheosis Expansion - Mock Implementation]"

        # Step 2: Symbolic Verification (Rust/Z3)
        verified_frame = await self.verify_with_gateway(expanded_content, branch, depth)

        if verified_frame:
            logger.info(f"🔐 Frame verified: {verified_frame.id}")
            # Broadcast to WebSocket clients
            await self.broadcast_frame(verified_frame)
        else:
            logger.warning("❌ Frame verification failed")

        return verified_frame

    async def verify_with_gateway(self, content: str, branch: int, depth: int) -> Optional[VerifiedFrame]:
        """Verify content through BIZRA Gateway (calls Rust FATE Sidecar)"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        # Add GoT metadata
        enriched_content = f"{content}\n\n[GoT: branch={branch}, depth={depth}]"

        payload = {
            "content": enriched_content,
            "got_branch": branch,
            "got_depth": depth
        }

        try:
            async with self.session.post(
                f"{self.gateway_url}/frames",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status == 201:
                    data = await response.json()
                    frame = VerifiedFrame(**data)
                    logger.info(f"✅ Verified frame created: {frame.id}")
                    return frame
                else:
                    error_data = await response.json()
                    logger.error(f"❌ Verification failed: {error_data}")
                    return None

        except Exception as e:
            logger.error(f"❌ Gateway communication failed: {e}")
            return None

    async def broadcast_frame(self, frame: VerifiedFrame):
        """Broadcast verified frame to WebSocket clients"""
        # In full implementation, this would use Iceoryx2 pub/sub
        # For now, the gateway handles broadcasting
        logger.info(f"📡 Frame {frame.id} broadcast via gateway")

    async def get_verified_frames(self) -> List[VerifiedFrame]:
        """Fetch all verified frames from gateway"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            async with self.session.get(f"{self.gateway_url}/frames") as response:
                if response.status == 200:
                    data = await response.json()
                    return [VerifiedFrame(**frame) for frame in data]
                else:
                    logger.error(f"❌ Failed to fetch frames: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"❌ Frame fetch failed: {e}")
            return []

    async def run_sape_session(self, initial_prompts: List[str]):
        """
        Run a complete SAPE session with multiple prompts
        Demonstrates the Graph of Thoughts workflow
        """
        logger.info("🚀 Starting SAPE Session - Phase Ω Creative Interface")

        results = []
        for i, prompt in enumerate(initial_prompts, 1):
            logger.info(f"📝 Processing prompt {i}/{len(initial_prompts)}")

            # Generate and verify
            frame = await self.generate_and_verify(prompt, branch=i, depth=1)
            if frame:
                results.append(frame)

                # Recursive expansion: use result as seed for next thought
                if i < len(initial_prompts):
                    next_prompt = f"Building on: {frame.content[:200]}..."
                    frame2 = await self.generate_and_verify(next_prompt, branch=i, depth=2)
                    if frame2:
                        results.append(frame2)

        logger.info(f"✨ SAPE Session complete: {len(results)} verified frames generated")
        return results

async def demo_sape_publisher():
    """Demo function showing SAPE Publisher in action"""
    logger.info("🎭 SAPE Publisher Demo - Neural-Symbolic Bridge")

    async with SAPEPublisher() as publisher:
        # Test single generation
        frame = await publisher.generate_and_verify(
            "Design a sovereign AI system with mathematical guarantees",
            branch=1,
            depth=1
        )

        if frame:
            logger.info(f"🎯 Generated frame: {frame.content[:100]}...")

        # Test session with multiple prompts
        prompts = [
            "What is the essence of digital sovereignty?",
            "How can AI achieve mathematical truth?",
            "What bridges neural creativity and symbolic logic?"
        ]

        results = await publisher.run_sape_session(prompts)
        logger.info(f"📊 Session results: {len(results)} verified frames")

        # Fetch all frames
        all_frames = await publisher.get_verified_frames()
        logger.info(f"📚 Total verified frames in system: {len(all_frames)}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_sape_publisher())
