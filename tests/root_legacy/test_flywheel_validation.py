#!/usr/bin/env python3
"""
BIZRA FLYWHEEL VALIDATION TEST
═══════════════════════════════════════════════════════════════════════════════

Tests the thermodynamic flywheel components:
1. Inference Gateway (PR1)
2. Epigenome Layer
3. Integration between components

Run: python3 test_flywheel_validation.py

Created: 2026-01-29 | BIZRA Sovereignty
"""

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.inference.gateway import (
    InferenceGateway,
    InferenceConfig,
    ComputeTier,
    InferenceStatus,
    TaskComplexity,
)
from core.pci.epigenome import (
    EpigeneticLayer,
    InterpretationType,
    GrowthNarrative,
)

# Test configuration
TEST_MODEL_PATH = None  # Set to actual GGUF path for full test


class FlywheelValidator:
    """Validates the thermodynamic flywheel components."""
    
    def __init__(self):
        self.results = {}
        self.temp_files = []
    
    def cleanup(self):
        """Clean up temp files."""
        for f in self.temp_files:
            try:
                f.unlink()
            except Exception:
                pass
    
    async def run_all(self):
        """Run all validation tests."""
        print("═" * 70)
        print("    BIZRA FLYWHEEL VALIDATION")
        print("═" * 70)
        print()
        
        # Test 1: Inference Gateway imports
        await self.test_inference_imports()
        
        # Test 2: Inference complexity routing
        await self.test_complexity_routing()
        
        # Test 3: Epigenome operations
        await self.test_epigenome_operations()
        
        # Test 4: Growth proof generation
        await self.test_growth_proof()
        
        # Test 5: Inference gateway initialization (no model)
        await self.test_gateway_init_offline()
        
        # Test 6: Full inference (if model available)
        if TEST_MODEL_PATH:
            await self.test_full_inference()
        
        # Summary
        self.print_summary()
        
        # Cleanup
        self.cleanup()
    
    async def test_inference_imports(self):
        """Test that inference gateway imports correctly."""
        test_name = "Inference Gateway Imports"
        try:
            from core.inference import InferenceGateway, ComputeTier, InferenceConfig
            self.results[test_name] = ("PASS", "All imports successful")
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    async def test_complexity_routing(self):
        """Test task complexity estimation and routing."""
        test_name = "Complexity Routing"
        try:
            gateway = InferenceGateway()
            
            # Simple task
            simple = gateway.estimate_complexity("What is 2+2?")
            simple_tier = gateway.route(simple)
            
            # Complex task
            complex_prompt = (
                "Analyze the mathematical foundations of cryptographic "
                "hash functions and explain why SHA-256 is considered "
                "secure against quantum computing attacks. Compare and "
                "prove the collision resistance properties."
            )
            complex_ = gateway.estimate_complexity(complex_prompt)
            complex_tier = gateway.route(complex_)
            
            # Validate
            assert simple.score < complex_.score, "Complex should have higher score"
            assert simple_tier in [ComputeTier.EDGE, ComputeTier.LOCAL]
            
            self.results[test_name] = (
                "PASS",
                f"Simple: {simple.score:.2f} → {simple_tier.value}, "
                f"Complex: {complex_.score:.2f} → {complex_tier.value}"
            )
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    async def test_epigenome_operations(self):
        """Test epigenome reframing and narratives."""
        test_name = "Epigenome Operations"
        try:
            # Create temp storage
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = Path(f.name)
                self.temp_files.append(temp_path)
            
            epigenome = EpigeneticLayer(storage_path=temp_path)
            
            # Test receipt hash
            receipt_hash = "b" * 64
            
            # Reframe 1: LEARNED
            r1 = epigenome.reframe(
                receipt_hash=receipt_hash,
                new_context="I learned patience from this challenge",
                interpretation_type=InterpretationType.LEARNED,
            )
            assert r1 is not None, "First reframe should succeed"
            
            # Reframe 2: RECONTEXTUALIZED
            r2 = epigenome.reframe(
                receipt_hash=receipt_hash,
                new_context="This experience shaped my resilience",
                interpretation_type=InterpretationType.RECONTEXTUALIZED,
            )
            assert r2 is not None, "Second reframe should succeed"
            
            # Get narrative
            narrative = epigenome.get_narrative(receipt_hash)
            assert narrative.current_interpretation is not None
            assert narrative.growth_score > 0
            
            self.results[test_name] = (
                "PASS",
                f"Reframes: 2, Growth score: {narrative.growth_score:.2f}"
            )
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    async def test_growth_proof(self):
        """Test ZK-style growth proof generation."""
        test_name = "Growth Proof Generation"
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = Path(f.name)
                self.temp_files.append(temp_path)
            
            epigenome = EpigeneticLayer(storage_path=temp_path)
            
            # Add some reframes
            for i in range(5):
                epigenome.reframe(
                    receipt_hash=f"{'c' * 63}{i}",
                    new_context=f"Growth interpretation {i}",
                    interpretation_type=InterpretationType.LEARNED,
                )
            
            # Generate proof
            proof = epigenome.generate_growth_proof()
            
            assert proof["interpretations_count"] == 5
            assert proof["growth_score_delta"] > 0
            assert proof["content_revealed"] == False
            assert len(proof["proof_hash"]) == 64
            
            self.results[test_name] = (
                "PASS",
                f"Count: {proof['interpretations_count']}, "
                f"Score: {proof['growth_score_delta']:.2f}, "
                f"Content hidden: ✓"
            )
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    async def test_gateway_init_offline(self):
        """Test gateway initialization without model (fail-closed)."""
        test_name = "Gateway Offline Mode"
        try:
            config = InferenceConfig(
                model_path="/nonexistent/path/model.gguf",
                require_local=True,
            )
            gateway = InferenceGateway(config)
            
            success = await gateway.initialize()
            
            # Should fail (no model) and go offline
            assert not success, "Should fail without model"
            assert gateway.status == InferenceStatus.OFFLINE
            
            # Try to infer (should raise)
            try:
                await gateway.infer("test")
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "fail-closed" in str(e).lower()
            
            self.results[test_name] = (
                "PASS",
                "Correctly failed closed when no model available"
            )
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    async def test_full_inference(self):
        """Test full inference with real model."""
        test_name = "Full Local Inference"
        try:
            config = InferenceConfig(
                model_path=TEST_MODEL_PATH,
                require_local=True,
            )
            gateway = InferenceGateway(config)
            
            success = await gateway.initialize()
            assert success, "Gateway should initialize with model"
            assert gateway.status == InferenceStatus.READY
            
            # Run inference
            start = time.time()
            result = await gateway.infer(
                "What is BIZRA in one sentence?",
                max_tokens=50,
            )
            latency = (time.time() - start) * 1000
            
            assert result.content, "Should have response content"
            assert result.backend.value == "llamacpp"
            
            self.results[test_name] = (
                "PASS",
                f"Latency: {latency:.0f}ms, "
                f"Speed: {result.tokens_per_second:.1f} tok/s"
            )
        except Exception as e:
            self.results[test_name] = ("FAIL", str(e))
    
    def print_summary(self):
        """Print test summary."""
        print()
        print("═" * 70)
        print("    RESULTS")
        print("═" * 70)
        
        passed = 0
        failed = 0
        
        for test_name, (status, details) in self.results.items():
            icon = "✅" if status == "PASS" else "❌"
            print(f"{icon} {test_name}")
            print(f"   {details}")
            print()
            
            if status == "PASS":
                passed += 1
            else:
                failed += 1
        
        print("═" * 70)
        print(f"    SUMMARY: {passed} passed, {failed} failed")
        print("═" * 70)
        
        if failed > 0:
            sys.exit(1)


async def main():
    validator = FlywheelValidator()
    await validator.run_all()


if __name__ == "__main__":
    asyncio.run(main())
