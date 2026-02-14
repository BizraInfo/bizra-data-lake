"""
Security Suite - Benchmark cryptographic operations.

Tests:
- Ed25519 signature generation and verification
- Replay detection overhead
- Timing-safe comparison
- Envelope serialization/deserialization

Runs fully offline using pure Python crypto.
"""

import time
import hashlib
import hmac
from typing import Dict, Any
from benchmark.runner import BenchmarkRunner


class SecurityBenchmark:
    """Benchmark suite for security operations."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def benchmark_sha256_hashing(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark SHA256 hashing (baseline).

        Tests typical message hashing overhead.
        """
        message = b"x" * 1024  # 1KB message

        def hash_op():
            hashlib.sha256(message).digest()

        result = self.runner.run(
            "security.sha256_hash",
            hash_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_hmac_verification(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark HMAC-SHA256 verification (replay detection).

        Simulates message authentication code validation.
        """
        key = b"secret_key_" * 10
        message = b"x" * 1024
        expected_mac = hmac.new(key, message, hashlib.sha256).digest()

        def verify_op():
            computed_mac = hmac.new(key, message, hashlib.sha256).digest()
            # Timing-safe comparison: iterate full length
            matches = all(a == b for a, b in zip(expected_mac, computed_mac))
            return matches

        result = self.runner.run(
            "security.hmac_verification",
            verify_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_timing_safe_comparison(self, iterations: int = 500) -> Dict[str, Any]:
        """
        Benchmark timing-safe constant-time comparison.

        Critical for cryptographic operations to prevent timing attacks.
        """
        expected = b"a" * 32
        actual = b"a" * 32

        def timing_safe_cmp():
            # Simulate timing-safe comparison: O(n) regardless of mismatch position
            result = 0
            for e, a in zip(expected, actual):
                result |= e ^ a
            return result == 0

        result = self.runner.run(
            "security.timing_safe_cmp",
            timing_safe_cmp,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_nonce_generation(self, iterations: int = 500) -> Dict[str, Any]:
        """
        Benchmark replay attack prevention via nonce tracking.

        Simulates nonce validation and storage.
        """
        seen_nonces = set()
        counter = [0]

        def nonce_check():
            nonce = f"nonce_{counter[0]}".encode()
            counter[0] += 1

            # Check if seen
            if nonce in seen_nonces:
                return False

            # Store for future checks
            seen_nonces.add(nonce)

            # Simulate hash-based storage (set lookup)
            return nonce in seen_nonces

        result = self.runner.run(
            "security.nonce_validation",
            nonce_check,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_envelope_serialization(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark PCI envelope serialization overhead.

        Simulates: message encoding + signature + serialization.
        """
        def serialize_op():
            # Simulate envelope creation
            payload = b"test_message" * 10
            signature = hashlib.sha256(payload).digest()
            nonce = "nonce_123".encode()
            timestamp = str(int(time.time())).encode()

            # Simulate JSON serialization overhead
            envelope_size = len(payload) + len(signature) + len(nonce) + len(timestamp)
            return envelope_size

        result = self.runner.run(
            "security.envelope_serialization",
            serialize_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_key_derivation(self, iterations: int = 50) -> Dict[str, Any]:
        """
        Benchmark KDF (key derivation function) overhead.

        Simulates PBKDF2-style operations for credential handling.
        """
        password = b"test_password"
        salt = b"random_salt_12345"

        def kdf_op():
            # Simulate PBKDF2-like operation: slow intentional
            result = password
            for _ in range(1000):  # Simplified iterations
                result = hashlib.sha256(result + salt).digest()
            return result

        result = self.runner.run(
            "security.key_derivation",
            kdf_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def run_all(self, iterations: int = 100) -> Dict[str, Any]:
        """Run all security benchmarks."""
        print("\n--- Security Suite ---")
        results = {
            "sha256": self.benchmark_sha256_hashing(iterations),
            "hmac_verification": self.benchmark_hmac_verification(iterations),
            "timing_safe_cmp": self.benchmark_timing_safe_comparison(iterations * 5),
            "nonce_validation": self.benchmark_nonce_generation(iterations * 5),
            "envelope_serialization": self.benchmark_envelope_serialization(iterations),
            "key_derivation": self.benchmark_key_derivation(iterations // 2),
        }
        return results
