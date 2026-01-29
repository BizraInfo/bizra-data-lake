"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Cryptographic Primitives (Ed25519 + BLAKE3)
"""

import json
import base64
import blake3
from typing import Any, Dict, Tuple
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

PCI_DOMAIN_PREFIX = "bizra-pci-v1:"

def canonical_json(data: Dict[str, Any]) -> bytes:
    """
    RFC 8785 JSON Canonicalization Scheme (JCS).
    - Keys sorted lexicographically
    - No whitespace
    - UTF-8
    """
    return json.dumps(
        data,
        separators=(',', ':'),
        sort_keys=True,
        ensure_ascii=False
    ).encode('utf-8')

def domain_separated_digest(canonical_data: bytes) -> str:
    """
    Compute domain-separated BLAKE3 digest.
    Prefix: bizra-pci-v1:
    """
    hasher = blake3.blake3()
    hasher.update(PCI_DOMAIN_PREFIX.encode('utf-8'))
    hasher.update(canonical_data)
    return hasher.hexdigest()

def sign_message(digest_hex: str, private_key_hex: str) -> str:
    """
    Sign a digest using Ed25519.
    input: hex digest, hex private key
    output: hex signature
    """
    priv_bytes = bytes.fromhex(private_key_hex)
    digest_bytes = bytes.fromhex(digest_hex)
    
    start_time = 0
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    signature = private_key.sign(digest_bytes)
    return signature.hex()

def verify_signature(digest_hex: str, signature_hex: str, public_key_hex: str) -> bool:
    """
    Verify an Ed25519 signature.
    """
    try:
        pub_bytes = bytes.fromhex(public_key_hex)
        sig_bytes = bytes.fromhex(signature_hex)
        digest_bytes = bytes.fromhex(digest_hex)
        
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        public_key.verify(sig_bytes, digest_bytes)
        return True
    except Exception:
        return False

def generate_keypair() -> Tuple[str, str]:
    """Generates (private_key_hex, public_key_hex)."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    
    priv_hex = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    ).hex()
    
    pub_hex = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    ).hex()
    
    return priv_hex, pub_hex
