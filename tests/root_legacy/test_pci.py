"""
Test PCI Protocol Implementation
"""
import sys
import os
sys.path.append(os.getcwd())

from core.pci import (
    generate_keypair, EnvelopeBuilder, PCIGateKeeper, 
    RejectCode, IHSAN_MINIMUM_THRESHOLD
)

def test_pci_flow():
    print("Generating Keypair...")
    priv_key, pub_key = generate_keypair()
    print(f"Public Key: {pub_key}")
    
    print("\nBuilding Envelope...")
    builder = EnvelopeBuilder()
    builder.with_sender("PAT", "pat-agent-01", pub_key)
    builder.with_payload(
        action="PROPOSE_PATTERN", 
        data={"pattern": "GraphOfThoughts"}, 
        policy_hash="mock_hash", 
        state_hash="mock_state"
    )
    builder.with_metadata(ihsan=0.97, snr=0.85)
    
    envelope = builder.build()
    print(f"Envelope ID: {envelope.envelope_id}")
    
    print("Signing Envelope...")
    envelope.sign(priv_key)
    print(f"Signature: {envelope.signature.value[:32]}...")
    
    print("\nVerifying Envelope...")
    keeper = PCIGateKeeper()
    result = keeper.verify(envelope)
    
    print(f"Result: {result.passed}")
    print(f"Code: {result.reject_code.name}")
    print(f"Gates: {result.gate_passed}")
    
    if result.passed:
        print("✅ PCI Protocol Verification PASSED")
    else:
        print("❌ PCI Protocol Verification FAILED")
        exit(1)

    # Test Rejection (Low Ihsan)
    print("\nTesting Rejection (Low Ihsan)...")
    envelope.metadata.ihsan_score = 0.90 # Below 0.95
    # Note: Signature is now invalid for the payload, but let's re-sign to test Ihsan gate explicitly
    envelope.sign(priv_key)
    
    result_fail = keeper.verify(envelope)
    print(f"Result: {result_fail.passed}")
    print(f"Code: {result_fail.reject_code.name}")
    
    if result_fail.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN:
        print("✅ Correctly Rejected for Low Ihsān")
    else:
        print("❌ Wrong Rejection Code")

if __name__ == "__main__":
    test_pci_flow()
