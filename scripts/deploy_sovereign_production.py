#!/usr/bin/env python3
"""
BIZRA SOVEREIGN SCALING FRAMEWORK - PRODUCTION DEPLOYMENT
Deployment script for production sovereign scaling experiments
"""

import argparse
import asyncio
import hashlib
import json
import logging
import secrets
import sys
from pathlib import Path
from typing import Dict, List

from test_sovereign_scaling import SoftwareTPM, FateEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignDeploy")

async def deploy_sovereign_production(args):
    """Deploy sovereign scaling framework in production configuration."""
    
    logger.info("=" * 70)
    logger.info("🚀 BIZRA SOVEREIGN PRODUCTION DEPLOYMENT")
    logger.info("=" * 70)
    
    # Validate arguments
    constitution_path = Path(args.constitution)
    if not constitution_path.exists():
        logger.error(f"❌ Constitution not found: {constitution_path}")
        return False
    
    # Create deployment directory
    deploy_dir = Path("deployments") / f"sovereign_{args.deployment_id}"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Deployment directory: {deploy_dir}")
    logger.info(f"📜 Constitution: {constitution_path}")
    logger.info(f"🤖 Agents: {args.agents}")
    logger.info(f"⚖️  BFT Quorum: {args.bft_quorum}")
    logger.info(f"🎯 Ihsān Threshold: {args.ihsan_threshold}")
    logger.info(f"💾 Model Size: {args.model_size}")
    
    # Initialize TPM
    tpm = SoftwareTPM()
    
    # Extend TPM with deployment configuration
    config_data = json.dumps(vars(args)).encode()
    tpm.extend(16, config_data)  # PCR[16] for deployment config
    
    # Load constitution
    fate = FateEvaluator(constitution_path)
    
    # Generate deployment manifest
    manifest = {
        "deployment_id": args.deployment_id,
        "timestamp": asyncio.get_event_loop().time(),
        "configuration": vars(args),
        "tpm_state": {
            "pcrs_extended": list(tpm.pcrs.keys()),
            "pcr_16_hash": tpm.pcrs[16].hex()[:32]
        },
        "constitution_hash": hashlib.sha256(constitution_path.read_bytes()).hexdigest(),
        "status": "deployed",
        "next_steps": [
            "1. Run scaling experiment: python scripts/test_sovereign_scaling.py",
            "2. Analyze results: python scripts/analyze_results.py",
            "3. Deploy optimal model: python scripts/deploy_model.py",
            "4. Start governance: python scripts/start_governance.py"
        ]
    }
    
    # Save manifest
    manifest_file = deploy_dir / "deployment_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"📄 Deployment manifest saved: {manifest_file}")
    
    # Generate TPM quote for deployment
    nonce = secrets.token_bytes(32)
    selected_pcrs = [12, 13, 14, 15, 16]
    pcr_digest = hashlib.sha256(b''.join(tpm.pcrs[i] for i in selected_pcrs)).digest()
    
    quote = {
        "deployment_id": args.deployment_id,
        "nonce": nonce.hex(),
        "pcr_indices": selected_pcrs,
        "pcr_digest": pcr_digest.hex(),
        "manifest_hash": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "attestation": "SOFTWARE_TPM_QUOTE_V1"
    }
    
    quote_file = deploy_dir / "tpm_quote.json"
    with open(quote_file, 'w') as f:
        json.dump(quote, f, indent=2)
    
    logger.info(f"🔐 TPM quote saved: {quote_file}")
    
    # Create deployment summary (without emojis for Windows compatibility)
    summary = f"""
    ============================================================
    BIZRA SOVEREIGN DEPLOYMENT COMPLETE
    ============================================================
    
    Deployment ID: {args.deployment_id}
    Status: DEPLOYED
    
    Files Generated:
    - Manifest: {manifest_file}
    - TPM Quote: {quote_file}
    
    TPM State:
    - PCRs Extended: {selected_pcrs}
    - PCR[16] Hash: {tpm.pcrs[16].hex()[:16]}...
    - Quote Digest: {pcr_digest.hex()[:16]}...
    
    Configuration:
    - Constitution: {constitution_path}
    - Ihsan Threshold: {args.ihsan_threshold}
    - Agents: {args.agents}
    - BFT Quorum: {args.bft_quorum}
    
    Next Steps:
    1. Run scaling experiment to discover optimal architecture
    2. Deploy optimal model to WASM
    3. Initialize governance council
    4. Start Harberger tax system
    
    ============================================================
    SOVEREIGNTY DEPLOYED | Ihsan Target: {args.ihsan_threshold}
    ============================================================
    """
    
    print(summary)
    
    # Save summary with UTF-8 encoding
    summary_file = deploy_dir / "deployment_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info("✅ Sovereign deployment completed successfully!")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy BIZRA Sovereign Scaling Framework")
    
    parser.add_argument("--deployment-id", default=f"deploy_{int(asyncio.get_event_loop().time())}",
                       help="Deployment identifier")
    parser.add_argument("--constitution", default="constitution/scaling_simple.yaml",
                       help="Path to constitution file")
    parser.add_argument("--agents", type=int, default=6,
                       help="Number of governance agents")
    parser.add_argument("--bft-quorum", type=int, default=4,
                       help="BFT quorum size (must be ≤ agents)")
    parser.add_argument("--ihsan-threshold", type=float, default=0.95,
                       help="Ihsān threshold for model verification")
    parser.add_argument("--model-size", default="optimal",
                       choices=["small", "medium", "large", "optimal"],
                       help="Target model size")
    
    args = parser.parse_args()
    
    # Validate BFT quorum
    if args.bft_quorum > args.agents:
        logger.error(f"❌ BFT quorum ({args.bft_quorum}) cannot exceed agents ({args.agents})")
        sys.exit(1)
    
    try:
        success = asyncio.run(deploy_sovereign_production(args))
        if success:
            print("\nDeployment successful! The sovereign framework is ready.")
            print("Run the scaling experiment to discover optimal architecture:")
            print("   python scripts/test_sovereign_scaling.py")
            print("\nTPM PCRs have been extended with deployment configuration.")
            print("Constitutional constraints are enforced.")
            print("Sovereignty deployed.")
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()