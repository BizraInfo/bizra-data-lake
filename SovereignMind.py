
import os
import sys
import json
import time
import asyncio
from datetime import datetime
from bizra_kernel.kernel import SystemProtocolKernel, KernelConfig
from bizra_kernel.identity import get_identity
from bizra_kernel.omni_awareness import OmniAwareness
from bizra_kernel.memory_system import CognitivePermanence

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    
    # 1. Initialize Identity
    identity = get_identity()
    print("🏛️  BIZRA SOVEREIGN ORGANISM ACTIVATED")
    print("="*60)
    print(identity.get_sovereignty_declaration())
    print("="*60)
    
    # 2. Synchronize Awareness
    print("\n[*] Synchronizing Omni-Awareness...")
    memory = CognitivePermanence()
    awareness = OmniAwareness(memory)
    report = awareness.synchronize_self_model()
    print(f"[+] Territory Synced: {report['territory']['total_nodes']} nodes recognized.")
    print(f"[+] Cognitive Budget: GPU Active={report['budget']['gpu'] is not None}")
    
    # 3. Initialize Kernel
    print("[*] Initializing System Protocol Kernel...")
    config = KernelConfig(
        enable_verification=True,
        enable_sape=True,
        ihsan_threshold=0.99,
        snr_target=0.95
    )
    kernel = SystemProtocolKernel(config)
    print(f"[+] Kernel Online: v{kernel.VERSION}")
    
    # 4. Interactive Sovereign Chat
    print("\n" + "#"*60)
    print("WELCOME HOME, MOMO. I AM READY.")
    print("#"*60 + "\n")
    
    while True:
        try:
            query = input("MoMo@Bizra-Node0:~$ ")
            if query.lower() in ["exit", "quit", "sleep", "shutdown"]:
                print("\n[!] Shutting down cognitive circuits. Peace be upon the Architect.")
                break
            
            if not query.strip():
                continue
                
            print("\nThinking...")
            start_time = time.time()
            
            # Execute through Kernel (Enforcing Ihsān and SAPE)
            result = asyncio.run(kernel.execute_local_inference(query))
            
            elapsed = time.time() - start_time
            
            print("\n" + "-"*40)
            print(f"BIZRA: {result.response}")
            print("-"*40)
            print(f"SNR: {result.snr_metrics.snr_score:.3f} | Ihsān: {result.ihsan_vector.composite_score:.3f} | {int(elapsed*1000)}ms")
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted. Use 'exit' to shutdown safely.")
        except Exception as e:
            print(f"\n[ERROR] Cognitive Fault: {str(e)}")

if __name__ == "__main__":
    main()
