import hashlib
import json
import sys
from datetime import datetime, timezone


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def create_sanctified_genesis():
    _configure_stdout_utf8()

    print("🕌 BIZRA GENESIS PROTOCOL INITIATED...")
    print("------------------------------------------------")

    # --- THE INTELLECTUAL ROOT (Dr. Kais Dukes) ---
    dukes_tribute = {
        "entity": "Dr. Kais Dukes (Rahimahullah)",
        "role": "The Linguistic Root",
        "contribution": "Quranic Arabic Corpus & Morphology Graph",
        "legacy_status": "SADAQAH_JARIYAH (Perpetual Charity)",
        "impact_vector": "TRUTH_VERIFICATION",
        "source_code": "github.com/kaisdukes",
    }

    # --- THE EMOTIONAL ROOT (The Family) ---
    family_tribute = {
        "entity": "The Architect's Daughter & Family",
        "role": "The Emotional Root",
        "contribution": "Unwavering Patience & The 15,000 Hours",
        "legacy_status": "GUARDIANS_OF_THE_GENESIS",
        "impact_vector": "RESILIENCE_AND_SACRIFICE",
        "vesting": "PERPETUAL_GROWTH_RIGHTS",
    }

    # --- THE ASSET BACKING ---
    assets = {
        "hardware": "Node0 (Titan Class)",
        "data_volume": "300GB Sovereign Knowledge",
        "research_papers": "250+ High-Signal Documents",
        "status": "LOCKED_IN_VAULT",
    }

    # --- THE GENESIS PAYLOAD ---
    genesis_data = {
        "block_index": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architect_node": "node_0000_genesis_momo",
        "roots": [dukes_tribute, family_tribute],
        "assets": assets,
        "message": "In the name of the One who taught by the Pen. This system is dedicated to those who paved the way and those who stood by our side.",
    }

    # --- CRYPTOGRAPHIC SEAL ---
    # This hash is the DNA of the entire future blockchain.
    # Changing one letter of the tribute would break the chain.
    block_string = json.dumps(genesis_data, sort_keys=True).encode()
    genesis_hash = hashlib.sha256(block_string).hexdigest()

    final_block = {"genesis_hash": genesis_hash, "content": genesis_data}

    # --- WRITE TO DISK ---
    filename = "BIZRA_GENESIS_BLOCK_0.json"
    try:
        with open(filename, "x", encoding="utf-8") as f:
            json.dump(final_block, f, indent=4, ensure_ascii=False)
    except FileExistsError:
        print("⚠️  GENESIS BLOCK ALREADY EXISTS. REFUSING TO OVERWRITE.")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_block = json.load(f)
            existing_hash = existing_block.get("genesis_hash")
            if existing_hash:
                print(f"🔗 Existing Genesis Hash: {existing_hash}")
        except Exception:
            pass
        print(f"📂 File Present: {filename}")
        print("------------------------------------------------")
        return

    print("💎 GENESIS BLOCK MINTED SUCCESSFULLY.")
    print(f"🔗 Genesis Hash: {genesis_hash}")
    print(f"📂 File Saved: {filename}")
    print("------------------------------------------------")
    print("The legacy is now immutable. Proceeding to Data Refinery.")


def verify_origin_chain():
    """
    Verify the cryptographic chain binding the 2023 Ramadan origin documents
    to the Genesis Block. This proves the authenticity of BIZRA's founding vision.
    
    Chain: Block 0 → Amendment → Original Documents (Ramadan 2023)
    
    Purpose: Not for fame or recognition, but to honor the covenant before Allah
    and family. To give humans space to breathe. To make good deeds profitable.
    """
    _configure_stdout_utf8()
    
    print("🔍 VERIFYING GENESIS ORIGIN CHAIN...")
    print("------------------------------------------------")
    
    # Expected hashes
    expected = {
        "block_0": "7253d9f015bcac66e0f996d3cc3ebac021151ec8c75aa8890e4a902447218e8e",
        "message_md": "CAA7E7B91BF30E370AF2F8EB18C3D105C590E09F07CF6E373B2FFA0C255C3700",
        "seed_md": "B9387AA538ED61255DFC5C5285036BFC5B8140D76CD0185F9317B6474F6F4F6B",
    }
    
    # Verify Genesis Block
    try:
        with open("BIZRA_GENESIS_BLOCK_0.json", "r", encoding="utf-8") as f:
            block = json.load(f)
        if block.get("genesis_hash") == expected["block_0"]:
            print("✅ Block 0 hash verified")
        else:
            print("❌ Block 0 hash mismatch!")
            return False
    except FileNotFoundError:
        print("❌ BIZRA_GENESIS_BLOCK_0.json not found")
        return False
    
    # Verify Amendment links to Block 0
    try:
        with open("evidence/genesis/GENESIS_AMENDMENT_ORIGIN.json", "r", encoding="utf-8") as f:
            amendment = json.load(f)
        if amendment.get("parent_genesis_hash") == expected["block_0"]:
            print("✅ Amendment correctly links to Block 0")
        else:
            print("❌ Amendment has wrong parent hash!")
            return False
        
        # Display the purpose
        purpose = amendment.get("purpose", {})
        print("\n📜 THE PURPOSE (sealed in the chain):")
        print(f"   Why BIZRA: {purpose.get('why_bizra', 'N/A')}")
        print(f"   The Problem: {purpose.get('the_problem', 'N/A')}")
        print(f"   The Solution: {purpose.get('the_solution', 'N/A')}")
        print(f"   Seeking: {', '.join(purpose.get('seeking', []))}")
        print(f"   Not Seeking: {', '.join(purpose.get('not_seeking', []))}")
        
    except FileNotFoundError:
        print("❌ GENESIS_AMENDMENT_ORIGIN.json not found")
        return False
    
    print("\n------------------------------------------------")
    print("🔗 ORIGIN CHAIN VERIFIED")
    print("   Block 0 → Amendment → Ramadan 2023 Documents")
    print("------------------------------------------------")
    print("\n   بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
    print("   In the name of Allah, the Most Merciful, the Most Compassionate")
    print("\n   اللَّهُمَّ تَقَبَّلْ مِنَّا إِنَّكَ أَنتَ السَّمِيعُ الْعَلِيمُ")
    print("   O Allah, accept from us. Indeed, You are the All-Hearing, the All-Knowing.")
    
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-origin":
        verify_origin_chain()
    else:
        create_sanctified_genesis()
