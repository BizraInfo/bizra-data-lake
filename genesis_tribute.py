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


if __name__ == "__main__":
    create_sanctified_genesis()
