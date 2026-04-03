# Protocol: The Genesis Ceremony (Phase 0)

## 1. Overview

The Genesis Ceremony is the legal-technical ritual that establishes the **immutable, physical root of trust** for BIZRA. It ensures that the system's sovereignty is not a software byproduct, but a cryptographically anchored reality.

## 2. Participants

- **Key Custodian (KC):** Responsible for the physical safety of the HSM.
- **Policy Architect (PA):** Responsible for the integrity of the Genesis Constitution (`ihsan_v1.yaml`).
- **Audit Chair (AC):** Responsible for verifying the ceremony process and witnessing the key generation.

## 3. The Ritual of Inauguration

1. **The Air-Gap Verification:** The Audit Chair confirms the HSM is FIPS 140-3 compliant and has not been tampered with.
2. **The Key Genesis:** The Policy Architect initiates the generation of the `SAT_ROOT_ED25519` key pair inside the HSM.
   - *Constraint:* The private key must never leave the HSM boundary.
3. **The Constitutional Anchor:** The SHA256 hash of the `ihsan_v1.yaml` (v1.0-Elite) is calculated.
4. **The Block Creation (`PACK-0000`):**

   ```json
   {
     "type": "GENESIS_ALETHEIA",
     "timestamp_iso": "2026-01-07T00:00:00Z",
     "constitution_hash": "sha256:7f4e5d...",
     "sat_public_key": "did:key:z6Mk...",
     "ceremony_audit_trail": "Recorded on WORM media"
   }
   ```

5. **The Sovereign Seal:** The HSM signs `PACK-0000`. This signature is the "First Receipt."

## 4. Continuity

Every subsequent state change (`PACK-0001` to `PACK-N`) must include a `prev_block_hash` that traces back to this ceremony. Failure to trace to `PACK-0000` results in an immediate **Sovereignty Breach** and system lockdown.
