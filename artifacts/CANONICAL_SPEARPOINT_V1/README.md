# CANONICAL_SPEARPOINT_V1

This bundle is the smallest honest contract for a canonical self-improving governed loop.

It is intentionally narrow:

- one mission
- two canonical runs
- one verified reward
- one bounded state adjustment
- one persisted receipt chain

It does **not** claim empirical proof yet. It is a frozen artifact template that defines what must be captured before any public claim of canonical empirical validation.

## What This Bundle Proves

The bundle is valid only if it proves this chain end to end:

`observe -> decide -> act -> verify -> reward -> improve -> persist`

If any step is missing, this artifact is incomplete.

## File Roles

- `mission_input.json`: the one replayable constitutional mission
- `pre_state.json`: the self-observation snapshot captured before action
- `run1_receipt.json`: the first canonical execution receipt
- `reward_calc.json`: the verified-only reward computation
- `state_delta.json`: the single bounded recursive adjustment
- `run2_receipt.json`: the replay receipt that must show the persisted effect
- `chain_verification.json`: the final proof that the chain is valid

## Frozen Mission Choice

The chosen mission is deliberately small and low-noise:

`Answer one BIZRA constitutional question through the canonical mission path.`

The specific question is frozen in `mission_input.json`.

## Frozen Reward Rule

The reward is computed only from verified facts:

- verified success
- SNR above floor
- latency relative to budget
- degraded execution status
- policy cleanliness and silent-fallback absence

No self-praise, narrative scoring, or unverifiable terms are allowed.

## Frozen Improvement Rule

Only one route-affecting preference may change in this version:

- `reflex_route_preference`

The adjustment is deliberately conservative:

- if reward is high enough and the run is policy-clean, compile exactly one persisted reflex entry for this mission pattern
- if that reflex write is verified, prefer the reflex route for the replay
- otherwise do not change anything

This creates a measurable, falsifiable replay effect on the actual organism path without changing mission identity.

## Genesis Sentinel

Run 1 is the genesis receipt of the artifact chain.

- `genesis_receipt: true`
- `prev_receipt_hash`: `0000...0000`
- `prev_receipt_hash_semantics`: `GENESIS_ZERO_HASH`

This removes ambiguity around the zero hash and makes the chain contract explicit.

## Authority Path

Each run receipt must carry an explicit authority field:

- `execution_authority`
- `authority_path`

For minimal canonical proof, the authority path must resolve to the runtime-owned organism path rather than an implicit or degraded fallback.

## State Source

The persisted delta must distinguish between:

- artifact bookkeeping state
- runtime-owned state that can actually affect replay behavior

This is why the state delta records:

- `state_store_kind`
- `state_store_key`
- `state_store_write_verified`

## Execution Order

1. Populate `pre_state.json` from live runtime state before action.
2. Execute the frozen mission through the canonical path and write `run1_receipt.json`.
3. Compute `reward_calc.json` from verified fields only.
4. Apply and persist the bounded update in `state_delta.json`.
5. Replay the same mission and write `run2_receipt.json`.
6. Validate the chain and write `chain_verification.json`.

## Replay Isolation

Replay validity assumes no external mutation between run 1 and run 2:

- no policy version drift
- no threshold drift
- no manual state edits
- no mission prompt edits

## Minimal Canonical Status

This artifact reaches minimal canonical status only if all of the following are true:

- two successful chained runs exist
- one verified reward calculation exists
- one persisted state delta exists
- run 2 shows measurable improvement or controlled behavioral change
- no constitutional violation occurred
- no silent fallback occurred
- all receipt hashes are valid

## Path Choice

The files live under `artifacts/CANONICAL_SPEARPOINT_V1/` to match the repo's existing artifact convention while preserving the exact eight-file spearpoint bundle requested.
