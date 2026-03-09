# Interaction Contract

## Input
1. `prompt` (internal prompt text)
2. optional evidence refs (retrieval context)
3. optional consent marker

## Output (`ShadowSessionRecord`)
1. `session_id`
2. `prompt_hash`
3. `response_hash`
4. `disclosure_id`
5. `uncertainty_summary`
6. `evidence_refs`
7. `redline_events`
8. `receipt_chain_head`

## Decision Rules
1. If claim-bearing and no evidence refs -> deny + redline.
2. If consent-sensitive and no consent -> deny + redline.
3. Otherwise answer with disclosure/uncertainty metadata.

## Chain Rule
`receipt_chain_head` must include predecessor hash linkage.
