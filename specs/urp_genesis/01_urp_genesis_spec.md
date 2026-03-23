# URP Genesis Specification

## Overview

When Node0 activates, it doesn't just create itself — it creates the entire ecosystem.
The URP (Universal Resource Protocol) is minted ONCE at genesis by Node0, then Node0
connects to it like any other node. This is the constitutional membrane from the CMN paper.

**Key Principle:** No node ever talks to another node directly.
Node → URP → Sea (resource pool). Never Node → Node.

## Genesis Sequence

```
PHASE 1: Node0 Identity
  node0.mint_identity()
    → Ed25519 keypair generated
    → BLAKE3 identity hash computed
    → Node0 is sovereign

PHASE 2: Agent Mint
  node0.mint_agents()
    → 7 PAT agents created (user's team)
       P1-Navigator, P2-Scholar, P3-Artisan, P4-Guardian,
       P5-Mentor, P6-Diplomat, P7-Oracle
    → 5 SAT agents created (system's workers)
       S1-Validator, S2-Oracle[frozen], S3-Mediator,
       S4-Archivist, S5-Sentinel
    → Each agent gets Ed25519 keypair (soulbound)

PHASE 3: URP Genesis Mint (ONE TIME ONLY)
  node0.mint_urp()
    → URP identity created (separate from Node0)
    → Constitutional spine initialized:
       - ZANN_ZERO: no unverified claims
       - RIBA_ZERO: no extractive economics
       - IHSAN_FLOOR: 0.95 minimum
       - GINI_CEILING: 0.35 maximum
    → Resource pool created (empty, ready for contributions)
    → House of Wisdom initialized (knowledge substrate)
    → SEED treasury minted (genesis allocation)

PHASE 4: SAT Deployment into URP
  node0.deploy_sat_to_urp()
    → S1-Validator → URP.validators pool
    → S2-Oracle[frozen] → URP.truth_oracle (constitutional axioms)
    → S3-Mediator → URP.dispute_resolution
    → S4-Archivist → URP.house_of_wisdom (knowledge curation)
    → S5-Sentinel → URP.security (threat detection)
    → Each SAT starts its work loop immediately

PHASE 5: Node0 Connects to URP
  node0.connect_to_urp()
    → Node0 registers as first participant
    → PAT agents gain access to URP resources (through membrane)
    → Node0 can now:
       - Submit knowledge → House of Wisdom (via S4-Archivist)
       - Request validation → S1-Validator
       - Get truth verification → S2-Oracle
       - Resolve disputes → S3-Mediator
       - Benefit from security → S5-Sentinel
    → Flywheel STARTS

PHASE 6: Self-Sustaining Loop
  loop forever:
    node0.execute_mission()
      → Mission produces receipt (BLAKE3 + Ed25519)
      → Receipt → URP via membrane (constitutional filtering)
      → SAT agents process receipt:
         S1 validates, S4 archives, S5 monitors
      → SEED earned → resource pool grows
      → Node0 learns from URP resource pool
      → Better missions → better receipts → stronger URP
```

## URP Service Architecture

```
┌──────────────────────────────────────────────────────┐
│                    URP (Membrane)                      │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  CONSTITUTIONAL SPINE                             │ │
│  │  ZANN_ZERO │ RIBA_ZERO │ IHSAN ≥ 0.95 │ GINI ≤ 0.35 │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐│
│  │  S1    │ │  S2    │ │  S3    │ │  S4    │ │  S5  ││
│  │Validatr│ │Oracle  │ │Mediatr │ │Archvst │ │Sentnl││
│  │        │ │[frozen]│ │        │ │        │ │      ││
│  │Verify  │ │Truth   │ │Resolve │ │Curate  │ │Guard ││
│  │receipts│ │from    │ │disputes│ │House of│ │the   ││
│  │+ state │ │axioms  │ │fairly  │ │Wisdom  │ │gates ││
│  └────────┘ └────────┘ └────────┘ └────────┘ └──────┘│
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  RESOURCE POOL (the "Sea")                        │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │ │
│  │  │ Knowledge│ │ SEED     │ │ Compiled Reflexes│  │ │
│  │  │ (H.o.W.) │ │ Treasury │ │ (shared patterns)│  │ │
│  │  └──────────┘ └──────────┘ └──────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  MEMBRANE (all node traffic passes through here)  │ │
│  │  Property 1: Fail-closed                          │ │
│  │  Property 2: Constitutional filtering             │ │
│  │  Property 3: Cryptographic authentication         │ │
│  │  Property 4: Provenance recording                 │ │
│  └──────────────────────────────────────────────────┘ │
│                           │                            │
│              ┌────────────┼────────────┐               │
│              │            │            │               │
│          ┌───▼───┐   ┌────▼───┐   ┌────▼───┐          │
│          │Node 0 │   │Node 1 │   │Node N │          │
│          │(7 PAT)│   │(7 PAT)│   │(7 PAT)│          │
│          └───────┘   └────────┘   └────────┘          │
└──────────────────────────────────────────────────────┘
```

## Implementation Modules

### Module 1: URP Service (`core/urp/service.py`)

```pseudocode
class URPService:
    """The Universal Resource Protocol — the constitutional membrane."""

    def __init__(self):
        self.identity = None          # URP's own Ed25519 identity
        self.constitution = None      # ZANN_ZERO, RIBA_ZERO, IHSAN, GINI
        self.resource_pool = None     # Knowledge + SEED + Reflexes
        self.house_of_wisdom = None   # Governed knowledge substrate
        self.sat_agents = {}          # Active SAT workers
        self.connected_nodes = {}     # Registered nodes
        self.membrane = None          # Constitutional filter
        self.genesis_complete = False

    def mint_genesis(self, founder_node_id, founder_sat_agents):
        """Called ONCE by Node0 at system genesis."""
        ASSERT not self.genesis_complete  # One-time only

        self.identity = generate_urp_identity()
        self.constitution = load_constitution()  # Maqasid invariants
        self.resource_pool = ResourcePool()
        self.house_of_wisdom = HouseOfWisdom(constitution=self.constitution)
        self.membrane = ConstitutionalMembrane(self.constitution)

        # Deploy founder's SAT agents
        for agent in founder_sat_agents:
            self.deploy_sat_agent(agent)

        # Genesis SEED allocation
        self.resource_pool.mint_genesis_seed(
            founder=founder_node_id,
            treasury=GENESIS_TREASURY_AMOUNT,
            zakat=GENESIS_TREASURY_AMOUNT * ZAKAT_RATE,
        )

        self.genesis_complete = True
        return URPGenesisReceipt(
            urp_id=self.identity.id,
            constitution_hash=self.constitution.hash(),
            sat_count=len(self.sat_agents),
            timestamp=now(),
        )

    def register_node(self, node_id, node_public_key):
        """A node connects to the URP through the membrane."""
        # Membrane check: is this node constitutionally admissible?
        IF NOT self.membrane.admit(node_id, node_public_key):
            RETURN Rejection("constitutional_filter")

        self.connected_nodes[node_id] = NodeRegistration(
            node_id=node_id,
            public_key=node_public_key,
            connected_at=now(),
        )
        RETURN ConnectionReceipt(node_id, self.identity.id)

    def submit_receipt(self, node_id, receipt):
        """Node submits a mission receipt through the membrane."""
        # Property 1: Fail-closed
        IF NOT self.membrane.verify_authority(receipt):
            RETURN Rejection("missing_authority")

        # Property 2: Constitutional filtering
        IF receipt.ihsan_score < IHSAN_FLOOR:
            RETURN Rejection("below_ihsan_threshold")

        # Property 3: Cryptographic authentication
        IF NOT receipt.verify_signature():
            RETURN Rejection("invalid_signature")

        # Property 4: Provenance recording
        self.resource_pool.record(receipt)

        # SAT agents process the receipt
        self.sat_agents["S1-Validator"].validate(receipt)
        self.sat_agents["S4-Archivist"].archive(receipt)
        self.sat_agents["S5-Sentinel"].monitor(receipt)

        RETURN Acknowledgment(receipt.id)

    def query_resource_pool(self, node_id, query):
        """Node requests knowledge from the House of Wisdom."""
        IF node_id NOT IN self.connected_nodes:
            RETURN Rejection("not_registered")

        # Membrane filters the query
        filtered_query = self.membrane.filter_query(query)

        # S4-Archivist retrieves from House of Wisdom
        results = self.house_of_wisdom.search(filtered_query)

        # S2-Oracle verifies truth claims
        verified = self.sat_agents["S2-Oracle"].verify_truth(results)

        RETURN verified
```

### Module 2: Resource Pool (`core/urp/resource_pool.py`)

```pseudocode
class ResourcePool:
    """The 'Sea' — collective resources all nodes draw from."""

    def __init__(self):
        self.knowledge = []       # Curated knowledge entries
        self.seed_treasury = 0    # SEED token pool
        self.shared_reflexes = {} # Compiled patterns from all nodes
        self.gini_state = 0.0     # Current inequality measure

    def contribute_knowledge(self, node_id, knowledge_entry, receipt):
        """Node contributes knowledge through membrane."""
        # Ihsan gate: only quality contributions
        IF receipt.ihsan_score < IHSAN_FLOOR:
            RETURN False

        # Deduplicate (BLAKE3 hash check)
        IF self.is_duplicate(knowledge_entry):
            RETURN False

        self.knowledge.append(KnowledgeEntry(
            content=knowledge_entry,
            contributor=node_id,
            ihsan=receipt.ihsan_score,
            timestamp=now(),
            provenance_receipt=receipt.id,
        ))
        RETURN True

    def contribute_reflex(self, node_id, reflex, confidence):
        """Node shares a compiled reflex pattern."""
        IF confidence < REFLEX_SHARE_THRESHOLD:
            RETURN False

        key = reflex.pattern_hash
        IF key IN self.shared_reflexes:
            # Merge: increase confidence if independently discovered
            self.shared_reflexes[key].adopt(node_id, confidence)
        ELSE:
            self.shared_reflexes[key] = SharedReflex(
                pattern=reflex,
                creator=node_id,
                confidence=confidence,
                adopters=[node_id],
            )
        RETURN True

    def draw_knowledge(self, query, requesting_node):
        """Node draws from the collective knowledge."""
        # Search by semantic similarity
        results = faiss_search(self.knowledge, query)

        # Filter: don't return node's own contributions
        # (you already know what you contributed)
        results = [r for r in results if r.contributor != requesting_node]

        RETURN results
```

### Module 3: House of Wisdom (`core/urp/house_of_wisdom.py`)

```pseudocode
class HouseOfWisdom:
    """Governed knowledge substrate with provenance-tracked retrieval."""

    def __init__(self, constitution):
        self.constitution = constitution
        self.entries = []           # All knowledge entries
        self.embeddings = None      # FAISS index
        self.provenance_chain = []  # Who contributed what, when

    def ingest(self, entry, contributor, receipt):
        """Add knowledge with full provenance."""
        # S2-Oracle frozen check: does this contradict axioms?
        IF self.contradicts_axioms(entry):
            RETURN Rejection("axiom_violation")

        # S1-Validator: is the receipt valid?
        IF NOT receipt.verify_full():
            RETURN Rejection("invalid_receipt")

        self.entries.append(entry)
        self.provenance_chain.append(ProvenanceRecord(
            entry_hash=blake3(entry),
            contributor=contributor,
            receipt_id=receipt.id,
            timestamp=now(),
        ))

        # Rebuild search index
        self.rebuild_embeddings()
        RETURN True

    def search(self, query, top_k=5):
        """Semantic search with provenance."""
        results = self.embeddings.search(query, top_k)

        # Attach provenance to each result
        FOR result IN results:
            result.provenance = self.get_provenance(result.entry_hash)

        RETURN results
```

## TDD Anchors

### Test: Genesis creates URP
```
GIVEN Node0 activates
WHEN node0.mint_urp() is called
THEN URP has identity, constitution, resource pool, membrane
AND 5 SAT agents are deployed and working
AND genesis receipt is chained
AND genesis can only happen ONCE (second call fails)
```

### Test: Node connects through membrane
```
GIVEN URP is minted
WHEN node0.connect_to_urp() is called
THEN node0 is registered in connected_nodes
AND membrane verified node0's identity
AND connection receipt is issued
```

### Test: Receipt flows through membrane
```
GIVEN node0 is connected to URP
WHEN node0 submits a mission receipt with ihsan=0.97
THEN S1-Validator validates the receipt
AND S4-Archivist archives it in House of Wisdom
AND S5-Sentinel logs the activity
AND resource pool grows
```

### Test: Below-threshold receipt rejected
```
GIVEN node0 is connected to URP
WHEN node0 submits receipt with ihsan=0.80
THEN membrane REJECTS (constitutional filtering)
AND receipt is NOT archived
AND rejection is logged (dead-letter evidence)
```

### Test: Knowledge flows back to node
```
GIVEN URP has knowledge from multiple contributions
WHEN node0 queries "consensus algorithms"
THEN House of Wisdom returns relevant entries
AND each entry has provenance (who contributed, when, receipt_id)
AND S2-Oracle verified truth claims
```

### Test: Self-sustaining flywheel
```
GIVEN node0 has completed 10 missions
THEN URP resource pool has 10 validated receipts
AND SEED treasury has grown
AND shared reflexes may have been contributed
AND node0 can draw knowledge that improves future missions
```

## Implementation Order

1. `core/urp/__init__.py` — package
2. `core/urp/constitution.py` — ZANN_ZERO, RIBA_ZERO, IHSAN, GINI
3. `core/urp/membrane.py` — 4-property constitutional filter
4. `core/urp/resource_pool.py` — the "Sea"
5. `core/urp/house_of_wisdom.py` — governed knowledge substrate
6. `core/urp/service.py` — URP service orchestrator
7. `core/urp/genesis.py` — one-time URP mint
8. Wire into `MissionExecutor` — receipts flow to URP after Stage 4 (SEED)

## The Flywheel

```
Mission → Receipt → Membrane → URP → SAT process → Pool grows
   ↑                                                    │
   └────────── Node draws from pool ←──────────────────┘
```

This is the self-sustaining loop. Node0 doesn't need Node1 for it to start.
Node0 creates the URP. Node0 connects to the URP. The flywheel begins.
When Node1 joins, it connects to the same URP. The sea grows.
No node ever talks to another node. Only to the URP. Only through the membrane.
