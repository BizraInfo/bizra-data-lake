-- 001_kg_core.sql
-- BIZRA Knowledge Substrate v1 — Core Schema
-- Postgres + pgvector + append-only receipts
-- P1.11 Implementation

BEGIN;

-- ══════════════════════════════════════════════════════════════════════════════
-- EXTENSIONS
-- ══════════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ══════════════════════════════════════════════════════════════════════════════
-- DOCUMENTS (raw sources)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_documents (
  doc_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source           TEXT NOT NULL,                 -- e.g. "chatgpt_md", "claude_json", "codebase"
  source_ref       TEXT NOT NULL,                 -- filename/id/path
  model            TEXT,                          -- optional extraction model (openai/claude/deepseek)
  sha256           TEXT NOT NULL UNIQUE,          -- content hash for dedup
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
  text             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS kg_documents_source_idx ON kg_documents(source);
CREATE INDEX IF NOT EXISTS kg_documents_created_idx ON kg_documents(created_at);
CREATE INDEX IF NOT EXISTS kg_documents_meta_gin ON kg_documents USING GIN(metadata);

-- ══════════════════════════════════════════════════════════════════════════════
-- CHUNKS / HYPEREDGES (atomic evidence units)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_chunks (
  chunk_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id           UUID NOT NULL REFERENCES kg_documents(doc_id) ON DELETE CASCADE,
  span_start       INT NOT NULL,
  span_end         INT NOT NULL,
  content          TEXT NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  tags             TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  provenance       JSONB NOT NULL DEFAULT '{}'::jsonb,
  -- Provenance must include: source_doc, timestamp, hash, model (if AI-extracted)
  CONSTRAINT chunk_span_valid CHECK (span_end >= span_start)
);

CREATE INDEX IF NOT EXISTS kg_chunks_doc_idx ON kg_chunks(doc_id);
CREATE INDEX IF NOT EXISTS kg_chunks_tags_gin ON kg_chunks USING GIN(tags);
CREATE INDEX IF NOT EXISTS kg_chunks_prov_gin ON kg_chunks USING GIN(provenance);
CREATE INDEX IF NOT EXISTS kg_chunks_created_idx ON kg_chunks(created_at);

-- ══════════════════════════════════════════════════════════════════════════════
-- ENTITIES (canonical concepts; bind to LexiconLedger++)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_entities (
  entity_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  canonical        TEXT NOT NULL UNIQUE,          -- must exist in lexicon_v*.yaml
  entity_type      TEXT NOT NULL,                 -- CONCEPT/PROTOCOL/MODULE/AGENT/METRIC/TOKEN/...
  aliases          TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  weight           DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS kg_entities_type_idx ON kg_entities(entity_type);
CREATE INDEX IF NOT EXISTS kg_entities_aliases_gin ON kg_entities USING GIN(aliases);
CREATE INDEX IF NOT EXISTS kg_entities_weight_idx ON kg_entities(weight DESC);

-- ══════════════════════════════════════════════════════════════════════════════
-- MENTIONS (entity evidence inside chunks)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_mentions (
  mention_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chunk_id         UUID NOT NULL REFERENCES kg_chunks(chunk_id) ON DELETE CASCADE,
  entity_id        UUID NOT NULL REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
  confidence       DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
  role             TEXT NOT NULL DEFAULT 'MENTION', -- DEF/REQ/RISK/MEASURE/REWARD/VERIFY/...
  span_start       INT,                           -- optional: position within chunk
  span_end         INT,
  evidence         JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kg_mentions_chunk_idx ON kg_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS kg_mentions_entity_idx ON kg_mentions(entity_id);
CREATE INDEX IF NOT EXISTS kg_mentions_role_idx ON kg_mentions(role);

-- ══════════════════════════════════════════════════════════════════════════════
-- GRAPH EDGES (lightweight adjacency-list graph layer)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_edges (
  edge_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  src_entity_id    UUID NOT NULL REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
  dst_entity_id    UUID NOT NULL REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
  edge_type        TEXT NOT NULL DEFAULT 'CO_OCCURS',  -- CO_OCCURS/DEPENDS_ON/VERIFIES/RISKS/MEASURES/REWARDS/DEFINES/IMPLEMENTS/...
  weight           DOUBLE PRECISION NOT NULL DEFAULT 1 CHECK (weight >= 0),
  evidence_chunk_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
  -- Prevent self-loops
  CONSTRAINT no_self_loops CHECK (src_entity_id != dst_entity_id)
);

CREATE INDEX IF NOT EXISTS kg_edges_src_idx ON kg_edges(src_entity_id);
CREATE INDEX IF NOT EXISTS kg_edges_dst_idx ON kg_edges(dst_entity_id);
CREATE INDEX IF NOT EXISTS kg_edges_type_idx ON kg_edges(edge_type);
CREATE INDEX IF NOT EXISTS kg_edges_weight_idx ON kg_edges(weight DESC);

-- Composite index for graph traversal
CREATE INDEX IF NOT EXISTS kg_edges_src_type_idx ON kg_edges(src_entity_id, edge_type);

-- ══════════════════════════════════════════════════════════════════════════════
-- EMBEDDINGS (pgvector)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_embeddings (
  chunk_id         UUID PRIMARY KEY REFERENCES kg_chunks(chunk_id) ON DELETE CASCADE,
  embedding_model  TEXT NOT NULL,
  dims             INT NOT NULL CHECK (dims > 0),
  embedding        VECTOR(768),                   -- 768 for most sentence-transformers; migrate if needed
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- HNSW index for fast approximate nearest neighbor (production-grade)
CREATE INDEX IF NOT EXISTS kg_embeddings_hnsw
  ON kg_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ══════════════════════════════════════════════════════════════════════════════
-- RECEIPTS (append-only, audit-grade)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_receipts (
  receipt_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  kind             TEXT NOT NULL,                 -- INGEST/QUERY/SAPE/ELEVATION/GATE/...
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  policy_hash      TEXT NOT NULL,                 -- hash of active policy at time of decision
  ihsan            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {score, tier, gates_passed}
  sape             JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {cycle_id, phase, vector}
  snr              JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {budget, input_tokens, output_tokens, ratio}
  decision         TEXT NOT NULL,                 -- ALLOWED/REJECTED/ESCALATED
  rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb, -- array of {code, severity, message, repair_hint}
  evidence_refs    JSONB NOT NULL DEFAULT '[]'::jsonb,  -- array of {type, id, hash}
  payload          JSONB NOT NULL DEFAULT '{}'::jsonb,  -- operation-specific data
  signature        TEXT,                          -- optional cryptographic signature (required in prod)
  -- Ensure decision is valid
  CONSTRAINT valid_decision CHECK (decision IN ('ALLOWED', 'REJECTED', 'ESCALATED'))
);

CREATE INDEX IF NOT EXISTS kg_receipts_kind_idx ON kg_receipts(kind);
CREATE INDEX IF NOT EXISTS kg_receipts_decision_idx ON kg_receipts(decision);
CREATE INDEX IF NOT EXISTS kg_receipts_created_idx ON kg_receipts(created_at);
CREATE INDEX IF NOT EXISTS kg_receipts_payload_gin ON kg_receipts USING GIN(payload);

-- ══════════════════════════════════════════════════════════════════════════════
-- APPEND-ONLY ENFORCEMENT (critical for audit integrity)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION forbid_updates_deletes() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'BIZRA INTEGRITY VIOLATION: append-only table "%" does not allow % operations',
    TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql;

-- Receipts: strictly append-only
DROP TRIGGER IF EXISTS kg_receipts_no_update ON kg_receipts;
CREATE TRIGGER kg_receipts_no_update
BEFORE UPDATE OR DELETE ON kg_receipts
FOR EACH ROW EXECUTE FUNCTION forbid_updates_deletes();

-- Documents: append-only (can add, cannot modify or delete)
DROP TRIGGER IF EXISTS kg_documents_no_update ON kg_documents;
CREATE TRIGGER kg_documents_no_update
BEFORE UPDATE OR DELETE ON kg_documents
FOR EACH ROW EXECUTE FUNCTION forbid_updates_deletes();

-- ══════════════════════════════════════════════════════════════════════════════
-- HELPER FUNCTIONS
-- ══════════════════════════════════════════════════════════════════════════════

-- 1-hop neighbors (for graph traversal)
CREATE OR REPLACE FUNCTION kg_neighbors(
  p_entity_id UUID,
  p_edge_types TEXT[] DEFAULT NULL,
  p_limit INT DEFAULT 100
)
RETURNS TABLE(
  entity_id UUID,
  canonical TEXT,
  entity_type TEXT,
  edge_type TEXT,
  weight DOUBLE PRECISION,
  direction TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    e.dst_entity_id AS entity_id,
    ent.canonical,
    ent.entity_type,
    e.edge_type,
    e.weight,
    'outgoing'::TEXT AS direction
  FROM kg_edges e
  JOIN kg_entities ent ON ent.entity_id = e.dst_entity_id
  WHERE e.src_entity_id = p_entity_id
    AND (p_edge_types IS NULL OR e.edge_type = ANY(p_edge_types))
  UNION ALL
  SELECT 
    e.src_entity_id AS entity_id,
    ent.canonical,
    ent.entity_type,
    e.edge_type,
    e.weight,
    'incoming'::TEXT AS direction
  FROM kg_edges e
  JOIN kg_entities ent ON ent.entity_id = e.src_entity_id
  WHERE e.dst_entity_id = p_entity_id
    AND (p_edge_types IS NULL OR e.edge_type = ANY(p_edge_types))
  ORDER BY weight DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- N-hop path traversal (recursive CTE wrapper)
CREATE OR REPLACE FUNCTION kg_paths(
  p_start_entity_id UUID,
  p_max_hops INT DEFAULT 3,
  p_edge_types TEXT[] DEFAULT NULL,
  p_limit INT DEFAULT 50
)
RETURNS TABLE(
  path UUID[],
  path_labels TEXT[],
  total_weight DOUBLE PRECISION,
  hops INT
) AS $$
BEGIN
  RETURN QUERY
  WITH RECURSIVE paths AS (
    -- Base case: start node
    SELECT 
      ARRAY[p_start_entity_id] AS path,
      ARRAY[(SELECT canonical FROM kg_entities WHERE entity_id = p_start_entity_id)] AS path_labels,
      0::DOUBLE PRECISION AS total_weight,
      0 AS hops
    
    UNION ALL
    
    -- Recursive case: extend path
    SELECT
      p.path || e.dst_entity_id,
      p.path_labels || ent.canonical,
      p.total_weight + e.weight,
      p.hops + 1
    FROM paths p
    JOIN kg_edges e ON e.src_entity_id = p.path[array_length(p.path, 1)]
    JOIN kg_entities ent ON ent.entity_id = e.dst_entity_id
    WHERE p.hops < p_max_hops
      AND NOT (e.dst_entity_id = ANY(p.path))  -- prevent cycles
      AND (p_edge_types IS NULL OR e.edge_type = ANY(p_edge_types))
  )
  SELECT paths.path, paths.path_labels, paths.total_weight, paths.hops
  FROM paths
  WHERE paths.hops > 0
  ORDER BY paths.total_weight DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ══════════════════════════════════════════════════════════════════════════════
-- VIEWS (convenience)
-- ══════════════════════════════════════════════════════════════════════════════

-- Entity with mention count and edge count
CREATE OR REPLACE VIEW kg_entity_stats AS
SELECT 
  e.entity_id,
  e.canonical,
  e.entity_type,
  e.weight,
  COALESCE(m.mention_count, 0) AS mention_count,
  COALESCE(edge_out.out_edges, 0) + COALESCE(edge_in.in_edges, 0) AS total_edges
FROM kg_entities e
LEFT JOIN (
  SELECT entity_id, COUNT(*) AS mention_count
  FROM kg_mentions
  GROUP BY entity_id
) m ON m.entity_id = e.entity_id
LEFT JOIN (
  SELECT src_entity_id, COUNT(*) AS out_edges
  FROM kg_edges
  GROUP BY src_entity_id
) edge_out ON edge_out.src_entity_id = e.entity_id
LEFT JOIN (
  SELECT dst_entity_id, COUNT(*) AS in_edges
  FROM kg_edges
  GROUP BY dst_entity_id
) edge_in ON edge_in.dst_entity_id = e.entity_id;

-- Recent receipts summary
CREATE OR REPLACE VIEW kg_recent_receipts AS
SELECT 
  receipt_id,
  kind,
  decision,
  created_at,
  policy_hash,
  ihsan->>'tier' AS ihsan_tier,
  snr->>'ratio' AS snr_ratio,
  jsonb_array_length(rejection_reasons) AS rejection_count
FROM kg_receipts
ORDER BY created_at DESC
LIMIT 100;

COMMIT;
