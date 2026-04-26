"""Output CSV schemas for the Claude Cognitive Archive Pilot.

Each constant below defines the exact column order for one output CSV. Future
lanes (OpenAI, Gemini) MUST use the same column names so downstream tools
(adjudication, review_pack, promotion) work across lanes.

Provenance columns appear in every candidate-level CSV:
    source_lane, source_conversation_uuid, source_message_uuid, source_created_at

Candidate IDs are deterministic:
    candidate_id = sha256(candidate_type | normalized_text | source_message_uuid)[:16]
"""

from typing import List

# Stage 1 — Inventory
CONVERSATION_INVENTORY_COLS: List[str] = [
    "conversation_uuid",
    "name",
    "project_uuid",
    "project_name",
    "created_at",
    "updated_at",
    "turn_count",
    "user_turn_count",
    "assistant_turn_count",
    "total_chars",
    "user_chars",
    "topic_buckets",  # pipe-separated bucket names
]

PROJECT_INVENTORY_COLS: List[str] = [
    "project_uuid",
    "name",
    "description",
    "is_starred",
    "created_at",
    "updated_at",
    "conversation_count",
]

TOPIC_BUCKET_COUNT_COLS: List[str] = [
    "bucket_name",
    "conversation_count",
    "user_message_count",
    "total_chars",
]

TOP_SIGNAL_SESSION_COLS: List[str] = [
    "rank",
    "conversation_uuid",
    "name",
    "signal_score",
    "turn_count",
    "user_turn_count",
    "total_chars",
    "topic_buckets",
]

# Stage 2 — Distillation
FACT_CANDIDATE_COLS: List[str] = [
    "candidate_id",
    "candidate_type",  # literal "fact"
    "content",
    "entity",
    "predicate",
    "normalized_text",
    "pattern_matched",
    "source_lane",
    "source_conversation_uuid",
    "source_conversation_name",
    "source_message_uuid",
    "source_created_at",
]

DECISION_CANDIDATE_COLS: List[str] = [
    "candidate_id",
    "candidate_type",  # literal "decision"
    "content",
    "normalized_text",
    "pattern_matched",
    "source_lane",
    "source_conversation_uuid",
    "source_conversation_name",
    "source_message_uuid",
    "source_created_at",
]

CONTRADICTION_CANDIDATE_COLS: List[str] = [
    "candidate_id",
    "candidate_type",  # literal "contradiction"
    "entity",
    "predicate",
    "value_a",
    "value_a_source_message_uuid",
    "value_a_created_at",
    "value_b",
    "value_b_source_message_uuid",
    "value_b_created_at",
    "source_lane",
    "source_conversation_uuids",  # pipe-separated
]

REASONING_EXEMPLAR_COLS: List[str] = [
    "candidate_id",
    "candidate_type",  # literal "reasoning_exemplar"
    "content_excerpt",  # first 600 chars
    "content_char_count",
    "speaker",  # human | assistant
    "marker_keywords_present",  # pipe-separated
    "source_lane",
    "source_conversation_uuid",
    "source_conversation_name",
    "source_message_uuid",
    "source_created_at",
]

# Stage 3 — Adjudication
CANONICAL_CANDIDATE_FACT_COLS: List[str] = [
    "cluster_id",
    "candidate_id",  # the canonical candidate's id from Stage 2
    "canonical_content",
    "entity",
    "predicate",
    "supporting_count",
    "supporting_candidate_ids",  # pipe-separated
    "most_recent_source_created_at",
    "earliest_source_created_at",
    "source_lane",
]

CANONICAL_CANDIDATE_DECISION_COLS: List[str] = [
    "cluster_id",
    "candidate_id",
    "canonical_content",
    "supporting_count",
    "supporting_candidate_ids",
    "most_recent_source_created_at",
    "earliest_source_created_at",
    "source_lane",
]

HYPOTHESIS_CANDIDATE_COLS: List[str] = [
    "candidate_id",
    "candidate_type_origin",  # fact | decision
    "content",
    "reason_flagged",
    "occurrences",
    "source_lane",
    "source_conversation_uuid",
    "source_message_uuid",
    "source_created_at",
]

OBSOLETE_CONFLICTED_CANDIDATE_COLS: List[str] = [
    "candidate_id",
    "candidate_type_origin",  # fact | decision
    "content",
    "entity",
    "predicate",
    "superseded_by_candidate_id",
    "superseded_by_content",
    "delta_days",
    "source_lane",
    "source_conversation_uuid",
    "source_message_uuid",
    "source_created_at",
]

CLUSTER_REGISTRY_COLS: List[str] = [
    "cluster_id",
    "cluster_type",  # fact | decision
    "member_count",
    "canonical_candidate_id",
    "entity",
    "predicate",
    "member_candidate_ids",  # pipe-separated
    "earliest_source_created_at",
    "most_recent_source_created_at",
]

# Stage 4 — Review Pack
REVIEW_WORKBOOK_COLS: List[str] = [
    "row_id",
    "candidate_type",  # fact | decision | hypothesis | obsolete
    "cluster_id",
    "candidate_id",
    "content",
    "entity",
    "predicate",
    "supporting_count",
    "provenance_conversation_uuids",  # pipe-separated
    "provenance_earliest",
    "provenance_most_recent",
    "source_lane",
    "review_status",  # pending_review (initial) | approved | rejected | needs_followup
    "reviewer_notes",  # initial blank
    "promote_to_canon",  # initial blank — NEVER set by the pipeline
]


# Literal strings used across modules. Keep in one place so refactors don't
# drift.
INITIAL_REVIEW_STATUS: str = "pending_review"
SOURCE_LANE_NAME: str = "claude"
