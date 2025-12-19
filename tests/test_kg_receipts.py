#!/usr/bin/env python3
"""
tests/test_kg_receipts.py — CI tests for Knowledge Substrate receipts

Tests:
1. Receipt insertion works
2. Append-only enforcement (UPDATE/DELETE fail)
3. Receipt structure validation
"""

import json
import os
import pytest
from datetime import datetime, timezone
from uuid import uuid4

# Skip if psycopg not available
psycopg = pytest.importorskip("psycopg")

PG_DSN = os.environ.get(
    "BIZRA_PG_DSN",
    "postgresql://bizra:bizra_dev_password@localhost:5432/bizra"
)


@pytest.fixture(scope="module")
def db_connection():
    """Create database connection for tests."""
    try:
        conn = psycopg.connect(PG_DSN)
        yield conn
        conn.close()
    except psycopg.OperationalError:
        pytest.skip("Database not available")


class TestReceiptAppendOnly:
    """Test append-only enforcement on receipts table."""
    
    def test_insert_receipt(self, db_connection):
        """Test that receipt insertion works."""
        conn = db_connection
        
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kg_receipts (kind, policy_hash, decision, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING receipt_id
                """,
                ("TEST", "test-policy-hash", "ALLOWED", json.dumps({"test": True}))
            )
            receipt_id = cur.fetchone()[0]
        
        conn.commit()
        assert receipt_id is not None
    
    def test_update_receipt_fails(self, db_connection):
        """Test that UPDATE on receipts raises exception."""
        conn = db_connection
        
        # First insert a receipt
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kg_receipts (kind, policy_hash, decision, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING receipt_id
                """,
                ("TEST", "test-policy-hash", "ALLOWED", json.dumps({"test": True}))
            )
            receipt_id = cur.fetchone()[0]
        conn.commit()
        
        # Attempt to UPDATE should fail
        with pytest.raises(psycopg.errors.RaiseException) as exc_info:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE kg_receipts 
                    SET decision = 'REJECTED' 
                    WHERE receipt_id = %s
                    """,
                    (receipt_id,)
                )
            conn.commit()
        
        conn.rollback()
        assert "append-only" in str(exc_info.value).lower()
    
    def test_delete_receipt_fails(self, db_connection):
        """Test that DELETE on receipts raises exception."""
        conn = db_connection
        
        # First insert a receipt
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kg_receipts (kind, policy_hash, decision, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING receipt_id
                """,
                ("TEST", "test-policy-hash", "ALLOWED", json.dumps({"test": True}))
            )
            receipt_id = cur.fetchone()[0]
        conn.commit()
        
        # Attempt to DELETE should fail
        with pytest.raises(psycopg.errors.RaiseException) as exc_info:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM kg_receipts 
                    WHERE receipt_id = %s
                    """,
                    (receipt_id,)
                )
            conn.commit()
        
        conn.rollback()
        assert "append-only" in str(exc_info.value).lower()


class TestDocumentAppendOnly:
    """Test append-only enforcement on documents table."""
    
    def test_insert_document(self, db_connection):
        """Test that document insertion works."""
        conn = db_connection
        
        unique_hash = f"test-{uuid4().hex}"
        
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kg_documents (source, source_ref, sha256, text)
                VALUES (%s, %s, %s, %s)
                RETURNING doc_id
                """,
                ("test", "test-ref", unique_hash, "Test document content")
            )
            doc_id = cur.fetchone()[0]
        
        conn.commit()
        assert doc_id is not None
    
    def test_update_document_fails(self, db_connection):
        """Test that UPDATE on documents raises exception."""
        conn = db_connection
        
        unique_hash = f"test-{uuid4().hex}"
        
        # First insert a document
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kg_documents (source, source_ref, sha256, text)
                VALUES (%s, %s, %s, %s)
                RETURNING doc_id
                """,
                ("test", "test-ref", unique_hash, "Test document content")
            )
            doc_id = cur.fetchone()[0]
        conn.commit()
        
        # Attempt to UPDATE should fail
        with pytest.raises(psycopg.errors.RaiseException) as exc_info:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE kg_documents 
                    SET text = 'Modified content' 
                    WHERE doc_id = %s
                    """,
                    (doc_id,)
                )
            conn.commit()
        
        conn.rollback()
        assert "append-only" in str(exc_info.value).lower()


class TestReceiptModule:
    """Test the kg.receipts module."""
    
    def test_emit_receipt(self, db_connection):
        """Test emit_receipt function."""
        # Import here to avoid import errors if psycopg not available
        from kg.receipts import emit_receipt, ReceiptKind, Decision
        
        receipt_id = emit_receipt(
            conn=db_connection,
            kind=ReceiptKind.QUERY,
            decision=Decision.ALLOWED,
            evidence_refs=[{"type": "test", "id": "test-1"}],
            payload={"test": True, "query": "test query"}
        )
        
        assert receipt_id is not None
        assert len(receipt_id) == 36  # UUID format
    
    def test_get_receipt(self, db_connection):
        """Test get_receipt function."""
        from kg.receipts import emit_receipt, get_receipt, ReceiptKind, Decision
        
        receipt_id = emit_receipt(
            conn=db_connection,
            kind=ReceiptKind.QUERY,
            decision=Decision.ALLOWED,
            evidence_refs=[{"type": "test", "id": "test-2"}],
            payload={"test": True}
        )
        
        receipt = get_receipt(db_connection, receipt_id)
        
        assert receipt is not None
        assert receipt["receipt_id"] == receipt_id
        assert receipt["kind"] == "QUERY"
        assert receipt["decision"] == "ALLOWED"
    
    def test_rejected_receipt(self, db_connection):
        """Test rejected receipt with rejection reasons."""
        from kg.receipts import emit_receipt, get_receipt, ReceiptKind, Decision, RejectionReason
        
        receipt_id = emit_receipt(
            conn=db_connection,
            kind=ReceiptKind.QUERY,
            decision=Decision.REJECTED,
            evidence_refs=[],
            payload={"query": "no results"},
            rejection_reasons=[
                RejectionReason(
                    code="INSUFFICIENT_EVIDENCE",
                    severity="HIGH",
                    message="No matching evidence",
                    repair_hint="Ingest more sources"
                )
            ]
        )
        
        receipt = get_receipt(db_connection, receipt_id)
        
        assert receipt["decision"] == "REJECTED"
        assert len(receipt["rejection_reasons"]) == 1
        assert receipt["rejection_reasons"][0]["code"] == "INSUFFICIENT_EVIDENCE"


class TestEmbeddings:
    """Test embedding module."""
    
    def test_null_embedder(self):
        """Test NullEmbedder produces consistent output."""
        from kg.embeddings import NullEmbedder
        
        embedder = NullEmbedder(dims=768)
        
        result1 = embedder.embed("test text")
        result2 = embedder.embed("test text")
        
        assert result1.vector == result2.vector  # Deterministic
        assert len(result1.vector) == 768
        assert result1.model == "null"
    
    def test_null_embedder_different_texts(self):
        """Test NullEmbedder produces different vectors for different texts."""
        from kg.embeddings import NullEmbedder
        
        embedder = NullEmbedder(dims=768)
        
        result1 = embedder.embed("text one")
        result2 = embedder.embed("text two")
        
        assert result1.vector != result2.vector
    
    def test_get_embedder_default(self):
        """Test get_embedder returns NullEmbedder by default."""
        from kg.embeddings import get_embedder, NullEmbedder
        
        # Clear any env override
        old_val = os.environ.pop("BIZRA_EMBEDDER", None)
        try:
            embedder = get_embedder()
            assert isinstance(embedder, NullEmbedder)
        finally:
            if old_val:
                os.environ["BIZRA_EMBEDDER"] = old_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
