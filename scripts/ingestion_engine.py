"""
Elite-grade ingestion pipeline with SNR gating, membrane filters, and graph projection.
Runs standalone for demo purposes; zero external dependencies.
"""

import json
import math
import re
import time
import uuid
import hashlib
import os
import logging
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# --- CONFIGURATION & CONSTANTS ---
SNR_THRESHOLD_REJECT = 0.4  # Noise
SNR_THRESHOLD_PROMOTE = 0.65  # Candidate for KG (calibrated to current SNR scale)
SNR_THRESHOLD_AUTO_CERT = 0.95  # Auto-cert gate
GATEWAY_HOST = os.environ.get("BIZRA_GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("BIZRA_GATEWAY_PORT", "8081"))
GATEWAY_BASE = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

# Regex Patterns for "The Membrane"
SENSITIVE_PATTERN = re.compile(r"(confidential|top secret|private key|password)", re.IGNORECASE)
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "seems", "likely", "guessing"}
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9_]+\b")  # Simple capitalized words as proxy for entities


class ArtifactStatus(Enum):
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    FLAGGED = "FLAGGED"
    PROVISIONAL = "PROVISIONAL"
    PASSED = "PASSED"  # Maintained for compatibility
    CERTIFIED = "CERTIFIED"


@dataclass
class IngestionPayload:
    source_id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, int] = field(default_factory=dict)

    # Internal Processing State
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: ArtifactStatus = ArtifactStatus.PENDING
    tags: Set[str] = field(default_factory=set)
    snr_score: float = 0.0
    got_trace: List[str] = field(default_factory=list)  # Trace of GoT nodes visited


@dataclass
class GraphNode:
    id: str
    type: str
    data: Dict[str, Any]


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class KnowledgeGraph:
    """A lightweight, zero-dependency in-memory Knowledge Graph."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.indices: Dict[str, List[str]] = defaultdict(list)  # Adjacency list

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node
        print(f"  [GRAPH] Node Added: {node.id} ({node.type})")

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0):
        edge = GraphEdge(source, target, relation, weight)
        self.edges.append(edge)
        self.indices[source].append(target)
        print(f"  [GRAPH] Edge Created: {source} --{relation}--> {target}")

    def get_stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


class SNRCalculator:
    """The Logic Engine for scoring Signal-to-Noise."""

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Approximation of Shannon entropy for text complexity."""
        if not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    @staticmethod
    def score(payload: IngestionPayload) -> float:
        text = payload.content
        words = text.split()
        if not words:
            return 0.0

        # Density (D): unique entities / tokens
        entities = ENTITY_PATTERN.findall(text)
        density_score = min(len(set(entities)) / len(words), 1.0)

        # Verifiability (V): citations, timestamps, signatures
        verifiability_score = 0.0
        if "http" in text or "cite:" in text or re.search(r"\d{4}-\d{2}-\d{2}", text):
            verifiability_score += 0.25
        if re.search(r"\d{4}-\d{2}-\d{2}", text):
            verifiability_score += 0.25
        if "sig:" in text or "signature" in text.lower():
            verifiability_score += 0.15
        verifiability_score = min(verifiability_score, 1.0)

        # Ambiguity (A): hedge words frequency
        hedge_count = sum(1 for w in words if w.lower() in HEDGE_WORDS)
        ambiguity_score = hedge_count / len(words)

        # SNR = (w1 * D + w2 * V) / (1 + w3 * A)
        w1, w2, w3 = 0.55, 0.55, 1.0
        snr = (w1 * density_score + w2 * verifiability_score) / (1 + w3 * ambiguity_score)
        return max(0.0, min(1.0, snr))  # Clamp between 0 and 1


class IngestionEngine:
    """The GoT Orchestrator for Ingestion."""

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.history_log: List[IngestionPayload] = []
        self.blocklist_sources: Set[str] = set()
        self.immutable_log: Set[str] = set()  # Content hashes for dedup
        self.metrics = {
            "ingested": 0,
            "rejected": 0,
            "flagged": 0,
            "provisional": 0,
            "certified": 0,
        }

    def log_step(self, payload: IngestionPayload, step_name: str, msg: str):
        entry = f"[{step_name.upper()}] {msg}"
        payload.got_trace.append(entry)

    # --- GoT NODE 1: INTAKE & MEMBRANE ---
    def node_membrane(self, payload: IngestionPayload) -> bool:
        self.log_step(payload, "Membrane", "Scanning regex patterns...")

        if payload.source_id in self.blocklist_sources:
            self.log_step(payload, "Membrane", "Blocked source detected.")
            payload.status = ArtifactStatus.REJECTED
            return False

        if SENSITIVE_PATTERN.search(payload.content):
            self.log_step(payload, "Membrane", "CRITICAL: Sensitive data pattern detected.")
            payload.status = ArtifactStatus.FLAGGED
            payload.tags.add("SECURITY_RISK")
            return False

        if "URGENT" in payload.content.upper():
            payload.tags.add("PRIORITY_HIGH")
        if "INTEL" in payload.content.upper():
            payload.tags.add("PRIORITY_INTEL")

        digest = hashlib.sha256(payload.content.encode("utf-8")).hexdigest()
        if digest in self.immutable_log:
            self.log_step(payload, "Membrane", "Duplicate payload detected (ImmutableLog).")
            payload.status = ArtifactStatus.REJECTED
            return False
        self.immutable_log.add(digest)

        return True

    # --- GoT NODE 2: THE PRISM (SCORING) ---
    def node_prism(self, payload: IngestionPayload) -> bool:
        self.log_step(payload, "Prism", "Calculating SNR...")
        score = SNRCalculator.score(payload)
        payload.snr_score = score
        self.log_step(payload, "Prism", f"SNR Score assigned: {score:.3f}")

        if score < SNR_THRESHOLD_REJECT:
            payload.status = ArtifactStatus.REJECTED
            self.log_step(payload, "Prism", "SNR too low. Rejecting (Noise).")
            return False
        if score < SNR_THRESHOLD_PROMOTE:
            payload.status = ArtifactStatus.PROVISIONAL
            payload.tags.add("RAW_LOG")
            self.log_step(payload, "Prism", "SNR logged as raw data (Provisional).")
            return False  # Do not promote further
        payload.status = ArtifactStatus.PASSED
        self.log_step(payload, "Prism", "High Signal detected. Candidate for Council.")
        return True

    # --- GoT NODE 3: COUNCIL REVIEW ---
    def node_council(self, payload: IngestionPayload) -> bool:
        self.log_step(payload, "Council", "Determining Certification Level...")

        if payload.status == ArtifactStatus.REJECTED:
            return False

        trust = payload.metadata.get("source_trust", 0.9)

        if payload.snr_score >= SNR_THRESHOLD_AUTO_CERT and trust > 0.9:
            payload.status = ArtifactStatus.CERTIFIED
            payload.metadata["certifier"] = "ALGORITHM_V1_AUTO"
            self.log_step(payload, "Council", "AUTO-CERTIFICATION GRANTED.")
            return True

        if payload.status == ArtifactStatus.FLAGGED or payload.tags.intersection({"SECURITY_RISK", "PRIORITY_HIGH"}):
            self.log_step(payload, "Council", "Routing to Level 2 review... (Simulation: Approved)")
            payload.status = ArtifactStatus.CERTIFIED
            payload.metadata["certifier"] = "HUMAN_OVERSIGHT_SIM"
            return True

        if payload.snr_score >= SNR_THRESHOLD_PROMOTE:
            payload.status = ArtifactStatus.CERTIFIED
            payload.metadata["certifier"] = "STANDARD_PROTOCOL"
            return True

        payload.status = ArtifactStatus.PROVISIONAL
        return False

    # --- GoT NODE 4: GRAPH PROJECTION ---
    def node_graph_projection(self, payload: IngestionPayload):
        if payload.status != ArtifactStatus.CERTIFIED:
            return

        self.log_step(payload, "Graph", "Projecting into Knowledge Graph...")

        source_node = GraphNode(id=payload.source_id, type="SOURCE", data={"trust": payload.metadata.get("source_trust", 0.9)})
        self.kg.add_node(source_node)

        content_id = f"claim_{payload.trace_id}"
        content_node = GraphNode(id=content_id, type="CLAIM", data={"text": payload.content[:50] + "..."})
        self.kg.add_node(content_node)

        self.kg.add_edge(payload.source_id, content_id, "DERIVED_FROM", weight=payload.snr_score)
        self.kg.add_edge(content_id, payload.metadata.get("certifier", "STANDARD_PROTOCOL"), "CERTIFIED_BY", weight=payload.snr_score)

        entities = set(ENTITY_PATTERN.findall(payload.content))
        for ent in entities:
            ent_id = f"ent_{ent.upper()}"
            if ent_id not in self.kg.nodes:
                self.kg.add_node(GraphNode(id=ent_id, type="ENTITY", data={"name": ent}))
            self.kg.add_edge(content_id, ent_id, "MENTIONS", weight=1.0)

        self.emit_frame(payload, content_id)

    def emit_frame(self, payload: IngestionPayload, claim_id: str):
        """Send certified frame to the Rust gateway."""
        frame = {
            "content": payload.content,
            "source_id": payload.source_id,
            "claim_id": claim_id,
            "snr": payload.snr_score,
            "timestamp": payload.timestamp,
            "certifier": payload.metadata.get("certifier", "STANDARD_PROTOCOL"),
            "tags": sorted(payload.tags),
        }
        req = urllib.request.Request(
            f"{GATEWAY_BASE}/frames",
            data=json.dumps(frame).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                logging.info(json.dumps({
                    "event": "gateway_emit_ok",
                    "status": resp.status,
                    "claim_id": claim_id,
                    "trace_id": payload.trace_id,
                    "snr": payload.snr_score,
                }))
        except urllib.error.URLError as e:
            logging.warning(json.dumps({
                "event": "gateway_emit_failed",
                "error": str(e),
                "claim_id": claim_id,
                "trace_id": payload.trace_id,
                "snr": payload.snr_score,
            }))

    # --- MASTER EXECUTION LOOP ---
    def ingest(self, source: str, data: str) -> IngestionPayload:
        self.metrics["ingested"] += 1
        logging.info(json.dumps({"event": "ingest_start", "source": source}))
        payload = IngestionPayload(source_id=source, content=data)

        t0 = time.perf_counter()
        mem_ok = self.node_membrane(payload)
        payload.stage_timings["membrane"] = time.perf_counter() - t0

        if mem_ok:
            t1 = time.perf_counter()
            prism_ok = self.node_prism(payload)
            payload.stage_timings["prism"] = time.perf_counter() - t1

            if prism_ok:
                t2 = time.perf_counter()
                council_ok = self.node_council(payload)
                payload.stage_timings["council"] = time.perf_counter() - t2

                if council_ok:
                    t3 = time.perf_counter()
                    self.node_graph_projection(payload)
                    payload.stage_timings["graph"] = time.perf_counter() - t3

        self.history_log.append(payload)
        status_key = payload.status.value.lower()
        if status_key in self.metrics:
            self.metrics[status_key] += 1
        logging.info(json.dumps({
            "event": "ingest_complete",
            "source": source,
            "status": payload.status.value,
            "trace": payload.got_trace,
            "timings_ms": {k: round(v * 1000, 2) for k, v in payload.stage_timings.items()},
            "snr": payload.snr_score,
            "tags": sorted(payload.tags),
            "metrics": self.metrics,
        }))
        return payload


# --- DEMONSTRATION RUNNER ---
def run_demo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    engine = IngestionEngine()

    data_1 = """
    The BIZRA Protocol v4.0 explicitly defines the interaction between Rust Kernels and Python Agents.
    Benchmarks indicate a 400% increase in throughput compared to legacy systems.
    Source: cite:whitepaper_2025, signature: sig:abc123, dated: 2025-12-01.
    """
    engine.ingest("agent_alpha", data_1)

    data_2 = """
    I think maybe the system serves some purpose, but I am guessing it might fail. 
    Possibly we should look at other things.
    """
    engine.ingest("agent_beta", data_2)

    data_3 = """
    Here is the TOP SECRET config for the genesis node.
    """
    engine.ingest("agent_gamma", data_3)

    data_4 = "BIZRA IHSAAN Z3 COUNCIL RUST KERNEL FATE cite:ihsan-ledger-2025 signature:sig-ledger-verified timestamp:2025-07-10"
    engine.ingest("agent_delta", data_4)

    logging.info(json.dumps({"event": "kg_stats", **engine.kg.get_stats()}))


if __name__ == "__main__":
    run_demo()
