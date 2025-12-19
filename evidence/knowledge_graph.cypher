// BIZRA Knowledge Graph
// Generated: 2025-12-19T18:11:05.176657
// Entities: 23
// Relationships: 16

CREATE (:Concept {id: 'concept_poi', "name": "PoI", "definition": "Proof-of-Impact Consensus", "confidence": 0.8, "ihsan_dimension": "auditability", "sape_module": null});
CREATE (:Concept {id: 'concept_proof_of_impact', "name": "Proof-of-Impact", "definition": "Proof-of-Impact Consensus", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Pattern {id: 'pattern_dual_agentic', "name": "Dual-Agentic", "definition": "Dual-Agentic Pattern", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Pattern {id: 'pattern_blockgraph', "name": "BlockGraph", "definition": "Hybrid Ledger Structure", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Concept {id: 'concept_mcp', "name": "MCP", "definition": "Model Context Protocol", "confidence": 0.8, "ihsan_dimension": "anti_centralization", "sape_module": null});
CREATE (:Concept {id: 'concept_a2a', "name": "A2A", "definition": "Agent-to-Agent Protocol", "confidence": 0.8, "ihsan_dimension": "anti_centralization", "sape_module": null});
CREATE (:Protocol {id: 'protocol_tmp', "name": "TMP", "definition": "Temporal Measurement Protocol", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Concept {id: 'concept_got', "name": "got", "definition": "Graph of Thoughts", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Concept {id: 'concept_إحسان', "name": "\u0625\u062d\u0633\u0627\u0646", "definition": "Excellence Principle", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Concept {id: 'concept_ihsan', "name": "ihsan", "definition": "Excellence Principle", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Concept {id: 'concept_node0', "name": "Node0", "definition": "Genesis Node", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Concept {id: 'concept_hrm_moe', "name": "HRM-MoE", "definition": "Hierarchical Reasoning Mixture-of-Experts", "confidence": 0.8, "ihsan_dimension": null, "sape_module": null});
CREATE (:Concept {id: 'concept_htdag', "name": "HTDAG", "definition": "Hierarchical Task Directed Acyclic Graph", "confidence": 0.8, "ihsan_dimension": "safety", "sape_module": null});
CREATE (:Pattern {id: 'pattern_causal_fabric', "name": "Causal Fabric", "definition": "Immutable Truth Ledger", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
CREATE (:Concept {id: 'concept_sape', "name": "SAPE", "definition": "Structured Agentic Prompt Engineering", "confidence": 0.8, "ihsan_dimension": null, "sape_module": null});
CREATE (:Agent {id: 'agent_reflector_agent', "name": "reflector-agent", "definition": "Learning Synthesizer", "confidence": 0.8, "ihsan_dimension": "safety", "sape_module": null});
CREATE (:Pattern {id: 'pattern_block_graph', "name": "Block graph", "definition": "Hybrid Ledger Structure", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Agent {id: 'agent_appagent', "name": "AppAgent", "definition": "Specialized Agent", "confidence": 0.8, "ihsan_dimension": null, "sape_module": null});
CREATE (:Concept {id: 'concept_hrmmoe', "name": "hrmMoE", "definition": "Hierarchical Reasoning Mixture-of-Experts", "confidence": 0.8, "ihsan_dimension": null, "sape_module": null});
CREATE (:Protocol {id: 'protocol_fate', "name": "FATE", "definition": "Fail-Closed Escalation Protocol", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Metric {id: 'metric_causal_drag', "name": "Causal Drag", "definition": "Structural Risk Metric", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Metric {id: 'metric_causaldrag', "name": "causalDrag", "definition": "Structural Risk Metric", "confidence": 0.8, "ihsan_dimension": "correctness", "sape_module": null});
CREATE (:Pattern {id: 'pattern_pat/sat', "name": "PAT/SAT", "definition": "Dual-Agentic Architecture", "confidence": 0.8, "ihsan_dimension": "user_benefit", "sape_module": null});
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_node0'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_causal_fabric'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_proof_of_impact'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_ihsan'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_mcp'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_ihsan'}), (b {id: 'protocol_tmp'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_ihsan'}), (b {id: 'metric_causal_drag'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'protocol_tmp'}), (b {id: 'metric_causal_drag'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_proof_of_impact'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_got'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_poi'}), (b {id: 'pattern_blockgraph'}) CREATE (a)-[:PART_OF]->(b);
MATCH (a {id: 'concept_proof_of_impact'}), (b {id: 'pattern_dual_agentic'}) CREATE (a)-[:PART_OF]->(b)