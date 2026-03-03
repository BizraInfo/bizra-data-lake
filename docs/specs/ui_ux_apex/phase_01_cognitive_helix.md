# UI/UX APEX — Phase 01: Cognitive Helix

> 3D living visualization of the user's unique Graph of Thoughts.
> Sprint priority: 3 (after Ghost Overlay + Iḥsān Gauge).

> Standing on Giants: Besta (GoT, 2024) · Shneiderman (information visualization, 1996) ·
> Tufte (data-ink ratio, 1983) · Al-Ghazali (self-knowledge via reflection)
> Repo anchors: `core/sovereign/graph_types.py`, `core/reasoning/graph_reasoner.py`

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| CH-01 | Render the user's live GoT as a 3D graph on BIZRA open |
| CH-02 | Node color encodes `ThoughtType` (hypothesis/evidence/synthesis…) |
| CH-03 | Node size encodes `snr_score` (0–1) |
| CH-04 | Edge opacity encodes `EdgeType` (supports=bright, refutes=dim) |
| CH-05 | HHMM prediction causes matched node to pulse bioluminescent gold |
| CH-06 | User can rotate/zoom/pan graph with mouse; tap-to-inspect node |
| CH-07 | "Semantic cluster" layout groups nodes by domain (Rust, Law, Finance…) |
| CH-08 | Helix idle animation: slow helical rotation at 3 rpm |
| CH-09 | Graceful empty state: Seed of Life SVG + "Your Helix is forming…" |
| CH-10 | Reduced-motion: freeze rotation, keep node colors and edges |

---

## 2. Edge Cases & Constraints

```
EDGE CASE: Graph has 0 nodes → render Seed of Life SVG placeholder (CH-09)
EDGE CASE: Graph has >10,000 nodes → enable WebGL LOD (Level-of-Detail)
           Nodes further than view-radius threshold render as billboard quads
EDGE CASE: HHMM pulse arrives faster than animation frame → queue pulses, max 3 concurrent
EDGE CASE: Node content too long → truncate at 80 chars in tooltip; full text in side panel
EDGE CASE: Duplicate node ids → deduplicate at the API boundary, log warning
CONSTRAINT: No raw GoT data stored in browser localStorage (privacy)
CONSTRAINT: Graph data fetched via authenticated WebSocket from Node0 kernel
CONSTRAINT: All node/edge rendering is client-side; server only streams graph diffs
CONSTRAINT: WebGL renderer must fall back to Canvas 2D if WebGL unavailable
```

---

## 3. Data Model (TypeScript, UI layer)

```typescript
// Mirrored from core/sovereign/graph_types.py (Python source of truth)

type ThoughtType =
  | "hypothesis" | "evidence" | "reasoning" | "synthesis"
  | "refinement" | "validation" | "conclusion" | "question" | "counterpoint";

type EdgeType =
  | "supports" | "refutes" | "derives" | "synthesizes"
  | "refines" | "questions" | "validates";

interface HelixNode {
  id: string;                  // UUID from Python ThoughtNode.id
  content: string;             // Truncated at 80 chars for label
  thought_type: ThoughtType;
  confidence: number;          // 0-1
  snr_score: number;           // 0-1 → maps to node radius
  domain_cluster: string;      // e.g. "rust", "law", "finance", "general"
  created_at: number;          // unix ms
  is_pulsing: boolean;         // set by HHMM prediction event
}

interface HelixEdge {
  id: string;
  from_node: string;
  to_node: string;
  edge_type: EdgeType;
  weight: number;              // 0-1
}

interface HelixGraph {
  nodes: HelixNode[];
  edges: HelixEdge[];
  snapshot_at: number;         // unix ms
  total_nodes: number;         // may exceed rendered count (LOD)
}
```

---

## 4. Pseudocode

### 4.1 HelixRenderer (WebGL)

```
MODULE HelixRenderer:

  CONSTANTS:
    NODE_TYPE_COLORS = {
      hypothesis:   #7a2ec9,  // Amethyst
      evidence:     #2e9aaa,  // Teal
      reasoning:    #C9A962,  // Genesis Gold
      synthesis:    #2eb86a,  // Operational Green
      refinement:   #c97e2e,  // Copper
      validation:   #8ac92e,  // Lime
      conclusion:   #F8F4EC,  // Ivory (brightest)
      question:     #2e56c9,  // Sovereign Blue
      counterpoint: #c93a4a,  // Alert Red
    }
    HELIX_RPM = 3.0
    MAX_RADIUS = 80           // WebGL units
    MIN_RADIUS = 8
    PULSE_DURATION_MS = 1200
    LOD_DISTANCE_THRESHOLD = 500

  STATE:
    graph: HelixGraph
    camera: Camera3D
    pulse_queue: Queue<{node_id, timestamp}>
    frame_time: number
    rotation_angle: number

  FUNCTION init(canvas_element):
    IF WebGL not supported:
      fallback_to_canvas2d()
      RETURN
    gl = WebGL2RenderingContext(canvas_element)
    compile_shaders(gl, NODE_SHADER, EDGE_SHADER, PULSE_SHADER)
    setup_camera_controls(canvas_element)  // orbit, zoom, pan
    websocket = open_kernel_ws("/api/v1/helix/stream")
    websocket.on("graph_snapshot", FUNCTION(data):
      graph = parse_helix_graph(data)
      layout_graph_3d(graph)
    )
    websocket.on("graph_diff", FUNCTION(diff):
      apply_diff(graph, diff)
      partial_relayout(diff.changed_nodes)
    )
    websocket.on("hhmm_prediction", FUNCTION(event):
      enqueue_pulse(event.node_id)
    )
    start_render_loop()

  FUNCTION layout_graph_3d(graph):
    // Force-directed layout with helical spine
    spine_y = linspace(-MAX_RADIUS, MAX_RADIUS, len(domain_clusters))
    FOR each domain_cluster:
      cluster_nodes = filter(graph.nodes, cluster == domain_cluster)
      center = (cos(cluster_angle) * 40, spine_y[i], sin(cluster_angle) * 40)
      apply_force_directed(cluster_nodes, center, radius=20)

  FUNCTION render_frame(timestamp):
    IF reduced_motion:
      rotation_angle = 0
    ELSE:
      rotation_angle += (HELIX_RPM * 2π / 60) * delta_ms / 1000

    clear(gl, color=#050B14)
    camera.rotate_y(rotation_angle)

    FOR each edge in graph.edges:
      opacity = edge_opacity(edge.edge_type, edge.weight)
      draw_edge(edge.from_pos, edge.to_pos, opacity)

    FOR each node in graph.nodes:
      IF distance(node.pos, camera) > LOD_DISTANCE_THRESHOLD:
        draw_billboard_quad(node)
      ELSE:
        radius = lerp(MIN_RADIUS, MAX_RADIUS, node.snr_score)
        color  = NODE_TYPE_COLORS[node.thought_type]
        pulse  = get_pulse_intensity(node.id, timestamp)
        draw_sphere(node.pos, radius, color, pulse_overlay=#C9A962 * pulse)

    drain_expired_pulses(timestamp)
    requestAnimationFrame(render_frame)

  FUNCTION get_pulse_intensity(node_id, now):
    FOR p in pulse_queue:
      IF p.node_id == node_id:
        elapsed = now - p.timestamp
        IF elapsed < PULSE_DURATION_MS:
          // Ease-out sine curve: bright at 0, zero at PULSE_DURATION_MS
          RETURN sin(π * (1 - elapsed/PULSE_DURATION_MS))
    RETURN 0.0

  FUNCTION enqueue_pulse(node_id):
    IF pulse_queue.count(active) >= 3:
      RETURN  // max concurrent pulses
    pulse_queue.push({node_id, timestamp: now()})

  FUNCTION on_node_click(node):
    open_node_detail_panel(node)  // side panel, full content, metadata

  FUNCTION edge_opacity(edge_type, weight):
    base = { supports: 1.0, refutes: 0.4, derives: 0.8,
             synthesizes: 0.9, refines: 0.7, questions: 0.5, validates: 0.85 }
    RETURN base[edge_type] * weight
```

### 4.2 Helix API Endpoint (Python — Node0 kernel)

```
MODULE HelixStreamEndpoint:
  // WebSocket endpoint: /api/v1/helix/stream

  ON connect(ws, user_id):
    graph = load_got_graph(user_id)           // from core/reasoning/graph_reasoner.py
    clusters = cluster_by_domain(graph)       // NLP domain labeling
    snapshot = build_helix_graph(graph, clusters)
    ws.send("graph_snapshot", snapshot)

    SUBSCRIBE to got_graph_updates(user_id):
      ON new_node(node):
        diff = {type:"add_node", node: to_helix_node(node)}
        ws.send("graph_diff", diff)
      ON new_edge(edge):
        diff = {type:"add_edge", edge: to_helix_edge(edge)}
        ws.send("graph_diff", diff)

    SUBSCRIBE to hhmm_predictions():
      ON prediction(node_id, confidence):
        IF confidence >= UNIFIED_SNR_THRESHOLD:   // from constants.py
          ws.send("hhmm_prediction", {node_id, confidence})

  FUNCTION cluster_by_domain(graph) -> Dict[node_id, cluster_label]:
    // Lightweight TF-IDF keyword matching against domain vocabulary
    // Domain vocab defined in config (not hardcoded)
    FOR node in graph.nodes:
      top_domain = argmax(tfidf_score(node.content, domain_vocabularies))
      node.domain_cluster = top_domain
    RETURN node_cluster_map
```

---

## 5. TDD Anchors

```python
# tests/ui_ux_apex/test_cognitive_helix.py

class TestHelixDataModel:
    def test_thought_types_match_python_enum(self):
        """TypeScript ThoughtType values must mirror core/sovereign/graph_types.ThoughtType."""
        from core.sovereign.graph_types import ThoughtType
        ts_types = {"hypothesis","evidence","reasoning","synthesis",
                    "refinement","validation","conclusion","question","counterpoint"}
        py_types = {t.value for t in ThoughtType}
        assert ts_types == py_types

    def test_node_snr_drives_radius(self):
        """snr_score=1.0 → MAX_RADIUS; snr_score=0.0 → MIN_RADIUS."""
        radius = lerp(MIN_RADIUS=8, MAX_RADIUS=80, t=1.0)
        assert radius == 80
        radius = lerp(MIN_RADIUS=8, MAX_RADIUS=80, t=0.0)
        assert radius == 8

class TestHelixStream:
    def test_empty_graph_returns_empty_snapshot(self, mock_ws, empty_got_graph):
        """0 nodes → snapshot with nodes=[], edges=[] (not error)."""
        endpoint = HelixStreamEndpoint()
        endpoint.on_connect(mock_ws, user_id="u_test")
        msg = mock_ws.last_sent("graph_snapshot")
        assert msg["nodes"] == []
        assert msg["edges"] == []

    def test_hhmm_prediction_below_threshold_not_sent(self, mock_ws, mock_hhmm):
        """Predictions below UNIFIED_SNR_THRESHOLD are not forwarded to UI."""
        from core.integration.constants import UNIFIED_SNR_THRESHOLD
        mock_hhmm.emit_prediction(node_id="n1", confidence=UNIFIED_SNR_THRESHOLD - 0.01)
        assert mock_ws.message_count("hhmm_prediction") == 0

    def test_pulse_queues_max_3_concurrent(self, helix_renderer):
        """Enqueuing 4 pulses when 3 are active drops the 4th."""
        for i in range(4):
            helix_renderer.enqueue_pulse(f"node_{i}")
        assert helix_renderer.pulse_queue.count_active() == 3
```
