# Phase 72.05: API Integration + Wiring

**Target files:** `core/sovereign/api.py` (patch), test files

## Purpose

Expose Node Value and Network Effect via REST endpoints. Wire the
NodeValueEngine into SovereignRuntime so it auto-records missions.

## New Endpoints

```pseudocode
# ─────────────────────────────────────────────────────────────
# GET /v1/node/value — Current node value composite KPI
# ─────────────────────────────────────────────────────────────

@app.get("/v1/node/value")
FUNCTION get_node_value(request: Request):
    _authenticate_http_request(request)  # existing auth guard
    nv_engine = runtime.node_value_engine
    IF nv_engine IS None:
        RETURN {"error": "NodeValueEngine not initialized"}, 503
    snapshot = nv_engine.compute()
    RETURN {
        "potential": snapshot.potential,
        "activation_rate": snapshot.activation_rate,
        "verification_quality": snapshot.verification_quality,
        "compounding_time": snapshot.compounding_time,
        "network_synergy": snapshot.network_synergy,
        "composite": snapshot.composite,
        "tier": snapshot.tier,
        "human_stage": snapshot.human_stage,
        "timestamp": snapshot.timestamp,
    }

# ─────────────────────────────────────────────────────────────
# GET /v1/node/lifecycle — Current stage with progress
# ─────────────────────────────────────────────────────────────

@app.get("/v1/node/lifecycle")
FUNCTION get_lifecycle(request: Request):
    _authenticate_http_request(request)
    seed = runtime.seed_engine
    IF seed IS None:
        RETURN {"error": "SeedEngine not initialized"}, 503
    pot = seed.potential()
    RETURN stage_progress(pot.sovereignty_score)

# ─────────────────────────────────────────────────────────────
# GET /v1/network/effect?nodes=N — Network effect projection
# ─────────────────────────────────────────────────────────────

@app.get("/v1/network/effect")
FUNCTION get_network_effect(request: Request, nodes: int = 1000):
    _authenticate_http_request(request)
    IF nodes < 1 OR nodes > 10_000_000_000:
        RETURN {"error": "nodes must be 1..10B"}, 400
    estimator = NetworkEffectEstimator()
    projection = estimator.project(nodes)
    RETURN {
        "nodes": projection.nodes,
        "skills_available": projection.skills_available,
        "compute_tflops": projection.compute_tflops,
        "latency_factor": projection.latency_factor,
        "intelligence_density": projection.intelligence_density,
        "cost_per_node": projection.cost_per_node,
    }

# ─────────────────────────────────────────────────────────────
# GET /v1/network/milestones — Standard milestone projections
# ─────────────────────────────────────────────────────────────

@app.get("/v1/network/milestones")
FUNCTION get_milestones(request: Request):
    _authenticate_http_request(request)
    estimator = NetworkEffectEstimator()
    milestones = estimator.project_milestones()
    RETURN {
        "milestones": [
            {
                "nodes": m.nodes,
                "skills": m.skills_available,
                "tflops": m.compute_tflops,
                "latency_factor": m.latency_factor,
            }
            FOR m IN milestones
        ]
    }
```

## Runtime Wiring

```pseudocode
# In SovereignRuntime.__init__() or _init_seed_engine():

FUNCTION _init_node_value_engine(self):
    """Wire NodeValueEngine to SeedEngine."""
    IF self._seed_engine IS NOT None:
        self._node_value_engine = NodeValueEngine(
            seed_engine=self._seed_engine,
            genesis_timestamp=self._genesis_timestamp,
        )
    ELSE:
        self._node_value_engine = None

# In mission completion handler:
FUNCTION _on_mission_complete(self, result):
    # ... existing seed_engine.record_episode() call ...
    IF self._node_value_engine IS NOT None:
        self._node_value_engine.record_mission()
```

## Health Integration

```pseudocode
# Add to existing /v1/health response:

health_response["node_value"] = (
    runtime.node_value_engine.health()
    IF runtime.node_value_engine
    ELSE {"active": False}
)
```

## TDD Anchors

```pseudocode
TEST "GET /v1/node/value returns valid JSON":
    client = TestClient(app)
    response = client.get("/v1/node/value", headers=auth_headers)
    ASSERT response.status_code == 200
    data = response.json()
    ASSERT "composite" IN data
    ASSERT "human_stage" IN data
    ASSERT "tier" IN data

TEST "GET /v1/node/lifecycle returns stage progress":
    client = TestClient(app)
    response = client.get("/v1/node/lifecycle", headers=auth_headers)
    ASSERT response.status_code == 200
    data = response.json()
    ASSERT "current_stage" IN data
    ASSERT "progress" IN data

TEST "GET /v1/network/effect requires auth":
    client = TestClient(app)
    response = client.get("/v1/network/effect")
    ASSERT response.status_code IN [401, 403]

TEST "GET /v1/network/effect with valid nodes":
    client = TestClient(app)
    response = client.get("/v1/network/effect?nodes=1000", headers=auth_headers)
    ASSERT response.status_code == 200
    data = response.json()
    ASSERT data["nodes"] == 1000
    ASSERT data["skills_available"] == 50000

TEST "GET /v1/network/effect rejects nodes=0":
    client = TestClient(app)
    response = client.get("/v1/network/effect?nodes=0", headers=auth_headers)
    ASSERT response.status_code == 400

TEST "GET /v1/network/milestones returns 8 entries":
    client = TestClient(app)
    response = client.get("/v1/network/milestones", headers=auth_headers)
    ASSERT response.status_code == 200
    data = response.json()
    ASSERT len(data["milestones"]) == 8

TEST "mission completion increments node value":
    # Simulate runtime with seed engine + node value engine
    runtime = create_test_runtime()
    initial = runtime.node_value_engine.compute()
    runtime.complete_mission({"snr": 0.95, "ihsan": 0.96})
    after = runtime.node_value_engine.compute()
    ASSERT after.composite >= initial.composite

TEST "health includes node_value section":
    client = TestClient(app)
    response = client.get("/v1/health")
    data = response.json()
    ASSERT "node_value" IN data
```
