# Phase 57.06: Test Plan — TDD Anchors

## Test Strategy

Three test tiers:
1. **Unit tests** — Each component in isolation (fast, no network, no AHK)
2. **Integration tests** — Pipeline stages wired together (may need LivingMemory)
3. **E2E tests** — Full mission flow (needs Python bridge, optional AHK/LLM)

## Unit Tests

### test_mission_orchestrator.py

```python
class TestMissionRequest:
    def test_mission_request_creation(self):
        """MissionRequest holds all required fields."""
        req = MissionRequest(
            mission_id="a" * 32,
            description="Test mission",
            context=DesktopContext("VS Code", "clipboard text", {}),
            timestamp=time.time(),
            source="test",
        )
        assert len(req.mission_id) == 32
        assert req.source == "test"

    def test_mission_id_is_hex(self):
        """Mission ID must be valid hex."""
        req = MissionRequest(...)
        assert all(c in "0123456789abcdef" for c in req.mission_id)


class TestMissionOrchestrator:
    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator with isolated storage."""
        return MissionOrchestrator({
            "memory_path": tmp_path / "memory",
            "evidence_path": tmp_path / "evidence.jsonl",
        })

    async def test_initialize_creates_memory(self, orchestrator, tmp_path):
        """Initialize creates LivingMemory database."""
        await orchestrator.initialize()
        assert (tmp_path / "memory" / "memory.db").exists()

    async def test_execute_returns_mission_result(self, orchestrator):
        """Execute returns a MissionResult with all required fields."""
        await orchestrator.initialize()
        request = MissionRequest(
            mission_id="b" * 32,
            description="Research AI frameworks",
            context=DesktopContext("Test", "", {}),
            timestamp=time.time(),
            source="test",
        )
        result = await orchestrator.execute(request)
        assert result.mission_id == "b" * 32
        assert result.status in ("COMPLETE", "PARTIAL", "FAILED")
        assert result.ihsan_score >= 0.0
        assert result.snr_score >= 0.0
        assert result.evidence_receipt_id != ""
        assert result.duration_ms > 0

    async def test_execute_without_gateway_uses_template(self, orchestrator):
        """Without LLM gateway, synthesis uses template mode."""
        await orchestrator.initialize()
        # No gateway injected — template synthesis expected
        request = MissionRequest(...)
        result = await orchestrator.execute(request)
        assert "BIZRA Mission Briefing" in result.synthesis

    async def test_execute_stores_episodic_memory(self, orchestrator):
        """Completed missions are stored as episodic memories."""
        await orchestrator.initialize()
        request = MissionRequest(...)
        result = await orchestrator.execute(request)
        # Retrieve the stored memory
        memories = orchestrator.memory.retrieve(
            query="Research AI frameworks",
            memory_type="EPISODIC",
            top_k=1,
        )
        assert len(memories) >= 1

    async def test_execute_emits_evidence_receipt(self, orchestrator, tmp_path):
        """Completed missions produce hash-chained evidence."""
        await orchestrator.initialize()
        request = MissionRequest(...)
        result = await orchestrator.execute(request)
        # Verify evidence chain
        valid, errors = orchestrator.evidence_ledger.verify_chain()
        assert valid
        assert len(errors) == 0

    async def test_execute_fails_closed_on_empty_description(self, orchestrator):
        """Empty mission description returns error, not crash."""
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc({"description": ""})
        assert "error" in result

    async def test_execute_truncates_clipboard_to_4kb(self, orchestrator):
        """Large clipboard content is truncated for safety."""
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc({
            "description": "Test",
            "context": {"clipboard": "x" * 10000},
        })
        # Should not crash, clipboard truncated internally
        assert result.get("status") is not None
```

### test_synthesis_engine.py

```python
class TestTemplateSynthesis:
    def test_template_includes_mission_description(self):
        """Template output includes the original mission description."""
        orch = MissionOrchestrator(...)
        synthesis = orch._template_synthesis(
            description="Research AI agents",
            browser_data=None,
            desktop_data=None,
        )
        assert "Research AI agents" in synthesis

    def test_template_includes_browser_results(self):
        """Template renders browser search results."""
        orch = MissionOrchestrator(...)
        browser_data = {
            "results": [
                {"title": "CrewAI", "url": "https://crewai.com", "snippet": "Agent framework"},
            ]
        }
        synthesis = orch._template_synthesis(
            description="Research",
            browser_data=browser_data,
            desktop_data=None,
        )
        assert "CrewAI" in synthesis
        assert "https://crewai.com" in synthesis

    def test_template_includes_proof_section(self):
        """Template always includes proof trace section."""
        orch = MissionOrchestrator(...)
        synthesis = orch._template_synthesis("Test", None, None)
        assert "Proof Trace" in synthesis
        assert "Ihsan" in synthesis
        assert "Ed25519" in synthesis

    def test_template_handles_empty_browser_results(self):
        """Template gracefully handles no browser results."""
        orch = MissionOrchestrator(...)
        synthesis = orch._template_synthesis(
            description="Test",
            browser_data={"results": []},
            desktop_data=None,
        )
        assert "Research Findings" not in synthesis  # Section omitted
```

### test_hda_client.py

```python
class TestHDAClient:
    async def test_connect_timeout_on_unreachable(self):
        """Connection to non-listening port times out gracefully."""
        client = HDAClient(host="127.0.0.1", port=59999, token="test")
        connected = await client.connect()
        assert connected is False

    async def test_send_command_builds_valid_jsonrpc(self):
        """Commands use JSON-RPC 2.0 with auth headers."""
        # Use a mock TCP server
        # Verify: jsonrpc="2.0", id=incremented, auth.token present
        pass

    async def test_send_command_timeout_on_no_response(self):
        """Command times out if HDA doesn't respond within 30s."""
        pass

    async def test_close_is_idempotent(self):
        """Closing an unconnected client doesn't crash."""
        client = HDAClient(host="127.0.0.1", port=59999, token="test")
        await client.close()  # Should not raise
```

## Integration Tests

### test_mission_pipeline.py

```python
class TestMissionPipeline:
    """Integration tests — full pipeline without AHK or LLM."""

    @pytest.fixture
    async def pipeline(self, tmp_path):
        """Boot a complete mission pipeline with isolated storage."""
        orch = MissionOrchestrator({
            "memory_path": tmp_path / "memory",
            "evidence_path": tmp_path / "evidence.jsonl",
        })
        await orch.initialize()
        return orch

    async def test_browser_channel_returns_results(self, pipeline):
        """Browser channel produces search results (mock or direct)."""
        request = MissionRequest(
            mission_id="c" * 32,
            description="Research distributed AI consensus",
            context=DesktopContext("Test", "", {}),
            timestamp=time.time(),
            source="test",
        )
        result = await pipeline.execute(request)
        browser_channel = next(
            (c for c in result.channels_executed if c.channel == "BROWSER"),
            None,
        )
        assert browser_channel is not None
        assert browser_channel.success is True
        assert browser_channel.data.get("results_count", 0) > 0

    async def test_desktop_channel_fallback_without_hda(self, pipeline):
        """Desktop channel falls back to Python I/O when HDA not connected."""
        request = MissionRequest(...)
        result = await pipeline.execute(request)
        desktop_channel = next(
            (c for c in result.channels_executed if c.channel == "DESKTOP"),
            None,
        )
        if desktop_channel:
            assert desktop_channel.data.get("fallback") == "python_file_io"

    async def test_briefing_file_created(self, pipeline, tmp_path):
        """Mission creates a briefing file on disk."""
        request = MissionRequest(...)
        result = await pipeline.execute(request)
        assert result.briefing_path is not None
        assert Path(result.briefing_path).exists()
        content = Path(result.briefing_path).read_text()
        assert "BIZRA Mission Briefing" in content

    async def test_evidence_chain_valid_after_mission(self, pipeline):
        """Evidence chain is valid after mission completion."""
        request = MissionRequest(...)
        await pipeline.execute(request)
        valid, errors = pipeline.evidence_ledger.verify_chain()
        assert valid
        assert len(errors) == 0

    async def test_multiple_missions_chain_evidence(self, pipeline):
        """Multiple missions produce a valid chain of evidence."""
        for i in range(3):
            request = MissionRequest(
                mission_id=f"{i:032x}",
                description=f"Research topic {i}",
                context=DesktopContext("Test", "", {}),
                timestamp=time.time(),
                source="test",
            )
            await pipeline.execute(request)

        valid, errors = pipeline.evidence_ledger.verify_chain()
        assert valid
        # 3 missions = 3 receipts
        assert pipeline.evidence_ledger.count() >= 3

    async def test_memory_retrieval_improves_second_mission(self, pipeline):
        """Second mission on same topic retrieves memory from first."""
        # First mission
        req1 = MissionRequest(
            mission_id="d" * 32,
            description="Research AI agent frameworks",
            ...
        )
        await pipeline.execute(req1)

        # Second mission on related topic
        req2 = MissionRequest(
            mission_id="e" * 32,
            description="Compare AI agent framework architectures",
            ...
        )
        result2 = await pipeline.execute(req2)

        # Memory should have been retrieved for second mission
        # (Verified via event bus or memory stats)
        stats = pipeline.memory.get_stats()
        assert stats.total_entries >= 2

    async def test_snr_gate_produces_valid_scores(self, pipeline):
        """SNR and Ihsan scores are in valid range."""
        request = MissionRequest(...)
        result = await pipeline.execute(request)
        assert 0.0 <= result.snr_score <= 1.0
        assert 0.0 <= result.ihsan_score <= 1.0

    async def test_channel_failure_isolated(self, pipeline):
        """One channel failing doesn't crash the entire mission."""
        # Force browser to fail by setting invalid mode
        pipeline.dispatcher._browser_client = None
        # Override to raise
        request = MissionRequest(
            mission_id="f" * 32,
            description="Research something",
            ...
        )
        result = await pipeline.execute(request)
        # Mission should complete (PARTIAL or COMPLETE)
        assert result.status in ("COMPLETE", "PARTIAL")
```

## E2E Tests (Manual / CI-Skip)

```python
@pytest.mark.slow
@pytest.mark.requires_network
class TestMissionE2E:
    """End-to-end tests that require network access."""

    async def test_real_web_search(self, pipeline):
        """Browser channel fetches real DuckDuckGo results."""
        pytest.importorskip("httpx")
        # Override browser client to direct mode
        pipeline.dispatcher._browser_client = BrowserMCPClient(mode="direct")
        request = MissionRequest(
            description="AI agent frameworks 2026",
            ...
        )
        result = await pipeline.execute(request)
        browser = next(c for c in result.channels_executed if c.channel == "BROWSER")
        assert browser.success
        # Real results should have actual URLs
        urls = [r.get("url", "") for r in browser.data.get("results", [])]
        assert any(url.startswith("http") for url in urls)


@pytest.mark.slow
@pytest.mark.requires_ollama
class TestMissionWithLLM:
    """Tests requiring an LLM backend."""

    async def test_llm_synthesis_quality(self, pipeline):
        """LLM synthesis produces higher quality than template."""
        from core.inference.gateway import InferenceGateway
        gateway = InferenceGateway()
        pipeline.gateway = gateway
        request = MissionRequest(
            description="Research AI consensus mechanisms",
            ...
        )
        result = await pipeline.execute(request)
        # LLM synthesis should NOT contain template markers
        assert "## Research Findings" not in result.synthesis  # Template marker
        assert result.ihsan_score >= 0.90  # LLM should score higher
```

## Acceptance Criteria

### Must-Have (v1 launch)
- [ ] `Win+Shift+B` hotkey triggers mission input
- [ ] Python bridge receives and processes `execute_mission`
- [ ] Browser channel returns search results (mock or real)
- [ ] Briefing file created on disk (Desktop preferred, fallback to ./missions/)
- [ ] Evidence receipt emitted with valid BLAKE3 chain
- [ ] AHK shows result tooltip with mission status and Ihsan score
- [ ] All unit tests pass
- [ ] All integration tests pass

### Should-Have (v1.1)
- [ ] HDA desktop context capture (active window, clipboard)
- [ ] Auto-open briefing file after creation
- [ ] LLM-powered synthesis (LM Studio)
- [ ] Memory retrieval enriches subsequent missions
- [ ] Event bus emits all lifecycle events

### Nice-to-Have (v2)
- [ ] Voice channel (PersonaPlex)
- [ ] Proof channel (OBS screenshot)
- [ ] PCI envelope wrapping the full mission
- [ ] Rust event bus integration (12 subscribers)
- [ ] Multiple concurrent missions
- [ ] Mission history UI (web dashboard)
