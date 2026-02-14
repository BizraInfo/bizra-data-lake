"""
Token System Integration Tests
================================

End-to-end integration tests for the token system wiring:

1. API Endpoints — /v1/token/{balance,supply,history,verify}
2. CLI Commands — wallet, tokens
3. PoI → Token Bridge — Full epoch distribution
4. Full Lifecycle — Genesis → Mint → Transfer → Burn → Verify

Each test class uses isolated tmp_path storage to prevent state leakage.

Standing on Giants:
- Nakamoto (2008): Hash chain verification patterns
- Al-Ghazali (1058-1111): Zakat distribution validation
- Shannon (1948): Signal-to-noise quality gates
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from core.token.ledger import TokenLedger
from core.token.mint import (
    COMMUNITY_FUND_ACCOUNT,
    GENESIS_NODE0_ACCOUNT,
    SYSTEM_TREASURY_ACCOUNT,
    TokenMinter,
)
from core.token.poi_bridge import PoITokenBridge
from core.token.types import (
    FOUNDER_GENESIS_ALLOCATION,
    SEED_SUPPLY_CAP_PER_YEAR,
    SYSTEM_TREASURY_ALLOCATION,
    ZAKAT_RATE,
    TokenOp,
    TokenType,
)


# =============================================================================
# HELPERS
# =============================================================================


def _create_minter(tmp_path: Path) -> TokenMinter:
    """Create a fresh minter with isolated storage."""
    return TokenMinter.create(
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test_ledger.jsonl",
    )


def _create_bridge(tmp_path: Path) -> PoITokenBridge:
    """Create a fresh PoI bridge with isolated storage."""
    return PoITokenBridge.create(
        db_path=tmp_path / "bridge.db",
        log_path=tmp_path / "bridge_ledger.jsonl",
    )


def _make_audit_trail(
    epoch_id: str = "epoch-1",
    contributors: Dict[str, float] | None = None,
):
    """Build a minimal AuditTrail for testing."""
    from core.proof_engine.poi_engine import AuditTrail, ProofOfImpact

    if contributors is None:
        contributors = {"alice": 0.6, "bob": 0.3, "carol": 0.1}

    poi_scores = []
    for cid, score in sorted(contributors.items()):
        poi_scores.append(
            ProofOfImpact(
                contributor_id=cid,
                contribution_score=score,
                reach_score=score * 0.5,
                longevity_score=score * 0.3,
                poi_score=score,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
                config_digest="test-config-hash",
                computation_id=f"comp-{cid}",
                epoch_id=epoch_id,
            )
        )

    return AuditTrail(
        epoch_id=epoch_id,
        poi_scores=poi_scores,
        gini_coefficient=0.35,
        rebalance_triggered=False,
        config_digest="test-config-hash",
    )


def _make_patched_ledger_cls(db_path: Path, log_path: Path):
    """Create a TokenLedger class factory that uses the given paths."""
    def _factory(*args, **kwargs):
        return TokenLedger(db_path=db_path, log_path=log_path)
    return _factory


# =============================================================================
# TEST: API TOKEN ENDPOINTS
# =============================================================================


class TestTokenAPIEndpoints:
    """Test the /v1/token/* endpoints via SovereignAPIServer."""

    def _make_server(self):
        """Create API server with mocked runtime."""
        from core.sovereign.api import SovereignAPIServer

        runtime = MagicMock()
        runtime.status.return_value = {"state": "running"}
        return SovereignAPIServer(runtime=runtime, port=0)

    @pytest.mark.asyncio
    async def test_token_balance_default_account(self, tmp_path: Path):
        """GET /v1/token/balance returns Node0 balances after genesis."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_balance({"account": GENESIS_NODE0_ACCOUNT})

        body = self._extract_json(response)
        assert body["account"] == GENESIS_NODE0_ACCOUNT
        assert "SEED" in body["balances"]
        assert body["balances"]["SEED"]["balance"] == FOUNDER_GENESIS_ALLOCATION

    @pytest.mark.asyncio
    async def test_token_balance_treasury(self, tmp_path: Path):
        """GET /v1/token/balance?account=SYSTEM-TREASURY returns treasury balance."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_balance({"account": SYSTEM_TREASURY_ACCOUNT})

        body = self._extract_json(response)
        assert body["account"] == SYSTEM_TREASURY_ACCOUNT
        assert body["balances"]["SEED"]["balance"] == SYSTEM_TREASURY_ALLOCATION

    @pytest.mark.asyncio
    async def test_token_supply_after_genesis(self, tmp_path: Path):
        """GET /v1/token/supply returns correct totals after genesis."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_supply()

        body = self._extract_json(response)
        assert "supply" in body
        seed_supply = body["supply"]["SEED"]
        total_seed = FOUNDER_GENESIS_ALLOCATION + SYSTEM_TREASURY_ALLOCATION
        zakat = total_seed * ZAKAT_RATE
        assert seed_supply["total_supply"] == total_seed + zakat
        assert seed_supply["yearly_cap"] == SEED_SUPPLY_CAP_PER_YEAR

    @pytest.mark.asyncio
    async def test_token_history(self, tmp_path: Path):
        """GET /v1/token/history returns genesis transactions."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_history({"limit": "10"})

        body = self._extract_json(response)
        assert body["count"] == 4  # founder SEED + treasury + zakat + founder IMPT

    @pytest.mark.asyncio
    async def test_token_history_filtered_by_account(self, tmp_path: Path):
        """GET /v1/token/history?account=X returns only that account's transactions."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_history(
                {"account": COMMUNITY_FUND_ACCOUNT, "limit": "10"}
            )

        body = self._extract_json(response)
        assert body["count"] == 1  # only the zakat transaction
        assert body["transactions"][0]["to"] == COMMUNITY_FUND_ACCOUNT

    @pytest.mark.asyncio
    async def test_token_verify_valid_chain(self, tmp_path: Path):
        """GET /v1/token/verify returns valid for intact chain."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_verify()

        body = self._extract_json(response)
        assert body["chain_valid"] is True
        assert body["transactions_verified"] == 4
        assert body["error"] is None

    @pytest.mark.asyncio
    async def test_token_balance_empty_account(self, tmp_path: Path):
        """GET /v1/token/balance for unknown account returns empty balances."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            server = self._make_server()
            response = await server._handle_token_balance({"account": "NONEXISTENT"})

        body = self._extract_json(response)
        assert body["account"] == "NONEXISTENT"
        assert body["balances"] == {}

    def _extract_json(self, http_response: str) -> Dict[str, Any]:
        """Extract JSON body from raw HTTP response."""
        parts = http_response.split("\r\n\r\n", 1)
        return json.loads(parts[1])


# =============================================================================
# TEST: CLI WALLET AND TOKEN COMMANDS
# =============================================================================


class TestTokenCLICommands:
    """Test the wallet and tokens CLI commands."""

    def test_wallet_command_shows_balances(self, tmp_path: Path, capsys):
        """'wallet' command displays token balances for all accounts."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            from core.sovereign.__main__ import _handle_wallet_command
            _handle_wallet_command()

        output = capsys.readouterr().out
        assert "BIZRA-00000000" in output
        assert "SYSTEM-TREASURY" in output
        assert "BIZRA-COMMUNITY-FUND" in output
        assert "SEED" in output
        assert "100,000.00" in output

    def test_tokens_command_shows_supply(self, tmp_path: Path, capsys):
        """'tokens' command displays supply, yearly cap, and chain validity."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        factory = _make_patched_ledger_cls(tmp_path / "test.db", tmp_path / "test_ledger.jsonl")
        with patch("core.token.ledger.TokenLedger", factory):
            from core.sovereign.__main__ import _handle_tokens_command
            _handle_tokens_command()

        output = capsys.readouterr().out
        assert "SEED" in output
        assert "VALID" in output
        assert "4 transactions" in output

    def test_wallet_command_no_genesis(self, tmp_path: Path, capsys):
        """'wallet' command gracefully handles empty ledger."""
        db_path = tmp_path / "empty.db"
        log_path = tmp_path / "empty_ledger.jsonl"
        TokenLedger(db_path=db_path, log_path=log_path)

        factory = _make_patched_ledger_cls(db_path, log_path)
        with patch("core.token.ledger.TokenLedger", factory):
            from core.sovereign.__main__ import _handle_wallet_command
            _handle_wallet_command()

        output = capsys.readouterr().out
        assert "BIZRA TOKEN WALLET" in output


# =============================================================================
# TEST: POI → TOKEN BRIDGE
# =============================================================================


class TestPoITokenBridge:
    """Test the PoI engine → Token minting bridge."""

    def test_distribute_epoch_basic(self, tmp_path: Path):
        """distribute_epoch() mints SEED + IMPT for all contributors."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-1")

        result = bridge.distribute_epoch(audit, epoch_reward=10000)

        summary = result["summary"]
        assert summary["epoch_id"] == "epoch-1"
        assert summary["epoch_reward"] == 10000
        assert summary["contributors"] == 3
        assert summary["seed_succeeded"] == 3
        assert summary["impt_succeeded"] == 3

    def test_distribute_epoch_proportional_allocation(self, tmp_path: Path):
        """SEED tokens distributed proportionally to PoI scores."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(
            epoch_id="epoch-2",
            contributors={"alice": 0.5, "bob": 0.5},
        )

        result = bridge.distribute_epoch(audit, epoch_reward=10000)

        ledger = bridge.minter.ledger
        alice_bal = ledger.get_balance("alice", TokenType.SEED)
        bob_bal = ledger.get_balance("bob", TokenType.SEED)
        assert abs(alice_bal.balance - bob_bal.balance) < 0.01

    def test_distribute_epoch_mints_impt(self, tmp_path: Path):
        """IMPT reputation tokens minted for each contributor."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-3")

        result = bridge.distribute_epoch(audit, epoch_reward=10000, impt_multiplier=100.0)

        ledger = bridge.minter.ledger
        alice_impt = ledger.get_balance("alice", TokenType.IMPT)
        assert alice_impt.balance == 60.0

        bob_impt = ledger.get_balance("bob", TokenType.IMPT)
        assert bob_impt.balance == 30.0

    def test_distribute_epoch_no_impt(self, tmp_path: Path):
        """mint_impt=False skips IMPT minting."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-4")

        result = bridge.distribute_epoch(audit, epoch_reward=10000, mint_impt=False)

        assert result["summary"]["impt_distributions"] == 0
        assert result["impt_receipts"] == []

    def test_distribute_epoch_chain_integrity(self, tmp_path: Path):
        """Hash chain remains valid after epoch distribution."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-5")

        bridge.distribute_epoch(audit, epoch_reward=10000)

        valid, count, err = bridge.minter.ledger.verify_chain()
        assert valid is True
        # 3 SEED mints + 3 zakat + 3 IMPT = 9 total transactions
        assert count == 9
        assert err is None

    def test_distribute_epoch_includes_zakat(self, tmp_path: Path):
        """Each SEED mint triggers computational zakat to community fund."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-6")

        bridge.distribute_epoch(audit, epoch_reward=10000)

        ledger = bridge.minter.ledger
        community_bal = ledger.get_balance(COMMUNITY_FUND_ACCOUNT, TokenType.SEED)
        assert community_bal.balance > 0

    def test_distribute_epoch_summary_has_gini(self, tmp_path: Path):
        """Summary includes the Gini coefficient from the PoI audit."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-7")

        result = bridge.distribute_epoch(audit, epoch_reward=10000)
        assert result["summary"]["gini_coefficient"] == 0.35

    def test_distribute_epoch_distribution_dict(self, tmp_path: Path):
        """Return value includes full distribution dict."""
        bridge = _create_bridge(tmp_path)
        audit = _make_audit_trail(epoch_id="epoch-8")

        result = bridge.distribute_epoch(audit, epoch_reward=10000)

        dist = result["distribution"]
        assert dist["epoch_id"] == "epoch-8"
        assert dist["epoch_reward"] == 10000
        assert "distributions" in dist
        assert "alice" in dist["distributions"]

    def test_bridge_status(self, tmp_path: Path):
        """Bridge status delegates to minter status."""
        bridge = _create_bridge(tmp_path)
        status = bridge.status()
        assert "total_supply" in status
        assert "genesis_minted" in status


# =============================================================================
# TEST: FULL TOKEN LIFECYCLE
# =============================================================================


class TestTokenLifecycle:
    """End-to-end lifecycle: Genesis → Mint → Transfer → Burn → Verify."""

    def test_full_lifecycle(self, tmp_path: Path):
        """Complete token lifecycle from genesis through unstake."""
        minter = _create_minter(tmp_path)

        # Step 1: Genesis mint
        receipts = minter.genesis_mint()
        assert len(receipts) == 4
        assert all(r.success for r in receipts)

        # Step 2: Verify genesis balances
        ledger = minter.ledger
        node0_seed = ledger.get_balance(GENESIS_NODE0_ACCOUNT, TokenType.SEED)
        assert node0_seed.balance == FOUNDER_GENESIS_ALLOCATION

        treasury_seed = ledger.get_balance(SYSTEM_TREASURY_ACCOUNT, TokenType.SEED)
        assert treasury_seed.balance == SYSTEM_TREASURY_ALLOCATION

        zakat_amount = (FOUNDER_GENESIS_ALLOCATION + SYSTEM_TREASURY_ALLOCATION) * ZAKAT_RATE
        community_seed = ledger.get_balance(COMMUNITY_FUND_ACCOUNT, TokenType.SEED)
        assert community_seed.balance == zakat_amount

        # Step 3: Transfer from Node0 to a new contributor
        transfer_receipt = minter.transfer(
            from_account=GENESIS_NODE0_ACCOUNT,
            to_account="contributor-001",
            token_type=TokenType.SEED,
            amount=1000.0,
            memo="Payment for contribution epoch-1",
        )
        assert transfer_receipt.success

        node0_after = ledger.get_balance(GENESIS_NODE0_ACCOUNT, TokenType.SEED)
        assert node0_after.balance == FOUNDER_GENESIS_ALLOCATION - 1000.0

        contrib_bal = ledger.get_balance("contributor-001", TokenType.SEED)
        assert contrib_bal.balance == 1000.0

        # Step 4: Burn some tokens
        burn_receipt = minter.burn(
            from_account="contributor-001",
            token_type=TokenType.SEED,
            amount=100.0,
            memo="Fee burn",
        )
        assert burn_receipt.success

        contrib_after_burn = ledger.get_balance("contributor-001", TokenType.SEED)
        assert contrib_after_burn.balance == 900.0

        # Step 5: Stake tokens
        stake_receipt = minter.stake(
            account_id="contributor-001",
            token_type=TokenType.SEED,
            amount=500.0,
        )
        assert stake_receipt.success

        contrib_staked = ledger.get_balance("contributor-001", TokenType.SEED)
        assert contrib_staked.balance == 900.0
        assert contrib_staked.staked == 500.0
        assert contrib_staked.available == 400.0

        # Step 6: Unstake
        unstake_receipt = minter.unstake(
            account_id="contributor-001",
            token_type=TokenType.SEED,
            amount=200.0,
        )
        assert unstake_receipt.success

        contrib_unstaked = ledger.get_balance("contributor-001", TokenType.SEED)
        assert contrib_unstaked.staked == 300.0
        assert contrib_unstaked.available == 600.0

        # Step 7: Verify entire chain
        valid, count, err = ledger.verify_chain()
        assert valid is True
        assert count >= 7  # genesis(4) + transfer + burn + stake + unstake
        assert err is None

    def test_genesis_then_poi_distribution(self, tmp_path: Path):
        """Genesis mint followed by PoI epoch distribution."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        bridge = PoITokenBridge(minter)
        audit = _make_audit_trail(epoch_id="epoch-1")

        result = bridge.distribute_epoch(audit, epoch_reward=10000)

        assert result["summary"]["seed_succeeded"] == 3
        assert result["summary"]["impt_succeeded"] == 3

        node0_seed = minter.ledger.get_balance(GENESIS_NODE0_ACCOUNT, TokenType.SEED)
        assert node0_seed.balance == FOUNDER_GENESIS_ALLOCATION

        alice_seed = minter.ledger.get_balance("alice", TokenType.SEED)
        assert alice_seed.balance > 0

        valid, count, err = minter.ledger.verify_chain()
        assert valid is True
        assert err is None

    def test_genesis_idempotency(self, tmp_path: Path):
        """Genesis mint cannot be executed twice."""
        minter = _create_minter(tmp_path)

        first = minter.genesis_mint()
        assert len(first) == 4

        second = minter.genesis_mint()
        # Idempotent: returns single failed receipt
        assert len(second) == 1
        assert second[0].success is False

        # Balances unchanged
        node0 = minter.ledger.get_balance(GENESIS_NODE0_ACCOUNT, TokenType.SEED)
        assert node0.balance == FOUNDER_GENESIS_ALLOCATION

    def test_impt_non_transferable(self, tmp_path: Path):
        """IMPT tokens cannot be transferred."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        receipt = minter.transfer(
            from_account=GENESIS_NODE0_ACCOUNT,
            to_account="someone",
            token_type=TokenType.IMPT,
            amount=10.0,
            memo="Attempted IMPT transfer",
        )
        assert receipt.success is False

    def test_chain_tamper_detection(self, tmp_path: Path):
        """Tampering with JSONL ledger is detected by verify_chain()."""
        minter = _create_minter(tmp_path)
        minter.genesis_mint()

        log_path = tmp_path / "test_ledger.jsonl"
        lines = log_path.read_text().strip().split("\n")
        if len(lines) >= 2:
            entry = json.loads(lines[1])
            entry["amount"] = 999999  # tamper
            lines[1] = json.dumps(entry)
            log_path.write_text("\n".join(lines) + "\n")

            ledger = TokenLedger(
                db_path=tmp_path / "test.db",
                log_path=log_path,
            )
            valid, count, err = ledger.verify_chain()
            assert valid is False

    def test_supply_cap_enforcement(self, tmp_path: Path):
        """Cannot mint beyond yearly SEED cap."""
        minter = _create_minter(tmp_path)

        large_amount = SEED_SUPPLY_CAP_PER_YEAR - 100
        receipt = minter.mint_seed(
            to_account="whale",
            amount=large_amount,
            epoch_id="epoch-cap-test",
            poi_score=0.99,
        )
        assert receipt.success

        over_cap_receipt = minter.mint_seed(
            to_account="whale",
            amount=SEED_SUPPLY_CAP_PER_YEAR,
            epoch_id="epoch-cap-test-2",
            poi_score=0.99,
        )
        assert over_cap_receipt.success is False


# =============================================================================
# TEST: CROSS-MODULE WIRING
# =============================================================================


class TestCrossModuleWiring:
    """Verify that token modules import and wire correctly."""

    def test_token_module_imports(self):
        """All public token types are importable from core.token."""
        from core.token import (
            TokenLedger,
            TokenMinter,
            TokenType,
            TokenOp,
            TransactionEntry,
            TokenReceipt,
            TokenBalance,
        )
        assert TokenType.SEED.value == "SEED"
        assert TokenOp.MINT.value == "mint"

    def test_poi_bridge_imports(self):
        """PoI bridge imports resolve correctly."""
        from core.token.poi_bridge import PoITokenBridge
        assert hasattr(PoITokenBridge, "distribute_epoch")

    def test_api_token_handlers_exist(self):
        """API server has all 4 token handler methods."""
        from core.sovereign.api import SovereignAPIServer
        assert hasattr(SovereignAPIServer, "_handle_token_balance")
        assert hasattr(SovereignAPIServer, "_handle_token_supply")
        assert hasattr(SovereignAPIServer, "_handle_token_history")
        assert hasattr(SovereignAPIServer, "_handle_token_verify")

    def test_cli_token_functions_exist(self):
        """CLI module has wallet and tokens handler functions."""
        from core.sovereign.__main__ import (
            _handle_wallet_command,
            _handle_tokens_command,
        )
        assert callable(_handle_wallet_command)
        assert callable(_handle_tokens_command)
