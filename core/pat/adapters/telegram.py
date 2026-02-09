"""
BIZRA Telegram Gateway Adapter — Sovereign Messaging Bridge

Connects the Telegram Bot API to the BIZRA sovereign runtime.
A human sends a Telegram message → the adapter routes it through
SovereignRuntime.query() → the response goes back to Telegram.

Article IX § 9.6: "Gateway operators earn from traffic, not from data."

Requirements:
    pip install httpx  (already in BIZRA deps)

Configuration:
    BIZRA_TELEGRAM_TOKEN=<bot_token>  (from @BotFather)

Standing on Giants: Telegram Bot API (Durov) + Hexagonal Architecture (Cockburn)
"""

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from ..bridge import ChannelType
from ..channels import ChannelAdapter, GatewayState, QueryFn

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
_API_BASE = "https://api.telegram.org/bot{token}"

# Command descriptions for BotFather /setcommands
BOT_COMMANDS = [
    {"command": "start", "description": "Create your sovereign identity"},
    {"command": "status", "description": "View your node status"},
    {"command": "agents", "description": "List your agentic team"},
    {"command": "help", "description": "Show available commands"},
]

# Rate limit: max messages per second to Telegram API
_SEND_RATE_LIMIT = 25  # Telegram allows 30/sec, we use 25 for safety


class TelegramAdapter(ChannelAdapter):
    """
    Telegram Bot adapter for the BIZRA sovereign gateway.

    Modes:
        - Long polling (default): Simple, works behind NAT/firewalls
        - Webhook: Production mode, requires public HTTPS endpoint

    Usage:
        adapter = TelegramAdapter(
            token="<bot_token>",
            query_fn=my_query_function,
        )
        await adapter.start()  # Blocks until stop() called
    """

    def __init__(
        self,
        token: str = "",
        query_fn: Optional[QueryFn] = None,
        node_id: str = "",
        webhook_url: str = "",
    ):
        if not token:
            token = os.environ.get("BIZRA_TELEGRAM_TOKEN", "")
        if not token:
            raise ValueError(
                "Telegram bot token required. "
                "Set BIZRA_TELEGRAM_TOKEN env var or pass token= parameter."
            )

        # Placeholder query_fn for when none provided (will be set before start)
        if query_fn is None:

            async def _placeholder(content: str, context: Optional[Dict] = None) -> str:
                return "Sovereign runtime not connected. Please configure query_fn."

            query_fn = _placeholder

        super().__init__(
            channel_type=ChannelType.TELEGRAM,
            query_fn=query_fn,
            node_id=node_id,
        )

        self._token = token
        self._api_base = _API_BASE.format(token=token)
        self._webhook_url = webhook_url
        self._poll_task: Optional[asyncio.Task] = None
        self._offset: int = 0  # Last processed update_id + 1
        self._http_client = None
        self._bot_info: Dict[str, Any] = {}
        self._started_at: Optional[float] = None
        # Rate limiter: track last N send timestamps (sliding window)
        self._send_timestamps: Deque[float] = deque(maxlen=_SEND_RATE_LIMIT)

    @property
    def bot_username(self) -> str:
        return self._bot_info.get("username", "unknown")

    async def _ensure_client(self):
        """Lazily create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is required for Telegram adapter. "
                    "Install with: pip install httpx"
                )
            self._http_client = httpx.AsyncClient(timeout=60.0)  # type: ignore[assignment]

    async def _api_call(
        self, method: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a Telegram Bot API call."""
        await self._ensure_client()
        url = f"{self._api_base}/{method}"

        try:
            if data:
                resp = await self._http_client.post(url, json=data)  # type: ignore[union-attr,attr-defined]
            else:
                resp = await self._http_client.get(url)  # type: ignore[union-attr,attr-defined]

            result = resp.json()

            if not result.get("ok"):
                desc = result.get("description", "Unknown error")
                logger.error(f"Telegram API error: {method} → {desc}")
                return {"ok": False, "description": desc}

            return result

        except Exception as e:
            # HIGH-2: Log only method + error type, never str(e) which may
            # contain the bot token if httpx embeds the URL in its error
            logger.error(f"Telegram API call failed: {method} ({type(e).__name__})")
            return {"ok": False, "description": f"API call failed: {type(e).__name__}"}

    async def start(self) -> None:
        """
        Start the Telegram gateway.

        In long-polling mode, this blocks until stop() is called.
        """
        self._state = GatewayState.STARTING
        self._started_at = time.time()
        self._metrics.started_at = datetime.now(timezone.utc).isoformat()

        # Verify token with getMe
        me = await self._api_call("getMe")
        if not me.get("ok"):
            self._state = GatewayState.ERROR
            raise ConnectionError(
                f"Invalid Telegram token: {me.get('description', 'auth failed')}"
            )

        self._bot_info = me.get("result", {})
        logger.info(
            f"Telegram gateway starting as @{self.bot_username} "
            f"(node: {self._node_id or 'unset'})"
        )

        # Register bot commands
        await self._api_call("setMyCommands", {"commands": BOT_COMMANDS})

        # Start polling or webhook
        self._state = GatewayState.RUNNING

        if self._webhook_url:
            await self._setup_webhook()
        else:
            await self._poll_loop()

    async def stop(self) -> None:
        """Stop the Telegram gateway gracefully."""
        self._state = GatewayState.STOPPING
        logger.info("Telegram gateway stopping...")

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()

        if self._webhook_url:
            await self._api_call("deleteWebhook")

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        # Clear sensitive token from memory (HIGH-1: prevent post-shutdown leakage)
        self._token = ""
        self._api_base = ""

        self._state = GatewayState.STOPPED
        logger.info("Telegram gateway stopped.")

    async def send_response(
        self,
        platform_user_id: str,
        content: str,
        parse_mode: str = "Markdown",
        **kwargs: Any,
    ) -> bool:
        """Send a response message to a Telegram chat."""
        # Enforce rate limit: max _SEND_RATE_LIMIT messages per second
        now = time.monotonic()
        # Evict timestamps older than 1 second
        while self._send_timestamps and self._send_timestamps[0] < now - 1.0:
            self._send_timestamps.popleft()
        if len(self._send_timestamps) >= _SEND_RATE_LIMIT:
            logger.warning(
                f"Rate limit reached ({_SEND_RATE_LIMIT}/s), dropping send to {platform_user_id}"
            )
            return False
        self._send_timestamps.append(now)

        data: Dict[str, Any] = {
            "chat_id": platform_user_id,
            "text": content[:4096],  # Telegram limit
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        reply_to = kwargs.get("reply_to_message_id")
        if reply_to:
            data["reply_to_message_id"] = reply_to

        result = await self._api_call("sendMessage", data)
        if not result.get("ok"):
            # Retry without parse_mode (Markdown can fail on special chars)
            if parse_mode:
                data.pop("parse_mode", None)
                result = await self._api_call("sendMessage", data)

        return result.get("ok", False)

    # ─── Long Polling ──────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Long-polling loop to receive updates from Telegram."""
        logger.info("Telegram gateway: long-polling mode active")

        while self._state == GatewayState.RUNNING:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._process_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._metrics.errors += 1
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(1.0)

    async def _get_updates(self) -> List[Dict[str, Any]]:
        """Fetch new updates via long polling."""
        params = {
            "offset": self._offset,
            "timeout": 30,
            "allowed_updates": ["message"],
        }
        result = await self._api_call("getUpdates", params)

        if not result.get("ok"):
            await asyncio.sleep(1.0)
            return []

        updates = result.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1

        return updates

    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single Telegram update."""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message["chat"]["id"])
        text = message.get("text", "")
        message_id = message.get("message_id")

        if not text:
            return

        # Metadata for context
        meta = {
            "chat_type": message["chat"].get("type", "private"),
            "first_name": message.get("from", {}).get("first_name", ""),
            "username": message.get("from", {}).get("username", ""),
            "message_id": message_id,
        }

        # Handle commands
        if text.startswith("/"):
            response = await self._handle_command(text, chat_id, meta)
        else:
            # Route through sovereign runtime
            response = await self.handle_incoming(
                platform_user_id=chat_id,
                content=text,
                platform_message_id=str(message_id) if message_id else "",
                metadata=meta,
            )

        # Send response
        await self.send_response(
            platform_user_id=chat_id,
            content=response,
            reply_to_message_id=message_id,
        )

    async def _handle_command(
        self, text: str, chat_id: str, meta: Dict[str, Any]
    ) -> str:
        """Handle Telegram bot commands."""
        command = text.split()[0].lower().split("@")[0]  # Strip @botname
        first_name = meta.get("first_name", "Human")

        if command == "/start":
            return (
                f"Welcome to BIZRA, {first_name}.\n\n"
                f"Every human is a node. Every node is a seed.\n\n"
                f"I am your sovereign gateway. Send me any message "
                f"and I will route it through your agentic team.\n\n"
                f"Commands:\n"
                f"/status — Your node status\n"
                f"/agents — Your agentic team\n"
                f"/help — Available commands"
            )

        if command == "/help":
            return (
                "BIZRA Sovereign Gateway\n\n"
                "/start — Initialize your connection\n"
                "/status — View node status\n"
                "/agents — List your agentic team\n"
                "/help — This message\n\n"
                "Or just send any message to query the sovereign runtime."
            )

        if command == "/status":
            return await self._cmd_status(chat_id)

        if command == "/agents":
            return await self._cmd_agents(chat_id)

        # Unknown command — route as query
        return await self.handle_incoming(
            platform_user_id=chat_id,
            content=text[1:],  # Strip leading /
            metadata=meta,
        )

    async def _cmd_status(self, chat_id: str) -> str:
        """Handle /status command."""
        session = self._sessions.get(chat_id)
        msg_count = session.message_count if session else 0
        uptime = time.time() - self._started_at if self._started_at else 0

        return (
            f"BIZRA Node Status\n"
            f"{'─' * 24}\n"
            f"Gateway: {self._state.value}\n"
            f"Bot: @{self.bot_username}\n"
            f"Node: {self._node_id or 'not set'}\n"
            f"Uptime: {uptime:.0f}s\n"
            f"Your messages: {msg_count}\n"
            f"Total users: {self._metrics.unique_users}\n"
            f"Total messages: {self._metrics.messages_received}"
        )

    async def _cmd_agents(self, chat_id: str) -> str:
        """Handle /agents command."""
        # Check if user has onboarded
        try:
            from ..onboarding import get_node_credentials

            creds = get_node_credentials()
            if creds:
                pat_lines = []
                for aid in creds.pat_agent_ids:
                    parts = aid.split("-")
                    atype = parts[2] if len(parts) >= 3 else "?"
                    names = {
                        "WRK": "Worker",
                        "RSC": "Researcher",
                        "GRD": "Guardian",
                        "SYN": "Synthesizer",
                        "VAL": "Validator",
                        "CRD": "Coordinator",
                        "EXC": "Executor",
                    }
                    pat_lines.append(f"  {names.get(atype, atype)}")

                return (
                    f"Your Agentic Team ({creds.node_id})\n"
                    f"{'─' * 30}\n"
                    f"PAT (Personal Agentic Team):\n"
                    + "\n".join(pat_lines)
                    + f"\n\nSAT (System): {len(creds.sat_agent_ids)} agents"
                    + f"\nTier: {creds.sovereignty_tier.upper()}"
                )
        except (ImportError, FileNotFoundError):
            pass  # Onboarding module or credentials unavailable
        except (KeyError, TypeError) as e:
            logger.debug(f"Agent listing failed: {type(e).__name__}")

        return (
            "No sovereign identity found.\n\n"
            "Run `bizra onboard` on your device first, "
            "then connect via this gateway."
        )

    # ─── Webhook Mode ──────────────────────────────────────────────

    async def _setup_webhook(self) -> None:
        """Configure Telegram webhook (for production behind reverse proxy)."""
        result = await self._api_call(
            "setWebhook",
            {"url": self._webhook_url, "allowed_updates": ["message"]},
        )
        if result.get("ok"):
            logger.info(f"Webhook set: {self._webhook_url}")
        else:
            raise ConnectionError(f"Webhook setup failed: {result.get('description')}")

    async def handle_webhook_update(self, update: Dict[str, Any]) -> Optional[str]:
        """
        Process a webhook update (called by HTTP server).

        Args:
            update: Raw Telegram update JSON

        Returns:
            Response text if any, None otherwise
        """
        message = update.get("message")
        if not message or not message.get("text"):
            return None

        chat_id = str(message["chat"]["id"])
        text = message["text"]
        message_id = message.get("message_id")

        meta = {
            "chat_type": message["chat"].get("type", "private"),
            "first_name": message.get("from", {}).get("first_name", ""),
            "username": message.get("from", {}).get("username", ""),
            "message_id": message_id,
        }

        if text.startswith("/"):
            response = await self._handle_command(text, chat_id, meta)
        else:
            response = await self.handle_incoming(
                platform_user_id=chat_id,
                content=text,
                platform_message_id=str(message_id) if message_id else "",
                metadata=meta,
            )

        await self.send_response(
            platform_user_id=chat_id,
            content=response,
            reply_to_message_id=message_id,
        )
        return response

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status with Telegram-specific info."""
        base = super().get_status()
        base["bot_username"] = self.bot_username
        base["mode"] = "webhook" if self._webhook_url else "polling"
        if self._started_at:
            self._metrics.uptime_seconds = time.time() - self._started_at
        return base
