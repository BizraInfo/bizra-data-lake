"""
Cross-Agent Memory Sync — Pub/sub via Redis synapse (port 6380).

Publisher broadcasts new MemoryRecords to a Redis channel.
Subscriber imports records from other agents into the local AgentDB.
Content-addressable IDs prevent self-import and duplication.

Usage:
    publisher = MemorySyncPublisher("agent_a")
    await publisher.connect()
    await publisher.publish(record)

    subscriber = MemorySyncSubscriber(db, "agent_b")
    await subscriber.start()

Standing on Giants: Redis pub/sub (Sanfilippo, 2009)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Set

from .types import MemoryKind, MemoryRecord, RecordState

if TYPE_CHECKING:
    from .agent_db import AgentDB

logger = logging.getLogger(__name__)


class MemorySyncPublisher:
    """Publishes new MemoryRecords to Redis for cross-agent sync.

    Buffers locally when Redis is unavailable, flushes on reconnect.
    """

    def __init__(
        self,
        agent_id: str,
        redis_url: str = "redis://localhost:6380",
        channel: str = "bizra:memory:new",
    ) -> None:
        self._agent_id = agent_id
        self._redis_url = redis_url
        self._channel = channel
        self._client = None
        self._connected = False
        self._buffer: List[str] = []
        self._max_buffer = 1000
        self._publish_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            await self._client.ping()
            self._connected = True
            logger.info(f"MemorySyncPublisher connected to {self._redis_url}")

            if self._buffer:
                logger.info(f"Flushing {len(self._buffer)} buffered messages")
                for msg in self._buffer:
                    await self._client.publish(self._channel, msg)
                self._buffer.clear()

        except ImportError:
            logger.warning("redis package not installed — sync disabled")
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
        ) as e:  # SEC-003 — async boundary
            logger.warning(f"Redis connection failed: {e}")

    async def publish(self, record: MemoryRecord) -> None:
        message = json.dumps(
            {
                "sender_id": self._agent_id,
                "record": record.to_dict(),
                "has_embedding": record.embedding is not None,
            }
        )

        if self._connected and self._client:
            try:
                await self._client.publish(self._channel, message)
                self._publish_count += 1
                return
            except (
                json.JSONDecodeError,
                OSError,
                ValueError,
            ) as e:  # SEC-003 — json boundary
                logger.warning(f"Publish failed: {e}")
                self._connected = False

        if len(self._buffer) < self._max_buffer:
            self._buffer.append(message)
        else:
            self._buffer.pop(0)
            self._buffer.append(message)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._connected = False

    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "published": self._publish_count,
            "buffered": len(self._buffer),
            "channel": self._channel,
        }


class MemorySyncSubscriber:
    """Listens for MemoryRecords from other agents via Redis pub/sub.

    Deduplicates by content-addressable ID and filters own messages.
    """

    def __init__(
        self,
        agent_db: AgentDB,
        agent_id: str,
        redis_url: str = "redis://localhost:6380",
        channel: str = "bizra:memory:new",
        accept_kinds: Optional[List[MemoryKind]] = None,
    ) -> None:
        self._db = agent_db
        self._agent_id = agent_id
        self._redis_url = redis_url
        self._channel = channel
        self._accept_kinds = accept_kinds
        self._client = None
        self._pubsub = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._imported_count = 0
        self._skipped_count = 0
        self._seen_ids: Set[str] = set()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def imported_count(self) -> int:
        return self._imported_count

    @property
    def skipped_count(self) -> int:
        return self._skipped_count

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
            )
            await self._client.ping()

            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(self._channel)

            self._running = True
            self._task = asyncio.create_task(self._listen_loop())
            logger.info(f"MemorySyncSubscriber listening on {self._channel}")

        except ImportError:
            logger.warning("redis package not installed — sync disabled")
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
        ) as e:  # SEC-003 — async boundary
            logger.warning(f"Subscriber start failed: {e}")

    async def _listen_loop(self) -> None:
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    await self._handle_message(message["data"])
            except asyncio.CancelledError:
                break
            except (
                asyncio.CancelledError,
                RuntimeError,
                OSError,
            ) as e:  # SEC-003 — async boundary
                logger.warning(f"Subscriber error: {e}")
                await asyncio.sleep(1.0)

    async def _handle_message(self, data: str) -> None:
        try:
            payload = json.loads(data)
            sender_id = payload.get("sender_id", "")
            record_dict = payload.get("record", {})

            if sender_id == self._agent_id:
                return

            record = self._dict_to_record(record_dict)
            if record is None:
                self._skipped_count += 1
                return

            if self._accept_kinds and record.kind not in self._accept_kinds:
                self._skipped_count += 1
                return

            if record.id in self._seen_ids:
                self._skipped_count += 1
                return

            existing = self._db.retrieve(record.id)
            if existing is not None:
                self._seen_ids.add(record.id)
                self._skipped_count += 1
                return

            self._db.store_record(record)
            self._seen_ids.add(record.id)
            self._imported_count += 1

            logger.debug(
                f"Synced record {record.id[:8]}... from {sender_id} "
                f"(kind={record.kind.value})"
            )

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in sync message")
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"Failed to handle sync message: {e}")

    @staticmethod
    def _dict_to_record(d: dict) -> Optional[MemoryRecord]:
        try:
            return MemoryRecord(
                id=d["id"],
                content=d["content"],
                kind=MemoryKind(d.get("kind", "semantic")),
                state=RecordState(d.get("state", "active")),
                embedding=None,
                ihsan_score=d.get("ihsan_score", 1.0),
                snr_score=d.get("snr_score", 1.0),
                importance=d.get("importance", 0.5),
                source=d.get("source", "sync"),
                source_id=d.get("source_id"),
                related_ids=d.get("related_ids", []),
                tags=list(set(d.get("tags", []) + ["synced"])),
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                last_accessed=datetime.fromisoformat(d["last_accessed"]),
                access_count=d.get("access_count", 0),
                metadata={
                    **d.get("metadata", {}),
                    "synced_from": d.get("source", "unknown"),
                },
            )
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"Failed to parse synced record: {e}")
            return None

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe(self._channel)
        if self._client:
            await self._client.aclose()
        logger.info(
            f"Subscriber stopped: {self._imported_count} imported, "
            f"{self._skipped_count} skipped"
        )

    def stats(self) -> dict:
        return {
            "running": self._running,
            "imported": self._imported_count,
            "skipped": self._skipped_count,
            "seen_cache_size": len(self._seen_ids),
            "channel": self._channel,
        }
