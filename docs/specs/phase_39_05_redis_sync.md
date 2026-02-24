# Phase 39 — Pseudocode Module 05: Cross-Agent Memory Sync (Redis Synapse)

**FR-05** | Priority: 5 | Risk: High (distributed) | New files: 2

---

## Overview

Enable multi-agent memory sharing via Redis pub/sub on port 6380 (synapse).
When Agent A stores a memory, Agent B receives it automatically.
Content-addressable IDs prevent self-import and duplication.

---

## Flow Diagram

```
Agent A                          Redis 6380                        Agent B
  │                                  │                                │
  ├─ db.store("fact X")              │                                │
  │   ├─ SQLite + HNSW (local)      │                                │
  │   └─ publisher.publish(record)──►│ channel: bizra:memory:new     │
  │                                  │──►subscriber.on_message()─────►│
  │                                  │     ├─ Dedup check (ID)       │
  │                                  │     ├─ Filter (sender_id)     │
  │                                  │     └─ db.store_record()      │
```

---

## Pseudocode: `core/memory/sync_publisher.py`

```
MODULE sync_publisher

IMPORT json, logging
FROM typing IMPORT Optional
FROM .types IMPORT MemoryRecord

LOG = logging.getLogger(__name__)


CLASS MemorySyncPublisher:
    """Publishes new MemoryRecords to Redis for cross-agent sync.

    Uses Redis pub/sub on the synapse channel (port 6380).
    Gracefully degrades if Redis unavailable.
    """

    CONSTRUCTOR(
        agent_id: str,
        redis_url: str = "redis://localhost:6380",
        channel: str = "bizra:memory:new",
    ):
        self._agent_id = agent_id
        self._redis_url = redis_url
        self._channel = channel
        self._client = None
        self._connected = False
        self._buffer: list = []  # Buffer for offline messages
        self._max_buffer = 1000

    ASYNC METHOD connect():
        """Connect to Redis. No-op if already connected or Redis unavailable."""
        IF self._connected:
            RETURN

        TRY:
            IMPORT redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            # Ping to verify connection
            AWAIT self._client.ping()
            self._connected = True
            LOG.info(f"MemorySyncPublisher connected to {self._redis_url}")

            # Flush buffer if we had offline messages
            IF self._buffer:
                LOG.info(f"Flushing {len(self._buffer)} buffered messages")
                FOR msg IN self._buffer:
                    AWAIT self._client.publish(self._channel, msg)
                self._buffer.clear()

        EXCEPT ImportError:
            LOG.warning("redis package not installed — sync disabled")
        EXCEPT Exception as e:
            LOG.warning(f"Redis connection failed: {e} — buffering locally")

    ASYNC METHOD publish(record: MemoryRecord):
        """Publish a record to the sync channel.

        If Redis unavailable, buffers locally for retry on reconnect.
        """
        message = json.dumps({
            "sender_id": self._agent_id,
            "record": record.to_dict(),
            "has_embedding": record.embedding IS NOT None,
        })

        IF self._connected AND self._client:
            TRY:
                AWAIT self._client.publish(self._channel, message)
                RETURN
            EXCEPT Exception as e:
                LOG.warning(f"Publish failed: {e}")
                self._connected = False

        # Buffer for retry
        IF len(self._buffer) < self._max_buffer:
            self._buffer.append(message)
        ELSE:
            LOG.warning("Sync buffer full — dropping oldest message")
            self._buffer.pop(0)
            self._buffer.append(message)

    ASYNC METHOD disconnect():
        IF self._client:
            AWAIT self._client.close()
            self._connected = False
```

---

## Pseudocode: `core/memory/sync_subscriber.py`

```
MODULE sync_subscriber

IMPORT asyncio, json, logging
FROM datetime IMPORT datetime
FROM typing IMPORT List, Optional, Set

FROM .agent_db IMPORT AgentDB
FROM .types IMPORT MemoryKind, MemoryRecord, RecordState

LOG = logging.getLogger(__name__)


CLASS MemorySyncSubscriber:
    """Listens for new MemoryRecords from other agents via Redis pub/sub.

    Automatically imports records into the local AgentDB.
    Deduplicates by content-addressable ID and filters own messages.
    """

    CONSTRUCTOR(
        agent_db: AgentDB,
        agent_id: str,
        redis_url: str = "redis://localhost:6380",
        channel: str = "bizra:memory:new",
        accept_kinds: Optional[List[MemoryKind]] = None,
    ):
        self._db = agent_db
        self._agent_id = agent_id
        self._redis_url = redis_url
        self._channel = channel
        self._accept_kinds = accept_kinds  # None = accept all
        self._client = None
        self._pubsub = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._imported_count = 0
        self._skipped_count = 0
        self._seen_ids: Set[str] = set()  # Fast dedup cache

    ASYNC METHOD start():
        """Start listening for sync messages."""
        TRY:
            IMPORT redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
            )
            AWAIT self._client.ping()

            self._pubsub = self._client.pubsub()
            AWAIT self._pubsub.subscribe(self._channel)

            self._running = True
            self._task = asyncio.create_task(self._listen_loop())
            LOG.info(f"MemorySyncSubscriber listening on {self._channel}")

        EXCEPT ImportError:
            LOG.warning("redis package not installed — sync disabled")
        EXCEPT Exception as e:
            LOG.warning(f"Subscriber start failed: {e}")

    ASYNC METHOD _listen_loop():
        """Main listen loop — processes incoming messages."""
        WHILE self._running:
            TRY:
                message = AWAIT self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                IF message AND message["type"] == "message":
                    AWAIT self._handle_message(message["data"])
            EXCEPT asyncio.CancelledError:
                BREAK
            EXCEPT Exception as e:
                LOG.warning(f"Subscriber error: {e}")
                AWAIT asyncio.sleep(1.0)  # Brief backoff

    ASYNC METHOD _handle_message(data: str):
        """Process a single sync message."""
        TRY:
            payload = json.loads(data)
            sender_id = payload.get("sender_id", "")
            record_dict = payload.get("record", {})

            # Skip own messages
            IF sender_id == self._agent_id:
                RETURN

            # Parse record
            record = self._dict_to_record(record_dict)
            IF record IS None:
                self._skipped_count += 1
                RETURN

            # Kind filter
            IF self._accept_kinds AND record.kind NOT IN self._accept_kinds:
                self._skipped_count += 1
                RETURN

            # Dedup: skip if already seen or already in DB
            IF record.id IN self._seen_ids:
                self._skipped_count += 1
                RETURN

            existing = self._db.retrieve(record.id)
            IF existing IS NOT None:
                self._seen_ids.add(record.id)
                self._skipped_count += 1
                RETURN

            # Import the record
            # Note: embedding may be None (sender may not include full vectors)
            self._db.store_record(record)
            self._seen_ids.add(record.id)
            self._imported_count += 1

            LOG.debug(
                f"Synced record {record.id[:8]}... from {sender_id} "
                f"(kind={record.kind.value})"
            )

        EXCEPT json.JSONDecodeError:
            LOG.warning("Invalid JSON in sync message")
        EXCEPT Exception as e:
            LOG.warning(f"Failed to handle sync message: {e}")

    METHOD _dict_to_record(d: dict) -> Optional[MemoryRecord]:
        """Reconstruct a MemoryRecord from a dict (reverse of to_dict)."""
        TRY:
            RETURN MemoryRecord(
                id=d["id"],
                content=d["content"],
                kind=MemoryKind(d.get("kind", "semantic")),
                state=RecordState(d.get("state", "active")),
                embedding=None,  # Don't sync embeddings (too large for pub/sub)
                ihsan_score=d.get("ihsan_score", 1.0),
                snr_score=d.get("snr_score", 1.0),
                importance=d.get("importance", 0.5),
                source=d.get("source", "sync"),
                source_id=d.get("source_id"),
                related_ids=d.get("related_ids", []),
                tags=d.get("tags", []) + ["synced"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                last_accessed=datetime.fromisoformat(d["last_accessed"]),
                access_count=d.get("access_count", 0),
                metadata={**d.get("metadata", {}), "synced_from": d.get("source", "unknown")},
            )
        EXCEPT Exception as e:
            LOG.warning(f"Failed to parse synced record: {e}")
            RETURN None

    ASYNC METHOD stop():
        self._running = False
        IF self._task:
            self._task.cancel()
            TRY:
                AWAIT self._task
            EXCEPT asyncio.CancelledError:
                PASS
        IF self._pubsub:
            AWAIT self._pubsub.unsubscribe(self._channel)
        IF self._client:
            AWAIT self._client.close()
        LOG.info(
            f"Subscriber stopped: {self._imported_count} imported, "
            f"{self._skipped_count} skipped"
        )

    METHOD stats() -> dict:
        RETURN {
            "running": self._running,
            "imported": self._imported_count,
            "skipped": self._skipped_count,
            "seen_cache_size": len(self._seen_ids),
            "channel": self._channel,
        }
```

---

## Config Additions: `core/memory/config.py`

```
IN MemoryConfig, ADD:

    # Cross-agent sync via Redis synapse
    sync_enabled: bool = False
    sync_redis_url: str = "redis://localhost:6380"
    sync_channel: str = "bizra:memory:new"
    sync_agent_id: str = "node0"
    sync_accept_kinds: Optional[List[str]] = None  # None = all kinds
```

---

## TDD Anchors

```
TEST test_publisher_connects_and_publishes:
    # Use fakeredis for test isolation
    publisher = MemorySyncPublisher("agent_a", redis_url="redis://localhost:6380")
    AWAIT publisher.connect()

    record = MemoryRecord(id="test123", content="shared knowledge")
    AWAIT publisher.publish(record)
    # Verify message published to channel (via fakeredis subscriber)

TEST test_publisher_buffers_when_offline:
    publisher = MemorySyncPublisher("agent_a", redis_url="redis://unreachable:9999")
    AWAIT publisher.connect()  # Should not crash

    record = MemoryRecord(id="test123", content="buffered")
    AWAIT publisher.publish(record)
    ASSERT len(publisher._buffer) == 1

TEST test_subscriber_imports_from_other_agent:
    db = AgentDB(config)
    db.initialize()

    subscriber = MemorySyncSubscriber(db, agent_id="agent_b")

    # Simulate incoming message from agent_a
    message = json.dumps({
        "sender_id": "agent_a",
        "record": MemoryRecord(id="abc123", content="fact from A").to_dict(),
    })
    AWAIT subscriber._handle_message(message)

    ASSERT subscriber._imported_count == 1
    ASSERT db.retrieve("abc123") IS NOT None

TEST test_subscriber_filters_own_messages:
    subscriber = MemorySyncSubscriber(db, agent_id="agent_b")

    message = json.dumps({
        "sender_id": "agent_b",  # Same as subscriber
        "record": MemoryRecord(id="self123", content="my own").to_dict(),
    })
    AWAIT subscriber._handle_message(message)

    ASSERT subscriber._imported_count == 0

TEST test_subscriber_dedup_by_id:
    subscriber = MemorySyncSubscriber(db, agent_id="agent_b")

    message = json.dumps({
        "sender_id": "agent_a",
        "record": MemoryRecord(id="dup123", content="same content").to_dict(),
    })
    # Send twice
    AWAIT subscriber._handle_message(message)
    AWAIT subscriber._handle_message(message)

    ASSERT subscriber._imported_count == 1
    ASSERT subscriber._skipped_count == 1

TEST test_subscriber_kind_filter:
    subscriber = MemorySyncSubscriber(
        db, agent_id="agent_b",
        accept_kinds=[MemoryKind.SEMANTIC]
    )

    # Send EPISODIC (should be filtered)
    msg = json.dumps({
        "sender_id": "agent_a",
        "record": MemoryRecord(
            id="ep1", content="episode", kind=MemoryKind.EPISODIC
        ).to_dict(),
    })
    AWAIT subscriber._handle_message(msg)
    ASSERT subscriber._skipped_count == 1

    # Send SEMANTIC (should be accepted)
    msg2 = json.dumps({
        "sender_id": "agent_a",
        "record": MemoryRecord(
            id="sem1", content="fact", kind=MemoryKind.SEMANTIC
        ).to_dict(),
    })
    AWAIT subscriber._handle_message(msg2)
    ASSERT subscriber._imported_count == 1

TEST test_reconnect_flushes_buffer:
    publisher = MemorySyncPublisher("agent_a")
    # Start offline
    publisher._buffer = [json.dumps({"sender_id": "agent_a", "record": {}})]

    # Connect (with fakeredis)
    AWAIT publisher.connect()

    # Buffer should be flushed
    ASSERT len(publisher._buffer) == 0
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Don't sync embeddings | Too large for pub/sub (768*4=3KB per msg). Let receiver auto-embed. |
| Add "synced" tag | Distinguish synced records from locally-originated ones. |
| Buffer on disconnect | Don't lose messages during transient Redis outages. |
| Seen-ID cache | O(1) dedup without DB lookup for repeated messages. |
| Optional feature | `sync_enabled=False` by default — opt-in only. |
| Port 6380 (synapse) | Dedicated inter-agent Redis, separate from cache (6379). |
