# core/receipt_batching.py - Receipt Batching Implementation
# Standing on Shoulders of Giants Protocol: Receipt batching and aggregation
# Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 100
BATCH_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
CONCURRENT_BATCHES = 5


class ReceiptBatch:
    def __init__(
        self,
        batch_id: str,
        batch_type: str = "standard",
    ):
        self.batch_id = batch_id
        self.batch_type = batch_type
        self.receipts: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.submitted_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def add_receipt(self, receipt: Dict[str, Any]) -> bool:
        if len(self.receipts) >= MAX_BATCH_SIZE:
            return False
        self.receipts.append(receipt)
        return True

    def size(self) -> int:
        return len(self.receipts)

    def is_full(self) -> bool:
        return self.size() >= MAX_BATCH_SIZE

    def is_ready(self, timeout_seconds: int = BATCH_TIMEOUT_SECONDS) -> bool:
        age = time.time() - self.created_at
        return age >= timeout_seconds or self.is_full()

    def total_ihsan_score(self) -> float:
        return sum(r.get("ihsan_score", 0.0) for r in self.receipts)

    def average_ihsan_score(self) -> float:
        count = self.size()
        return self.total_ihsan_score() / count if count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_type": self.batch_type,
            "receipts": self.receipts,
            "size": self.size(),
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "average_ihsan_score": self.average_ihsan_score(),
        }


class ReceiptBatcher:
    def __init__(
        self,
        max_batch_size: int = MAX_BATCH_SIZE,
        batch_timeout: int = BATCH_TIMEOUT_SECONDS,
        concurrent_batches: int = CONCURRENT_BATCHES,
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.concurrent_batches = concurrent_batches
        self.pending_batches: Dict[str, ReceiptBatch] = {}
        self.completed_batches: Dict[str, ReceiptBatch] = {}
        self.dedup_set: set = set()
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_receipts_processed": 0,
        }
        self._lock = asyncio.Lock()

    def _compute_receipt_hash(self, receipt: Dict[str, Any]) -> str:
        content = json.dumps(receipt, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_batch_id(self, batch_type: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"batch_{batch_type}_{timestamp}"

    async def add_receipt(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        receipt_hash = self._compute_receipt_hash(receipt)
        
        async with self._lock:
            if receipt_hash in self.dedup_set:
                return {
                    "accepted": False,
                    "reason": "duplicate",
                    "receipt_hash": receipt_hash,
                }
            
            self.dedup_set.add(receipt_hash)
        
        receipt_id = receipt.get("receipt_id", f"receipt_{int(time.time() * 1000)}")
        
        async with self._lock:
            for batch in self.pending_batches.values():
                if not batch.is_full():
                    if batch.add_receipt(receipt):
                        return {
                            "accepted": True,
                            "batch_id": batch.batch_id,
                            "batch_size": batch.size(),
                        }
            
            new_batch_id = self._generate_batch_id("standard")
            new_batch = ReceiptBatch(new_batch_id, "standard")
            new_batch.add_receipt(receipt)
            self.pending_batches[new_batch_id] = new_batch
            
            return {
                "accepted": True,
                "batch_id": new_batch_id,
                "batch_size": 1,
            }

    async def submit_batch(self, batch_id: str) -> Dict[str, Any]:
        async with self._lock:
            batch = self.pending_batches.get(batch_id)
            if not batch:
                return {
                    "success": False,
                    "reason": "batch_not_found",
                }
            
            if batch.size() == 0:
                return {
                    "success": False,
                    "reason": "empty_batch",
                }
        
        batch.submitted_at = time.time()
        
        processed = await self._process_batch(batch)
        
        async with self._lock:
            if processed.get("success"):
                batch.completed_at = time.time()
                self.completed_batches[batch_id] = batch
                del self.pending_batches[batch_id]
                self.stats["total_submitted"] += 1
                self.stats["total_completed"] += 1
                self.stats["total_receipts_processed"] += batch.size()
                
                for receipt in batch.receipts:
                    receipt_hash = self._compute_receipt_hash(receipt)
                    self.dedup_set.discard(receipt_hash)
            else:
                self.stats["total_failed"] += 1
        
        return processed

    async def _process_batch(self, batch: ReceiptBatch) -> Dict[str, Any]:
        try:
            await asyncio.sleep(0.1)
            
            success_count = batch.size()
            total_score = batch.total_ihsan_score()
            avg_score = batch.average_ihsan_score()
            
            return {
                "success": True,
                "batch_id": batch.batch_id,
                "receipts_processed": success_count,
                "total_ihsan_score": total_score,
                "average_ihsan_score": avg_score,
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                "success": False,
                "batch_id": batch.batch_id,
                "error": str(e),
            }

    async def flush_ready_batches(self) -> List[Dict[str, Any]]:
        results = []
        
        async with self._lock:
            ready_batch_ids = [
                bid for bid, batch in self.pending_batches.items()
                if batch.is_ready(self.batch_timeout)
            ]
        
        for batch_id in ready_batch_ids:
            result = await self.submit_batch(batch_id)
            results.append(result)
        
        return results

    async def flush_all_batches(self) -> List[Dict[str, Any]]:
        batch_ids = list(self.pending_batches.keys())
        results = []
        
        for batch_id in batch_ids:
            result = await self.submit_batch(batch_id)
            results.append(result)
        
        return results

    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if batch_id in self.pending_batches:
                batch = self.pending_batches[batch_id]
                return {
                    "status": "pending",
                    "batch_id": batch.batch_id,
                    "size": batch.size(),
                    "created_at": batch.created_at,
                }
            elif batch_id in self.completed_batches:
                batch = self.completed_batches[batch_id]
                return {
                    "status": "completed",
                    "batch_id": batch.batch_id,
                    "size": batch.size(),
                    "submitted_at": batch.submitted_at,
                    "completed_at": batch.completed_at,
                }
        
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pending_batches": len(self.pending_batches),
            "completed_batches": len(self.completed_batches),
            "total_submitted": self.stats["total_submitted"],
            "total_completed": self.stats["total_completed"],
            "total_failed": self.stats["total_failed"],
            "total_receipts_processed": self.stats["total_receipts_processed"],
            "dedup_set_size": len(self.dedup_set),
        }

    def get_pending_count(self) -> int:
        return sum(b.size() for b in self.pending_batches.values())

    async def clear_completed(self, max_age_seconds: int = 3600) -> int:
        cleared = 0
        now = time.time()
        
        async with self._lock:
            expired_batches = [
                bid for bid, batch in self.completed_batches.items()
                if batch.completed_at and (now - batch.completed_at) > max_age_seconds
            ]
            
            for bid in expired_batches:
                del self.completed_batches[bid]
                cleared += 1
        
        return cleared


class ReceiptBatchingService:
    def __init__(self):
        self.batcher = ReceiptBatcher()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            await self._task
        await self.batcher.flush_all_batches()

    async def _run(self):
        while self._running:
            try:
                await self.batcher.flush_ready_batches()
            except Exception as e:
                logger.error(f"Batching error: {e}")
            await asyncio.sleep(1)

    async def submit(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        return await self.batcher.add_receipt(receipt)

    def get_stats(self) -> Dict[str, Any]:
        return self.batcher.get_stats()


async def create_receipt(
    agent_id: str,
    operation: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    ihsan_score: float,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    
    input_content = json.dumps(input_data, sort_keys=True)
    input_hash = hashlib.sha256(input_content.encode()).hexdigest()
    
    output_content = json.dumps(output_data, sort_keys=True)
    output_hash = hashlib.sha256(output_content.encode()).hexdigest()
    
    return {
        "receipt_id": f"receipt_{int(time.time() * 1000000)}",
        "timestamp": timestamp,
        "agent_id": agent_id,
        "operation": operation,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "ihsan_score": ihsan_score,
    }


async def demo():
    service = ReceiptBatchingService()
    await service.start()
    
    for i in range(150):
        receipt = await create_receipt(
            agent_id=f"agent_{i % 5}",
            operation="execute",
            input_data={"task": f"task_{i}"},
            output_data={"result": f"result_{i}"},
            ihsan_score=0.75 + (i % 25) * 0.01,
        )
        result = await service.submit(receipt)
        print(f"Receipt {i}: {result}")
    
    await asyncio.sleep(2)
    stats = service.get_stats()
    print(f"Final stats: {json.dumps(stats, indent=2)}")
    
    await service.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())