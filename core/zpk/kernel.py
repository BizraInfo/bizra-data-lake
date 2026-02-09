"""
Zero Point Kernel (ZPK) v0.1
============================
Userland-first bootstrap kernel for sovereign nodes.

Flow:
    identity -> fetch -> verify -> policy gate -> execute -> receipt
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

from core.pci.crypto import (
    canonicalize_json,
    generate_keypair,
    sign_message,
    verify_digest_match,
    verify_signature,
)

logger = logging.getLogger("zpk.kernel")

TPMQuoteProvider = Callable[[], Dict[str, Any] | Awaitable[Dict[str, Any]]]


@dataclass
class ZPKConfig:
    """Execution and persistence settings."""

    worker_timeout_seconds: float = 20.0
    max_restarts: int = 1
    receipts_relpath: str = "receipts/zpk_receipts.jsonl"
    identity_relpath: str = "identity/zpk_identity.json"
    lkg_manifest_relpath: str = "lkg/manifest.json"
    lkg_worker_relpath: str = "lkg/worker.py"


@dataclass
class ZPKPolicy:
    """Policy gate rules applied before execution."""

    allowed_versions: Optional[Set[str]] = None
    min_policy_version: int = 1
    min_ihsan_policy: float = 0.95
    ruleset_version: str = "zpk-policy-v1"


@dataclass
class WorkerArtifact:
    """Fetched and verified worker payload."""

    version: str
    worker_code: str
    worker_hash: str
    worker_signature: str
    policy_version: int
    ihsan_policy: float
    source_uri: str


@dataclass
class AttestationReceipt:
    device_id: str
    pubkey: str
    tpm_quote: Optional[str]
    pcrs: Optional[Dict[str, Any]]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "pubkey": self.pubkey,
            "tpm_quote": self.tpm_quote,
            "pcrs": self.pcrs,
            "timestamp": self.timestamp,
        }


@dataclass
class AttestationChallenge:
    challenge_id: str
    verifier_id: str
    nonce: str
    issued_at: float
    expires_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "verifier_id": self.verifier_id,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


@dataclass
class AttestationResponse:
    challenge_id: str
    verifier_id: str
    nonce: str
    device_id: str
    pubkey: str
    timestamp: float
    tpm_quote: Optional[str]
    pcrs: Optional[Dict[str, Any]]
    digest: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "verifier_id": self.verifier_id,
            "nonce": self.nonce,
            "device_id": self.device_id,
            "pubkey": self.pubkey,
            "timestamp": self.timestamp,
            "tpm_quote": self.tpm_quote,
            "pcrs": self.pcrs,
            "digest": self.digest,
            "signature": self.signature,
        }


@dataclass
class FetchReceipt:
    url: str
    hash: Optional[str]
    signature_ok: bool
    bytes: int
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "hash": self.hash,
            "signature_ok": self.signature_ok,
            "bytes": self.bytes,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PolicyReceipt:
    ruleset_version: str
    decision: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ruleset_version": self.ruleset_version,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


@dataclass
class ExecutionReceipt:
    worker_version: str
    exit_code: int
    runtime_ms: float
    health: Dict[str, Any]
    rollback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_version": self.worker_version,
            "exit_code": self.exit_code,
            "runtime_ms": self.runtime_ms,
            "health": dict(self.health),
            "rollback_used": self.rollback_used,
        }


@dataclass
class BootstrapResult:
    success: bool
    executed_version: Optional[str]
    rollback_used: bool
    reason: str


class ZeroPointKernel:
    """
    Minimal bootstrap kernel with strict verify-before-execute guarantees.

    Acceptance guarantees:
    - signature fail -> no execute
    - policy fail -> no execute
    - fetch fail -> rollback to last-known-good
    - each execute -> append-only receipt
    """

    def __init__(
        self,
        state_dir: Path,
        release_public_key_hex: str,
        config: Optional[ZPKConfig] = None,
        event_bus: Optional[Any] = None,
        event_topic: str = "zpk.bootstrap.receipt",
        tpm_quote_provider: Optional[TPMQuoteProvider] = None,
    ) -> None:
        self.state_dir = state_dir
        self.config = config or ZPKConfig()
        self.release_public_key_hex = release_public_key_hex
        self._event_bus = event_bus
        self._event_topic = event_topic
        self._tpm_quote_provider = tpm_quote_provider

        self._receipts_path = self.state_dir / self.config.receipts_relpath
        self._identity_path = self.state_dir / self.config.identity_relpath
        self._lkg_manifest_path = self.state_dir / self.config.lkg_manifest_relpath
        self._lkg_worker_path = self.state_dir / self.config.lkg_worker_relpath

        self._identity_private_key, self._identity_public_key = self._load_or_create_identity()
        self._device_id = self._compute_device_id(self._identity_public_key)
        self._last_receipt_hash = self._load_last_receipt_hash()

    async def bootstrap(
        self,
        manifest_uri: str,
        policy: Optional[ZPKPolicy] = None,
    ) -> BootstrapResult:
        """
        Execute one ZPK bootstrap cycle.

        Returns a compact decision outcome for orchestrators.
        """
        policy = policy or ZPKPolicy()
        tpm_quote, pcrs = await self._collect_tpm_attestation()

        attestation = AttestationReceipt(
            device_id=self._device_id,
            pubkey=self._identity_public_key,
            tpm_quote=tpm_quote,
            pcrs=pcrs,
            timestamp=time.time(),
        )
        await self._append_receipt("attestation", attestation.to_dict())

        artifact, fetch_receipt = await self._fetch_and_verify(manifest_uri)
        await self._append_receipt("fetch", fetch_receipt.to_dict())

        if artifact is None:
            lkg = self._load_last_known_good()
            if lkg is None:
                return BootstrapResult(
                    success=False,
                    executed_version=None,
                    rollback_used=False,
                    reason=f"fetch failed: {fetch_receipt.error or 'unknown'}",
                )
            execution = await self._execute_worker(lkg, rollback_used=True)
            await self._append_receipt("execution", execution.to_dict())
            return BootstrapResult(
                success=execution.exit_code == 0,
                executed_version=lkg.version,
                rollback_used=True,
                reason="rollback_to_last_known_good",
            )

        policy_receipt = self._evaluate_policy(artifact, policy)
        await self._append_receipt("policy", policy_receipt.to_dict())
        if policy_receipt.decision != "allow":
            return BootstrapResult(
                success=False,
                executed_version=None,
                rollback_used=False,
                reason="policy_denied",
            )

        execution = await self._execute_worker(artifact, rollback_used=False)
        await self._append_receipt("execution", execution.to_dict())
        if execution.exit_code == 0:
            self._persist_last_known_good(artifact)
            return BootstrapResult(
                success=True,
                executed_version=artifact.version,
                rollback_used=False,
                reason="executed",
            )

        return BootstrapResult(
            success=False,
            executed_version=artifact.version,
            rollback_used=False,
            reason="execution_failed",
        )

    async def _collect_tpm_attestation(self) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Collect optional TPM quote/pcrs via pluggable provider."""
        if not self._tpm_quote_provider:
            return None, None

        try:
            payload = self._tpm_quote_provider()
            if inspect.isawaitable(payload):
                payload = await payload
            if not isinstance(payload, dict):
                return None, None
            tpm_quote = payload.get("quote")
            pcrs = payload.get("pcrs")
            if tpm_quote is not None and not isinstance(tpm_quote, str):
                tpm_quote = str(tpm_quote)
            if pcrs is not None and not isinstance(pcrs, dict):
                pcrs = None
            return tpm_quote, pcrs
        except Exception as e:
            logger.warning("TPM quote collection failed: %s", e)
            return None, None

    def issue_attestation_challenge(
        self,
        verifier_id: str,
        ttl_seconds: int = 120,
        now: Optional[float] = None,
    ) -> AttestationChallenge:
        """Issue a time-bounded remote attestation challenge."""
        issued_at = now if now is not None else time.time()
        nonce_seed = f"{self._device_id}:{verifier_id}:{issued_at}:{self._last_receipt_hash or ''}"
        nonce = hashlib.sha256(nonce_seed.encode("utf-8")).hexdigest()
        challenge_id = hashlib.sha256(
            f"challenge:{nonce}:{issued_at}".encode("utf-8")
        ).hexdigest()[:24]
        return AttestationChallenge(
            challenge_id=challenge_id,
            verifier_id=verifier_id,
            nonce=nonce,
            issued_at=issued_at,
            expires_at=issued_at + max(10, int(ttl_seconds)),
        )

    async def answer_attestation_challenge(
        self,
        challenge: AttestationChallenge,
        now: Optional[float] = None,
    ) -> AttestationResponse:
        """
        Produce signed attestation response for a verifier challenge.

        Includes optional TPM quote/PCR evidence when available.
        """
        timestamp = now if now is not None else time.time()
        tpm_quote, pcrs = await self._collect_tpm_attestation()
        body = {
            "challenge_id": challenge.challenge_id,
            "verifier_id": challenge.verifier_id,
            "nonce": challenge.nonce,
            "device_id": self._device_id,
            "pubkey": self._identity_public_key,
            "timestamp": timestamp,
            "tpm_quote": tpm_quote,
            "pcrs": pcrs,
        }
        digest = self._digest_record(body)
        signature = sign_message(digest, self._identity_private_key)
        return AttestationResponse(
            challenge_id=challenge.challenge_id,
            verifier_id=challenge.verifier_id,
            nonce=challenge.nonce,
            device_id=self._device_id,
            pubkey=self._identity_public_key,
            timestamp=timestamp,
            tpm_quote=tpm_quote,
            pcrs=pcrs,
            digest=digest,
            signature=signature,
        )

    @staticmethod
    def verify_attestation_response(
        challenge: AttestationChallenge,
        response: AttestationResponse,
        require_tpm_quote: bool = False,
        now: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Verify a remote attestation response against challenge constraints."""
        now_ts = now if now is not None else time.time()
        if now_ts > challenge.expires_at:
            return False, "challenge_expired"

        if response.challenge_id != challenge.challenge_id:
            return False, "challenge_id_mismatch"
        if response.verifier_id != challenge.verifier_id:
            return False, "verifier_mismatch"
        if response.nonce != challenge.nonce:
            return False, "nonce_mismatch"

        body = {
            "challenge_id": response.challenge_id,
            "verifier_id": response.verifier_id,
            "nonce": response.nonce,
            "device_id": response.device_id,
            "pubkey": response.pubkey,
            "timestamp": response.timestamp,
            "tpm_quote": response.tpm_quote,
            "pcrs": response.pcrs,
        }
        computed_digest = ZeroPointKernel._digest_record(body)
        if not verify_digest_match(computed_digest, response.digest):
            return False, "digest_mismatch"

        if not verify_signature(response.digest, response.signature, response.pubkey):
            return False, "signature_invalid"

        if require_tpm_quote and not response.tpm_quote:
            return False, "missing_tpm_quote"

        return True, "ok"

    async def _fetch_and_verify(self, manifest_uri: str) -> tuple[Optional[WorkerArtifact], FetchReceipt]:
        """Fetch manifest + worker, then verify digest and detached signature."""
        try:
            manifest_bytes = await self._fetch_bytes(manifest_uri)
            manifest = json.loads(manifest_bytes.decode("utf-8"))

            worker_uri = self._resolve_worker_uri(manifest_uri, manifest.get("worker_uri", ""))
            worker_bytes = await self._fetch_bytes(worker_uri)
            worker_hash = hashlib.sha256(worker_bytes).hexdigest()

            expected_hash = str(manifest.get("worker_hash", ""))
            if not verify_digest_match(worker_hash, expected_hash):
                return None, FetchReceipt(
                    url=worker_uri,
                    hash=worker_hash,
                    signature_ok=False,
                    bytes=len(worker_bytes),
                    success=False,
                    error="digest_mismatch",
                )

            signature = str(manifest.get("worker_signature", ""))
            signature_ok = verify_signature(worker_hash, signature, self.release_public_key_hex)
            if not signature_ok:
                return None, FetchReceipt(
                    url=worker_uri,
                    hash=worker_hash,
                    signature_ok=False,
                    bytes=len(worker_bytes),
                    success=False,
                    error="signature_invalid",
                )

            artifact = WorkerArtifact(
                version=str(manifest.get("version", "unknown")),
                worker_code=worker_bytes.decode("utf-8"),
                worker_hash=worker_hash,
                worker_signature=signature,
                policy_version=int(manifest.get("policy_version", 1)),
                ihsan_policy=float(manifest.get("ihsan_policy", 0.95)),
                source_uri=worker_uri,
            )
            return artifact, FetchReceipt(
                url=worker_uri,
                hash=worker_hash,
                signature_ok=True,
                bytes=len(worker_bytes),
                success=True,
                error=None,
            )

        except Exception as e:
            return None, FetchReceipt(
                url=manifest_uri,
                hash=None,
                signature_ok=False,
                bytes=0,
                success=False,
                error=str(e),
            )

    async def _fetch_bytes(self, uri: str) -> bytes:
        """
        Fetch bytes from local file or HTTPS endpoint.

        For remote sources, plain HTTP is rejected.
        """
        parsed = urlparse(uri)
        if parsed.scheme in ("http", "https"):
            if parsed.scheme != "https":
                raise ValueError("remote fetch requires HTTPS")
            return await asyncio.to_thread(self._download_https, uri)

        if parsed.scheme == "file":
            path = Path(parsed.path)
        else:
            path = Path(uri)

        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _download_https(uri: str) -> bytes:
        with urlopen(uri, timeout=10.0) as response:
            return response.read()

    @staticmethod
    def _resolve_worker_uri(manifest_uri: str, worker_uri: str) -> str:
        """Resolve worker URI relative to manifest location when needed."""
        if not worker_uri:
            raise ValueError("manifest missing worker_uri")

        parsed_worker = urlparse(worker_uri)
        if parsed_worker.scheme in ("http", "https", "file"):
            return worker_uri

        parsed_manifest = urlparse(manifest_uri)
        if parsed_manifest.scheme in ("http", "https"):
            base = manifest_uri.rsplit("/", 1)[0] + "/"
            return urljoin(base, worker_uri)
        if parsed_manifest.scheme == "file":
            manifest_path = Path(parsed_manifest.path)
        else:
            manifest_path = Path(manifest_uri)
        return str((manifest_path.parent / worker_uri).resolve())

    def _evaluate_policy(self, artifact: WorkerArtifact, policy: ZPKPolicy) -> PolicyReceipt:
        """Evaluate policy constraints before execution."""
        reasons: List[str] = []

        if (
            policy.allowed_versions is not None
            and artifact.version not in policy.allowed_versions
        ):
            reasons.append(f"version_not_allowed:{artifact.version}")

        if artifact.policy_version < policy.min_policy_version:
            reasons.append(
                f"policy_version_too_old:{artifact.policy_version}<{policy.min_policy_version}"
            )

        if artifact.ihsan_policy < policy.min_ihsan_policy:
            reasons.append(
                f"ihsan_policy_too_low:{artifact.ihsan_policy:.3f}<{policy.min_ihsan_policy:.3f}"
            )

        return PolicyReceipt(
            ruleset_version=policy.ruleset_version,
            decision="allow" if not reasons else "deny",
            reasons=reasons,
        )

    async def _execute_worker(self, artifact: WorkerArtifact, rollback_used: bool) -> ExecutionReceipt:
        """
        Execute worker script with supervision and restart budget.

        Worker entrypoint contract:
        - `main(context: dict)` or `run(context: dict)`
        """
        attempts = self.config.max_restarts + 1
        last_error = ""
        start_all = time.perf_counter()

        for attempt in range(1, attempts + 1):
            try:
                namespace: Dict[str, Any] = {}
                exec(artifact.worker_code, namespace)
                entrypoint = namespace.get("main") or namespace.get("run")
                if not callable(entrypoint):
                    raise RuntimeError("worker missing callable main/run entrypoint")

                context = {
                    "attempt": attempt,
                    "version": artifact.version,
                    "rollback": rollback_used,
                }
                result = entrypoint(context)
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=self.config.worker_timeout_seconds)
                elapsed_ms = (time.perf_counter() - start_all) * 1000
                return ExecutionReceipt(
                    worker_version=artifact.version,
                    exit_code=0,
                    runtime_ms=round(elapsed_ms, 2),
                    health={"attempts": attempt, "last_error": None},
                    rollback_used=rollback_used,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning("worker attempt %d failed: %s", attempt, e)

        elapsed_ms = (time.perf_counter() - start_all) * 1000
        return ExecutionReceipt(
            worker_version=artifact.version,
            exit_code=1,
            runtime_ms=round(elapsed_ms, 2),
            health={"attempts": attempts, "last_error": last_error},
            rollback_used=rollback_used,
        )

    def _persist_last_known_good(self, artifact: WorkerArtifact) -> None:
        """Persist last-known-good worker for fetch-failure rollback."""
        self._lkg_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._lkg_worker_path.parent.mkdir(parents=True, exist_ok=True)
        self._lkg_manifest_path.write_text(
            json.dumps(
                {
                    "version": artifact.version,
                    "worker_hash": artifact.worker_hash,
                    "worker_signature": artifact.worker_signature,
                    "policy_version": artifact.policy_version,
                    "ihsan_policy": artifact.ihsan_policy,
                    "source_uri": artifact.source_uri,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._lkg_worker_path.write_text(artifact.worker_code, encoding="utf-8")

    def _load_last_known_good(self) -> Optional[WorkerArtifact]:
        """Load last-known-good worker artifact if available."""
        if not self._lkg_manifest_path.exists() or not self._lkg_worker_path.exists():
            return None
        try:
            manifest = json.loads(self._lkg_manifest_path.read_text(encoding="utf-8"))
            code = self._lkg_worker_path.read_text(encoding="utf-8")
            return WorkerArtifact(
                version=str(manifest.get("version", "unknown")),
                worker_code=code,
                worker_hash=str(manifest.get("worker_hash", "")),
                worker_signature=str(manifest.get("worker_signature", "")),
                policy_version=int(manifest.get("policy_version", 1)),
                ihsan_policy=float(manifest.get("ihsan_policy", 0.95)),
                source_uri=str(manifest.get("source_uri", "lkg")),
            )
        except Exception:
            return None

    def _load_or_create_identity(self) -> tuple[str, str]:
        """Load node identity keypair or create one if missing."""
        if self._identity_path.exists():
            data = json.loads(self._identity_path.read_text(encoding="utf-8"))
            return str(data["private_key_hex"]), str(data["public_key_hex"])

        private_key_hex, public_key_hex = generate_keypair()
        self._identity_path.parent.mkdir(parents=True, exist_ok=True)
        self._identity_path.write_text(
            json.dumps(
                {
                    "private_key_hex": private_key_hex,
                    "public_key_hex": public_key_hex,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return private_key_hex, public_key_hex

    @staticmethod
    def _compute_device_id(public_key_hex: str) -> str:
        host = socket.gethostname()
        digest = hashlib.sha256(f"{public_key_hex}:{host}".encode("utf-8")).hexdigest()
        return f"zpk-{digest[:16]}"

    def _load_last_receipt_hash(self) -> Optional[str]:
        """Load chain tail hash for append-only receipt chaining."""
        if not self._receipts_path.exists():
            return None
        last_hash: Optional[str] = None
        with self._receipts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        last_hash = rec.get("hash") or last_hash
                except json.JSONDecodeError:
                    continue
        return last_hash

    async def _append_receipt(self, receipt_type: str, body: Dict[str, Any]) -> None:
        """Append signed receipt to JSONL chain."""
        record = {
            "type": receipt_type,
            "body": body,
            "prev_hash": self._last_receipt_hash,
            "timestamp": time.time(),
            "signer_pubkey": self._identity_public_key,
        }

        digest = self._digest_record(record)
        signature = sign_message(digest, self._identity_private_key)
        record["signature"] = signature
        record["hash"] = digest

        self._receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with self._receipts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self._last_receipt_hash = digest

        if self._event_bus is not None and hasattr(self._event_bus, "publish"):
            try:
                from core.sovereign.event_bus import Event, EventPriority

                event = Event(
                    topic=self._event_topic,
                    payload={"receipt": record},
                    priority=EventPriority.HIGH,
                    source="zpk",
                )
                publish_result = self._event_bus.publish(event)
                if inspect.isawaitable(publish_result):
                    await publish_result
            except Exception as e:
                logger.debug("Receipt event publish failed: %s", e)

    @staticmethod
    def _digest_record(record: Dict[str, Any]) -> str:
        canonical = canonicalize_json(record, ensure_ascii=True)
        return hashlib.sha256(canonical).hexdigest()
