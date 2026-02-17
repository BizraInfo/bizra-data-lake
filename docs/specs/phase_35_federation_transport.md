# Phase 35: Federation Transport — DTLS Handshake + DoS Protection + Rust Bridge

> Completes the secure transport layer for multi-node federation with full DTLS/QUIC handshake, cookie-based DoS protection, and Rust-accelerated cryptographic operations.

Standing on Giants: Lamport (1982, Byzantine fault tolerance) + Rescorla & Modadugu (2012, DTLS 1.2) + Bernstein (2006, Ed25519 signatures) + Shannon (1949, communication theory of secrecy)

## Context

`core/federation/secure_transport.py` (1,504 lines) has two critical gaps:
1. **Line 574**: Handshake raises `HandshakeError("Handshake initiated, awaiting response")` — placeholder, not a real async handshake
2. **Line 1028**: `# TODO: Implement cookie verification for DoS protection` — no cookie-based amplification resistance

The Rust federation crate (`bizra-omega/bizra-federation/`) has clean implementations but **no PyO3 bridge** to Python. Node-to-node federation currently runs Python-only.

## Gaps Addressed

| Gap | Current State | Target State |
|-----|--------------|--------------|
| DTLS handshake | Placeholder raises exception | Full async ClientHello → HelloVerifyRequest → ServerHello flow |
| DoS cookie | `# TODO` comment | HMAC-SHA256 cookie with IP+port binding |
| Rust federation bindings | Absent | PyO3 wrappers for gossip + consensus |
| Multi-node integration test | Absent | 3-node gossip convergence test |

## 1. DTLS Handshake Completion

```
ASYNC FUNCTION perform_handshake(self, peer_addr: tuple) -> SecureSession:
  """
  Full DTLS 1.2-style handshake (simplified for sovereignty use case).

  Flow:
    Client                        Server
      |-- ClientHello ------------->|
      |<-- HelloVerifyRequest ------|  (with cookie)
      |-- ClientHello + cookie ---->|
      |<-- ServerHello + Cert ------|
      |-- ClientKeyExchange ------->|
      |-- Finished ---------------->|
      |<-- Finished ----------------|

  Standing on Giants: Rescorla & Modadugu (DTLS)
  Artifact: core/federation/secure_transport.py
  """

  # Step 1: Send ClientHello
  client_random = os.urandom(32)
  client_hello = HandshakeMessage(
    msg_type=HandshakeType.CLIENT_HELLO,
    payload={
      "protocol_version": (1, 2),
      "random": client_random,
      "session_id": b"",
      "cipher_suites": [CipherSuite.ED25519_AES256_SHA256],
    }
  )
  AWAIT self._send(peer_addr, client_hello.serialize())

  # Step 2: Receive HelloVerifyRequest with cookie
  response = AWAIT asyncio.wait_for(
    self._receive_from(peer_addr),
    timeout=self.config.handshake_timeout
  )
  verify_request = HandshakeMessage.deserialize(response)
  IF verify_request.msg_type != HandshakeType.HELLO_VERIFY_REQUEST:
    RAISE HandshakeError(f"Expected HelloVerifyRequest, got {verify_request.msg_type}")

  cookie = verify_request.payload["cookie"]

  # Step 3: Resend ClientHello with cookie
  client_hello.payload["cookie"] = cookie
  AWAIT self._send(peer_addr, client_hello.serialize())

  # Step 4: Receive ServerHello + Certificate
  server_hello_raw = AWAIT asyncio.wait_for(
    self._receive_from(peer_addr),
    timeout=self.config.handshake_timeout
  )
  server_hello = HandshakeMessage.deserialize(server_hello_raw)
  server_random = server_hello.payload["random"]
  server_cert = server_hello.payload["certificate"]  # Ed25519 public key

  # Step 5: Verify server identity against known peers
  IF NOT self._verify_peer_identity(peer_addr, server_cert):
    RAISE HandshakeError("Server certificate not in trusted peer set")

  # Step 6: Key exchange (X25519 ECDHE)
  private_key = X25519PrivateKey.generate()
  public_key = private_key.public_key()
  key_exchange = HandshakeMessage(
    msg_type=HandshakeType.CLIENT_KEY_EXCHANGE,
    payload={"public_key": public_key.public_bytes_raw()}
  )
  AWAIT self._send(peer_addr, key_exchange.serialize())

  # Step 7: Derive session keys
  server_public = X25519PublicKey.from_public_bytes(
    server_hello.payload["key_exchange"]
  )
  shared_secret = private_key.exchange(server_public)
  session_keys = self._derive_keys(shared_secret, client_random, server_random)

  # Step 8: Send Finished (HMAC of handshake transcript)
  transcript_hash = self._hash_transcript()
  finished = HandshakeMessage(
    msg_type=HandshakeType.FINISHED,
    payload={"verify_data": hmac_sha256(session_keys.client_write, transcript_hash)}
  )
  AWAIT self._send(peer_addr, finished.serialize())

  # Step 9: Receive server Finished
  server_finished_raw = AWAIT asyncio.wait_for(
    self._receive_from(peer_addr),
    timeout=self.config.handshake_timeout
  )
  server_finished = HandshakeMessage.deserialize(server_finished_raw)
  expected = hmac_sha256(session_keys.server_write, transcript_hash)
  IF server_finished.payload["verify_data"] != expected:
    RAISE HandshakeError("Server Finished verification failed")

  # Step 10: Session established
  session = SecureSession(
    peer_addr=peer_addr,
    session_keys=session_keys,
    peer_identity=server_cert,
    established_at=datetime.utcnow(),
  )
  self._sessions[peer_addr] = session
  RETURN session
```

---

## 2. Cookie-Based DoS Protection

```
CLASS DoSCookieVerifier:
  """
  HMAC-SHA256 cookie bound to client IP:port to prevent
  amplification attacks on the handshake protocol.

  Standing on Giants: Rescorla (DTLS cookie mechanism)
  Artifact: core/federation/secure_transport.py
  """

  FUNCTION __init__(self, secret: bytes = None, ttl_seconds: int = 60):
    self._secret = secret OR os.urandom(32)
    self._ttl = ttl_seconds

  FUNCTION generate(self, client_addr: tuple) -> bytes:
    """Generate a cookie for the given client address."""
    timestamp = int(time.time())
    data = f"{client_addr[0]}:{client_addr[1]}:{timestamp}".encode()
    mac = hmac.new(self._secret, data, hashlib.sha256).digest()
    RETURN struct.pack("!I", timestamp) + mac    # 4 + 32 = 36 bytes

  FUNCTION verify(self, client_addr: tuple, cookie: bytes) -> bool:
    """Verify cookie authenticity and freshness."""
    IF len(cookie) != 36:
      RETURN False

    timestamp = struct.unpack("!I", cookie[:4])[0]
    now = int(time.time())

    # Check TTL
    IF now - timestamp > self._ttl:
      RETURN False

    # Recompute HMAC
    data = f"{client_addr[0]}:{client_addr[1]}:{timestamp}".encode()
    expected_mac = hmac.new(self._secret, data, hashlib.sha256).digest()
    RETURN hmac.compare_digest(cookie[4:], expected_mac)

  FUNCTION rotate_secret(self):
    """Rotate the HMAC secret. Old cookies become invalid."""
    self._secret = os.urandom(32)
```

---

## 3. Server-Side Handshake Handler

```
ASYNC FUNCTION handle_client_hello(self, client_addr: tuple,
                                    client_hello: HandshakeMessage) -> None:
  """
  Server-side handshake processing.
  """

  # Phase 1: If no cookie, send HelloVerifyRequest
  IF "cookie" NOT IN client_hello.payload OR NOT client_hello.payload["cookie"]:
    cookie = self._cookie_verifier.generate(client_addr)
    verify_request = HandshakeMessage(
      msg_type=HandshakeType.HELLO_VERIFY_REQUEST,
      payload={"cookie": cookie}
    )
    AWAIT self._send(client_addr, verify_request.serialize())
    RETURN  # Wait for client to resend with cookie

  # Phase 2: Verify cookie
  IF NOT self._cookie_verifier.verify(client_addr, client_hello.payload["cookie"]):
    self.logger.warning(f"Invalid cookie from {client_addr} — dropping")
    RETURN  # Silent drop (DoS resistance)

  # Phase 3: Cookie valid — proceed with full handshake
  server_random = os.urandom(32)
  ecdhe_private = X25519PrivateKey.generate()

  server_hello = HandshakeMessage(
    msg_type=HandshakeType.SERVER_HELLO,
    payload={
      "random": server_random,
      "certificate": self._node_identity.public_key_bytes(),
      "key_exchange": ecdhe_private.public_key().public_bytes_raw(),
      "cipher_suite": CipherSuite.ED25519_AES256_SHA256,
    }
  )
  AWAIT self._send(client_addr, server_hello.serialize())

  # Phase 4: Await ClientKeyExchange + Finished (handled in message loop)
  self._pending_handshakes[client_addr] = PendingHandshake(
    ecdhe_private=ecdhe_private,
    client_random=client_hello.payload["random"],
    server_random=server_random,
    started_at=datetime.utcnow(),
  )
```

---

## 4. Rust Federation PyO3 Bridge

```rust
// bizra-python/src/federation.rs — NEW

#[pyclass(name = "GossipProtocol")]
pub struct PyGossipProtocol {
    inner: GossipProtocol,
}

#[pymethods]
impl PyGossipProtocol {
    #[new]
    fn new(node_id: &str, fanout: usize) -> Self {
        Self { inner: GossipProtocol::new(NodeId::from_str(node_id), fanout) }
    }

    /// Receive a gossip message and return messages to forward.
    fn receive(&mut self, from_node: &str, payload: &[u8]) -> Vec<(String, Vec<u8>)> {
        self.inner.receive(NodeId::from_str(from_node), payload)
            .into_iter()
            .map(|(node, data)| (node.to_string(), data))
            .collect()
    }

    /// Check if a message has been seen (bloom filter).
    fn has_seen(&self, message_id: &[u8]) -> bool {
        self.inner.has_seen(message_id)
    }
}
```

---

## 5. TDD Anchors

```
TEST test_cookie_generate_verify_roundtrip:
  verifier = DoSCookieVerifier()
  addr = ("192.168.1.1", 9000)
  cookie = verifier.generate(addr)
  ASSERT verifier.verify(addr, cookie) IS True

TEST test_cookie_rejects_wrong_address:
  verifier = DoSCookieVerifier()
  cookie = verifier.generate(("192.168.1.1", 9000))
  ASSERT verifier.verify(("10.0.0.1", 9000), cookie) IS False

TEST test_cookie_rejects_expired:
  verifier = DoSCookieVerifier(ttl_seconds=1)
  cookie = verifier.generate(("192.168.1.1", 9000))
  time.sleep(2)
  ASSERT verifier.verify(("192.168.1.1", 9000), cookie) IS False

TEST test_handshake_full_roundtrip:
  # Two in-memory transport instances
  server = SecureTransport(identity=server_key)
  client = SecureTransport(identity=client_key)
  # Add each other as trusted peers
  server.add_trusted_peer(client.identity)
  client.add_trusted_peer(server.identity)
  # Perform handshake
  session = AWAIT client.perform_handshake(server.address)
  ASSERT session.peer_identity == server.identity.public_key_bytes()

TEST test_handshake_rejects_untrusted_server:
  client = SecureTransport(identity=client_key)
  # Do NOT add server as trusted
  ASSERT_RAISES(HandshakeError, AWAIT client.perform_handshake(rogue.address))

TEST test_server_drops_invalid_cookie:
  server = SecureTransport(identity=server_key)
  bad_hello = HandshakeMessage(CLIENT_HELLO, {"cookie": b"garbage"})
  # Should silently drop, not crash
  AWAIT server.handle_client_hello(("10.0.0.1", 9000), bad_hello)
  ASSERT ("10.0.0.1", 9000) NOT IN server._sessions

TEST test_3node_gossip_convergence:
  # Integration: 3 nodes, one proposes, all converge
  nodes = [GossipNode(f"node-{i}") for i in range(3)]
  connect_mesh(nodes)
  nodes[0].broadcast(b"hello")
  # Simulate rounds
  FOR round IN range(10):
    FOR node IN nodes:
      node.process_inbox()
  ASSERT all(node.has_seen(b"hello") for node in nodes)

TEST test_rust_gossip_has_seen:
  proto = GossipProtocol("node-0", fanout=2)
  ASSERT proto.has_seen(b"msg1") IS False
  proto.receive("node-1", b"msg1")
  ASSERT proto.has_seen(b"msg1") IS True
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Handshake completion | Full 10-step DTLS flow, no placeholders |
| DoS protection | Cookie-based, HMAC-SHA256, TTL-enforced |
| Rust federation bindings | Gossip + BFT consensus exposed to Python |
| Integration test | 3-node gossip convergence in < 10 rounds |
| Test count | +8 Python tests, +3 Rust tests |
