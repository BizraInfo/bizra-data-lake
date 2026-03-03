# Phase 56.01: Critical — Transport Auth + WebSocket Bridges

> Standing on Giants: Perrin (Noise Protocol, 2018) · Rescorla (DTLS 1.3, RFC 9147) · Bernstein (Ed25519) · Saltzer & Schroeder (complete mediation, 1975)

## F1: Transport MITM — No Peer Identity Binding

### Current State

File: `core/federation/secure_transport.py:1097-1103`

```
peer_static_public=client_e_public,  # Using ephemeral as static for simplicity
```

The DTLS/Noise handshake generates ephemeral X25519 keypairs and derives session keys,
but never proves the peer owns a long-term Ed25519 identity. The `peer_static_public`
field stores the ephemeral key — any active network attacker can MITM.

The module docstring (line 16) claims "Identity binding via Ed25519 static keys"
but this is aspirational, not implemented.

### Required Behavior

After ephemeral key exchange completes, each side MUST:
1. Sign its ephemeral public key with its long-term Ed25519 signing key
2. Send the signature + Ed25519 public key to the peer
3. Verify the peer's signature before completing the session
4. Store the verified Ed25519 public key as `peer_static_public`

### Pseudocode

```
FUNCTION complete_handshake_with_identity(
    shared_secret, client_random, server_random,
    local_identity: Ed25519PrivateKey,
    peer_claimed_pubkey: Ed25519PublicKey  # from gossip/config
):
    # Existing: derive session keys from DH
    client_key, server_key = derive_keys(shared_secret, client_random, server_random)

    # NEW: identity proof
    my_ephemeral_pub = get_local_ephemeral_public()
    proof = local_identity.sign(
        b"BIZRA-TRANSPORT-BIND-v1" + my_ephemeral_pub + client_random + server_random
    )

    # Send: [existing_hello | ed25519_pubkey(32) | signature(64)]
    identity_payload = local_identity.public_key_bytes() + proof

    # On receive: verify peer's identity proof
    FUNCTION verify_peer_identity(peer_identity_payload, peer_ephemeral_pub):
        peer_pubkey = peer_identity_payload[:32]
        peer_sig = peer_identity_payload[32:96]

        IF peer_claimed_pubkey IS NOT None AND peer_pubkey != peer_claimed_pubkey:
            RAISE HandshakeError("peer identity mismatch")

        message = b"BIZRA-TRANSPORT-BIND-v1" + peer_ephemeral_pub + client_random + server_random
        IF NOT ed25519_verify(peer_pubkey, message, peer_sig):
            RAISE HandshakeError("peer identity signature invalid")

        RETURN peer_pubkey  # verified

    session.peer_static_public = verify_peer_identity(...)
    RETURN session
```

### Files Modified

| File | Change |
|------|--------|
| `core/federation/secure_transport.py` | Add identity proof to `process_client_hello` and `process_handshake_response` |
| `core/federation/secure_transport.py` | Update `SecureSession` dataclass — `peer_static_public` now stores verified Ed25519 key |

### TDD Anchors

```python
# tests/core/federation/test_secure_transport.py

def test_handshake_binds_to_identity():
    """Session peer_static_public matches the verifying Ed25519 key, not the ephemeral."""
    initiator = DTLSTransport(identity=alice_id)
    responder = DTLSTransport(identity=bob_id)
    session = complete_handshake(initiator, responder)
    assert session.peer_static_public == bob_id.public_key_bytes()

def test_handshake_rejects_wrong_identity():
    """MITM with different identity key is rejected."""
    initiator = DTLSTransport(identity=alice_id)
    responder = DTLSTransport(identity=mallory_id)
    with pytest.raises(HandshakeError, match="identity"):
        complete_handshake(initiator, responder, expected_peer=bob_id)

def test_handshake_rejects_forged_signature():
    """Tampered signature over ephemeral key is rejected."""
    # Intercept and modify signature bytes before verify
    with pytest.raises(HandshakeError, match="signature invalid"):
        ...

def test_backward_compat_no_identity_mode():
    """When identity=None, handshake still works (dev/test mode) with warning."""
    ...
```

### Migration

Existing peers without identity keys will fail handshake. Add a config flag
`require_identity_binding: bool = True` with a deprecation warning when `False`.
Default to `True` in production, `False` only in test fixtures.

---

## F2: Unauthenticated WebSocket Bridge (bizra-bridge.mjs)

### Current State

File: `filedfs/bizra-bridge.mjs:362-419`

```javascript
const wss = new WebSocketServer({ server: httpServer });
wss.on("connection", (ws, req) => {
    // No origin check, no token — any client sends commands to bizra-node
    const response = await node.send(protocolLine);
    // SHUTDOWN verb = kill the node
});
httpServer.listen(config.port);  // binds all interfaces
```

Any process on the machine (or any website via cross-origin WS to localhost) can
connect, send arbitrary protocol commands including SHUTDOWN, and receive full
responses from the node.

### Required Behavior

1. Bind to `127.0.0.1` explicitly (not all interfaces)
2. Validate `Origin` header — reject non-localhost origins
3. Require bearer token from `BIZRA_BRIDGE_TOKEN` env var on WS upgrade
4. SHUTDOWN command requires a separate confirmation token

### Pseudocode

```
CONST ALLOWED_ORIGINS = [
    "http://localhost", "https://localhost",
    "http://127.0.0.1", "https://127.0.0.1",
    null  // null Origin = same-machine CLI tools
]

FUNCTION validate_upgrade(req):
    origin = req.headers.origin OR null
    IF origin IS NOT null AND NOT any(origin.startsWith(allowed) FOR allowed IN ALLOWED_ORIGINS):
        RETURN { allowed: false, reason: "origin_rejected" }

    expected_token = process.env.BIZRA_BRIDGE_TOKEN
    IF expected_token:
        # Check Authorization header or ?token= query param
        auth = req.headers.authorization OR url.searchParams.get("token")
        IF auth != "Bearer " + expected_token AND auth != expected_token:
            RETURN { allowed: false, reason: "token_invalid" }

    RETURN { allowed: true }

# On server creation:
httpServer.listen(config.port, "127.0.0.1")   # <-- explicit localhost bind

# On connection:
wss.on("connection", (ws, req) => {
    validation = validate_upgrade(req)
    IF NOT validation.allowed:
        ws.close(4001, validation.reason)
        RETURN
    ...existing handler...
})
```

### Files Modified

| File | Change |
|------|--------|
| `filedfs/bizra-bridge.mjs` | Add `validate_upgrade()`, bind to `127.0.0.1`, origin/token checks |

### TDD Anchors

```javascript
// tests/filedfs/test_bridge_auth.mjs (new)

test("rejects connections from non-localhost origins", async () => {
    const ws = new WebSocket(bridgeUrl, { headers: { Origin: "https://evil.com" } });
    await expect(ws).toClose(4001);
});

test("accepts connections with valid BIZRA_BRIDGE_TOKEN", async () => {
    process.env.BIZRA_BRIDGE_TOKEN = "test-secret-123";
    const ws = new WebSocket(bridgeUrl, { headers: { Authorization: "Bearer test-secret-123" } });
    const msg = await firstMessage(ws);
    expect(msg.event).toBe("connected");
});

test("rejects connections with invalid token", async () => {
    process.env.BIZRA_BRIDGE_TOKEN = "correct-token";
    const ws = new WebSocket(bridgeUrl, { headers: { Authorization: "Bearer wrong" } });
    await expect(ws).toClose(4001);
});

test("binds to 127.0.0.1 not 0.0.0.0", async () => {
    // Verify httpServer.address().address === "127.0.0.1"
});
```

---

## F12: Localhost Bridge Drive-By WS + Protocol Injection (bridge.mjs)

### Current State

File: `filedfs/bridge.mjs:195-224`

```javascript
this.wss = new WebSocketServer({ port: this.config.port, host: this.config.host });
// ...
ws.on("message", (data) => {
    try {
        const msg = JSON.parse(data.toString());
        this.handleClientMessage(ws, msg);
    } catch {
        // Raw command string — sent directly to node stdin
        const cmd = data.toString().trim();
        if (cmd) this.node.send(cmd);
    }
});
```

The catch block sends **raw unparsed strings** directly to the node's stdin.
Tab/newline injection can frame multiple protocol commands in one message.
Additionally, `this.config.host` may not default to localhost.

File: `filedfs/useBizraNode.js:176` — client-side helper does not sanitize
outbound messages.

### Required Behavior

1. Default `host` config to `127.0.0.1`
2. Remove raw-string fallback in catch block — all commands must be valid JSON
3. Add origin validation (same pattern as F2)
4. Sanitize protocol lines: strip `\t`, `\n`, `\r` from any interpolated values

### Pseudocode

```
FUNCTION sanitize_protocol_value(value: string): string:
    RETURN value.replace(/[\t\n\r]/g, "")

# In message handler — REMOVE raw string fallback:
ws.on("message", (data) => {
    LET msg
    TRY:
        msg = JSON.parse(data.toString())
    CATCH:
        ws.send(JSON.stringify({ ok: false, code: "PARSE_ERROR", message: "JSON required" }))
        RETURN

    # Sanitize all string fields before building protocol line
    FOR key IN msg:
        IF typeof msg[key] === "string":
            msg[key] = sanitize_protocol_value(msg[key])

    this.handleClientMessage(ws, msg)
})
```

### Files Modified

| File | Change |
|------|--------|
| `filedfs/bridge.mjs` | Default host to `127.0.0.1`, remove raw-string fallback, add sanitization |
| `filedfs/useBizraNode.js` | Add client-side sanitization of outbound messages |

### TDD Anchors

```javascript
test("rejects non-JSON messages with PARSE_ERROR", async () => {
    ws.send("raw string command");
    const response = await nextMessage(ws);
    expect(response.code).toBe("PARSE_ERROR");
});

test("strips tab/newline injection from protocol fields", async () => {
    ws.send(JSON.stringify({ verb: "QUERY", key: "x\ty\nz" }));
    // Verify the protocol line sent to node has no \t or \n in the key
});

test("default host is 127.0.0.1", () => {
    const bridge = new BizraBridge();
    expect(bridge.config.host).toBe("127.0.0.1");
});
```
