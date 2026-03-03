# Phase 47: Infrastructure Wiring

> Standing on Giants: Fielding (REST, 2000) · Fette & Melnikov (WebSocket RFC 6455, 2011) · Lamport (distributed reliability, 1978) · Berners-Lee (offline-first web, 2003) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

The filedfs frontend has infrastructure scaffolding but critical gaps prevent
production deployment. Four overlapping bridges, no unified client, no reconnection
strategy, incomplete PWA, and no SAP protocol integration at the transport layer.

| Gap | Current | Target |
|-----|---------|--------|
| WebSocket client | 2 bridge files (bizra-bridge.mjs, bridge.mjs) | Unified `BizraClient` with reconnect |
| Transport selection | Hardcoded per-file | Auto-detect: Tauri IPC → WebSocket → HTTP fallback |
| SAP protocol layer | Verbs defined in useNode.js but no transport | Full SAP v0 over BizraClient |
| Offline queue | Basic IndexedDB (queue.js, 1.2K) | Robust: retry backoff, conflict resolution, sync |
| Service worker | Basic cache-first (1.8K) | Background sync, push notifications stub, versioned cache |
| PWA install | manifest.json exists | Install prompt, update notification, A2HS |
| Connection health | No monitoring | Heartbeat, latency tracking, auto-reconnect with backoff |
| Error boundaries | None | React error boundaries per feature zone |

### Existing Infrastructure Files

| File | LOC | What It Does | Keep/Replace |
|------|-----|-------------|-------------|
| `bizra-bridge.mjs` | 14K | Node.js child_process → WS bridge | REPLACE → BizraClient |
| `bridge.mjs` | 12K | Simpler WS bridge + BizraClient emitter | REPLACE → BizraClient |
| `useNode.js` | 25K | React hook with 40+ verbs | REFACTOR → useNode uses BizraClient |
| `useBizraNode.js` | 6.8K | Lower-level transport abstraction | MERGE into BizraClient |
| `llm_bridge.js` | 15K | LLM proxy (Ollama/LM Studio) | KEEP → server-side only |
| `offline/queue.js` | 1.2K | IndexedDB queue | REPLACE → robust OfflineManager |
| `service-worker.js` | 1.8K | Basic PWA SW | REPLACE → versioned SW |

---

## 2. Architecture

```
src/
├── lib/
│   ├── client/
│   │   ├── BizraClient.js          # Unified client (Tauri | WS | HTTP)
│   │   ├── TauriTransport.js       # Tauri invoke adapter
│   │   ├── WebSocketTransport.js   # WebSocket adapter with reconnect
│   │   ├── HttpTransport.js        # HTTP fallback adapter
│   │   └── TransportSelector.js    # Auto-detect best transport
│   ├── offline/
│   │   ├── OfflineManager.js       # Queue + retry + sync
│   │   ├── ConflictResolver.js     # Last-write-wins or merge
│   │   └── SyncEngine.js           # Background sync orchestrator
│   ├── protocol/
│   │   ├── verbs.js                # Verb definitions (RECEIVE, TEACH, SAP_*, etc.)
│   │   ├── codec.js                # Encode/decode tab-delimited protocol
│   │   └── sap.js                  # SAP v0 session management
│   └── health/
│       ├── ConnectionMonitor.js    # Heartbeat, latency, state machine
│       └── ErrorBoundary.jsx       # React error boundary component
├── hooks/
│   └── useNode.js                  # REFACTORED — thin wrapper over BizraClient
├── service-worker.js               # REPLACED — versioned, background sync
└── manifest.json                   # UPDATED — screenshots, shortcuts
```

### Transport Selection Flow

```
┌─────────────────────────────────────┐
│         TransportSelector            │
│                                     │
│  1. Check window.__TAURI__          │
│     → YES → TauriTransport          │
│     → NO  ↓                         │
│  2. Check WebSocket availability    │
│     → ws://localhost:9800 reachable │
│     → YES → WebSocketTransport      │
│     → NO  ↓                         │
│  3. Fallback to HTTP                │
│     → HttpTransport (polling)       │
└─────────────────────────────────────┘
```

---

## 3. Pseudocode: BizraClient

```
CLASS BizraClient:
    PRIVATE transport: Transport
    PRIVATE offlineManager: OfflineManager
    PRIVATE connectionMonitor: ConnectionMonitor
    PRIVATE listeners: Map<string, Set<Function>>
    PRIVATE pendingRequests: Map<string, { resolve, reject, timeout }>

    CONSTRUCTOR(options):
        transport = TransportSelector.select(options)
        offlineManager = new OfflineManager()
        connectionMonitor = new ConnectionMonitor(transport)

        transport.onMessage = (raw) => handleMessage(raw)
        transport.onDisconnect = () => handleDisconnect()
        transport.onReconnect = () => handleReconnect()

    ASYNC connect():
        TRY:
            await transport.connect()
            connectionMonitor.start()
            # Flush offline queue
            await offlineManager.flush((verb, args) => this.send(verb, args))
        CATCH error:
            connectionMonitor.markDisconnected()
            EMIT 'connection:error', error

    ASYNC send(verb, args = {}):
        requestId = nanoid()
        encoded = codec.encode(verb, args)

        IF NOT transport.isConnected():
            # Queue for later
            await offlineManager.enqueue({ verb, args, requestId, timestamp: Date.now() })
            RETURN { ok: false, queued: true, requestId }

        TRY:
            raw = await transport.send(encoded, { timeout: 30_000 })
            response = codec.decode(raw)

            IF response.status == 'OK':
                RETURN { ok: true, ...response.fields }
            ELSE:
                RETURN { ok: false, error: response.message }

        CATCH TimeoutError:
            await offlineManager.enqueue({ verb, args, requestId, timestamp: Date.now() })
            RETURN { ok: false, queued: true, timeout: true }

    FUNCTION on(event, callback):
        IF NOT listeners.has(event): listeners.set(event, new Set())
        listeners.get(event).add(callback)
        RETURN () => listeners.get(event).delete(callback)  # unsubscribe

    PRIVATE handleMessage(raw):
        decoded = codec.decode(raw)
        # Emit to listeners
        EMIT decoded.verb, decoded.fields

    PRIVATE handleDisconnect():
        EMIT 'connection:lost'

    PRIVATE handleReconnect():
        EMIT 'connection:restored'
        offlineManager.flush((verb, args) => this.send(verb, args))

    ASYNC disconnect():
        connectionMonitor.stop()
        await transport.disconnect()
```

---

## 4. Pseudocode: WebSocket Transport

```
CLASS WebSocketTransport IMPLEMENTS Transport:
    PRIVATE ws: WebSocket | null
    PRIVATE url: string
    PRIVATE reconnectAttempts: number = 0
    PRIVATE maxReconnectAttempts: number = 10
    PRIVATE baseDelay: number = 1000  # ms

    CONSTRUCTOR(url):
        this.url = url

    ASYNC connect():
        RETURN new Promise((resolve, reject) => {
            ws = new WebSocket(url)

            ws.onopen = () => {
                reconnectAttempts = 0
                resolve()
            }

            ws.onclose = (event) => {
                IF NOT event.wasClean:
                    scheduleReconnect()
                onDisconnect?.()
            }

            ws.onerror = (error) => {
                reject(error)
            }

            ws.onmessage = (event) => {
                onMessage?.(event.data)
            }
        })

    ASYNC send(data, options):
        IF ws?.readyState != WebSocket.OPEN:
            THROW ConnectionError('WebSocket not connected')

        RETURN new Promise((resolve, reject) => {
            timeout = setTimeout(() => reject(TimeoutError()), options.timeout)

            # For request-response pattern, use correlation ID
            ws.send(data)

            # Listen for next message as response (simplified)
            oneTimeListener = (event) => {
                clearTimeout(timeout)
                ws.removeEventListener('message', oneTimeListener)
                resolve(event.data)
            }
            ws.addEventListener('message', oneTimeListener)
        })

    FUNCTION isConnected():
        RETURN ws?.readyState == WebSocket.OPEN

    PRIVATE scheduleReconnect():
        IF reconnectAttempts >= maxReconnectAttempts:
            EMIT 'reconnect:exhausted'
            RETURN

        # Exponential backoff with jitter
        delay = baseDelay * Math.pow(2, reconnectAttempts) + Math.random() * 1000
        reconnectAttempts += 1

        setTimeout(async () => {
            TRY:
                await connect()
                onReconnect?.()
            CATCH:
                scheduleReconnect()
        }, delay)

    ASYNC disconnect():
        IF ws:
            ws.close(1000, 'Client disconnect')
            ws = null
```

---

## 5. Pseudocode: Protocol Codec

```
MODULE codec:

    # Tab-delimited protocol:
    # Request:  VERB\targ1\targ2\t...
    # Response: OK\tkey1=val1\tkey2=val2\t...
    # Error:    ERR\tCODE\tMessage

    FUNCTION encode(verb, args):
        parts = [verb]
        FOR EACH [key, value] IN Object.entries(args):
            parts.push(String(value))
        RETURN parts.join('\t')

    FUNCTION decode(raw):
        parts = raw.split('\t')
        status = parts[0]

        IF status == 'OK':
            fields = {}
            FOR i FROM 1 TO parts.length:
                IF parts[i].includes('='):
                    [key, value] = parts[i].split('=', 2)
                    fields[key] = value
                ELSE:
                    fields[`arg${i}`] = parts[i]
            RETURN { status: 'OK', fields }

        ELSE IF status == 'ERR':
            RETURN { status: 'ERR', code: parts[1], message: parts[2] }

        ELSE:
            # Server-initiated message
            RETURN { verb: status, fields: parseFields(parts.slice(1)) }

    FUNCTION parseFields(parts):
        fields = {}
        FOR EACH part IN parts:
            IF part.includes('='):
                [key, value] = part.split('=', 2)
                fields[key] = value
        RETURN fields
```

---

## 6. Pseudocode: Offline Manager

```
CLASS OfflineManager:
    PRIVATE DB_NAME = 'bizra-offline'
    PRIVATE STORE_NAME = 'queue'
    PRIVATE maxRetries = 5
    PRIVATE retryBackoff = [1000, 5000, 15000, 60000, 300000]  # ms

    ASYNC enqueue(action):
        db = await openDB()
        await db.put(STORE_NAME, {
            ...action,
            id: action.requestId,
            retryCount: 0,
            createdAt: Date.now(),
            nextRetryAt: Date.now(),
        })

    ASYNC flush(sendFn):
        db = await openDB()
        actions = await db.getAll(STORE_NAME)
        actions.sort((a, b) => a.createdAt - b.createdAt)

        FOR EACH action IN actions:
            IF action.retryCount >= maxRetries:
                # Move to dead letter queue
                await db.delete(STORE_NAME, action.id)
                EMIT 'offline:dead-letter', action
                CONTINUE

            IF Date.now() < action.nextRetryAt:
                CONTINUE  # Not ready for retry

            TRY:
                result = await sendFn(action.verb, action.args)
                IF result.ok:
                    await db.delete(STORE_NAME, action.id)
                    EMIT 'offline:synced', action
                ELSE:
                    # Increment retry
                    action.retryCount += 1
                    action.nextRetryAt = Date.now() + retryBackoff[min(action.retryCount, 4)]
                    await db.put(STORE_NAME, action)
            CATCH:
                action.retryCount += 1
                action.nextRetryAt = Date.now() + retryBackoff[min(action.retryCount, 4)]
                await db.put(STORE_NAME, action)

    ASYNC count():
        db = await openDB()
        RETURN db.count(STORE_NAME)

    ASYNC clear():
        db = await openDB()
        await db.clear(STORE_NAME)
```

---

## 7. Pseudocode: Connection Monitor

```
CLASS ConnectionMonitor:
    PRIVATE transport: Transport
    PRIVATE intervalId: number | null
    PRIVATE pingInterval = 5000   # ms
    PRIVATE timeoutMs = 3000      # ms
    PRIVATE latencyHistory: number[] = []  # last 10 measurements
    PRIVATE state: 'connected' | 'degraded' | 'disconnected' = 'disconnected'

    CONSTRUCTOR(transport):
        this.transport = transport

    start():
        intervalId = setInterval(async () => {
            startTime = Date.now()
            TRY:
                await transport.send(codec.encode('PING'), { timeout: timeoutMs })
                latency = Date.now() - startTime
                latencyHistory.push(latency)
                IF latencyHistory.length > 10: latencyHistory.shift()

                avgLatency = average(latencyHistory)
                IF avgLatency > 2000:
                    state = 'degraded'
                ELSE:
                    state = 'connected'

            CATCH:
                state = 'disconnected'
                EMIT 'health:disconnected'

        }, pingInterval)

    stop():
        IF intervalId: clearInterval(intervalId)

    getState(): RETURN state

    getAverageLatency():
        IF latencyHistory.length == 0: RETURN null
        RETURN average(latencyHistory)

    markDisconnected():
        state = 'disconnected'
```

---

## 8. Pseudocode: Service Worker (versioned)

```
CONST CACHE_VERSION = 'bizra-v1'
CONST SHELL_ASSETS = [
    '/',
    '/index.html',
    '/src/tokens/index.css',
    '/manifest.json',
]
CONST API_PATTERNS = ['/v1/', '/api/', 'localhost:11434', '192.168.56.1:1234']

# Install: cache shell
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_VERSION).then((cache) => cache.addAll(SHELL_ASSETS))
    )
    self.skipWaiting()
})

# Activate: purge old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            RETURN Promise.all(
                keys.filter(k => k != CACHE_VERSION).map(k => caches.delete(k))
            )
        })
    )
    self.clients.claim()
})

# Fetch: strategy per request type
self.addEventListener('fetch', (event) => {
    url = new URL(event.request.url)

    IF API_PATTERNS.some(p => url.href.includes(p)):
        # Network-first for API calls
        event.respondWith(
            fetch(event.request)
                .catch(() => caches.match(event.request))
                .catch(() => new Response(JSON.stringify({ error: 'offline' }), {
                    headers: { 'Content-Type': 'application/json' }
                }))
        )
    ELSE:
        # Cache-first for static assets
        event.respondWith(
            caches.match(event.request)
                .then((cached) => cached || fetch(event.request).then((response) => {
                    cache = await caches.open(CACHE_VERSION)
                    cache.put(event.request, response.clone())
                    RETURN response
                }))
                .catch(() => caches.match('/index.html'))  # SPA fallback
        )
})

# Background sync: flush offline queue
self.addEventListener('sync', (event) => {
    IF event.tag == 'bizra-sync':
        event.waitUntil(flushOfflineQueue())
})

# Push notification stub
self.addEventListener('push', (event) => {
    data = event.data?.json() || { title: 'BIZRA Node0', body: 'New notification' }
    event.waitUntil(
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: '/public/icon-192.png',
            badge: '/public/icon-192.png',
        })
    )
})
```

---

## 9. Pseudocode: Error Boundary

```
CLASS ErrorBoundary EXTENDS React.Component:
    STATE hasError = false
    STATE error = null

    STATIC getDerivedStateFromError(error):
        RETURN { hasError: true, error }

    componentDidCatch(error, errorInfo):
        # Log to monitoring (future: Sentry, or local log)
        console.error('[BIZRA ErrorBoundary]', error, errorInfo)
        # Could emit to BizraClient for node-level logging

    RENDER:
        IF hasError:
            <div className="error-boundary">
                <SeedOfLife size={60} opacity={0.1} />
                <h3>Something went wrong</h3>
                <p className="error-message">{error?.message}</p>
                <Button onClick={() => this.setState({ hasError: false, error: null })}>
                    Try Again
                </Button>
                <Button variant="ghost" onClick={() => window.location.reload()}>
                    Reload App
                </Button>
            </div>
        ELSE:
            RETURN this.props.children

# Usage: wrap each feature zone
<ErrorBoundary>
    <ChatContainer />
</ErrorBoundary>
<ErrorBoundary>
    <Dashboard />
</ErrorBoundary>
```

---

## 10. Pseudocode: PWA Install Prompt

```
PROCEDURE usePWAInstall():
    STATE deferredPrompt = null
    STATE isInstalled = false

    ON_MOUNT:
        # Capture install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault()
            deferredPrompt = e
        })

        # Detect if already installed
        window.addEventListener('appinstalled', () => {
            isInstalled = true
            deferredPrompt = null
        })

        # Check standalone mode
        IF window.matchMedia('(display-mode: standalone)').matches:
            isInstalled = true

    FUNCTION promptInstall():
        IF deferredPrompt:
            deferredPrompt.prompt()
            result = await deferredPrompt.userChoice
            IF result.outcome == 'accepted':
                isInstalled = true
            deferredPrompt = null

    RETURN { canInstall: deferredPrompt != null, isInstalled, promptInstall }

PROCEDURE InstallBanner():
    { canInstall, promptInstall } = usePWAInstall()

    IF NOT canInstall: RETURN null

    RENDER:
        <div className="install-banner">
            <SeedOfLife size={24} />
            <span>Install BIZRA Node0 for the best experience</span>
            <Button size="sm" onClick={promptInstall}>Install</Button>
            <button className="dismiss" onClick={() => dismissed = true}>×</button>
        </div>
```

---

## 11. Pseudocode: Refactored useNode Hook

```
PROCEDURE useNode():
    # Thin wrapper over BizraClient — replaces 25K LOC useNode.js
    CONST client = useMemo(() => new BizraClient(), [])
    STATE connected = false
    STATE health = null
    STATE knowsMe = 0
    STATE queuedCount = 0

    ON_MOUNT:
        await client.connect()

        client.on('connection:restored', () => { connected = true })
        client.on('connection:lost', () => { connected = false })
        client.on('offline:synced', updateQueueCount)

        # Initial health check
        refreshHealth()

    FUNCTION receive(content):
        RETURN client.send('RECEIVE', { content, timestamp: Date.now() })

    FUNCTION teach(kind, content, confidence = 0.8):
        RETURN client.send('TEACH', { kind, content, confidence, timestamp: Date.now() })

    FUNCTION refreshHealth():
        result = await client.send('HEALTH')
        IF result.ok: health = result
        knows = await client.send('KNOWS_ME')
        IF knows.ok: knowsMe = parseFloat(knows.score)

    # SAP verbs
    FUNCTION sapOpen(profile, role):
        RETURN client.send('SAP_MEET_OPEN', { profile, initiator_role: role, timestamp: Date.now() })

    FUNCTION sapMessage(sessionId, content):
        RETURN client.send('SAP_MESSAGE', { session_id: sessionId, content, timestamp: Date.now() })

    FUNCTION sapClose(sessionId):
        RETURN client.send('SAP_SESSION_CLOSE', { session_id: sessionId })

    ON_UNMOUNT:
        client.disconnect()

    RETURN {
        connected, health, knowsMe, queuedCount,
        receive, teach, refreshHealth,
        sapOpen, sapMessage, sapClose,
        client,  # escape hatch for advanced verbs
    }
```

---

## 12. Migration Plan

```
PROCEDURE migrate_infrastructure():
    # Step 1: Create BizraClient + transports (new files, no breaking changes)
    CREATE lib/client/BizraClient.js
    CREATE lib/client/TauriTransport.js
    CREATE lib/client/WebSocketTransport.js
    CREATE lib/client/HttpTransport.js
    CREATE lib/client/TransportSelector.js

    # Step 2: Create protocol codec (extract from useNode.js)
    CREATE lib/protocol/verbs.js
    CREATE lib/protocol/codec.js
    CREATE lib/protocol/sap.js

    # Step 3: Create OfflineManager (replace queue.js)
    CREATE lib/offline/OfflineManager.js
    CREATE lib/offline/SyncEngine.js

    # Step 4: Refactor useNode.js → thin wrapper
    BACKUP useNode.js → useNode.legacy.js
    REWRITE useNode.js to use BizraClient

    # Step 5: Update all components to use refactored useNode
    # (API is compatible — receive(), teach(), etc. same signatures)
    # No component changes needed if hook API preserved

    # Step 6: Replace service-worker.js
    REPLACE service-worker.js with versioned implementation
    UPDATE index.html SW registration

    # Step 7: Remove old bridges
    DELETE bizra-bridge.mjs (replaced by BizraClient)
    DELETE bridge.mjs (replaced by BizraClient)
    DELETE useBizraNode.js (merged into BizraClient)
    DELETE offline/queue.js (replaced by OfflineManager)

    # Step 8: Verify
    RUN all tests
    CHECK WebSocket connection in dev mode
    CHECK Tauri build
    CHECK offline queue → reconnect → flush
    CHECK PWA install prompt
```

---

## 13. TDD Anchors

```
TEST_SUITE infrastructure_wiring:

    # --- BizraClient ---
    TEST "BizraClient auto-selects transport":
        # In browser without Tauri
        client = new BizraClient()
        ASSERT client.transport INSTANCEOF WebSocketTransport

    TEST "BizraClient queues when disconnected":
        client = new BizraClient()
        # Don't connect
        result = await client.send('RECEIVE', { content: 'hello' })
        ASSERT result.queued == true
        ASSERT offlineManager.count() == 1

    TEST "BizraClient flushes queue on reconnect":
        client = new BizraClient()
        client.send('RECEIVE', { content: 'hello' })  # queued
        await client.connect()
        ASSERT offlineManager.count() == 0

    # --- Codec ---
    TEST "encode produces tab-delimited string":
        result = codec.encode('RECEIVE', { content: 'hello', timestamp: 123 })
        ASSERT result == 'RECEIVE\thello\t123'

    TEST "decode parses OK response":
        result = codec.decode('OK\tcontent=world\tconfidence=0.9')
        ASSERT result.status == 'OK'
        ASSERT result.fields.content == 'world'
        ASSERT result.fields.confidence == '0.9'

    TEST "decode parses ERR response":
        result = codec.decode('ERR\tBAD_COMMAND\tUnknown verb')
        ASSERT result.status == 'ERR'
        ASSERT result.code == 'BAD_COMMAND'

    # --- WebSocket Transport ---
    TEST "reconnects with exponential backoff":
        mock WebSocket → close immediately
        transport = new WebSocketTransport('ws://localhost:9800')
        await transport.connect() → fails
        ASSERT reconnectAttempts == 1
        advance timer 1000ms
        ASSERT reconnect attempted
        advance timer 2000ms
        ASSERT reconnect attempted again

    TEST "stops reconnecting after max attempts":
        mock WebSocket → always fail
        transport = new WebSocketTransport('ws://localhost:9800')
        FOR 10 times: advance timer, fail
        ASSERT reconnectAttempts == 10
        ASSERT 'reconnect:exhausted' event emitted

    # --- Offline Manager ---
    TEST "enqueue stores in IndexedDB":
        manager = new OfflineManager()
        await manager.enqueue({ verb: 'RECEIVE', args: { content: 'hello' }, requestId: 'abc' })
        ASSERT await manager.count() == 1

    TEST "flush sends and removes on success":
        manager = new OfflineManager()
        await manager.enqueue(action)
        await manager.flush(mockSendFn)  # mockSendFn returns { ok: true }
        ASSERT await manager.count() == 0

    TEST "flush retries with backoff on failure":
        manager = new OfflineManager()
        await manager.enqueue(action)
        await manager.flush(failingSendFn)
        item = await getFromDB()
        ASSERT item.retryCount == 1
        ASSERT item.nextRetryAt > Date.now()

    TEST "dead letter after max retries":
        seed action with retryCount = 5
        await manager.flush(sendFn)
        ASSERT 'offline:dead-letter' emitted
        ASSERT await manager.count() == 0

    # --- Service Worker ---
    TEST "caches shell assets on install":
        trigger install event
        ASSERT cache contains '/', '/index.html', '/manifest.json'

    TEST "purges old caches on activate":
        create cache 'bizra-v0'
        trigger activate event
        ASSERT cache 'bizra-v0' deleted
        ASSERT cache 'bizra-v1' exists

    TEST "API requests use network-first":
        trigger fetch '/v1/health'
        ASSERT network request made first

    TEST "static assets use cache-first":
        seed cache with '/index.html'
        trigger fetch '/index.html'
        ASSERT served from cache

    # --- Connection Monitor ---
    TEST "heartbeat detects latency":
        monitor = new ConnectionMonitor(mockTransport)
        monitor.start()
        mockTransport.respond(PING) after 100ms
        advance timer 5000ms
        ASSERT monitor.getAverageLatency() ≈ 100

    TEST "heartbeat detects disconnect":
        monitor = new ConnectionMonitor(mockTransport)
        monitor.start()
        mockTransport.timeout(PING)
        advance timer 5000ms
        ASSERT monitor.getState() == 'disconnected'

    # --- Error Boundary ---
    TEST "catches child errors":
        render <ErrorBoundary><ThrowingComponent /></ErrorBoundary>
        ASSERT query('.error-boundary') EXISTS
        ASSERT query('h3').text == 'Something went wrong'

    TEST "retry resets error state":
        render <ErrorBoundary><ThrowingComponent /></ErrorBoundary>
        click 'Try Again'
        ASSERT hasError == false

    # --- PWA Install ---
    TEST "captures install prompt":
        emit 'beforeinstallprompt' event
        ASSERT canInstall == true

    TEST "install banner shown when available":
        emit 'beforeinstallprompt'
        render <InstallBanner />
        ASSERT banner visible with 'Install' button

    # --- Refactored useNode ---
    TEST "useNode().receive calls BizraClient":
        mock BizraClient
        { receive } = useNode()
        await receive('hello')
        ASSERT BizraClient.send CALLED_WITH('RECEIVE', { content: 'hello', ... })

    TEST "useNode().teach fires TEACH verb":
        { teach } = useNode()
        await teach('expertise', 'I am a developer')
        ASSERT BizraClient.send CALLED_WITH('TEACH', { kind: 'expertise', ... })

    TEST "useNode().sapOpen fires SAP_MEET_OPEN":
        { sapOpen } = useNode()
        await sapOpen(profile, 'peer')
        ASSERT BizraClient.send CALLED_WITH('SAP_MEET_OPEN', ...)
```

---

## 14. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | BizraClient auto-detects transport | Test in Tauri + browser |
| 2 | WebSocket reconnects with backoff | Kill WS server → observe reconnect |
| 3 | Offline queue persists across refresh | Disconnect → send → refresh → check IndexedDB |
| 4 | Queue flushes on reconnect | Reconnect → verify messages delivered |
| 5 | Dead letter after 5 retries | Seed failed action → verify removal |
| 6 | Service worker caches shell assets | Lighthouse PWA audit |
| 7 | Old cache purged on update | Deploy new version → verify cleanup |
| 8 | PWA install prompt works | Chrome → check install banner |
| 9 | Error boundaries catch without crashing app | Trigger error in one zone → others still work |
| 10 | useNode API backward compatible | Existing components work without changes |
| 11 | Old bridge files removed | `ls bizra-bridge.mjs bridge.mjs` → not found |
| 12 | Heartbeat monitors latency | Check ConnectionMonitor reports in DevTools |

---

## 15. Dependency Chain

```
Phase 42 (Brand Tokens)        ← Error boundary uses tokens
Phase 43 (Onboarding)          ← Onboarding uses refactored useNode
Phase 44 (Chat)                ← Chat uses BizraClient for RECEIVE
Phase 45 (Dashboard)           ← Dashboard uses HEALTH polling
Phase 46 (Node/Community)      ← Node controls + SAP use BizraClient
Phase 47 (This phase)          ← Foundational — all above depend on this

NOTE: Phase 47 should ideally be built FIRST or in parallel with Phase 42,
since all feature phases depend on BizraClient. However, the current useNode.js
is functional enough for Phase 42-43 to proceed independently.

Recommended build order:
  42 (tokens) + 47 (infra) in parallel
  → 43 (onboarding) + 44 (chat)
  → 45 (dashboard) + 46 (node/community/legacy)
```
