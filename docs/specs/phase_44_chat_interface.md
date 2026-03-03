# Phase 44: Chat Interface

> Standing on Giants: Raskin (humane interface, 2000) · Abramov (React hooks, 2018) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

BIZRA Node0 has no dedicated conversational UI. Chat exists only as a
mini-component in onboarding Step 4 (`FirstChatStep.jsx`, 448 LOC). Users need a
full-screen, production-grade chat interface that serves as the primary interaction
surface for their sovereign node — combining conversational AI with transparency
about which agents contributed, what was learned, and current sovereignty score.

| Dimension | Current | Target |
|-----------|---------|--------|
| Chat UI | Embedded in onboarding only | Dedicated `/chat` route |
| Message rendering | Plain text bubbles | Rich bubbles: markdown, code blocks, citations |
| Agent attribution | Hidden | Inline badges: which PAT agents consulted |
| Guardian veto | Hidden | Visual indicator when response was vetoed/modified |
| KnowsMe integration | Mini gauge only | Persistent sidebar gauge + trait growth indicators |
| Input modalities | Text only | Text + voice intent (future) + file attach stub |
| History | In-memory, lost on refresh | Persisted via IndexedDB + optional backend sync |
| Streaming | Not supported | Token-by-token streaming via WebSocket |
| Context window | Not visible | Token budget indicator |

---

## 2. Architecture

```
src/
├── pages/
│   └── ChatPage.jsx             # Full-screen chat layout
├── features/
│   └── chat/
│       ├── ChatContainer.jsx    # State management, message orchestration
│       ├── MessageList.jsx      # Virtualized message list
│       ├── MessageBubble.jsx    # Individual message rendering
│       ├── InputBar.jsx         # Text input + actions
│       ├── AgentBadges.jsx      # Agent attribution chips
│       ├── VetoIndicator.jsx    # Guardian veto visual
│       ├── ChatSidebar.jsx      # KnowsMe gauge + session info
│       ├── StreamingText.jsx    # Token-by-token text reveal
│       └── TypingIndicator.jsx  # Agent thinking animation
├── hooks/
│   ├── useChat.js               # Chat state + message send/receive
│   └── useChatHistory.js        # IndexedDB persistence
└── utils/
    └── markdown.js              # Lightweight markdown → React nodes
```

### Data Flow

```
User types → InputBar → useChat.send()
                          ↓
                   BizraClient.send('RECEIVE', { content, timestamp })
                          ↓
                   [WebSocket/Tauri IPC]
                          ↓
                   Node0 backend processes
                          ↓
                   Response: { content, confidence, agents_consulted,
                               guardian_approved, knows_me, fragments_extracted }
                          ↓
                   MessageList renders new bubble
                          ↓
                   ChatSidebar updates KnowsMe gauge
```

---

## 3. Message Schema

```
DEFINE Message:
    id:         string          # nanoid() — unique per message
    role:       'user' | 'assistant' | 'system'
    content:    string          # plain text or markdown
    timestamp:  number          # Unix ms
    metadata:   {
        confidence:        number | null      # 0.0–1.0
        agents_consulted:  number | null      # count of PAT agents
        agent_names:       string[] | null    # ['Scribe', 'Guardian', ...]
        guardian_approved: boolean | null
        guardian_veto:     boolean | null      # true if response was modified
        veto_reason:       string | null
        fragments_extracted: number | null    # knowledge fragments learned
        knows_me:          number | null      # updated KnowsMe score
        tokens_used:       number | null      # token budget consumed
    }

DEFINE ChatSession:
    id:         string          # session identifier
    startedAt:  number
    messages:   Message[]
    knowsMe:    number          # latest KnowsMe score
    totalTokens: number         # cumulative token usage
```

---

## 4. Pseudocode: Chat Container

```
PROCEDURE ChatContainer():
    STATE session = useChatSession()
    STATE messages = useChatHistory(session.id)
    STATE isStreaming = false
    STATE streamBuffer = ''
    CONST bizraClient = useBizraClient()

    FUNCTION sendMessage(text):
        # Create user message
        userMsg = Message(role='user', content=text, timestamp=Date.now())
        messages.append(userMsg)
        messages.save()  # IndexedDB persist

        # Create placeholder assistant message
        assistantId = nanoid()
        placeholder = Message(id=assistantId, role='assistant', content='', timestamp=Date.now())
        messages.append(placeholder)
        isStreaming = true

        # Send to backend
        TRY:
            response = await bizraClient.send('RECEIVE', {
                content: text,
                timestamp: userMsg.timestamp
            })

            # Update placeholder with real response
            placeholder.content = response.content
            placeholder.metadata = {
                confidence: parseFloat(response.confidence),
                agents_consulted: parseInt(response.agents_consulted),
                guardian_approved: response.guardian_approved == 'true',
                fragments_extracted: parseInt(response.fragments_extracted),
                knows_me: parseFloat(response.knows_me),
            }

            # Update sidebar gauge
            session.knowsMe = placeholder.metadata.knows_me
            session.totalTokens += placeholder.metadata.tokens_used || 0

        CATCH error:
            placeholder.content = "I couldn't process that. Your message has been queued for retry."
            placeholder.metadata = { guardian_veto: false }
            # Queue for offline retry
            await offlineQueue.enqueue({ verb: 'RECEIVE', content: text, timestamp: userMsg.timestamp })

        FINALLY:
            isStreaming = false
            messages.save()

    RENDER:
        <div className="chat-page">
            <ChatSidebar
                knowsMe={session.knowsMe}
                totalTokens={session.totalTokens}
                messageCount={messages.length}
            />
            <div className="chat-main">
                <MessageList
                    messages={messages}
                    isStreaming={isStreaming}
                    streamBuffer={streamBuffer}
                />
                <InputBar onSend={sendMessage} disabled={isStreaming} />
            </div>
        </div>
```

---

## 5. Pseudocode: Message Bubble

```
PROCEDURE MessageBubble({ message }):
    isUser = message.role == 'user'
    isSystem = message.role == 'system'
    meta = message.metadata

    RENDER:
        <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}
             data-confidence={meta?.confidence}>

            # Content with markdown rendering
            <div className="bubble-content">
                IF isUser:
                    <span>{message.content}</span>
                ELSE:
                    <MarkdownRenderer content={message.content} />
            </div>

            # Agent attribution (assistant only)
            IF NOT isUser AND meta?.agents_consulted > 0:
                <AgentBadges
                    count={meta.agents_consulted}
                    names={meta.agent_names}
                />

            # Guardian indicator
            IF meta?.guardian_veto:
                <VetoIndicator reason={meta.veto_reason} />
            ELSE IF meta?.guardian_approved:
                <span className="guardian-ok" title="Guardian approved">✓</span>

            # Confidence meter (assistant only)
            IF NOT isUser AND meta?.confidence != null:
                <ConfidenceMeter value={meta.confidence} />

            # Fragments learned indicator
            IF meta?.fragments_extracted > 0:
                <span className="fragments-badge">
                    +{meta.fragments_extracted} learned
                </span>

            # Timestamp
            <time className="bubble-time">{formatTime(message.timestamp)}</time>
        </div>
```

---

## 6. Pseudocode: Input Bar

```
PROCEDURE InputBar({ onSend, disabled }):
    STATE text = ''
    REF inputRef = useRef()
    REF textareaRef = useRef()

    FUNCTION handleSend():
        IF text.trim() == '' OR disabled: RETURN
        onSend(text.trim())
        text = ''
        textareaRef.current.style.height = 'auto'

    FUNCTION handleKeyDown(e):
        IF e.key == 'Enter' AND NOT e.shiftKey:
            e.preventDefault()
            handleSend()

    FUNCTION autoResize():
        # Grow textarea to fit content, max 6 lines
        el = textareaRef.current
        el.style.height = 'auto'
        el.style.height = min(el.scrollHeight, 6 * 24) + 'px'

    RENDER:
        <div className="input-bar">
            <textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => { text = e.target.value; autoResize(); }}
                onKeyDown={handleKeyDown}
                placeholder="Message your node..."
                disabled={disabled}
                rows={1}
                aria-label="Chat message input"
            />
            <button
                onClick={handleSend}
                disabled={disabled OR text.trim() == ''}
                aria-label="Send message"
            >
                <SendIcon />
            </button>
        </div>
```

---

## 7. Pseudocode: Chat Sidebar

```
PROCEDURE ChatSidebar({ knowsMe, totalTokens, messageCount }):
    RENDER:
        <aside className="chat-sidebar">
            # KnowsMe gauge (large, 8-segment)
            <KnowsMeGauge score={knowsMe} size={160} />

            # Session stats
            <div className="sidebar-stats">
                <StatRow label="Messages" value={messageCount} />
                <StatRow label="Tokens Used" value={formatNumber(totalTokens)} />
                <StatRow label="Confidence" value={`${(avgConfidence * 100).toFixed(0)}%`} />
            </div>

            # Sovereignty tier
            <TierBadge tier={computeTier(knowsMe)} />

            # Quick actions
            <div className="sidebar-actions">
                <ActionButton icon="📝" label="Teach" onClick={() => navigate('/teach')} />
                <ActionButton icon="📊" label="Dashboard" onClick={() => navigate('/dashboard')} />
                <ActionButton icon="🔒" label="Privacy" onClick={() => navigate('/settings')} />
            </div>
        </aside>
```

---

## 8. Pseudocode: Chat History Persistence

```
PROCEDURE useChatHistory(sessionId):
    CONST DB_NAME = 'bizra-chat'
    CONST STORE_NAME = 'messages'

    STATE messages = []

    ON_MOUNT:
        db = await openDB(DB_NAME, 1, {
            upgrade(db) {
                store = db.createObjectStore(STORE_NAME, { keyPath: 'id' })
                store.createIndex('session', 'sessionId')
                store.createIndex('timestamp', 'timestamp')
            }
        })
        # Load messages for this session
        messages = await db.getAllFromIndex(STORE_NAME, 'session', sessionId)
        messages.sort((a, b) => a.timestamp - b.timestamp)

    FUNCTION append(message):
        msg = { ...message, sessionId }
        messages.push(msg)
        await db.put(STORE_NAME, msg)

    FUNCTION save():
        # Batch write all messages
        tx = db.transaction(STORE_NAME, 'readwrite')
        FOR EACH msg IN messages:
            tx.store.put({ ...msg, sessionId })
        await tx.done

    FUNCTION clear():
        tx = db.transaction(STORE_NAME, 'readwrite')
        index = tx.store.index('session')
        cursor = await index.openCursor(sessionId)
        WHILE cursor:
            cursor.delete()
            cursor = await cursor.continue()
        messages = []

    RETURN { messages, append, save, clear }
```

---

## 9. Pseudocode: Lightweight Markdown Renderer

```
PROCEDURE MarkdownRenderer({ content }):
    # Minimal markdown → React nodes (no heavy deps)
    FUNCTION parse(text):
        lines = text.split('\n')
        result = []
        inCodeBlock = false
        codeBuffer = ''

        FOR EACH line IN lines:
            IF line.startsWith('```'):
                IF inCodeBlock:
                    result.push(<CodeBlock content={codeBuffer} />)
                    codeBuffer = ''
                inCodeBlock = NOT inCodeBlock
                CONTINUE

            IF inCodeBlock:
                codeBuffer += line + '\n'
                CONTINUE

            # Inline formatting
            formatted = line
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')

            IF line.startsWith('- '):
                result.push(<li>{formatted.slice(2)}</li>)
            ELSE IF line == '':
                result.push(<br />)
            ELSE:
                result.push(<p dangerouslySetInnerHTML={formatted} />)

        RETURN result

    RENDER:
        <div className="markdown-body">
            {parse(content)}
        </div>
```

---

## 10. TDD Anchors

```
TEST_SUITE chat_interface:

    TEST "sending message fires RECEIVE verb":
        mock bizraClient
        render <ChatContainer />
        type "Hello" → send
        ASSERT bizraClient.send CALLED_WITH('RECEIVE', { content: 'Hello', ... })

    TEST "assistant response renders with metadata":
        mock response { content: 'Hi', confidence: '0.92', agents_consulted: '3', guardian_approved: 'true' }
        render <ChatContainer />
        send "Hello" → wait
        bubble = query('[data-role="assistant"]')
        ASSERT bubble.text CONTAINS 'Hi'
        ASSERT query('.agent-badges').text CONTAINS '3 agents'
        ASSERT query('.guardian-ok') EXISTS

    TEST "guardian veto shows indicator":
        mock response { guardian_veto: true, veto_reason: 'low confidence' }
        render <MessageBubble message={vetoedMsg} />
        ASSERT query('.veto-indicator') EXISTS
        ASSERT query('.veto-indicator').title CONTAINS 'low confidence'

    TEST "messages persist to IndexedDB":
        render <ChatContainer />
        send "Hello" → wait
        messages = await getFromDB('bizra-chat', 'messages')
        ASSERT messages.length >= 2  # user + assistant

    TEST "messages restore on remount":
        seed IndexedDB with 5 messages
        render <ChatContainer />
        bubbles = queryAll('.bubble')
        ASSERT bubbles.length == 5

    TEST "sidebar KnowsMe updates after response":
        mock response { knows_me: '0.55' }
        render <ChatContainer />
        send message → wait
        ASSERT sidebar gauge value == 0.55

    TEST "Enter sends, Shift+Enter adds newline":
        render <InputBar />
        focus input → type "line1" → press Enter
        ASSERT onSend CALLED_WITH 'line1'
        type "line1" → press Shift+Enter → type "line2" → press Enter
        ASSERT onSend CALLED_WITH 'line1\nline2'

    TEST "input disabled during streaming":
        render <InputBar disabled={true} />
        ASSERT textarea.disabled == true
        ASSERT sendButton.disabled == true

    TEST "offline fallback queues message":
        mock bizraClient.send → throws NetworkError
        render <ChatContainer />
        send "Hello" → wait
        ASSERT offlineQueue has 1 item
        ASSERT assistant bubble CONTAINS 'queued for retry'

    TEST "markdown renders code blocks":
        render <MessageBubble message={{ content: '```\nconst x = 1\n```' }} />
        ASSERT query('code') EXISTS
        ASSERT query('code').text == 'const x = 1'

    TEST "chat sidebar shows correct stats":
        render <ChatSidebar knowsMe={0.6} totalTokens={1500} messageCount={10} />
        ASSERT gauge value == 0.6
        ASSERT text CONTAINS '1,500'
        ASSERT text CONTAINS '10'
```

---

## 11. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/chat` route renders full-screen chat | Navigate to /chat, visual check |
| 2 | RECEIVE verb fires on every user message | Network/WebSocket inspector |
| 3 | Agent attribution badges visible on responses | Visual + test |
| 4 | Guardian veto indicator shown when applicable | Mock veto response, visual |
| 5 | KnowsMe gauge updates live in sidebar | Send 3 messages, observe score change |
| 6 | Messages persist across page refresh | Refresh → messages still there |
| 7 | Shift+Enter creates newline, Enter sends | Keyboard test |
| 8 | Offline messages queued in IndexedDB | Disconnect network → send → check queue |
| 9 | Markdown (bold, italic, code blocks) renders | Send markdown content |
| 10 | Each component < 300 LOC | `wc -l` on all files |
