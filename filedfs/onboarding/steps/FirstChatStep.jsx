// ============================================================
// FirstChatStep — First conversation with your node
// ============================================================
// Simple chat interface: greeting, text input, RECEIVE command.
// Shows response bubble with metadata (agents, fragments).
// KnowsMe mini-gauge updates in real-time.
// "Continue to Dashboard" after first exchange.
// ============================================================

import { useState, useCallback, useRef, useEffect } from 'react';

// ── Mini KnowsMe Gauge ───────────────────────────────────────

const MiniGauge = ({ score, size = 80 }) => {
  const r = (size - 12) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <defs>
          <linearGradient id="onb-ggrad" x1="0%" y1="0%" x2="100%">
            <stop offset="0%" stopColor="#D4A547" />
            <stop offset="50%" stopColor="#F0D68A" />
            <stop offset="100%" stopColor="#D4A547" />
          </linearGradient>
        </defs>
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="rgba(212,165,71,0.08)"
          strokeWidth="4"
        />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="url(#onb-ggrad)"
          strokeWidth="4"
          strokeDasharray={c}
          strokeDashoffset={c - score * c}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 16,
          fontWeight: 700,
          color: '#F0D68A',
          letterSpacing: -0.5,
        }}>
          {(score * 100).toFixed(1)}
        </span>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 7,
          color: 'rgba(212,165,71,0.5)',
          letterSpacing: 1.5,
          textTransform: 'uppercase',
          marginTop: 1,
        }}>
          knows me
        </span>
      </div>
    </div>
  );
};

// ── Chat Bubble ───────────────────────────────────────────────

const ChatBubble = ({ role, content, meta }) => {
  const isUser = role === 'user';
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
      maxWidth: '85%',
      alignSelf: isUser ? 'flex-end' : 'flex-start',
      animation: 'onb-fadeUp 0.3s ease',
    }}>
      <div style={{
        padding: '10px 14px',
        background: isUser
          ? 'rgba(212,165,71,0.12)'
          : 'rgba(255,255,255,0.04)',
        border: `1px solid ${
          isUser
            ? 'rgba(212,165,71,0.2)'
            : 'rgba(255,255,255,0.06)'
        }`,
        borderRadius: isUser
          ? '14px 14px 4px 14px'
          : '14px 14px 14px 4px',
        color: 'rgba(255,255,255,0.88)',
        fontFamily: 'var(--sans)',
        fontSize: 13.5,
        lineHeight: 1.55,
      }}>
        {content}
      </div>
      {meta && (
        <div style={{ display: 'flex', gap: 10, marginTop: 4, padding: '0 4px' }}>
          {meta.agents && (
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(212,165,71,0.4)',
            }}>
              {meta.agents} agents
            </span>
          )}
          {meta.fragments > 0 && (
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(91,186,111,0.5)',
            }}>
              +{meta.fragments} learned
            </span>
          )}
          {meta.confidence && (
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(255,255,255,0.25)',
            }}>
              {(parseFloat(meta.confidence) * 100).toFixed(0)}% conf
            </span>
          )}
        </div>
      )}
    </div>
  );
};

// ── Typing Indicator ──────────────────────────────────────────

const TypingIndicator = () => (
  <div style={{
    display: 'flex',
    gap: 4,
    padding: '10px 14px',
    background: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '14px 14px 14px 4px',
    alignSelf: 'flex-start',
    marginBottom: 12,
  }}>
    {[0, 1, 2].map((i) => (
      <div
        key={i}
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: 'rgba(212,165,71,0.4)',
          animation: `onb-typing 1.2s ease infinite ${i * 0.2}s`,
        }}
      />
    ))}
  </div>
);

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function FirstChatStep({ node, state, setState, onNext }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [knowsMe, setKnowsMe] = useState(0);
  const [hasExchange, setHasExchange] = useState(false);
  const chatEndRef = useRef(null);

  // Scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  // Fetch initial knows_me
  useEffect(() => {
    const fetchKnowsMe = async () => {
      try {
        const result = await node.send('KNOWS_ME');
        if (result?.ok && result.fields?.score) {
          setKnowsMe(parseFloat(result.fields.score));
        }
      } catch (err) {
        // Silently handle - gauge stays at 0
      }
    };
    fetchKnowsMe();
  }, [node]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || sending) return;
    const text = input.trim();
    setInput('');

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setSending(true);

    try {
      const result = await node.send('RECEIVE', {
        content: text,
        timestamp: Date.now(),
      });

      if (result?.ok && result.fields) {
        const f = result.fields;

        setMessages((prev) => [...prev, {
          role: 'node',
          content: f.content || '...',
          meta: {
            agents: f.agents_consulted,
            fragments: parseInt(f.fragments_extracted || '0', 10),
            confidence: f.confidence,
          },
        }]);

        // Update knows_me
        if (f.knows_me) {
          setKnowsMe(parseFloat(f.knows_me));
        }

        setHasExchange(true);
        setState({ firstChatComplete: true });
      }
    } catch (err) {
      setMessages((prev) => [...prev, {
        role: 'node',
        content: 'Connection issue. Please try again.',
      }]);
    }

    setSending(false);
  }, [input, sending, node, setState]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 16,
      height: '100%',
      minHeight: 380,
    }}>
      {/* Header with gauge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div>
          <h2 style={{
            fontFamily: 'var(--sans)',
            fontSize: 20,
            fontWeight: 600,
            color: 'rgba(255,255,255,0.88)',
            margin: '0 0 4px 0',
          }}>
            Your First Conversation
          </h2>
          <p style={{
            fontFamily: 'var(--sans)',
            fontSize: 12,
            color: 'rgba(255,255,255,0.3)',
            margin: 0,
          }}>
            Every message teaches your node who you are.
          </p>
        </div>
        <MiniGauge score={knowsMe} />
      </div>

      {/* Chat area */}
      <div style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        padding: '12px 0',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}>
        {/* Greeting */}
        {messages.length === 0 && !sending && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            gap: 8,
            marginBottom: 16,
          }}>
            <div style={{
              padding: '12px 16px',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '14px 14px 14px 4px',
              color: 'rgba(255,255,255,0.8)',
              fontFamily: 'var(--sans)',
              fontSize: 14,
              lineHeight: 1.6,
              maxWidth: '90%',
            }}>
              I'm your sovereign AI. Ask me anything — the more you share, the better I understand you.
            </div>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(255,255,255,0.15)',
              padding: '0 4px',
            }}>
              Try: "What can you do?" or tell me about your work
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <ChatBubble
            key={i}
            role={msg.role}
            content={msg.content}
            meta={msg.meta}
          />
        ))}

        {/* Typing indicator */}
        {sending && <TypingIndicator />}

        <div ref={chatEndRef} />
      </div>

      {/* Input area */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 12,
        padding: '6px 6px 6px 16px',
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Talk to your node..."
          disabled={sending}
          style={{
            flex: 1,
            background: 'none',
            border: 'none',
            outline: 'none',
            fontFamily: 'var(--sans)',
            fontSize: 14,
            color: 'rgba(255,255,255,0.88)',
            padding: '6px 0',
          }}
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || sending}
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            border: 'none',
            background: input.trim() && !sending
              ? 'linear-gradient(135deg, #D4A547, #8B6914)'
              : 'rgba(255,255,255,0.04)',
            color: input.trim() && !sending ? '#0A0B0F' : 'rgba(255,255,255,0.15)',
            cursor: input.trim() && !sending ? 'pointer' : 'default',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 16,
            fontWeight: 700,
            flexShrink: 0,
            transition: 'all 0.2s ease',
          }}
        >
          {String.fromCharCode(8593)}
        </button>
      </div>

      {/* Continue to Dashboard */}
      {hasExchange && (
        <button
          onClick={onNext}
          style={{
            alignSelf: 'center',
            padding: '12px 32px',
            background: 'linear-gradient(135deg, #D4A547, #8B6914)',
            border: 'none',
            borderRadius: 10,
            fontFamily: 'var(--sans)',
            fontSize: 14,
            fontWeight: 600,
            color: '#0A0B0F',
            cursor: 'pointer',
            boxShadow: '0 4px 20px rgba(212,165,71,0.25)',
            transition: 'all 0.3s ease',
            animation: 'onb-fadeUp 0.4s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          Continue to Dashboard
        </button>
      )}

      <style>{`
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes onb-typing {
          0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
          30% { opacity: 1; transform: translateY(-3px); }
        }
      `}</style>
    </div>
  );
}
