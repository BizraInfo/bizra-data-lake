// ============================================================
// ProviderStep — LLM provider selection
// ============================================================
// Four radio-card options: Ollama, LM Studio, Anthropic, OpenAI.
// Cloud providers show an API key input field.
// Selected card gets a gold border highlight.
// ============================================================

import { useState, useCallback } from 'react';

const PROVIDERS = [
  {
    id: 'local-ollama',
    name: 'Ollama',
    subtitle: 'Local inference',
    description: 'Run models locally with Ollama. No data leaves your machine.',
    icon: 'O',
    iconColor: '#5BBA6F',
    recommended: true,
    needsKey: false,
  },
  {
    id: 'local-lmstudio',
    name: 'LM Studio',
    subtitle: 'Local inference',
    description: 'Connect to LM Studio for local model hosting and management.',
    icon: 'L',
    iconColor: '#6B9BF7',
    recommended: false,
    needsKey: false,
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    subtitle: 'Claude API',
    description: 'Use Claude models via the Anthropic API. Requires an API key.',
    icon: 'A',
    iconColor: '#A78BFA',
    recommended: false,
    needsKey: true,
  },
  {
    id: 'openai',
    name: 'OpenAI',
    subtitle: 'GPT API',
    description: 'Connect to OpenAI GPT models. Requires an API key.',
    icon: 'G',
    iconColor: '#F59E42',
    recommended: false,
    needsKey: true,
  },
];

// ── Provider Card ─────────────────────────────────────────────

const ProviderCard = ({ provider, selected, onSelect }) => {
  const isSelected = selected === provider.id;
  return (
    <button
      onClick={() => onSelect(provider.id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        width: '100%',
        padding: '14px 16px',
        background: isSelected
          ? 'rgba(212,165,71,0.06)'
          : 'rgba(255,255,255,0.02)',
        border: `1.5px solid ${
          isSelected
            ? 'rgba(212,165,71,0.35)'
            : 'rgba(255,255,255,0.06)'
        }`,
        borderRadius: 12,
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.25s ease',
        position: 'relative',
        overflow: 'hidden',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)';
          e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
          e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
        }
      }}
    >
      {/* Icon */}
      <div style={{
        width: 40,
        height: 40,
        borderRadius: 10,
        background: isSelected
          ? `${provider.iconColor}18`
          : 'rgba(255,255,255,0.04)',
        border: `1px solid ${
          isSelected
            ? `${provider.iconColor}30`
            : 'rgba(255,255,255,0.06)'
        }`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'var(--mono)',
        fontSize: 16,
        fontWeight: 700,
        color: isSelected ? provider.iconColor : 'rgba(255,255,255,0.3)',
        flexShrink: 0,
        transition: 'all 0.25s ease',
      }}>
        {provider.icon}
      </div>

      {/* Text content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <span style={{
            fontFamily: 'var(--sans)',
            fontSize: 14,
            fontWeight: 600,
            color: isSelected ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.6)',
            transition: 'color 0.2s ease',
          }}>
            {provider.name}
          </span>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'rgba(255,255,255,0.2)',
            letterSpacing: 0.5,
          }}>
            {provider.subtitle}
          </span>
          {provider.recommended && (
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 8,
              color: '#D4A547',
              background: 'rgba(212,165,71,0.1)',
              border: '1px solid rgba(212,165,71,0.2)',
              borderRadius: 4,
              padding: '1px 6px',
              letterSpacing: 0.5,
              textTransform: 'uppercase',
            }}>
              Recommended
            </span>
          )}
        </div>
        <div style={{
          fontFamily: 'var(--sans)',
          fontSize: 11,
          color: 'rgba(255,255,255,0.3)',
          lineHeight: 1.4,
        }}>
          {provider.description}
        </div>
      </div>

      {/* Radio indicator */}
      <div style={{
        width: 18,
        height: 18,
        borderRadius: '50%',
        border: `2px solid ${
          isSelected ? '#D4A547' : 'rgba(255,255,255,0.1)'
        }`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        transition: 'border-color 0.25s ease',
      }}>
        <div style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: isSelected ? '#D4A547' : 'transparent',
          transition: 'all 0.25s ease',
          transform: isSelected ? 'scale(1)' : 'scale(0)',
        }} />
      </div>
    </button>
  );
};

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function ProviderStep({ node, state, setState, onNext }) {
  const [selected, setSelected] = useState(state.provider || 'local-ollama');
  const [apiKey, setApiKey] = useState(state.apiKey || '');
  const [keyVisible, setKeyVisible] = useState(false);

  const selectedProvider = PROVIDERS.find((p) => p.id === selected);
  const needsKey = selectedProvider?.needsKey || false;
  const canContinue = !needsKey || apiKey.trim().length >= 10;

  const handleSelect = useCallback((id) => {
    setSelected(id);
    // Clear API key when switching to a provider that does not need one
    const provider = PROVIDERS.find((p) => p.id === id);
    if (!provider?.needsKey) {
      setApiKey('');
    }
  }, []);

  const handleContinue = useCallback(() => {
    setState({
      provider: selected,
      apiKey: needsKey ? apiKey.trim() : '',
      model: selected.startsWith('local') ? 'auto' : '',
    });
    onNext();
  }, [selected, apiKey, needsKey, setState, onNext]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Title */}
      <div style={{ textAlign: 'center', marginBottom: 4 }}>
        <h2 style={{
          fontFamily: 'var(--sans)',
          fontSize: 20,
          fontWeight: 600,
          color: 'rgba(255,255,255,0.88)',
          margin: '0 0 6px 0',
        }}>
          Choose Your Provider
        </h2>
        <p style={{
          fontFamily: 'var(--sans)',
          fontSize: 13,
          color: 'rgba(255,255,255,0.35)',
          margin: 0,
          lineHeight: 1.5,
        }}>
          Select how your node processes inference. Local keeps data sovereign.
        </p>
      </div>

      {/* Provider cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {PROVIDERS.map((provider) => (
          <ProviderCard
            key={provider.id}
            provider={provider}
            selected={selected}
            onSelect={handleSelect}
          />
        ))}
      </div>

      {/* API key input (conditionally shown) */}
      {needsKey && (
        <div style={{
          padding: '16px',
          background: 'rgba(167,139,250,0.04)',
          border: '1px solid rgba(167,139,250,0.12)',
          borderRadius: 10,
          animation: 'onb-fadeUp 0.3s ease',
        }}>
          <label style={{
            display: 'block',
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: 'rgba(255,255,255,0.35)',
            letterSpacing: 1,
            textTransform: 'uppercase',
            marginBottom: 8,
          }}>
            {selectedProvider.name} API Key
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type={keyVisible ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`sk-...`}
              autoComplete="off"
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 8,
                padding: '10px 14px',
                fontFamily: 'var(--mono)',
                fontSize: 12,
                color: 'rgba(255,255,255,0.8)',
                outline: 'none',
                transition: 'border-color 0.2s ease',
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = 'rgba(212,165,71,0.3)'; }}
              onBlur={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; }}
            />
            <button
              onClick={() => setKeyVisible(!keyVisible)}
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 8,
                padding: '0 12px',
                fontFamily: 'var(--mono)',
                fontSize: 10,
                color: 'rgba(255,255,255,0.35)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'rgba(255,255,255,0.35)'; }}
            >
              {keyVisible ? 'Hide' : 'Show'}
            </button>
          </div>
          <p style={{
            fontFamily: 'var(--sans)',
            fontSize: 10,
            color: 'rgba(255,255,255,0.2)',
            margin: '8px 0 0 0',
            lineHeight: 1.4,
          }}>
            Your key is stored locally and never sent to any third party.
          </p>
        </div>
      )}

      {/* Sovereignty notice for local providers */}
      {!needsKey && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 14px',
          background: 'rgba(91,186,111,0.04)',
          border: '1px solid rgba(91,186,111,0.1)',
          borderRadius: 8,
        }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: '#5BBA6F',
            flexShrink: 0,
          }} />
          <span style={{
            fontFamily: 'var(--sans)',
            fontSize: 11,
            color: 'rgba(91,186,111,0.7)',
            lineHeight: 1.4,
          }}>
            Full sovereignty: all inference runs on your hardware.
          </span>
        </div>
      )}

      {/* Continue button */}
      <button
        onClick={handleContinue}
        disabled={!canContinue}
        style={{
          alignSelf: 'center',
          marginTop: 8,
          padding: '12px 36px',
          background: canContinue
            ? 'linear-gradient(135deg, #D4A547, #8B6914)'
            : 'rgba(255,255,255,0.04)',
          border: 'none',
          borderRadius: 10,
          fontFamily: 'var(--sans)',
          fontSize: 14,
          fontWeight: 600,
          color: canContinue ? '#0A0B0F' : 'rgba(255,255,255,0.15)',
          cursor: canContinue ? 'pointer' : 'default',
          boxShadow: canContinue ? '0 4px 20px rgba(212,165,71,0.25)' : 'none',
          transition: 'all 0.3s ease',
        }}
        onMouseEnter={(e) => {
          if (canContinue) {
            e.currentTarget.style.boxShadow = '0 6px 28px rgba(212,165,71,0.35)';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }
        }}
        onMouseLeave={(e) => {
          if (canContinue) {
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(212,165,71,0.25)';
            e.currentTarget.style.transform = 'translateY(0)';
          }
        }}
      >
        Continue
      </button>

      <style>{`
        @keyframes onb-fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
