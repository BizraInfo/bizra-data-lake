"""
BIZRA LOCAL LLM GATEWAY (Data Lake Integration)
"Bridging BIZRA-TaskMaster's LLM Stack to the Data Lake"

This module provides the Data Lake with access to the local LLM infrastructure
already configured in BIZRA-TaskMaster:
- LM Studio (http://192.168.56.1:1234) - OpenAI-compatible API, 10+ models
- Ollama (http://localhost:11434) - Native Ollama API

Supports offline operation with automatic model discovery and fallback.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
LM_STUDIO_URL = "http://192.168.56.1:1234"
OLLAMA_URL = "http://localhost:11434"

class LLMProvider(Enum):
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OFFLINE = "offline"

@dataclass
class LocalModel:
    """Represents a locally available model."""
    id: str
    provider: LLMProvider
    size_label: str = ""  # e.g., "7B", "13B", "30B"
    context_length: int = 4096
    capabilities: List[str] = field(default_factory=list)

@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int = 0
    latency_ms: float = 0.0

# ═══════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
class LocalLLMGateway:
    """
    Unified gateway to all local LLM providers.
    Automatically discovers available models and routes requests.
    """
    
    def __init__(self):
        self.available_models: List[LocalModel] = []
        self.active_provider: Optional[LLMProvider] = None
        self.active_model: Optional[str] = None
        self._discover_models()
    
    def _discover_models(self):
        """Discover all available models from LM Studio and Ollama."""
        print("🔍 Discovering Local LLM Infrastructure...")
        
        # 1. Try LM Studio (OpenAI-compatible)
        lm_models = self._probe_lm_studio()
        
        # 2. Try Ollama
        ollama_models = self._probe_ollama()
        
        self.available_models = lm_models + ollama_models
        
        if self.available_models:
            # Select best model (prefer larger models)
            self._select_best_model()
        else:
            print("   ⚠️ No local LLM available. Operating in OFFLINE mode (retrieval only).")
            self.active_provider = LLMProvider.OFFLINE
    
    def _probe_lm_studio(self) -> List[LocalModel]:
        """Probe LM Studio for available models."""
        models = []
        try:
            req = urllib.request.Request(
                f"{LM_STUDIO_URL}/v1/models",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                
                for m in data.get("data", []):
                    model_id = m.get("id", "unknown")
                    # Parse size from model name (e.g., "qwen2.5-32b" -> "32B")
                    size = self._parse_model_size(model_id)
                    
                    models.append(LocalModel(
                        id=model_id,
                        provider=LLMProvider.LM_STUDIO,
                        size_label=size,
                        context_length=m.get("context_length", 4096),
                        capabilities=["chat", "completion"]
                    ))
                
                print(f"   ✅ LM Studio: {len(models)} models available")
                for m in models[:5]:  # Show first 5
                    print(f"      - {m.id} ({m.size_label})")
                if len(models) > 5:
                    print(f"      ... and {len(models) - 5} more")
                    
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"   ⚠️ LM Studio not reachable: {type(e).__name__}")
        except Exception as e:
            print(f"   ⚠️ LM Studio error: {e}")
        
        return models
    
    def _probe_ollama(self) -> List[LocalModel]:
        """Probe Ollama for available models."""
        models = []
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/tags",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                
                for m in data.get("models", []):
                    model_name = m.get("name", "unknown")
                    size = self._parse_model_size(model_name)
                    
                    models.append(LocalModel(
                        id=model_name,
                        provider=LLMProvider.OLLAMA,
                        size_label=size,
                        capabilities=["chat", "embedding"]
                    ))
                
                print(f"   ✅ Ollama: {len(models)} models available")
                for m in models[:5]:
                    print(f"      - {m.id} ({m.size_label})")
                    
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"   ⚠️ Ollama not reachable: {type(e).__name__}")
        except Exception as e:
            print(f"   ⚠️ Ollama error: {e}")
        
        return models
    
    def _parse_model_size(self, model_id: str) -> str:
        """Extract size label from model ID."""
        import re
        match = re.search(r'(\d+)[bB]', model_id)
        if match:
            return f"{match.group(1)}B"
        return "?"
    
    def _select_best_model(self):
        """Select the best available model (prioritize larger, then LM Studio)."""
        if not self.available_models:
            return
        
        # Sort by size (descending), prefer LM Studio for tie-breaking
        def model_score(m: LocalModel) -> tuple:
            try:
                size = int(m.size_label.replace('B', '').replace('?', '0'))
            except:
                size = 0
            provider_score = 1 if m.provider == LLMProvider.LM_STUDIO else 0
            return (size, provider_score)
        
        sorted_models = sorted(self.available_models, key=model_score, reverse=True)
        best = sorted_models[0]
        
        self.active_model = best.id
        self.active_provider = best.provider
        
        print(f"   🎯 Selected Model: {best.id} ({best.size_label}) via {best.provider.value}")
    
    def is_online(self) -> bool:
        """Check if any LLM is available."""
        return self.active_provider != LLMProvider.OFFLINE and self.active_model is not None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return [
            {
                "id": m.id,
                "provider": m.provider.value,
                "size": m.size_label,
                "capabilities": m.capabilities
            }
            for m in self.available_models
        ]
    
    # ═══════════════════════════════════════════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    def generate(self, prompt: str, system: str = None, model: str = None, 
                 max_tokens: int = 1024, temperature: float = 0.7) -> LLMResponse:
        """
        Generate a response from the local LLM.
        Falls back to retrieval-only if no LLM available.
        """
        if not self.is_online():
            return LLMResponse(
                content="[OFFLINE MODE] No local LLM available. Use retrieval-only capabilities.",
                model="none",
                provider=LLMProvider.OFFLINE
            )
        
        target_model = model or self.active_model
        target_provider = self._get_provider_for_model(target_model)
        
        if target_provider == LLMProvider.LM_STUDIO:
            return self._generate_lm_studio(prompt, system, target_model, max_tokens, temperature)
        elif target_provider == LLMProvider.OLLAMA:
            return self._generate_ollama(prompt, system, target_model, max_tokens, temperature)
        else:
            return LLMResponse(
                content="[ERROR] Unknown provider",
                model=target_model,
                provider=LLMProvider.OFFLINE
            )
    
    def _get_provider_for_model(self, model_id: str) -> LLMProvider:
        """Find which provider hosts a given model."""
        for m in self.available_models:
            if m.id == model_id:
                return m.provider
        return self.active_provider or LLMProvider.OFFLINE
    
    def _generate_lm_studio(self, prompt: str, system: str, model: str, 
                            max_tokens: int, temperature: float) -> LLMResponse:
        """Generate using LM Studio (OpenAI-compatible API)."""
        import time
        start = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                latency = (time.time() - start) * 1000
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=LLMProvider.LM_STUDIO,
                    tokens_used=tokens,
                    latency_ms=latency
                )
                
        except Exception as e:
            return LLMResponse(
                content=f"[ERROR] LM Studio generation failed: {e}",
                model=model,
                provider=LLMProvider.LM_STUDIO
            )
    
    def _generate_ollama(self, prompt: str, system: str, model: str,
                         max_tokens: int, temperature: float) -> LLMResponse:
        """Generate using Ollama native API."""
        import time
        start = time.time()
        
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                
                content = data.get("response", "")
                latency = (time.time() - start) * 1000
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=LLMProvider.OLLAMA,
                    latency_ms=latency
                )
                
        except Exception as e:
            return LLMResponse(
                content=f"[ERROR] Ollama generation failed: {e}",
                model=model,
                provider=LLMProvider.OLLAMA
            )
    
    # ═══════════════════════════════════════════════════════════════════════
    # RAG-ENHANCED GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    def generate_with_context(self, query: str, context: str, 
                              system: str = None, model: str = None) -> LLMResponse:
        """
        RAG-style generation: Inject retrieved context into prompt.
        """
        rag_system = system or """You are BIZRA PRIME, an intelligent assistant with access to a vast knowledge base.
Answer the user's question based on the provided context. If the context doesn't contain 
relevant information, say so honestly. Be concise and accurate."""
        
        rag_prompt = f"""CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:"""
        
        return self.generate(rag_prompt, system=rag_system, model=model)


# ═══════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 70)
    print("   🔌 BIZRA LOCAL LLM GATEWAY - INITIALIZATION")
    print("═" * 70)
    
    gateway = LocalLLMGateway()
    
    print(f"\n📊 INFRASTRUCTURE STATUS:")
    print(f"   Provider: {gateway.active_provider.value if gateway.active_provider else 'NONE'}")
    print(f"   Model: {gateway.active_model or 'NONE'}")
    print(f"   Online: {'✅ YES' if gateway.is_online() else '❌ NO'}")
    print(f"   Total Models: {len(gateway.available_models)}")
    
    if gateway.is_online():
        print("\n🧪 TEST GENERATION:")
        response = gateway.generate(
            "What is the capital of France? Answer in one word.",
            system="You are a helpful assistant. Be brief.",
            max_tokens=50
        )
        print(f"   Model: {response.model}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print(f"   Response: {response.content}")
    
    print("\n" + "═" * 70)
    print("   ✅ LOCAL LLM GATEWAY: READY")
    print("═" * 70)
