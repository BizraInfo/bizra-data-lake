/**
 * BIZRA IPC Module
 * 
 * Inter-process communication for the Sovereign LLM ecosystem.
 * 
 * Current: Stdio JSON protocol (fallback)
 * Target: Iceoryx2 zero-copy shared memory (< 250ns latency)
 */

export {
  SandboxClient,
  createSandboxClient,
  InferenceRequest,
  InferenceResponse,
  SandboxConfig,
} from './sandbox-client';
