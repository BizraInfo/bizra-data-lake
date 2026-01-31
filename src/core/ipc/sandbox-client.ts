/**
 * Sandbox IPC Client
 * 
 * Connects TypeScript orchestrator to Python inference worker.
 * Uses stdin/stdout JSON protocol (fallback until Iceoryx2 is compiled).
 * 
 * Target: < 55ms end-to-end latency
 * Fallback: < 100ms via stdio
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import * as path from 'path';

export interface InferenceRequest {
  id: string;
  prompt: string;
  model_id: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
}

export interface InferenceResponse {
  id: string;
  content: string;
  model_id: string;
  tokens_generated: number;
  generation_time_ms: number;
  ihsan_score: number;
  snr_score: number;
  success: boolean;
  error?: string;
}

export interface SandboxConfig {
  pythonPath: string;
  workerScript: string;
  modelStorePath?: string;
  env?: Record<string, string>;
}

export class SandboxClient extends EventEmitter {
  private process: ChildProcess | null = null;
  private pendingRequests: Map<string, {
    resolve: (value: InferenceResponse) => void;
    reject: (reason: Error) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();
  private requestTimeoutMs: number = 60000;
  private isReady: boolean = false;

  constructor(private config: SandboxConfig) {
    super();
  }

  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      const env = {
        ...process.env,
        BIZRA_SANDBOX: '1',
        BIZRA_MODEL_STORE: this.config.modelStorePath || '',
        ...this.config.env,
      };

      this.process = spawn(this.config.pythonPath, [this.config.workerScript], {
        env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let buffer = '';

      this.process.stdout?.on('data', (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            this.handleMessage(line.trim());
          }
        }
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        const message = data.toString().trim();
        console.error(`[Sandbox] ${message}`);
        this.emit('stderr', message);
      });

      this.process.on('error', (error) => {
        this.emit('error', error);
        reject(error);
      });

      this.process.on('exit', (code) => {
        this.isReady = false;
        this.emit('exit', code);
        
        // Reject all pending requests
        for (const [id, pending] of this.pendingRequests) {
          pending.reject(new Error(`Sandbox exited with code ${code}`));
          clearTimeout(pending.timeout);
        }
        this.pendingRequests.clear();
      });

      // Wait for ready signal
      const checkReady = (line: string) => {
        try {
          const msg = JSON.parse(line);
          if (msg.status === 'ready') {
            this.isReady = true;
            this.emit('ready', msg);
            resolve();
          }
        } catch {
          // Ignore parse errors during startup
        }
      };

      this.process.stdout?.once('data', (data: Buffer) => {
        checkReady(data.toString().trim());
      });

      // Timeout if not ready in 30s
      setTimeout(() => {
        if (!this.isReady) {
          reject(new Error('Sandbox failed to start within 30s'));
        }
      }, 30000);
    });
  }

  private handleMessage(line: string): void {
    try {
      const msg = JSON.parse(line);

      if (msg.id && this.pendingRequests.has(msg.id)) {
        const pending = this.pendingRequests.get(msg.id)!;
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(msg.id);

        if (msg.type === 'error') {
          pending.reject(new Error(msg.message));
        } else {
          pending.resolve(msg as InferenceResponse);
        }
      } else if (msg.type) {
        this.emit(msg.type, msg);
      }
    } catch (error) {
      this.emit('parseError', { line, error });
    }
  }

  async infer(request: InferenceRequest): Promise<InferenceResponse> {
    if (!this.isReady || !this.process) {
      throw new Error('Sandbox not ready');
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(request.id);
        reject(new Error(`Inference timeout after ${this.requestTimeoutMs}ms`));
      }, this.requestTimeoutMs);

      this.pendingRequests.set(request.id, { resolve, reject, timeout });

      const message = JSON.stringify({
        type: 'inference',
        request,
      });

      this.process!.stdin!.write(message + '\n');
    });
  }

  async loadModel(modelId: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Load model timeout'));
      }, 30000);

      const handler = (msg: any) => {
        if (msg.type === 'model_loaded' && msg.model_id === modelId) {
          clearTimeout(timeout);
          this.off('model_loaded', handler);
          resolve(msg.success);
        }
      };

      this.on('model_loaded', handler);

      this.process!.stdin!.write(JSON.stringify({
        type: 'load_model',
        model_id: modelId,
      }) + '\n');
    });
  }

  async unloadModel(modelId: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Unload model timeout'));
      }, 10000);

      const handler = (msg: any) => {
        if (msg.type === 'model_unloaded' && msg.model_id === modelId) {
          clearTimeout(timeout);
          this.off('model_unloaded', handler);
          resolve(msg.success);
        }
      };

      this.on('model_unloaded', handler);

      this.process!.stdin!.write(JSON.stringify({
        type: 'unload_model',
        model_id: modelId,
      }) + '\n');
    });
  }

  async listModels(tier?: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('List models timeout'));
      }, 5000);

      const handler = (msg: any) => {
        if (msg.type === 'model_list') {
          clearTimeout(timeout);
          this.off('model_list', handler);
          resolve(msg.models);
        }
      };

      this.on('model_list', handler);

      const message: any = { type: 'list_models' };
      if (tier) message.tier = tier;

      this.process!.stdin!.write(JSON.stringify(message) + '\n');
    });
  }

  async shutdown(): Promise<void> {
    if (!this.process) return;

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        this.process?.kill('SIGTERM');
        resolve();
      }, 5000);

      const handler = (msg: any) => {
        if (msg.type === 'shutdown') {
          clearTimeout(timeout);
          resolve();
        }
      };

      this.on('shutdown', handler);

      this.process!.stdin!.write(JSON.stringify({ type: 'shutdown' }) + '\n');
    });
  }

  stop(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
    this.isReady = false;
  }

  get healthy(): boolean {
    return this.isReady && this.process !== null && !this.process.killed;
  }
}

// Factory function
export async function createSandboxClient(
  modelStorePath?: string
): Promise<SandboxClient> {
  const isWindows = process.platform === 'win32';
  
  const config: SandboxConfig = {
    pythonPath: isWindows ? 'python' : 'python3',
    workerScript: path.join(process.cwd(), 'sandbox', 'inference_worker.py'),
    modelStorePath,
  };

  const client = new SandboxClient(config);
  await client.start();
  return client;
}
