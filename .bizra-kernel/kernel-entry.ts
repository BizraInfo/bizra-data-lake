import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// BIZRA KERNEL ENTRY POINT
// This script initializes the kernel for the current directory context.

interface KernelConfig {
  modules: Record<string, boolean>;
}

interface MemoryStore {
  shortTerm: { sessionLog: string[] };
  longTerm: { vectorIndex: string };
}

interface TriggerConfig {
  watchers: Array<{
    id: string;
    pattern: string;
    events: string[];
    action: string;
  }>;
}

class BizraKernel {
  private rootDir: string;
  private config: KernelConfig;
  private memory: MemoryStore | null = null;
  private context: string = '';
  private triggers: TriggerConfig | null = null;

  constructor(rootDir: string) {
    this.rootDir = rootDir;
    this.config = this.loadConfig();
  }

  private loadConfig(): KernelConfig {
    const configPath = path.join(this.rootDir, '.bizra-kernel', 'config', 'core.json');
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    }
    throw new Error('Kernel configuration not found. Is the kernel installed?');
  }

  public async init() {
    console.log(`[BIZRA-KERNEL] Initializing in ${this.rootDir}...`);
    
    if (this.config.modules.memory) await this.initMemory();
    if (this.config.modules.context) await this.initContext();
    if (this.config.modules.triggers) await this.initTriggers();
    
    console.log('[BIZRA-KERNEL] System Active.');
    
    // Demonstration: Execute security scan hook
    await this.executeHook('pre-commit');

    // Self-test: Query the Oracle
    await this.queryOracle("Kernel initialized. Status report?");
  }

  private async initMemory() {
    const memoryPath = path.join(this.rootDir, '.bizra-kernel', 'memory', 'local_store.json');
    const historyPath = path.join(this.rootDir, '.bizra-kernel', 'memory', 'history_index.json');
    
    this.memory = { shortTerm: { sessionLog: [] }, longTerm: { vectorIndex: '' } };

    if (fs.existsSync(memoryPath)) {
      const localStore = JSON.parse(fs.readFileSync(memoryPath, 'utf-8'));
      this.memory = { ...this.memory, ...localStore };
      console.log('[MODULE] Memory System: Online (Local Store Loaded)');
    }

    if (fs.existsSync(historyPath)) {
      const history = JSON.parse(fs.readFileSync(historyPath, 'utf-8'));
      console.log(`[MODULE] Memory System: History Ingested (${history.length} High-SNR Genes Active)`);
    } else {
      console.warn('[MODULE] Memory System: History Index not found. Run ingestion script.');
    }

    const evidencePath = path.join(this.rootDir, '.bizra-kernel', 'memory', 'evidence_index.json');
    if (fs.existsSync(evidencePath)) {
      const evidence = JSON.parse(fs.readFileSync(evidencePath, 'utf-8'));
      console.log(`[MODULE] Memory System: Evidence Vault Active (${evidence.length} Proof Assets)`);
    }

    const graphPath = path.join(this.rootDir, '.bizra-kernel', 'memory', 'knowledge_graph.json');
    if (fs.existsSync(graphPath)) {
      const graph = JSON.parse(fs.readFileSync(graphPath, 'utf-8'));
      console.log(`[MODULE] Memory System: Knowledge Graph Synthesized (${graph.stats.connections} Synapses)`);
    }
  }

  private async initContext() {
    const contextPath = path.join(this.rootDir, '.bizra-kernel', 'context', 'agent_manifest.md');
    if (fs.existsSync(contextPath)) {
      this.context = fs.readFileSync(contextPath, 'utf-8');
      console.log('[MODULE] Agentic Context: Loaded (Manifest Active)');
    }
  }

  private async initTriggers() {
    const triggersPath = path.join(this.rootDir, '.bizra-kernel', 'triggers', 'watchers.json');
    if (fs.existsSync(triggersPath)) {
      this.triggers = JSON.parse(fs.readFileSync(triggersPath, 'utf-8'));
      console.log(`[MODULE] Trigger System: Watching ${this.triggers?.watchers.length} patterns`);
      
      // Simple watcher implementation (Proof of Concept)
      this.triggers?.watchers.forEach(watcher => {
        // In a real implementation, we would use chokidar here.
        // For now, we just log that we are aware of the pattern.
        console.log(`  > Active Watcher: ${watcher.id} on ${watcher.pattern}`);
      });
    }
  }

  public async executeHook(hookName: string) {
    const registryPath = path.join(this.rootDir, '.bizra-kernel', 'hooks', 'registry.json');
    if (!fs.existsSync(registryPath)) return;

    const registry = JSON.parse(fs.readFileSync(registryPath, 'utf-8'));
    const scripts = registry[hookName] || [];

    for (const script of scripts) {
      const scriptPath = path.join(this.rootDir, '.bizra-kernel', script);
      console.log(`[HOOK] Executing ${script}...`);
      try {
        // Determine execution engine based on extension
        const cmd = script.endsWith('.js') ? `node "${scriptPath}"` : `"${scriptPath}"`;
        const { stdout, stderr } = await execAsync(cmd);
        if (stdout) console.log(`  [OUT] ${stdout.trim()}`);
        if (stderr) console.error(`  [ERR] ${stderr.trim()}`);
      } catch (error: any) {
        console.error(`  [FAIL] Hook execution failed: ${error.message}`);
      }
    }
  }

  public async queryOracle(prompt: string): Promise<string> {
    // The Neural Bridge: Connects the Kernel to the Local LLM
    console.log(`[ORACLE] Querying Local Intelligence: "${prompt}"`);
    
    try {
      // Try to connect to Ollama (DeepSeek/Bizra-Planner)
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'deepseek-r1', // Default to reasoning model
          prompt: `[SYSTEM: You are the Kernel Agent for ${this.rootDir}. Context: ${this.context.substring(0, 200)}...] USER: ${prompt}`,
          stream: false
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`[ORACLE] Response: ${data.response.substring(0, 50)}...`);
        return data.response;
      }
    } catch (e) {
      console.log('[ORACLE] Local LLM offline. Using fallback logic.');
    }
    return "Oracle Offline";
  }
}

// Auto-start if run directly
// In ES modules, we can just call it if we are the entry point
const isMainModule = import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}`;

if (true) { // Simplified for immediate execution in this context
  const kernel = new BizraKernel(process.cwd());
  kernel.init().catch(console.error);
}

export default BizraKernel;
