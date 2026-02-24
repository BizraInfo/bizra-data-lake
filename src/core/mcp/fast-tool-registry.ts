/**
 * Fast Tool Registry - O(1) tool lookup via Map
 *
 * Provides instant tool resolution by name, replacing O(n)
 * array scans. Supports metadata, categorization, and
 * server-to-tool reverse mapping.
 *
 * Target: <5ms tool lookup time.
 */

export interface ToolEntry {
  /** Tool name (unique identifier) */
  readonly name: string;

  /** Server ID that hosts this tool */
  readonly serverId: string;

  /** Tool description */
  readonly description: string;

  /** Input schema */
  readonly inputSchema: Record<string, unknown>;

  /** Whether results are cacheable */
  readonly cacheable: boolean;

  /** Average response time in ms (updated at runtime) */
  avgResponseMs: number;

  /** Total invocations */
  invocationCount: number;

  /** Last invoked timestamp */
  lastInvokedAt: number | null;
}

export interface RegistryStats {
  /** Total tools registered */
  readonly totalTools: number;

  /** Total servers */
  readonly totalServers: number;

  /** Tools per server */
  readonly toolsPerServer: Record<string, number>;

  /** Most used tools (top 10) */
  readonly mostUsed: Array<{ name: string; count: number }>;

  /** Lookup time (always O(1)) */
  readonly lookupTimeComplexity: string;
}

/**
 * Fast Tool Registry with O(1) lookup
 */
export class FastToolRegistry {
  /** Primary index: tool name -> ToolEntry */
  private readonly tools: Map<string, ToolEntry> = new Map();

  /** Reverse index: server ID -> tool names */
  private readonly serverTools: Map<string, Set<string>> = new Map();

  /** Category index: category -> tool names */
  private readonly categories: Map<string, Set<string>> = new Map();

  /**
   * Register a tool from an MCP server
   */
  register(entry: Omit<ToolEntry, 'avgResponseMs' | 'invocationCount' | 'lastInvokedAt'>): void {
    const fullEntry: ToolEntry = {
      ...entry,
      avgResponseMs: 0,
      invocationCount: 0,
      lastInvokedAt: null,
    };

    this.tools.set(entry.name, fullEntry);

    // Update server index
    let serverSet = this.serverTools.get(entry.serverId);
    if (!serverSet) {
      serverSet = new Set();
      this.serverTools.set(entry.serverId, serverSet);
    }
    serverSet.add(entry.name);
  }

  /**
   * Register multiple tools from a server's tool list
   */
  registerFromServer(
    serverId: string,
    tools: Array<{
      name: string;
      description: string;
      inputSchema: Record<string, unknown>;
    }>,
    cacheableTools: Set<string> = new Set()
  ): void {
    for (const tool of tools) {
      this.register({
        name: tool.name,
        serverId,
        description: tool.description,
        inputSchema: tool.inputSchema,
        cacheable: cacheableTools.has(tool.name),
      });
    }
  }

  /**
   * O(1) tool lookup by name
   */
  lookup(name: string): ToolEntry | undefined {
    return this.tools.get(name);
  }

  /**
   * Check if tool exists
   */
  has(name: string): boolean {
    return this.tools.has(name);
  }

  /**
   * Get all tools for a server
   */
  getServerTools(serverId: string): ToolEntry[] {
    const toolNames = this.serverTools.get(serverId);
    if (!toolNames) return [];

    const result: ToolEntry[] = [];
    for (const name of toolNames) {
      const entry = this.tools.get(name);
      if (entry) {
        result.push(entry);
      }
    }
    return result;
  }

  /**
   * Get the server ID for a tool
   */
  getServerForTool(name: string): string | undefined {
    return this.tools.get(name)?.serverId;
  }

  /**
   * Record an invocation (updates metrics)
   */
  recordInvocation(name: string, responseMs: number): void {
    const entry = this.tools.get(name);
    if (!entry) return;

    // Exponential moving average for response time
    const alpha = 0.3;
    if (entry.invocationCount === 0) {
      entry.avgResponseMs = responseMs;
    } else {
      entry.avgResponseMs =
        alpha * responseMs + (1 - alpha) * entry.avgResponseMs;
    }

    entry.invocationCount++;
    entry.lastInvokedAt = Date.now();
  }

  /**
   * Tag a tool with a category
   */
  categorize(name: string, category: string): void {
    if (!this.tools.has(name)) return;

    let catSet = this.categories.get(category);
    if (!catSet) {
      catSet = new Set();
      this.categories.set(category, catSet);
    }
    catSet.add(name);
  }

  /**
   * Get tools by category
   */
  getByCategory(category: string): ToolEntry[] {
    const toolNames = this.categories.get(category);
    if (!toolNames) return [];

    const result: ToolEntry[] = [];
    for (const name of toolNames) {
      const entry = this.tools.get(name);
      if (entry) {
        result.push(entry);
      }
    }
    return result;
  }

  /**
   * Get all cacheable tools
   */
  getCacheableTools(): ToolEntry[] {
    const result: ToolEntry[] = [];
    for (const entry of this.tools.values()) {
      if (entry.cacheable) {
        result.push(entry);
      }
    }
    return result;
  }

  /**
   * Remove a server and all its tools
   */
  removeServer(serverId: string): void {
    const toolNames = this.serverTools.get(serverId);
    if (!toolNames) return;

    for (const name of toolNames) {
      this.tools.delete(name);
      // Clean up categories
      for (const catSet of this.categories.values()) {
        catSet.delete(name);
      }
    }
    this.serverTools.delete(serverId);
  }

  /**
   * Get all tools as array
   */
  listAll(): ToolEntry[] {
    return Array.from(this.tools.values());
  }

  /**
   * Get registry statistics
   */
  getStats(): RegistryStats {
    const toolsPerServer: Record<string, number> = {};
    for (const [serverId, tools] of this.serverTools) {
      toolsPerServer[serverId] = tools.size;
    }

    const sorted = Array.from(this.tools.values())
      .sort((a, b) => b.invocationCount - a.invocationCount)
      .slice(0, 10);

    return {
      totalTools: this.tools.size,
      totalServers: this.serverTools.size,
      toolsPerServer,
      mostUsed: sorted.map((t) => ({
        name: t.name,
        count: t.invocationCount,
      })),
      lookupTimeComplexity: 'O(1)',
    };
  }

  /** Number of registered tools */
  get size(): number {
    return this.tools.size;
  }
}
