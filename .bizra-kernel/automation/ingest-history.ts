import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

// BIZRA INGESTION AUTOMATION: SLOT-ALPHA (HISTORY)
// Implements the "Golden Sieve" protocol to ingest high-SNR conversation history.

interface ThoughtGene {
  id: string;
  timestamp: number;
  role: 'user' | 'assistant';
  content: string;
  significance: number;
  tags: string[];
}

class HistoryIngestor {
  private sourcePath: string;
  private kernelRoot: string;
  private memoryPath: string;
  private stats = {
    processed: 0,
    accepted: 0,
    rejected: 0,
    totalTokens: 0
  };

  constructor(kernelRoot: string, sourceFile: string) {
    this.kernelRoot = kernelRoot;
    this.sourcePath = sourceFile;
    this.memoryPath = path.join(kernelRoot, '.bizra-kernel', 'memory', 'history_index.json');
  }

  // The "Golden Sieve" Heuristic
  private calculateSignificance(content: string): number {
    let score = 0.5; // Base score

    // Boost for Code Blocks (High value technical context)
    if (content.includes('```')) score += 0.3;
    
    // Boost for Key Concepts (The "Genes" of BIZRA)
    const keyGenes = ['Ihsan', 'Rust', 'Consensus', 'Architecture', 'Security', 'Protocol', 'Node 0'];
    const geneMatches = keyGenes.filter(g => content.toLowerCase().includes(g.toLowerCase()));
    score += (geneMatches.length * 0.05);

    // Penalty for short/chatter messages
    if (content.length < 50) score -= 0.3;
    if (content.includes("Thank you") || content.includes("Hello")) score -= 0.1;

    return Math.min(1.0, Math.max(0.0, score));
  }

  private extractTags(content: string): string[] {
    const tags = new Set<string>();
    if (content.includes('```')) tags.add('code');
    if (content.toLowerCase().includes('error')) tags.add('debug');
    if (content.toLowerCase().includes('plan')) tags.add('strategy');
    return Array.from(tags);
  }

  public async ingest() {
    console.log(`[INGEST] Starting ingestion of ${this.sourcePath}`);
    console.log(`[INGEST] Applying "Golden Sieve" Protocol...`);

    if (!fs.existsSync(this.sourcePath)) {
      console.error(`[ERROR] Source file not found: ${this.sourcePath}`);
      return;
    }

    // Since we can't easily use a streaming JSON parser library without npm install,
    // we will use a robust line-reader approach assuming standard JSONL or formatted JSON.
    // NOTE: For a standard ChatGPT export (one giant JSON array), this is tricky without a parser.
    // We will assume for this script that we are reading a simplified or pre-processed version,
    // OR we read the whole file if memory permits (105MB is manageable in Node.js 14+).
    
    try {
      console.log('[INGEST] Reading file into memory (105MB is safe for modern heap)...');
      const rawData = fs.readFileSync(this.sourcePath, 'utf-8');
      const conversations = JSON.parse(rawData);

      console.log(`[INGEST] Found ${conversations.length} conversation threads.`);

      const memoryBank: ThoughtGene[] = [];

      for (const convo of conversations) {
        // Handle Custom Export Structure (uuid, chat_messages array)
        const messages = convo.chat_messages;
        if (!messages || !Array.isArray(messages)) continue;

        for (const msg of messages) {
          const content = msg.text; // Key is 'text', not 'content'
          if (!content) continue;

          const significance = this.calculateSignificance(content);

          // THRESHOLD: Only keep High SNR thoughts (> 0.6)
          if (significance > 0.6) {
            memoryBank.push({
              id: msg.uuid || convo.uuid,
              timestamp: msg.created_at ? new Date(msg.created_at).getTime() : (convo.created_at ? new Date(convo.created_at).getTime() : Date.now()),
              role: msg.sender === 'human' ? 'user' : 'assistant', // Infer role if possible, else default
              content: content.substring(0, 1000), // Truncate for index
              significance,
              tags: this.extractTags(content)
            });
            this.stats.accepted++;
          } else {
            this.stats.rejected++;
          }
          this.stats.processed++;
        }
      }

      this.saveMemory(memoryBank);

    } catch (error: any) {
      console.error(`[FAIL] Ingestion error: ${error.message}`);
    }
  }

  private saveMemory(data: ThoughtGene[]) {
    console.log(`[INGEST] Saving ${data.length} High-SNR genes to ${this.memoryPath}`);
    fs.writeFileSync(this.memoryPath, JSON.stringify(data, null, 2));
    
    console.log('--- INGESTION REPORT ---');
    console.log(`Total Messages Scanned: ${this.stats.processed}`);
    console.log(`Retained (High SNR):    ${this.stats.accepted}`);
    console.log(`Discarded (Noise):      ${this.stats.rejected}`);
    console.log(`Retention Rate:         ${((this.stats.accepted / this.stats.processed) * 100).toFixed(2)}%`);
  }
}

// Execution
const ingestor = new HistoryIngestor(
  process.cwd(), 
  process.argv[2] || "C:\\Users\\BIZRA-OS\\Downloads\\extracted-history\\conversations.json"
);
ingestor.ingest();
