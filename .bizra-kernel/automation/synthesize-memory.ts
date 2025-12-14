import * as fs from 'fs';
import * as path from 'path';

// BIZRA KERNEL SYNTHESIS ENGINE
// Correlates "Thought Genes" (History) with "Evidence Genes" (Files) to build the Knowledge Graph.

interface ThoughtGene {
  id: string;
  timestamp: number;
  content: string;
  tags: string[];
}

interface EvidenceGene {
  id: string;
  filename: string;
  created: number;
  tags: string[];
}

interface GraphNode {
  id: string;
  type: 'thought' | 'evidence';
  label: string;
  data: any;
}

interface GraphEdge {
  source: string;
  target: string;
  relation: 'validates' | 'relates_to' | 'generated_by';
  weight: number; // 0.0 - 1.0 (Confidence)
}

interface KnowledgeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    thoughts: number;
    evidence: number;
    connections: number;
  };
}

class SynthesisEngine {
  private kernelRoot: string;
  private historyPath: string;
  private evidencePath: string;
  private graphPath: string;

  constructor(kernelRoot: string) {
    this.kernelRoot = kernelRoot;
    this.historyPath = path.join(kernelRoot, '.bizra-kernel', 'memory', 'history_index.json');
    this.evidencePath = path.join(kernelRoot, '.bizra-kernel', 'memory', 'evidence_index.json');
    this.graphPath = path.join(kernelRoot, '.bizra-kernel', 'memory', 'knowledge_graph.json');
  }

  public synthesize() {
    console.log('[SYNTHESIS] Initiating Neural-Symbolic Bridge...');

    if (!fs.existsSync(this.historyPath) || !fs.existsSync(this.evidencePath)) {
      console.error('[ERROR] Missing memory indices. Cannot synthesize.');
      return;
    }

    const thoughts: ThoughtGene[] = JSON.parse(fs.readFileSync(this.historyPath, 'utf-8'));
    const evidence: EvidenceGene[] = JSON.parse(fs.readFileSync(this.evidencePath, 'utf-8'));

    console.log(`[SYNTHESIS] Loaded ${thoughts.length} Thoughts and ${evidence.length} Evidence items.`);

    const graph: KnowledgeGraph = {
      nodes: [],
      edges: [],
      stats: { thoughts: 0, evidence: 0, connections: 0 }
    };

    // 1. Hydrate Nodes
    thoughts.forEach(t => {
      graph.nodes.push({
        id: t.id,
        type: 'thought',
        label: t.content.substring(0, 50) + '...',
        data: t
      });
    });
    graph.stats.thoughts = thoughts.length;

    evidence.forEach(e => {
      graph.nodes.push({
        id: e.id,
        type: 'evidence',
        label: e.filename,
        data: e
      });
    });
    graph.stats.evidence = evidence.length;

    // 2. Generate Edges (The "Synapse")
    console.log('[SYNTHESIS] Weaving connections...');
    
    let connections = 0;

    // Heuristic A: Tag Overlap (Symbolic Matching)
    for (const t of thoughts) {
      for (const e of evidence) {
        const sharedTags = t.tags.filter(tag => e.tags.includes(tag));
        
        if (sharedTags.length > 0) {
          graph.edges.push({
            source: e.id,
            target: t.id,
            relation: 'relates_to',
            weight: 0.3 + (sharedTags.length * 0.1)
          });
          connections++;
        }

        // Heuristic B: Keyword Resonance (Neural Approximation)
        // Check if thought content explicitly mentions the filename (minus extension)
        const nameStem = e.filename.split('.')[0].toLowerCase();
        if (nameStem.length > 4 && t.content.toLowerCase().includes(nameStem)) {
           graph.edges.push({
            source: t.id,
            target: e.id,
            relation: 'generated_by', // Thought generated the file
            weight: 0.9 // High confidence
          });
          connections++;
        }
      }
    }

    graph.stats.connections = connections;
    this.saveGraph(graph);
  }

  private saveGraph(graph: KnowledgeGraph) {
    console.log(`[SYNTHESIS] Graph Construction Complete.`);
    console.log(`  > Nodes: ${graph.nodes.length}`);
    console.log(`  > Edges: ${graph.edges.length}`);
    
    fs.writeFileSync(this.graphPath, JSON.stringify(graph, null, 2));
    console.log(`[SYNTHESIS] Knowledge Graph saved to ${this.graphPath}`);
  }
}

// Execution
const engine = new SynthesisEngine(process.cwd());
engine.synthesize();
