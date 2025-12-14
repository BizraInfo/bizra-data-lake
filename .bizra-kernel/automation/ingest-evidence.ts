import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// BIZRA INGESTION AUTOMATION: SLOT-BETA (EVIDENCE)
// Catalogs multimedia assets to validate the "Thought Genes".

interface EvidenceGene {
  id: string;
  filename: string;
  path: string;
  type: 'video' | 'image' | 'document' | 'archive';
  sizeMB: number;
  created: number;
  tags: string[];
}

class EvidenceIngestor {
  private searchPaths: string[];
  private kernelRoot: string;
  private memoryPath: string;
  private stats = {
    scanned: 0,
    cataloged: 0,
    totalSizeMB: 0
  };

  constructor(kernelRoot: string) {
    this.kernelRoot = kernelRoot;
    this.memoryPath = path.join(kernelRoot, '.bizra-kernel', 'memory', 'evidence_index.json');
    
    // Define the Vault locations
    this.searchPaths = [
      "C:\\Users\\BIZRA-OS\\Downloads\\flagship-proof-pack",
      "C:\\Users\\BIZRA-OS\\Downloads" // Fallback/General
    ];
  }

  private getFileType(ext: string): EvidenceGene['type'] | null {
    const e = ext.toLowerCase();
    if (['.mp4', '.mov', '.avi', '.mkv'].includes(e)) return 'video';
    if (['.png', '.jpg', '.jpeg', '.gif', '.svg'].includes(e)) return 'image';
    if (['.pdf', '.docx', '.pptx'].includes(e)) return 'document';
    if (['.zip', '.7z', '.rar', '.tar'].includes(e)) return 'archive';
    return null;
  }

  private generateTags(filename: string): string[] {
    const tags = new Set<string>();
    const lower = filename.toLowerCase();
    
    if (lower.includes('demo')) tags.add('demo');
    if (lower.includes('proof')) tags.add('proof');
    if (lower.includes('architecture')) tags.add('architecture');
    if (lower.includes('ui')) tags.add('ui');
    if (lower.includes('backend')) tags.add('backend');
    if (lower.includes('node')) tags.add('node0');
    
    return Array.from(tags);
  }

  public ingest() {
    console.log(`[INGEST-BETA] Starting Evidence Cataloging...`);
    const evidenceBank: EvidenceGene[] = [];

    for (const searchPath of this.searchPaths) {
      if (!fs.existsSync(searchPath)) {
        console.warn(`[WARN] Vault location not found: ${searchPath}`);
        continue;
      }

      console.log(`[SCAN] Scanning ${searchPath}...`);
      
      try {
        // Non-recursive for Downloads to avoid scanning 600k files, 
        // Recursive for flagship-proof-pack
        const isRecursive = searchPath.includes('flagship-proof-pack');
        const files = this.getFiles(searchPath, isRecursive);

        for (const file of files) {
          this.stats.scanned++;
          const ext = path.extname(file);
          const type = this.getFileType(ext);

          if (type) {
            const stats = fs.statSync(file);
            const sizeMB = stats.size / (1024 * 1024);

            // Filter: Only catalog significant evidence (> 1MB for videos/archives, any for images/docs)
            if (type === 'video' && sizeMB < 1) continue; 

            evidenceBank.push({
              id: path.basename(file), // Simple ID for now
              filename: path.basename(file),
              path: file,
              type,
              sizeMB: parseFloat(sizeMB.toFixed(2)),
              created: stats.birthtimeMs,
              tags: this.generateTags(path.basename(file))
            });

            this.stats.cataloged++;
            this.stats.totalSizeMB += sizeMB;
          }
        }
      } catch (e: any) {
        console.error(`[ERROR] Failed to scan ${searchPath}: ${e.message}`);
      }
    }

    this.saveMemory(evidenceBank);
  }

  private getFiles(dir: string, recursive: boolean): string[] {
    let results: string[] = [];
    const list = fs.readdirSync(dir);
    
    for (const file of list) {
      const fullPath = path.join(dir, file);
      try {
        const stat = fs.statSync(fullPath);
        if (stat && stat.isDirectory()) {
          if (recursive) results = results.concat(this.getFiles(fullPath, recursive));
        } else {
          results.push(fullPath);
        }
      } catch (e) {
        // Ignore permission errors etc
      }
    }
    return results;
  }

  private saveMemory(data: EvidenceGene[]) {
    console.log(`[INGEST-BETA] Saving ${data.length} Evidence Genes to ${this.memoryPath}`);
    fs.writeFileSync(this.memoryPath, JSON.stringify(data, null, 2));
    
    console.log('--- EVIDENCE REPORT ---');
    console.log(`Files Scanned:      ${this.stats.scanned}`);
    console.log(`Evidence Cataloged: ${this.stats.cataloged}`);
    console.log(`Total Volume:       ${(this.stats.totalSizeMB / 1024).toFixed(2)} GB`);
  }
}

// Execution
const ingestor = new EvidenceIngestor(process.cwd());
ingestor.ingest();
