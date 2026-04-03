import { MemoryDirectory } from '../types';

const NOW = new Date().toISOString();

export const INITIAL_MEMORY_ROOT: MemoryDirectory = {
  name: 'memory_docs',
  type: 'directory',
  path: '/mnt/data/memory_docs',
  children: [
    {
      name: 'README.md',
      type: 'file',
      path: '/mnt/data/memory_docs/README.md',
      status: 'active',
      lastModified: NOW,
      content: `# BIZRA Memory Bank (Node0)\n\nSingle source of truth for long-horizon continuity.`
    },
    {
      name: 'codeMap_root.md',
      type: 'file',
      path: '/mnt/data/memory_docs/codeMap_root.md',
      status: 'active',
      lastModified: NOW,
      content: `---
timestamp: ${NOW}
status: APOTHEOSIS_ACTIVE
mode: RECURSIVE_EVOLUTION
---
# CodeMap Root
- **Active Task:** TASK_004 (Recursive Expansion)
- **Paradigm:** Cognitive MMORPG`
    },
    {
        name: 'docs',
        type: 'directory',
        path: '/mnt/data/memory_docs/docs',
        children: [
            {
                name: 'recursive_dynamics.md',
                type: 'file',
                path: '/mnt/data/memory_docs/docs/recursive_dynamics.md',
                status: 'active',
                lastModified: NOW,
                content: `# Recursive Capacity Dynamics\n\nTarget: 0.55 -> 0.91 Synergy Gain.`
            }
        ]
    }
  ]
};