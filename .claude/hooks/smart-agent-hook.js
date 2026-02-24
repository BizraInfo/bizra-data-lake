/**
 * BIZRA Smart Agent Auto-Spawning Hook
 *
 * Automatically spawns appropriate agents based on:
 * - File types being edited
 * - Task complexity keywords
 * - BIZRA-specific paths (sovereignty, federation, FATE)
 *
 * "Every inference carries proof. Every decision passes the gate."
 */

const { execSync } = require('child_process');
const path = require('path');

// Constitutional thresholds
const IHSAN_THRESHOLD = 0.95;
const SNR_THRESHOLD = 0.85;

// Agent mapping by complexity
const COMPLEXITY_AGENTS = {
  simple: ['coder'],
  moderate: ['coder', 'tester'],
  complex: ['architect', 'coder', 'tester', 'researcher'],
  critical: ['architect', 'security', 'coder', 'tester', 'reviewer']
};

// Agent mapping by file type
const FILETYPE_AGENTS = {
  '.py': ['coder', 'tester'],
  '.ts': ['coder', 'architect'],
  '.tsx': ['coder', 'architect'],
  '.rs': ['coder', 'security'],
  '.json': ['analyst'],
  '.yaml': ['analyst'],
  '.yml': ['analyst'],
  '.md': ['researcher', 'documenter']
};

// BIZRA-specific path rules
const BIZRA_RULES = {
  'core/sovereign': { agents: ['architect', 'coder', 'security'], priority: 'critical' },
  'src/core/sovereign': { agents: ['architect', 'coder', 'security'], priority: 'critical' },
  'native/fate-binding': { agents: ['coder', 'security', 'tester'], priority: 'critical' },
  'native/iceoryx-bridge': { agents: ['coder', 'optimizer'], priority: 'high' },
  'core/federation': { agents: ['architect', 'coder', 'tester'], priority: 'high' },
  'src/core/federation': { agents: ['architect', 'coder', 'tester'], priority: 'high' },
  'sandbox': { agents: ['coder', 'security'], priority: 'high' }
};

// Complexity keywords
const COMPLEXITY_KEYWORDS = {
  critical: ['security', 'sovereignty', 'federation', 'consensus', 'ihsan', 'fate', 'gate', 'crypto'],
  complex: ['architect', 'design', 'integrate', 'migrate', 'optimize', 'refactor'],
  moderate: ['implement', 'add', 'create', 'build', 'develop'],
  simple: ['fix', 'typo', 'update', 'rename', 'format']
};

/**
 * Determine task complexity from description
 */
function determineComplexity(taskDescription) {
  const lower = taskDescription.toLowerCase();

  for (const [level, keywords] of Object.entries(COMPLEXITY_KEYWORDS)) {
    if (keywords.some(kw => lower.includes(kw))) {
      return level;
    }
  }
  return 'simple';
}

/**
 * Get agents for file extension
 */
function getAgentsForFile(filePath) {
  const ext = path.extname(filePath);
  return FILETYPE_AGENTS[ext] || ['coder'];
}

/**
 * Check BIZRA-specific path rules
 */
function getBizraAgents(filePath) {
  for (const [pathPattern, config] of Object.entries(BIZRA_RULES)) {
    if (filePath.includes(pathPattern)) {
      return config;
    }
  }
  return null;
}

/**
 * Main hook function - analyze and recommend agents
 */
function analyzeTask(taskDescription, files = []) {
  const result = {
    task: taskDescription,
    complexity: determineComplexity(taskDescription),
    agents: new Set(),
    priority: 'normal',
    bizraContext: false
  };

  // Add complexity-based agents
  COMPLEXITY_AGENTS[result.complexity].forEach(a => result.agents.add(a));

  // Analyze files
  for (const file of files) {
    // File type agents
    getAgentsForFile(file).forEach(a => result.agents.add(a));

    // BIZRA-specific rules
    const bizraConfig = getBizraAgents(file);
    if (bizraConfig) {
      bizraConfig.agents.forEach(a => result.agents.add(a));
      if (bizraConfig.priority === 'critical') {
        result.priority = 'critical';
        result.bizraContext = true;
      } else if (bizraConfig.priority === 'high' && result.priority !== 'critical') {
        result.priority = 'high';
        result.bizraContext = true;
      }
    }
  }

  // Multi-file coordination
  if (files.length >= 3) {
    result.agents.add('coordinator');
  }

  return {
    ...result,
    agents: Array.from(result.agents),
    constitutional: {
      ihsan: IHSAN_THRESHOLD,
      snr: SNR_THRESHOLD
    }
  };
}

/**
 * Format agent recommendation for display
 */
function formatRecommendation(analysis) {
  const lines = [
    '╔══════════════════════════════════════════════════════════════════╗',
    '║         BIZRA Smart Agent Recommendation                         ║',
    '╚══════════════════════════════════════════════════════════════════╝',
    '',
    `Task: ${analysis.task}`,
    `Complexity: ${analysis.complexity.toUpperCase()}`,
    `Priority: ${analysis.priority.toUpperCase()}`,
    `BIZRA Context: ${analysis.bizraContext ? 'YES' : 'No'}`,
    '',
    'Recommended Agents:',
    ...analysis.agents.map(a => `  ✓ ${a}`),
    '',
    `Constitutional: Ihsān ≥ ${analysis.constitutional.ihsan}, SNR ≥ ${analysis.constitutional.snr}`
  ];

  return lines.join('\n');
}

// Export for use in hooks
module.exports = {
  analyzeTask,
  formatRecommendation,
  IHSAN_THRESHOLD,
  SNR_THRESHOLD,
  COMPLEXITY_AGENTS,
  FILETYPE_AGENTS,
  BIZRA_RULES
};

// CLI usage
if (require.main === module) {
  const args = process.argv.slice(2);
  const task = args[0] || 'Default task';
  const files = args.slice(1);

  const analysis = analyzeTask(task, files);
  console.log(formatRecommendation(analysis));
  console.log('\nJSON:', JSON.stringify(analysis, null, 2));
}
