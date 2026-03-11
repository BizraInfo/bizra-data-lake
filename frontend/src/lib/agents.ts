/**
 * PAT-7 Agent Definitions + SAT-5 System Agents
 * JARVIS personality engine with idle chatter and working sequences.
 */

import type { AgentDef, AgentId, SatAgent, Skill, ScheduledMission, TeachQuestion } from '../types';
import { color } from '../tokens';

export const PAT: Record<AgentId, AgentDef> = {
  P1: {
    name: 'Planner', callsign: 'ATLAS', domain: 'Strategy',
    bootMsg: 'Strategic planning ready.', icon: '\u25C8', color: color.sapphire,
    idle: ['Analyzing priority queue...', 'Three pending objectives identified.', 'Shall I restructure your schedule?', 'Your roadmap has a dependency conflict I can resolve.'],
    working: ['Decomposing into subtasks...', 'Dependency graph resolved.', 'Execution order optimized.', 'Critical path identified \u2014 3 steps.'],
  },
  P2: {
    name: 'Researcher', callsign: 'ORACLE', domain: 'Knowledge',
    bootMsg: 'Knowledge systems nominal.', icon: '\u25C9', color: color.cyan,
    idle: ['I found something interesting in your domain...', 'Three new papers match your interests.', 'Your knowledge graph has grown 12% this week.', 'Shall I deep-dive on that topic from yesterday?'],
    working: ['Scanning knowledge base...', 'Cross-referencing 47 sources.', 'Signal-to-noise ratio: 0.94.', 'Synthesis complete. Key findings extracted.'],
  },
  P3: {
    name: 'Coder', callsign: 'FORGE', domain: 'Build',
    bootMsg: 'Compiler initialized.', icon: '\u2B21', color: color.emerald,
    idle: ['Your test suite has 3 flaky tests I can fix.', 'I spotted a refactoring opportunity in the kernel.', 'Build pipeline is green. All 219 tests passing.', 'Dependency update available \u2014 no breaking changes.'],
    working: ['Generating implementation...', 'Running test suite...', 'All assertions pass.', 'Code quality: Ihsan 0.97.'],
  },
  P4: {
    name: 'Evaluator', callsign: 'JUDGE', domain: 'Quality',
    bootMsg: 'Quality gates armed.', icon: '\u25C7', color: color.amber,
    idle: ['Your average Ihsan is trending up \u2014 0.983 this week.', "I've benchmarked 3 alternatives for your last approach.", 'Quality score: top 5% of all nodes.', 'Recommending a peer review for your latest reflex.'],
    working: ['Running quality assessment...', 'Shannon entropy: above threshold.', 'Scoring against rubric...', 'Verdict: exceeds constitutional floor.'],
  },
  P5: {
    name: 'Ethicist', callsign: 'CROWN', domain: 'Ethics',
    bootMsg: 'All invariants holding.', icon: '\u2657', color: color.ruby,
    idle: ['All invariants satisfied. System is constitutional.', 'I-3 check: Gini at 0.31 \u2014 well within bounds.', 'No ethical flags in your recent actions.', 'The covenant holds. Integrity verified.'],
    working: ['Scanning against I-1 through I-7...', 'Shariah compliance: verified.', 'No bias detected in output.', 'Constitutional clearance granted.'],
  },
  P6: {
    name: 'Publisher', callsign: 'HERALD', domain: 'Deliver',
    bootMsg: 'Delivery channels open.', icon: '\u25C6', color: color.flame,
    idle: ['Your last report scored 4.8/5.0 readability.', "I've drafted three versions of your response.", 'Format optimized for your audience.', 'Feedback from the last delivery was excellent.'],
    working: ['Structuring output...', 'Formatting for clarity...', 'Final polish applied.', 'Ready for delivery.'],
  },
  P7: {
    name: 'Integrator', callsign: 'NEXUS', domain: 'Orchestrate',
    bootMsg: 'All agents reporting.', icon: '\u2726', color: color.amethyst,
    idle: ['All seven agents nominal.', 'Memory utilization: optimal.', "I've pre-loaded context from your last session.", 'Cross-agent coordination score: 94%.'],
    working: ['Routing to specialist...', 'Context bridge established.', 'Agent handoff complete.', 'Aggregating results from all sources.'],
  },
};

export const SAT_AGENTS: SatAgent[] = [
  { name: 'Sentinel', color: color.ruby },
  { name: 'Oracle', color: color.gold },
  { name: 'Ledger', color: color.amber },
  { name: 'Conductor', color: color.sapphire },
  { name: 'Ambassador', color: color.cyan },
];

export const AGENT_IDS = Object.keys(PAT) as AgentId[];

/** Keyword-to-agent routing table */
export const ROUTING_KEYWORDS: Record<AgentId, string[]> = {
  P1: ['plan', 'organize', 'strategy', 'roadmap', 'schedule'],
  P2: ['research', 'find', 'analyze', 'study', 'paper'],
  P3: ['code', 'build', 'test', 'fix', 'deploy', 'debug', 'implement'],
  P4: ['evaluate', 'score', 'review', 'audit', 'benchmark', 'assess'],
  P5: ['check', 'ethics', 'compliance', 'constitution', 'verify'],
  P6: ['write', 'draft', 'report', 'document', 'publish', 'present'],
  P7: [],
};

export function routeToAgent(task: string): AgentId {
  const lower = task.toLowerCase();
  let best: AgentId = 'P2';
  let bestScore = 0;
  for (const [id, keywords] of Object.entries(ROUTING_KEYWORDS) as [AgentId, string[]][]) {
    const score = keywords.filter(w => lower.includes(w)).length;
    if (score > bestScore) { best = id; bestScore = score; }
  }
  return best;
}

// ═══ Skills Data ═══

export const SKILLS: Skill[] = [
  { id: 'open_app', name: 'Open App', tier: 0, icon: '\u{1F680}', unlocked: true, hda: true },
  { id: 'switch_window', name: 'Switch Window', tier: 0, icon: '\u{1FA9F}', unlocked: true, hda: true },
  { id: 'type_text', name: 'Type Text', tier: 0, icon: '\u2328\uFE0F', unlocked: true, hda: true },
  { id: 'click_element', name: 'Click Element', tier: 1, icon: '\u{1F5B1}\uFE0F', hda: true },
  { id: 'screenshot', name: 'Screenshot', tier: 1, icon: '\u{1F4F8}', hda: true },
  { id: 'read_clipboard', name: 'Clipboard', tier: 1, icon: '\u{1F4CB}', hda: true },
  { id: 'file_open', name: 'File Open', tier: 2, icon: '\u{1F4D6}', hda: true },
  { id: 'browser_nav', name: 'Browser Nav', tier: 2, icon: '\u{1F310}', hda: true },
  { id: 'powershell', name: 'PowerShell', tier: 3, icon: '\u26A1' },
  { id: 'multistep', name: 'Multi-Step', tier: 3, icon: '\u{1F517}' },
  { id: 'crossapp', name: 'Cross-App', tier: 4, icon: '\u{1F504}' },
  { id: 'network', name: 'Network', tier: 4, icon: '\u{1F4E1}' },
  { id: 'governance', name: 'Governance', tier: 4, icon: '\u{1F3DB}\uFE0F' },
  { id: 'selfmod', name: 'Self-Modify', tier: 5, icon: '\u{1F9EC}' },
  { id: 'validator', name: 'Validator', tier: 5, icon: '\u{1F6E1}\uFE0F' },
  { id: 'federation', name: 'Federation', tier: 5, icon: '\u{1F30D}' },
];

// ═══ Scheduled Missions ═══

export const SCHEDULED_MISSIONS: ScheduledMission[] = [
  { id: 'morning-brief', name: 'Morning Brief', cron: '08:00 weekdays', icon: '\u2600\uFE0F', seedReward: '0.50', description: 'Overnight alerts + priority tasks', auto: false, agents: ['ATLAS', 'ORACLE', 'CROWN'] },
  { id: 'standup', name: 'Daily Standup', cron: '10:00 weekdays', icon: '\u{1F4CB}', seedReward: '0.30', description: 'Progress, blockers, plan', auto: false, agents: ['ATLAS', 'ORACLE'] },
  { id: 'health-check', name: 'Health Check', cron: 'Every 15 min', icon: '\u{1F49A}', seedReward: '0.05', description: 'Node0 subsystem monitoring', auto: true, agents: ['ORACLE'] },
  { id: 'weekly-review', name: 'Weekly Review', cron: '16:00 Friday', icon: '\u{1F4CA}', seedReward: '1.00', description: 'Accomplishments, metrics, next week', auto: false, agents: ['ATLAS', 'ORACLE', 'CROWN'] },
];

// ═══ TEACH Questions ═══

export const TEACH_QUESTIONS: TeachQuestion[] = [
  { id: 'work_schedule', prompt: 'What is your typical work schedule?', type: 'text', default: '8:00-18:00', icon: '\u{1F550}' },
  { id: 'primary_tools', prompt: 'Which applications do you use most?', type: 'multi', options: ['VS Code', 'Chrome', 'Slack', 'Terminal', 'Notion', 'Figma', 'Excel'], icon: '\u{1F6E0}\uFE0F' },
  { id: 'communication_pref', prompt: 'How should I communicate with you?', type: 'single', options: ['Concise bullet points', 'Detailed explanations', 'Only when critical'], default: 'Concise bullet points', icon: '\u{1F4AC}' },
  { id: 'priority_domains', prompt: 'What are your top priority domains?', type: 'multi', options: ['Engineering', 'Business strategy', 'Marketing', 'Operations', 'Research'], icon: '\u{1F3AF}' },
  { id: 'autonomy', prompt: 'How much autonomy should I have?', type: 'single', options: ['Ask before every action', 'Auto low-risk, ask high-risk', 'Full autonomous within budget'], default: 'Auto low-risk, ask high-risk', icon: '\u{1F916}' },
];

export const QUICK_MISSIONS = [
  'Research sovereign AI architectures',
  'Build constitutional invariant tests',
  'Plan Alpha-100 rollout strategy',
  'Evaluate deployment pipeline',
  'Draft quarterly progress report',
  'Review authentication security',
  'Check minting parameter compliance',
];
