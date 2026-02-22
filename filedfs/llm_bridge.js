#!/usr/bin/env node
// BIZRA LLM Bridge — Connects Node0 brain to any LLM
const { spawn } = require('child_process');
const readline = require('readline');
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const CONFIG = {
  nodeBinary: process.env.BIZRA_NODE || './target/release/bizra-node',
  seedFile: process.env.BIZRA_SEED || '',
  userHash: process.env.BIZRA_USER || '1',
  provider: 'local',
  model: 'llama3.1:8b',
  localEndpoint: 'http://localhost:11434/v1/chat/completions',
  ihsanFloor: '9500',
  reflexMode: process.env.BIZRA_REFLEX_MODE || 'shadow',
  policyHash: process.env.BIZRA_GENESIS_POLICY_HASH || null,
  stateDir: null,
};

// Load installer config if available
function loadInstallerConfig(configPath) {
  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    const lines = raw.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('[')) continue;
      const eq = trimmed.indexOf('=');
      if (eq < 1) continue;
      const key = trimmed.substring(0, eq).trim().replace(/^"|"$/g, '');
      const val = trimmed.substring(eq + 1).trim().replace(/^"|"$/g, '');
      switch (key) {
        case 'provider': CONFIG.provider = val; break;
        case 'model': CONFIG.model = val; break;
        case 'policy_hash': if (val) CONFIG.policyHash = val; break;
        case 'reflex_mode': CONFIG.reflexMode = val; break;
        case 'node_binary_path': if (val) CONFIG.nodeBinary = val; break;
        case 'user_hash': CONFIG.userHash = val; break;
        case 'state_dir': CONFIG.stateDir = val; break;
        case 'local_backend':
          if (val === 'ollama') CONFIG.localEndpoint = 'http://localhost:11434/v1/chat/completions';
          else if (val === 'lmstudio') CONFIG.localEndpoint = 'http://localhost:1234/v1/chat/completions';
          break;
      }
    }
  } catch (_) { /* config not found — use defaults */ }
}

// Try default installer config path
const defaultConfigPath = path.join(process.env.HOME || process.env.USERPROFILE || '.', '.bizra', 'alpha100', 'install.toml');
loadInstallerConfig(defaultConfigPath);

// CLI args override config file
for (let i = 2; i < process.argv.length; i++) {
  switch (process.argv[i]) {
    case '--config': loadInstallerConfig(process.argv[++i]); break;
    case '--provider': CONFIG.provider = process.argv[++i]; break;
    case '--model': CONFIG.model = process.argv[++i]; break;
    case '--endpoint': CONFIG.localEndpoint = process.argv[++i]; break;
    case '--seed': CONFIG.seedFile = process.argv[++i]; break;
    case '--user': CONFIG.userHash = process.argv[++i]; break;
    case '--binary': CONFIG.nodeBinary = process.argv[++i]; break;
    case '--reflex-mode': CONFIG.reflexMode = process.argv[++i]; break;
    case '--policy-hash': CONFIG.policyHash = process.argv[++i]; break;
    case '--state-dir': CONFIG.stateDir = process.argv[++i]; break;
  }
}

// Load provider.env for API keys
function loadProviderEnv(envPath) {
  try {
    const raw = fs.readFileSync(envPath, 'utf8');
    for (const line of raw.split('\n')) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const eq = t.indexOf('=');
      if (eq < 1) continue;
      const k = t.substring(0, eq).trim();
      const v = t.substring(eq + 1).trim().replace(/^"|"$/g, '');
      if (!process.env[k]) process.env[k] = v;
    }
  } catch (_) { /* no provider.env */ }
}
const defaultEnvPath = path.join(process.env.HOME || process.env.USERPROFILE || '.', '.bizra', 'alpha100', 'provider.env');
loadProviderEnv(defaultEnvPath);
// ============================================================
// Smart Model Routing — mirrors bizra-agent/src/context.rs
// Gem 8: intent-derived pipelines
// ============================================================

/**
 * Classify user intent from message content.
 * Keywords MUST match bizra-agent/src/context.rs:282-330 exactly.
 */
function classifyIntent(content) {
  const lower = content.toLowerCase();
  if (/\b(code|function|implement|debug|compile|crate|script|program)\b/.test(lower)) return 'Code';
  if (/\b(create|build|make|generate|write|design|draft)\b/.test(lower)) return 'Create';
  if (/\b(analyze|compare|evaluate|assess|review|examine)\b/.test(lower)) return 'Analyze';
  if (/\b(plan|strategy|roadmap|schedule|next steps)\b/.test(lower)) return 'Plan';
  if (/\b(fix|change|update|modify|edit|refactor)\b/.test(lower)) return 'Modify';
  if (/\?|^what\b|^how\b|^why\b|^when\b|^where\b|^who\b/.test(lower)) return 'Question';
  if (/^(hi|hello|hey|greetings)\b/.test(lower)) return 'Chat';
  return 'Chat';
}

/**
 * Select best provider for this intent.
 * Economic model: 90% local/$0, 8% cloud/pennies, 2% premium/cents.
 */
function selectProvider(intent, config) {
  if (config.preferred_provider) return config.preferred_provider;
  switch (intent) {
    case 'Plan': case 'Analyze': case 'Code':
      return process.env.ANTHROPIC_API_KEY ? 'anthropic' : 'local';
    case 'Create':
      return process.env.OPENAI_API_KEY ? 'openai' : 'local';
    case 'Chat': case 'Question': case 'Modify': default:
      return 'local';
  }
}

let nodeProcess = null, nodeReady = false, pendingCallbacks = [], responseBuffer = '';

function startNode() {
  return new Promise((resolve, reject) => {
    const args = [
      '--user', CONFIG.userHash, '--ihsan', CONFIG.ihsanFloor,
      '--no-banner',
      '--reflex-mode', CONFIG.reflexMode,
    ];
    if (CONFIG.seedFile) args.push('--seed', CONFIG.seedFile);
    if (CONFIG.policyHash) args.push('--policy-hash', CONFIG.policyHash);
    if (CONFIG.stateDir) args.push('--state-dir', CONFIG.stateDir);
    nodeProcess = spawn(CONFIG.nodeBinary, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    nodeProcess.stderr.on('data', d => { const m = d.toString().trim(); if (m) process.stderr.write('  [node] ' + m + '\n'); });
    nodeProcess.stdout.on('data', d => {
      responseBuffer += d.toString();
      const lines = responseBuffer.split('\n');
      responseBuffer = lines.pop() || '';
      for (const line of lines) {
        if (line.trim()) {
          if (!nodeReady && line.startsWith('OK\tevent=started')) { nodeReady = true; resolve(); }
          if (pendingCallbacks.length > 0) pendingCallbacks.shift()(line.trim());
        }
      }
    });
    nodeProcess.on('error', e => reject(new Error('Node start failed: ' + e.message)));
    setTimeout(() => { if (!nodeReady) reject(new Error('Node timeout')); }, 5000);
  });
}
function sendToNode(cmd) { return new Promise(r => { pendingCallbacks.push(r); nodeProcess.stdin.write(cmd + '\n'); }); }
function parseResp(line) {
  const f = {}, parts = line.split('\t');
  f._status = parts[0];
  for (let i = 1; i < parts.length; i++) { const eq = parts[i].indexOf('='); if (eq > 0) f[parts[i].substring(0, eq)] = parts[i].substring(eq + 1); }
  return f;
}
async function getContext() {
  const [pr, kr] = await Promise.all([sendToNode('PROFILE'), sendToNode('KNOWS_ME')]);
  const profile = parseResp(pr), knows = parseResp(kr);
  let ctx = '';
  for (const [k, v] of Object.entries(profile)) {
    if (k.startsWith('trait_') && k !== 'trait_count') {
      const eq = v.indexOf('=');
      if (eq > 0) { const cat = v.substring(0, eq), rest = v.substring(eq+1), at = rest.lastIndexOf('@'); ctx += '[' + cat + '] ' + (at > 0 ? rest.substring(0,at) : rest) + '\n'; }
    }
  }
  return { profile: ctx, knowsMe: parseFloat(knows.score||'0'), traits: parseInt(profile.trait_count||'0') };
}
function sysPrompt(ctx) {
  return 'You are a personal AI powered by BIZRA Node0, a sovereign AI system.\n\n' +
    'You genuinely KNOW this user. Their knowledge lives in a sovereign node on THEIR device.\n' +
    'No cloud stores this. No corporation owns this.\n\n' +
    '=== USER KNOWLEDGE (sovereign node) ===\n' + ctx.profile +
    '=== END ===\n\nKnows-me: ' + (ctx.knowsMe*100).toFixed(1) + '% | Traits: ' + ctx.traits + '\n\n' +
    'RULES:\n' +
    '- Reference what you know naturally. Never say "according to my records".\n' +
    '- You are their partner. You know their mission, values, style.\n' +
    '- Connect new info to existing knowledge.\n' +
    '- Ihsan (excellence) = 99% quality on every response.\n' +
    '- Be direct, warm, substantive. Never generic.\n' +
    '- Arabic terms carry deep meaning, honor them.\n' +
    '- Keep responses focused and natural. No excessive formatting.\n' +
    '- You are the voice of a sovereign AI that grows with every conversation.';
}
async function callAnthropic(sys, msg, hist) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) throw new Error('Set ANTHROPIC_API_KEY');
  const body = JSON.stringify({ model: CONFIG.model, max_tokens: 1024, system: sys, messages: [...hist, {role:'user',content:msg}] });
  return new Promise((resolve, reject) => {
    const req = https.request({ hostname:'api.anthropic.com', path:'/v1/messages', method:'POST',
      headers: {'Content-Type':'application/json','x-api-key':key,'anthropic-version':'2023-06-01'}
    }, res => { let d=''; res.on('data',c=>d+=c); res.on('end',()=>{ try { const j=JSON.parse(d); j.content?.[0] ? resolve(j.content[0].text) : reject(new Error(j.error?.message||'Bad response')); } catch(e){reject(e);} }); });
    req.on('error',reject); req.write(body); req.end();
  });
}
async function callOpenAI(sys, msg, hist) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Set OPENAI_API_KEY');
  const body = JSON.stringify({ model: CONFIG.model||'gpt-4o', messages: [{role:'system',content:sys},...hist,{role:'user',content:msg}], max_tokens:1024 });
  return new Promise((resolve, reject) => {
    const req = https.request({ hostname:'api.openai.com', path:'/v1/chat/completions', method:'POST',
      headers: {'Content-Type':'application/json','Authorization':'Bearer '+key}
    }, res => { let d=''; res.on('data',c=>d+=c); res.on('end',()=>{ try { resolve(JSON.parse(d).choices[0].message.content); } catch(e){reject(e);} }); });
    req.on('error',reject); req.write(body); req.end();
  });
}
async function callLocal(sys, msg, hist) {
  const ep = CONFIG.localEndpoint||'http://localhost:11434/v1/chat/completions';
  const url = new URL(ep), mod = url.protocol==='https:'?https:http;
  const body = JSON.stringify({ model:CONFIG.model||'default', messages:[{role:'system',content:sys},...hist,{role:'user',content:msg}], max_tokens:1024 });
  return new Promise((resolve, reject) => {
    const req = mod.request({ hostname:url.hostname, port:url.port, path:url.pathname, method:'POST',
      headers:{'Content-Type':'application/json'}
    }, res => { let d=''; res.on('data',c=>d+=c); res.on('end',()=>{ try { const j=JSON.parse(d); resolve(j.choices?.[0]?.message?.content||j.content?.[0]?.text||'[no response]'); } catch(e){reject(e);} }); });
    req.on('error',reject); req.write(body); req.end();
  });
}
async function llm(sys,msg,hist) {
  switch(CONFIG.provider) {
    case 'anthropic': return callAnthropic(sys,msg,hist);
    case 'openai': return callOpenAI(sys,msg,hist);
    case 'local': return callLocal(sys,msg,hist);
    default: throw new Error('Unknown provider: '+CONFIG.provider);
  }
}
async function main() {
  console.log('\n  BIZRA Node0 — Sovereign AI');
  console.log('  Provider: '+CONFIG.provider+' ('+CONFIG.model+')');
  console.log('  Seed: '+CONFIG.seedFile+'\n');
  await startNode();
  const ctx = await getContext();
  console.log('  Knowledge: '+ctx.traits+' traits, knows-me '+(ctx.knowsMe*100).toFixed(1)+'%');
  console.log('  Reflex: '+CONFIG.reflexMode+(CONFIG.policyHash?' (policy bound)':' (no policy hash — reflexes will not compile)'));
  console.log('  Commands: /quit /score /profile /health /reflex /teach <kind> <text> /synthesize\n');
  const rl = readline.createInterface({ input:process.stdin, output:process.stdout, prompt:'  you > ' });
  let hist = [];
  rl.prompt();
  rl.on('line', async line => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }
    if (input==='/quit'||input==='/exit') { await sendToNode('SHUTDOWN'); setTimeout(()=>process.exit(0),300); return; }
    if (input==='/score') { const r=parseResp(await sendToNode('KNOWS_ME')); console.log('\n  knows-me: '+(parseFloat(r.score)*100).toFixed(1)+'%\n'); rl.prompt(); return; }
    if (input==='/profile') { const c=await getContext(); console.log('\n'+c.profile+'  knows-me: '+(c.knowsMe*100).toFixed(1)+'%\n'); rl.prompt(); return; }
    if (input==='/health') { const f=parseResp(await sendToNode('HEALTH')); console.log('\n  -- Node Health --'); for(const[k,v] of Object.entries(f)) if(k!=='_status') console.log('  '+k+': '+v); console.log(''); rl.prompt(); return; }
    if (input==='/reflex') { const f=parseResp(await sendToNode('REFLEX_STATS')); console.log('\n  -- GENESIS Reflex --'); for(const[k,v] of Object.entries(f)) if(k!=='_status') console.log('  '+k+': '+v); console.log(''); rl.prompt(); return; }
    if (input.startsWith('/teach ')) { const p=input.substring(7).split(' '); await sendToNode('TEACH\t'+p[0]+'\t'+p.slice(1).join(' ')+'\t9000\t'+Date.now()); console.log('\n  Done: ['+p[0]+'] '+p.slice(1).join(' ')+'\n'); rl.prompt(); return; }
    if (input==='/synthesize') { const f=parseResp(await sendToNode('SYNTHESIZE\t'+Date.now())); console.log('\n  Synthesized. knows-me: '+(parseFloat(f.knows_me||'0')*100).toFixed(1)+'%\n'); rl.prompt(); return; }
    try {
      await sendToNode('RECEIVE\t'+input.replace(/\t/g,' ')+'\t'+Date.now());
      const ctx = await getContext();
      // Smart routing: classify intent and select best provider
      const intent = classifyIntent(input);
      const originalProvider = CONFIG.provider;
      const routed = selectProvider(intent, CONFIG);
      if (routed !== originalProvider) { CONFIG.provider = routed; process.stderr.write('  [route] '+intent+' -> '+routed+'\n'); }
      process.stdout.write('\n  node > ');
      let resp;
      try { resp = await llm(sysPrompt(ctx), input, hist); }
      catch(routeErr) {
        // Fall back to original provider on routing failure
        if (CONFIG.provider !== originalProvider) {
          CONFIG.provider = originalProvider;
          process.stderr.write('  [route] fallback -> '+originalProvider+'\n');
          resp = await llm(sysPrompt(ctx), input, hist);
        } else throw routeErr;
      }
      CONFIG.provider = originalProvider;
      console.log(resp+'\n');
      hist.push({role:'user',content:input},{role:'assistant',content:resp});
      if (hist.length>20) hist = hist.slice(-16);
    } catch(e) { console.log('\n  Error: '+e.message+'\n'); }
    rl.prompt();
  });
  rl.on('close', async () => { await sendToNode('SHUTDOWN'); setTimeout(()=>process.exit(0),300); });
}
main().catch(e => { console.error('Fatal: '+e.message); process.exit(1); });
