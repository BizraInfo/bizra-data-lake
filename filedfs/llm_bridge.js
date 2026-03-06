#!/usr/bin/env node
// BIZRA LLM Bridge — Connects Node0 brain to any LLM
const { spawn } = require('child_process');
const readline = require('readline');
const https = require('https');
const http = require('http');

const CONFIG = {
  nodeBinary: process.env.BIZRA_NODE || './target/release/bizra-node',
  seedFile: process.env.BIZRA_SEED || './genesis_mumo.seed',
  userHash: process.env.BIZRA_USER || '1',
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  localEndpoint: null,
  ihsanFloor: '9500',
};
for (let i = 2; i < process.argv.length; i++) {
  switch (process.argv[i]) {
    case '--provider': CONFIG.provider = process.argv[++i]; break;
    case '--model': CONFIG.model = process.argv[++i]; break;
    case '--endpoint': CONFIG.localEndpoint = process.argv[++i]; break;
    case '--seed': CONFIG.seedFile = process.argv[++i]; break;
    case '--user': CONFIG.userHash = process.argv[++i]; break;
    case '--binary': CONFIG.nodeBinary = process.argv[++i]; break;
  }
}
let nodeProcess = null, nodeReady = false, pendingCallbacks = [], responseBuffer = '';

function startNode() {
  return new Promise((resolve, reject) => {
    nodeProcess = spawn(CONFIG.nodeBinary, [
      '--user', CONFIG.userHash, '--ihsan', CONFIG.ihsanFloor,
      '--seed', CONFIG.seedFile, '--no-banner',
    ], { stdio: ['pipe', 'pipe', 'pipe'] });
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
  const ep = CONFIG.localEndpoint||'http://localhost:8080/v1/chat/completions';
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
  console.log('  Commands: /quit /score /profile /health /teach <kind> <text> /synthesize\n');
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
    if (input.startsWith('/teach ')) { const p=input.substring(7).split(' '); await sendToNode('TEACH\t'+p[0]+'\t'+p.slice(1).join(' ')+'\t9000\t'+Date.now()); console.log('\n  Done: ['+p[0]+'] '+p.slice(1).join(' ')+'\n'); rl.prompt(); return; }
    if (input==='/synthesize') { const f=parseResp(await sendToNode('SYNTHESIZE\t'+Date.now())); console.log('\n  Synthesized. knows-me: '+(parseFloat(f.knows_me||'0')*100).toFixed(1)+'%\n'); rl.prompt(); return; }
    try {
      await sendToNode('RECEIVE\t'+input.replace(/\t/g,' ')+'\t'+Date.now());
      const ctx = await getContext();
      process.stdout.write('\n  node > ');
      const resp = await llm(sysPrompt(ctx), input, hist);
      console.log(resp+'\n');
      hist.push({role:'user',content:input},{role:'assistant',content:resp});
      if (hist.length>20) hist = hist.slice(-16);
    } catch(e) { console.log('\n  Error: '+e.message+'\n'); }
    rl.prompt();
  });
  rl.on('close', async () => { await sendToNode('SHUTDOWN'); setTimeout(()=>process.exit(0),300); });
}
main().catch(e => { console.error('Fatal: '+e.message); process.exit(1); });
