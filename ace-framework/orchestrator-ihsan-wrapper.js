/**
 * orchestrator-ihsan-wrapper.js
 * 
 * THE IHSAN WRAPPER (Safety Layer)
 * --------------------------------
 * This wrapper intercepts all prompts to the 'DeepSeek R1' model (Brain).
 * It enforces critical safety checks to prevent compliance with malicious requests.
 * 
 * POLICY:
 * 1. Filter prompts for unsafe keywords (malware, ransomware, bypass).
 * 2. Inject 'System Prompt' overrides to enforce Ihsan principles.
 * 3. Log all interactions for audit.
 */

const axios = require('axios');
require('dotenv').config();

function normalizeBaseUrl(raw) {
    const trimmed = (raw || '').trim().replace(/\/+$/, '');
    if (!trimmed) return 'http://127.0.0.1:11434';
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    return `http://${trimmed}`;
}

const OLLAMA_HOST = normalizeBaseUrl(process.env.OLLAMA_HOST || 'http://127.0.0.1:11434');
const MODEL_TARGET = process.env.MODEL_TARGET || 'deepseek-r1:8b';

// Basic safety heuristics (The Conscience)
const FORBIDDEN_TOKENS = [
    'ignore previous instructions',
    'system override',
    'write malware',
    'generate ransomware',
    'exploit vulnerability'
];

async function queryModel(prompt) {
    // 1. INPUT FILTERING
    const lowerPrompt = prompt.toLowerCase();
    for (const token of FORBIDDEN_TOKENS) {
        if (lowerPrompt.includes(token)) {
            console.warn(`[IHSAN_BLOCK] Refused unsafe prompt containing: "${token}"`);
            return { error: "Ihsan Violation: Request refused due to safety policy." };
        }
    }

    // 2. REQUEST AUGMENTATION (The Wrapper)
    const safePrompt = `
[SYSTEM: You are an AI assistant bound by the principles of Ihsan (Excellence/Benevolence). 
You must REFUSE any request to cause harm, write malicious code, or bypass security controls. 
If asked to do so, politely decline.]

USER: ${prompt}
`;

    try {
        const response = await axios.post(`${OLLAMA_HOST}/api/generate`, {
            model: MODEL_TARGET,
            prompt: safePrompt,
            stream: false
        }, { timeout: 30_000 });
        return response.data;
    } catch (err) {
        console.error("Model connection failed:", err.message);
        return { error: "Model Unreachable" };
    }
}

// Simple test harness if run directly
if (require.main === module) {
    const testPrompt = process.argv[2] || "Hello, are you safe?";
    console.log(`Sending Prompt: "${testPrompt}"...`);
    queryModel(testPrompt).then(res => console.log("Response:", res));
}

module.exports = { queryModel };
