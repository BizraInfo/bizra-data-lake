/**
 * ace-framework/team-runner.js
 *
 * Minimal “personal agentic team” runner (local, Ollama-backed).
 * - Reads an aggregated chat index (aggregate.json) produced by tools/summarize_chat_index.py
 * - Delegates high-SNR analysis prompts to multiple role agents via orchestrator-ihsan-wrapper.js
 * - Writes receipts (hashes + metadata) without embedding raw chat history into repo by default
 *
 * Usage:
 *   node ace-framework/team-runner.js --aggregate "C:\\BIZRA-DATA-LAKE\\03_INDEXED\\chat_history\\<run>\\aggregate.json"
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const { queryModel } = require("./orchestrator-ihsan-wrapper");

function sha256String(input) {
  return crypto.createHash("sha256").update(input, "utf8").digest("hex");
}

function sha256File(filePath) {
  const h = crypto.createHash("sha256");
  const data = fs.readFileSync(filePath);
  h.update(data);
  return h.digest("hex");
}

function utcNowIso() {
  return new Date().toISOString();
}

function parseArgs(argv) {
  const args = { aggregate: "", out: "", context: [] };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--aggregate") args.aggregate = argv[++i] || "";
    else if (a === "--out") args.out = argv[++i] || "";
    else if (a === "--context") args.context.push(argv[++i] || "");
  }
  if (!args.aggregate) {
    throw new Error("Missing required flag: --aggregate <path>");
  }
  return args;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeText(filePath, content) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, "utf8");
}

function writeJson(filePath, obj) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(obj, null, 2), "utf8");
}

function takeTop(rows, n) {
  if (!Array.isArray(rows)) return [];
  return rows.slice(0, n);
}

function renderContext(agg) {
  const topTerms = takeTop(agg.top_terms, 25)
    .map((r) => `- ${r.key}: total=${r.total_count}, convs=${r.conversation_count}`)
    .join("\n");
  const topPaths = takeTop(agg.top_paths, 20)
    .map((r) => `- ${r.key} (convs=${r.conversation_count})`)
    .join("\n");
  const topCmds = takeTop(agg.top_commands, 20)
    .map((r) => `- ${r.key} (convs=${r.conversation_count})`)
    .join("\n");

  return [
    "EVIDENCE_CONTEXT (DERIVED, SNR-focused):",
    `generated_at: ${agg.generated_at}`,
    `inputs: ${JSON.stringify(agg.inputs || [])}`,
    "",
    "Top terms:",
    topTerms || "(none)",
    "",
    "Top paths:",
    topPaths || "(none)",
    "",
    "Top commands:",
    topCmds || "(none)",
  ].join("\n");
}

function readOptionalTextFile(filePath) {
  if (!filePath) return "";
  const full = path.resolve(filePath);
  if (!fs.existsSync(full)) return "";
  const stat = fs.statSync(full);
  if (!stat.isFile()) return "";
  const maxBytes = 256 * 1024;
  const data = fs.readFileSync(full);
  const slice = data.length > maxBytes ? data.subarray(0, maxBytes) : data;
  const txt = slice.toString("utf8");
  if (data.length > maxBytes) {
    return `${txt}\n\n[TRUNCATED: ${data.length - maxBytes} bytes omitted]`;
  }
  return txt;
}

function extractJsonCandidate(text) {
  const raw = (text || "").trim();
  if (!raw) {
    return { ok: false, error: "empty_response" };
  }

  // Direct parse
  try {
    return { ok: true, extracted: false, value: JSON.parse(raw) };
  } catch {}

  // Code-fence extraction ```json ... ```
  const fence = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence && fence[1]) {
    const inner = fence[1].trim();
    try {
      return { ok: true, extracted: true, value: JSON.parse(inner) };
    } catch (e) {
      // fallthrough
    }
  }

  // Object substring extraction (first '{' .. last '}')
  const firstObj = raw.indexOf("{");
  const lastObj = raw.lastIndexOf("}");
  if (firstObj !== -1 && lastObj !== -1 && lastObj > firstObj) {
    const inner = raw.slice(firstObj, lastObj + 1);
    try {
      return { ok: true, extracted: true, value: JSON.parse(inner) };
    } catch {}
  }

  // Array substring extraction (first '[' .. last ']')
  const firstArr = raw.indexOf("[");
  const lastArr = raw.lastIndexOf("]");
  if (firstArr !== -1 && lastArr !== -1 && lastArr > firstArr) {
    const inner = raw.slice(firstArr, lastArr + 1);
    try {
      return { ok: true, extracted: true, value: JSON.parse(inner) };
    } catch {}
  }

  return { ok: false, error: "json_parse_failed", raw_excerpt: raw.slice(0, 2000) };
}

function validateTeamOutput(obj, expectedRole) {
  const errors = [];
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
    errors.push("Top-level output is not a JSON object");
    return errors;
  }

  const truthLabels = new Set(["MEASURED", "VERIFIED", "DERIVED", "PLANNED", "ASSUMED", "UNKNOWN"]);
  const riskLevels = new Set(["low", "medium", "high", "critical"]);

  if (typeof obj.role !== "string" || !obj.role.trim()) errors.push("Missing/invalid: role");
  if (typeof expectedRole === "string" && obj.role && obj.role !== expectedRole) {
    errors.push(`Role mismatch: expected=${expectedRole} actual=${obj.role}`);
  }
  if (typeof obj.summary !== "string") errors.push("Missing/invalid: summary");
  if (!Array.isArray(obj.insights)) errors.push("Missing/invalid: insights (array)");
  if (!Array.isArray(obj.next_7_days)) errors.push("Missing/invalid: next_7_days (array)");
  if (!Array.isArray(obj.blocked_by)) errors.push("Missing/invalid: blocked_by (array)");

  if (Array.isArray(obj.insights)) {
    obj.insights.forEach((ins, idx) => {
      if (!ins || typeof ins !== "object" || Array.isArray(ins)) {
        errors.push(`insights[${idx}] is not an object`);
        return;
      }
      if (typeof ins.claim !== "string") errors.push(`insights[${idx}].claim invalid`);
      if (typeof ins.action !== "string") errors.push(`insights[${idx}].action invalid`);
      if (!truthLabels.has(ins.truth_label)) errors.push(`insights[${idx}].truth_label invalid`);
      if (!riskLevels.has(ins.risk)) errors.push(`insights[${idx}].risk invalid`);
      if (!Array.isArray(ins.evidence)) errors.push(`insights[${idx}].evidence invalid`);
    });
  }

  return errors;
}

function buildResultEnvelope(roleName, responseObj, parsed) {
  return {
    schema_version: 1,
    truth_label: "DERIVED",
    generated_at: utcNowIso(),
    role: roleName,
    model_target: process.env.MODEL_TARGET || "deepseek-r1:8b",
    ollama_host: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
    parse: {
      ok: !!parsed.ok,
      extracted: !!parsed.extracted,
      error: parsed.ok ? null : parsed.error || "unknown",
      raw_excerpt: parsed.ok ? null : parsed.raw_excerpt || null,
    },
    validation: {
      ok: false,
      errors: [],
    },
    data: parsed.ok ? parsed.value : null,
    raw: {
      has_error: !!(responseObj && responseObj.error),
      error: responseObj && responseObj.error ? String(responseObj.error) : null,
    },
  };
}

function buildPrompt(roleName, roleFocus, contextBlock) {
  return [
    "You are a specialized sub-agent operating under Ihsan (excellence, benevolence, trust).",
    "Hard rules:",
    "- Do not hallucinate. If you cannot verify a claim from the provided context, label it ASSUMED or UNKNOWN.",
    "- Output MUST be valid JSON only (no markdown).",
    "- Use truth labels: MEASURED, VERIFIED, DERIVED, PLANNED, ASSUMED, UNKNOWN.",
    "",
    `ROLE: ${roleName}`,
    `FOCUS: ${roleFocus}`,
    "",
    contextBlock,
    "",
    "Task: Produce a high-SNR action plan with evidence mapping.",
    "JSON schema:",
    "{",
    '  "role": "string",',
    '  "summary": "string",',
    '  "insights": [',
    "    {",
    '      "claim": "string",',
    '      "truth_label": "MEASURED|VERIFIED|DERIVED|PLANNED|ASSUMED|UNKNOWN",',
    '      "evidence": ["string"],',
    '      "risk": "low|medium|high|critical",',
    '      "action": "string"',
    "    }",
    "  ],",
    '  "next_7_days": ["string"],',
    '  "blocked_by": ["string"]',
    "}",
  ].join("\n");
}

async function main() {
  const { aggregate, out, context } = parseArgs(process.argv);
  const aggPath = path.resolve(aggregate);
  if (!fs.existsSync(aggPath)) {
    throw new Error(`Aggregate file not found: ${aggPath}`);
  }

  const repoRoot = path.resolve(__dirname, "..");
  const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+$/, "").replace("T", "_");
  const outDir = out ? path.resolve(out) : path.join(repoRoot, "docs", "evidence", "receipts", `llm_team_${ts}`);
  ensureDir(outDir);

  const agg = JSON.parse(fs.readFileSync(aggPath, "utf8"));
  const extraBlocks = (context || [])
    .map((p) => readOptionalTextFile(p))
    .filter((t) => t && t.trim())
    .map((t, idx) => `EXTRA_CONTEXT_${idx + 1} (MEASURED):\n${t}`);

  const contextBlock = [renderContext(agg), ...extraBlocks].join("\n\n---\n\n");

  const roles = [
    {
      name: "ARCHIVIST",
      focus: "Organize knowledge assets into a vault + index workflow; propose durable schemas and retrieval primitives.",
    },
    {
      name: "SECURITY",
      focus: "Identify likely security risks and hardening priorities; align with evidence-first and safe defaults.",
    },
    {
      name: "ARCHITECT",
      focus: "Map the system into clear components/contracts and identify architectural tensions + resolution strategy.",
    },
    {
      name: "PMO",
      focus: "Convert insights into a prioritized execution roadmap with owner roles and measurable gates.",
    },
    {
      name: "IHSAN_AUDITOR",
      focus: "Check for assumptions, missing evidence, and ways to increase trust (receipts, truth labels, verification).",
    },
  ];

  const runManifest = {
    run_type: "llm_team_runner",
    generated_at: utcNowIso(),
    aggregate: { path: aggPath, sha256: sha256File(aggPath) },
    outputs: [],
    truth_label: "MEASURED",
  };

  writeText(path.join(outDir, "00_context.txt"), contextBlock);
  runManifest.outputs.push({ file: "00_context.txt", sha256: sha256File(path.join(outDir, "00_context.txt")) });

  for (const role of roles) {
    const prompt = buildPrompt(role.name, role.focus, contextBlock);
    const promptPath = path.join(outDir, `prompt_${role.name}.txt`);
    writeText(promptPath, prompt);

    const response = await queryModel(prompt);
    const responsePath = path.join(outDir, `response_${role.name}.json`);
    writeJson(responsePath, response);

    const parsed = extractJsonCandidate(response && response.response ? response.response : "");
    const result = buildResultEnvelope(role.name, response, parsed);
    if (parsed.ok) {
      const errors = validateTeamOutput(parsed.value, role.name);
      result.validation.errors = errors;
      result.validation.ok = errors.length === 0;
    }

    const resultPath = path.join(outDir, `result_${role.name}.json`);
    writeJson(resultPath, result);

    const receipt = {
      receipt_version: 1,
      type: "llm_task",
      generated_at: utcNowIso(),
      truth_label: "MEASURED",
      role: role.name,
      model_target: process.env.MODEL_TARGET || "deepseek-r1:8b",
      ollama_host: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
      prompt_sha256: sha256String(prompt),
      response_sha256: sha256File(responsePath),
      files: {
        prompt: { path: promptPath, sha256: sha256File(promptPath) },
        response: { path: responsePath, sha256: sha256File(responsePath) },
        result: { path: resultPath, sha256: sha256File(resultPath) },
      },
      inputs: [{ path: aggPath, sha256: sha256File(aggPath) }],
    };

    const receiptPath = path.join(outDir, `receipt_${role.name}.json`);
    writeJson(receiptPath, receipt);

    runManifest.outputs.push(
      { file: path.basename(promptPath), sha256: sha256File(promptPath) },
      { file: path.basename(responsePath), sha256: sha256File(responsePath) },
      { file: path.basename(resultPath), sha256: sha256File(resultPath) },
      { file: path.basename(receiptPath), sha256: sha256File(receiptPath) }
    );
  }

  const manifestPath = path.join(outDir, "ZZ_RUN_MANIFEST.json");
  writeJson(manifestPath, runManifest);
  console.log(outDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
