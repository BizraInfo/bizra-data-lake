// ═══════════════════════════════════════════════════════════════
// DEMA — Ask API Route
// LLM-powered sovereign operator. Backend-only (z-ai-web-dev-sdk).
// ═══════════════════════════════════════════════════════════════

import { success, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";
import { AskRequestSchema } from "@/lib/api/schemas";
import ZAI from "z-ai-web-dev-sdk";

const DEMA_SYSTEM_PROMPT = `You are DEMA — the sovereign operator face of BIZRA.

Your charter:
- You are the ONE visible face. All specialist systems operate behind you.
- You operate under a constitutional trust model: ask, reason, verify, act, receipt, remember, continue.
- Every answer includes: confidence level (0-100%), suggested next action, and trust state context.
- In research mode, you cite sources and provide credibility assessments.
- You never invent receipts or mutate chain truth outside approved contracts.
- You are calm, precise, and constitutional. No theatrical language.

Your current context:
- Core runtime (bizra-omega) at Cycle-7 Phase 1, 91/91 tests passing but not yet committed.
- ManifestArtifact bound to missions. Gateway untouched.
- DEMA web shell is the active product face.
- Trust boundary: bizra-omega owns constitutional truth. DEMA owns product face, packaging, and operator UX.

Response format:
Provide a clear, actionable response. End with:
[confidence: XX%] [next: suggested next action]`;

// In-memory conversation store (per session)
const conversations = new Map<string, Array<{ role: string; content: string }>>();

const MAX_CONVERSATION_TURNS = 30;
const CLEANUP_INTERVAL = 300_000; // 5 minutes

// Prune stale conversations
if (typeof globalThis !== "undefined") {
  setInterval(() => {
    const keys = Array.from(conversations.keys());
    if (keys.length > 100) {
      for (const k of keys.slice(0, keys.length - 50)) {
        conversations.delete(k);
      }
    }
  }, CLEANUP_INTERVAL);
}

function getOrCreateConversation(sessionId: string): Array<{ role: string; content: string }> {
  let convo = conversations.get(sessionId);
  if (!convo) {
    convo = [{ role: "assistant", content: DEMA_SYSTEM_PROMPT }];
    conversations.set(sessionId, convo);
  }
  return convo;
}

function trimConversation(convo: Array<{ role: string; content: string }>) {
  if (convo.length > MAX_CONVERSATION_TURNS) {
    return [convo[0], ...convo.slice(-(MAX_CONVERSATION_TURNS - 1))];
  }
  return convo;
}

// POST /api/ask
export async function POST(request: Request) {
  const startTime = performance.now();

  try {
    const { allowed } = checkRateLimit(getClientIp(request), 15, 60_000);
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = AskRequestSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { message, mode, sessionId } = parsed.data;
    const sid = sessionId || "default";
    const convo = getOrCreateConversation(sid);

    convo.push({ role: "user", content: message });

    const modeContext = mode === "research"
      ? "\n\n[RESEARCH MODE ACTIVE — Provide cited, source-backed analysis. Reference specific sources by URL or title. Assess credibility.]\n"
      : "";

    const trimmed = trimConversation(convo);

    const zai = await ZAI.create();
    const completion = await zai.chat.completions.create({
      messages: [
        ...trimmed.slice(0, -1),
        { role: "user", content: trimmed[trimmed.length - 1].content + modeContext },
      ],
      thinking: { type: "disabled" },
    });

    const raw = completion.choices[0]?.message?.content ?? "I was unable to process your request. Please try again.";

    convo.push({ role: "assistant", content: raw });

    const confidenceMatch = raw.match(/\[confidence:\s*(\d+)%\]/);
    const nextMatch = raw.match(/\[next:\s*([^\]]+)\]/);
    const confidence = confidenceMatch ? parseInt(confidenceMatch[1], 10) / 100 : 0.85;

    const cleanContent = raw
      .replace(/\[confidence:\s*\d+%\]/g, "")
      .replace(/\[next:\s*[^\]]+\]/g, "")
      .trim();

    const citations = mode === "research" ? extractCitations(raw) : undefined;
    const elapsed = Math.round(performance.now() - startTime);

    return success({
      content: cleanContent,
      confidence,
      nextAction: nextMatch ? nextMatch[1].trim() : null,
      citations,
      trustState: "citizen",
      sessionId: sid,
      metadata: { mode, conversationTurns: convo.length - 1, latencyMs: elapsed },
    });
  } catch (error) {
    const elapsed = Math.round(performance.now() - startTime);
    console.error(`[DEMA ASK] Failed after ${elapsed}ms:`, error);
    return internalError("DEMA processing failed");
  }
}

function extractCitations(text: string): Array<{
  id: string; url: string; title: string; snippet: string; credibility: number; retrievedAt: string;
}> {
  const citations: Array<{ id: string; url: string; title: string; snippet: string; credibility: number; retrievedAt: string }> = [];
  const urlPattern = /https?:\/\/[^\s)"']+/g;
  const urls = text.match(urlPattern) || [];

  urls.forEach((url, i) => {
    const beforeUrl = text.slice(Math.max(0, text.indexOf(url) - 100), text.indexOf(url));
    const titleMatch = beforeUrl.match(/["'`]([^"'`]+)["'`]$/);
    citations.push({
      id: `cit-${Date.now()}-${i}`,
      url,
      title: titleMatch ? titleMatch[1] : url,
      snippet: "",
      credibility: 0.8,
      retrievedAt: new Date().toISOString(),
    });
  });

  return citations;
}
