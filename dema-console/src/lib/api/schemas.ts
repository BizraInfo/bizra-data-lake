// ═══════════════════════════════════════════════════════════════
// DEMA — Zod Validation Schemas
// Every API contract is a typed, validated, enforceable boundary.
// ═══════════════════════════════════════════════════════════════

import { z } from "zod";

// ─── Receipt ──────────────────────────────────────────────────

export const CreateReceiptSchema = z.object({
  missionId: z.string().optional(),
  type: z.enum(["action", "verification", "delegation", "completion", "error"]),
  title: z.string().min(1).max(256),
  description: z.string().max(2048).optional(),
  evidence: z.string().max(8192).optional(),
  expiresAt: z.string().datetime().optional(),
});

export type CreateReceiptInput = z.infer<typeof CreateReceiptSchema>;

// ─── Manifest ─────────────────────────────────────────────────

export const ManifestArtifactSchema = z.object({
  name: z.string().min(1).max(256),
  type: z.string().min(1).max(64),
  path: z.string().max(512).optional(),
  hash: z.string().max(128).optional(),
});

export const CreateManifestSchema = z.object({
  missionId: z.string().optional(),
  title: z.string().min(1).max(256),
  description: z.string().max(4096).optional(),
  artifacts: z.array(ManifestArtifactSchema).max(100).optional(),
});

export type CreateManifestInput = z.infer<typeof CreateManifestSchema>;

// ─── Resource ─────────────────────────────────────────────────

export const CreateResourceSchema = z.object({
  name: z.string().min(1).max(256),
  type: z.enum(["file", "url", "credential", "service", "knowledge", "browser", "terminal"]),
  path: z.string().max(1024).optional(),
  metadata: z.record(z.unknown()).optional(),
});

export type CreateResourceInput = z.infer<typeof CreateResourceSchema>;

// ─── Memory Entry ─────────────────────────────────────────────

export const CreateMemoryEntrySchema = z.object({
  category: z.enum(["preference", "context", "knowledge", "poi"]),
  title: z.string().min(1).max(256),
  content: z.string().min(1).max(8192),
  confidence: z.number().min(0).max(1).optional().default(0.5),
  relevance: z.number().min(0).max(1).optional().default(0.5),
  source: z.string().max(256).optional(),
  tags: z.array(z.string().max(64)).max(20).optional(),
});

export type CreateMemoryEntryInput = z.infer<typeof CreateMemoryEntrySchema>;

// ─── Action Log ───────────────────────────────────────────────

export const CreateActionLogSchema = z.object({
  mode: z.enum(["browser", "computer", "code", "research"]),
  action: z.string().min(1).max(256),
  description: z.string().max(2048).optional(),
  permission: z.enum(["auto", "explicit", "denied"]).optional().default("explicit"),
  evidence: z.string().max(4096).optional(),
});

export type CreateActionLogInput = z.infer<typeof CreateActionLogSchema>;

// ─── Trust State ──────────────────────────────────────────────

export const UpdateTrustStateSchema = z.object({
  id: z.string().min(1),
  principalId: z.string().max(128).optional(),
  principalName: z.string().max(128).optional(),
  level: z.enum(["visitor", "citizen", "operator", "admin"]).optional(),
  score: z.number().int().min(0).max(100).optional(),
  lastVerified: z.string().datetime().nullable().optional(),
});

export type UpdateTrustStateInput = z.infer<typeof UpdateTrustStateSchema>;

// ─── Ask (LLM Request) ────────────────────────────────────────

export const AskRequestSchema = z.object({
  message: z.string().min(1).max(4096),
  mode: z.enum(["ask", "research"]).optional().default("ask"),
  sessionId: z.string().optional(),
});

export type AskRequestInput = z.infer<typeof AskRequestSchema>;
