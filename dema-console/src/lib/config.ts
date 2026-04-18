// ═══════════════════════════════════════════════════════════════
// DEMA — Environment Configuration
// Centralized config with type safety. No magic strings.
// ═══════════════════════════════════════════════════════════════

export const env = {
  // Application
  NODE_ENV: process.env.NODE_ENV ?? "development",
  APP_VERSION: "0.1.0",
  APP_PHASE: "R1",

  // Database
  DATABASE_URL: process.env.DATABASE_URL ?? "file:./db/custom.db",

  // Rate Limiting
  RATE_LIMIT_WINDOW_MS: 60_000,
  RATE_LIMIT_MAX_REQUESTS: 30,

  // Ask LLM
  ASK_RATE_LIMIT: 15,
  ASK_MAX_CONVERSATION_TURNS: 30,

  // API
  API_DEFAULT_PAGE_SIZE: 50,
  API_MAX_PAGE_SIZE: 200,

  // Trust
  TRUST_DEFAULT_LEVEL: "visitor" as const,
  TRUST_MAX_SCORE: 100,
} as const;

export type Env = typeof env;
