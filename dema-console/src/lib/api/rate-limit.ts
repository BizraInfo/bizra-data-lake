// ═══════════════════════════════════════════════════════════════
// DEMA — In-Memory Rate Limiter
// Token bucket with sliding window. No external deps.
// ═══════════════════════════════════════════════════════════════

interface Bucket {
  tokens: number;
  lastRefill: number;
}

const buckets = new Map<string, Bucket>();

const DEFAULT_RATE = 30;       // requests per window
const DEFAULT_WINDOW = 60_000; // 1 minute in ms
const BUCKET_MAX = DEFAULT_RATE * 2;
const CLEANUP_INTERVAL = 120_000; // purge stale entries every 2 min

// Periodic cleanup to prevent memory leak
if (typeof globalThis !== "undefined") {
  setInterval(() => {
    const now = Date.now();
    for (const [key, bucket] of buckets) {
      if (now - bucket.lastRefill > DEFAULT_WINDOW * 3) {
        buckets.delete(key);
      }
    }
  }, CLEANUP_INTERVAL);
}

/**
 * Check rate limit for a given key (e.g., IP address or session ID).
 * Returns true if the request is allowed, false if rate-limited.
 * Thread-safe for single-node deployments (sufficient for SQLite-backed app).
 */
export function checkRateLimit(
  key: string,
  rate = DEFAULT_RATE,
  windowMs = DEFAULT_WINDOW,
): { allowed: boolean; remaining: number; resetIn: number } {
  const now = Date.now();
  let bucket = buckets.get(key);

  if (!bucket) {
    bucket = { tokens: rate, lastRefill: now };
    buckets.set(key, bucket);
  }

  // Refill tokens based on elapsed time
  const elapsed = now - bucket.lastRefill;
  const refillTokens = Math.floor((elapsed / windowMs) * rate);
  if (refillTokens > 0) {
    bucket.tokens = Math.min(BUCKET_MAX, bucket.tokens + refillTokens);
    bucket.lastRefill = now;
  }

  if (bucket.tokens <= 0) {
    const resetIn = Math.ceil((windowMs - (now - bucket.lastRefill)) / 1000);
    return { allowed: false, remaining: 0, resetIn: Math.max(1, resetIn) };
  }

  bucket.tokens -= 1;
  return {
    allowed: true,
    remaining: bucket.tokens,
    resetIn: Math.ceil(windowMs / 1000),
  };
}

/**
 * Extract client IP from request headers.
 * Works behind proxies (X-Forwarded-For, CF-Connecting-IP).
 */
export function getClientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    return forwarded.split(",")[0].trim();
  }
  const cfIp = request.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp;
  return "unknown";
}
