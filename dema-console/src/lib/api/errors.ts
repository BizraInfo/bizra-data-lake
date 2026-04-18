// ═══════════════════════════════════════════════════════════════
// DEMA — Typed API Error Factory
// Consistent error shape across all routes.
// ═══════════════════════════════════════════════════════════════

import { NextResponse } from "next/server";
import { ZodError } from "zod";

export interface DEMA_API_Error {
  success: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

/**
 * Creates a standardized error response.
 * Every error includes a machine-readable `code` and human-readable `message`.
 */
export function apiError(
  code: string,
  message: string,
  status: number,
  details?: unknown,
) {
  return NextResponse.json(
    {
      success: false,
      error: { code, message, ...(details !== undefined && { details }) },
    },
    { status },
  );
}

export function badRequest(message: string, details?: unknown) {
  return apiError("BAD_REQUEST", message, 400, details);
}

export function unauthorized(message = "Authentication required") {
  return apiError("UNAUTHORIZED", message, 401);
}

export function forbidden(message = "Insufficient permissions") {
  return apiError("FORBIDDEN", message, 403);
}

export function notFound(resource: string) {
  return apiError("NOT_FOUND", `${resource} not found`, 404);
}

export function conflict(message: string) {
  return apiError("CONFLICT", message, 409);
}

export function rateLimited(retryAfterSeconds = 60) {
  return NextResponse.json(
    {
      success: false,
      error: { code: "RATE_LIMITED", message: "Too many requests. Slow down." },
    },
    {
      status: 429,
      headers: { "Retry-After": String(retryAfterSeconds) },
    },
  );
}

export function internalError(message = "Internal server error", details?: unknown) {
  if (process.env.NODE_ENV === "development") {
    console.error("[DEMA API]", message, details);
  }
  return apiError("INTERNAL_ERROR", message, 500);
}

export function validationError(err: ZodError) {
  return NextResponse.json(
    {
      success: false,
      error: {
        code: "VALIDATION_ERROR",
        message: "Input validation failed",
        details: err.issues.map((i) => ({
          path: i.path.join("."),
          message: i.message,
          code: i.code,
        })),
      },
    },
    { status: 400 },
  );
}

export function success<T>(data: T, status = 200) {
  return NextResponse.json({ success: true, data }, { status });
}

export function created<T>(data: T) {
  return NextResponse.json({ success: true, data }, { status: 201 });
}
