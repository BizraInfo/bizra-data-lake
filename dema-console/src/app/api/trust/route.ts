import { db } from "@/lib/db";
import { UpdateTrustStateSchema } from "@/lib/api/schemas";
import { success, badRequest, notFound, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/trust
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const trustState = await db.trustState.findFirst({
      orderBy: { updatedAt: "desc" },
    });

    if (!trustState) {
      return success({
        level: "visitor",
        score: 0,
        principalName: null,
        lastVerified: null,
      });
    }

    return success(trustState);
  } catch (error) {
    return internalError("Failed to fetch trust state");
  }
}

// PUT /api/trust
export async function PUT(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = UpdateTrustStateSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { id, ...updates } = parsed.data;

    const existing = await db.trustState.findUnique({ where: { id } });
    if (!existing) {
      return notFound("TrustState");
    }

    const trustState = await db.trustState.update({
      where: { id },
      data: {
        ...(updates.principalId !== undefined ? { principalId: updates.principalId } : {}),
        ...(updates.principalName !== undefined ? { principalName: updates.principalName } : {}),
        ...(updates.level !== undefined ? { level: updates.level } : {}),
        ...(updates.score !== undefined ? { score: updates.score } : {}),
        ...(updates.lastVerified !== undefined
          ? { lastVerified: updates.lastVerified ? new Date(updates.lastVerified) : null }
          : {}),
      },
    });

    return success(trustState);
  } catch (error) {
    return internalError("Failed to update trust state");
  }
}
