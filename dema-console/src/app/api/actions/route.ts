import { db } from "@/lib/db";
import { CreateActionLogSchema } from "@/lib/api/schemas";
import { success, created, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/actions?mode=research&status=completed&limit=50
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const mode = searchParams.get("mode");
    const status = searchParams.get("status");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    const actions = await db.actionLog.findMany({
      where: {
        ...(mode ? { mode } : {}),
        ...(status ? { status } : {}),
      },
      orderBy: { createdAt: "desc" },
      take: limit,
    });

    return success(actions);
  } catch (error) {
    return internalError("Failed to fetch action log");
  }
}

// POST /api/actions
export async function POST(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = CreateActionLogSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { mode, action, description, permission, evidence } = parsed.data;

    const entry = await db.actionLog.create({
      data: {
        mode,
        action,
        description: description ?? null,
        permission: permission ?? "explicit",
        evidence: evidence ?? null,
      },
    });

    return created(entry);
  } catch (error) {
    return internalError("Failed to log action");
  }
}
