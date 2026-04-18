import { db } from "@/lib/db";
import { CreateReceiptSchema } from "@/lib/api/schemas";
import { success, created, badRequest, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/receipts?status=pending&limit=50
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    const receipts = await db.receipt.findMany({
      where: status ? { status } : undefined,
      orderBy: { createdAt: "desc" },
      take: limit,
    });

    return success(receipts);
  } catch (error) {
    return internalError("Failed to fetch receipts");
  }
}

// POST /api/receipts
export async function POST(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = CreateReceiptSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { missionId, type, title, description, evidence, expiresAt } = parsed.data;

    const receipt = await db.receipt.create({
      data: {
        missionId: missionId ?? null,
        type,
        title,
        description: description ?? null,
        evidence: evidence ?? null,
        status: "pending",
        expiresAt: expiresAt ? new Date(expiresAt) : null,
      },
    });

    return created(receipt);
  } catch (error) {
    return internalError("Failed to create receipt");
  }
}
