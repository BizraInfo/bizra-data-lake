import { db } from "@/lib/db";
import { CreateMemoryEntrySchema } from "@/lib/api/schemas";
import { success, created, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/memory?category=preference&limit=50
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    const entries = await db.memoryEntry.findMany({
      where: category ? { category } : undefined,
      orderBy: { updatedAt: "desc" },
      take: limit,
    });

    return success(entries);
  } catch (error) {
    return internalError("Failed to fetch memory entries");
  }
}

// POST /api/memory
export async function POST(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = CreateMemoryEntrySchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { category, title, content, confidence, relevance, source, tags } = parsed.data;

    const entry = await db.memoryEntry.create({
      data: {
        category,
        title,
        content,
        confidence: confidence ?? 0.5,
        relevance: relevance ?? 0.5,
        source: source ?? null,
        tags: tags ? JSON.stringify(tags) : null,
      },
    });

    return created(entry);
  } catch (error) {
    return internalError("Failed to add memory entry");
  }
}
