import { db } from "@/lib/db";
import { CreateResourceSchema } from "@/lib/api/schemas";
import { success, created, badRequest, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/resources?type=service&status=active&limit=50
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const type = searchParams.get("type");
    const status = searchParams.get("status");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    const resources = await db.resource.findMany({
      where: {
        ...(type ? { type } : {}),
        ...(status ? { status } : {}),
      },
      orderBy: { createdAt: "desc" },
      take: limit,
    });

    return success(resources);
  } catch (error) {
    return internalError("Failed to fetch resources");
  }
}

// POST /api/resources
export async function POST(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = CreateResourceSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { name, type, path, metadata } = parsed.data;

    const resource = await db.resource.create({
      data: {
        name,
        type,
        path: path ?? null,
        metadata: metadata ? JSON.stringify(metadata) : null,
      },
    });

    return created(resource);
  } catch (error) {
    return internalError("Failed to register resource");
  }
}

// DELETE /api/resources?id=<resourceId>
export async function DELETE(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");

    if (!id) {
      return badRequest("Missing required query param: id");
    }

    await db.resource.delete({ where: { id } });
    return success({ deleted: true });
  } catch (error) {
    return internalError("Failed to delete resource");
  }
}
