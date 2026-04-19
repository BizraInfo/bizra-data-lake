import { db } from "@/lib/db";
import { CreateManifestSchema } from "@/lib/api/schemas";
import { success, created, validationError, internalError } from "@/lib/api/errors";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

// GET /api/manifests?status=active&limit=50
export async function GET(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status");
    const rawLimit = searchParams.get("limit");
    const limit = rawLimit ? Math.min(parseInt(rawLimit, 10) || 50, 200) : 50;

    const manifests = await db.manifest.findMany({
      where: status ? { status } : undefined,
      include: { artifacts: true },
      orderBy: { createdAt: "desc" },
      take: limit,
    });

    return success(manifests);
  } catch (error) {
    return internalError("Failed to fetch manifests");
  }
}

// POST /api/manifests
export async function POST(request: Request) {
  try {
    const { allowed } = checkRateLimit(getClientIp(request));
    if (!allowed) return internalError("Rate limited");

    const body = await request.json();
    const parsed = CreateManifestSchema.safeParse(body);

    if (!parsed.success) {
      return validationError(parsed.error);
    }

    const { missionId, title, description, artifacts } = parsed.data;

    const manifest = await db.manifest.create({
      data: {
        missionId: missionId ?? null,
        title,
        description: description ?? null,
        artifacts: artifacts
          ? {
              create: artifacts.map((a) => ({
                name: a.name,
                type: a.type,
                path: a.path ?? null,
                hash: a.hash ?? null,
              })),
            }
          : undefined,
      },
      include: { artifacts: true },
    });

    return created(manifest);
  } catch (error) {
    return internalError("Failed to create manifest");
  }
}
