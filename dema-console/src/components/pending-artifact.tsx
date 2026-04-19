// Shared component for artifacts whose source lives on a different
// branch (cycle-8) and hasn't merged to PR #28 yet.
//
// Honest status: artifact is drafted, not yet on this branch.
// Renders a link to the source-of-truth location + a brief summary.
// NO_SHADOW_STATE: no invented content body, no placeholder prose.

import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface PendingArtifactProps {
  title: string;
  summary: string;
  sourcePath: string;
  sourceBranch: string;
  repoUrl?: string;
}

const DEFAULT_REPO_URL = "https://github.com/BizraInfo/bizra-data-lake";

export function PendingArtifact({
  title,
  summary,
  sourcePath,
  sourceBranch,
  repoUrl = DEFAULT_REPO_URL,
}: PendingArtifactProps) {
  const sourceUrl = `${repoUrl}/blob/${sourceBranch}/${sourcePath}`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-2xl mx-auto px-6 py-12 space-y-6">
        <header className="space-y-2">
          <Link
            href="/landing"
            className="text-xs text-muted-foreground hover:text-foreground underline decoration-dotted"
          >
            ← DEMA landing
          </Link>
          <h1 className="text-xl font-semibold">{title}</h1>
        </header>

        <Card className="border-border/50 bg-muted/10">
          <CardHeader className="pb-2">
            <Badge variant="outline" className="w-fit">Draft — pending merge</Badge>
          </CardHeader>
          <CardContent className="p-4 pt-0 space-y-3 text-sm text-muted-foreground">
            <p>{summary}</p>

            <p>
              The full text lives on the{" "}
              <code className="font-mono bg-muted/30 px-1 py-0.5 rounded">{sourceBranch}</code>{" "}
              branch at{" "}
              <code className="font-mono bg-muted/30 px-1 py-0.5 rounded break-all">
                {sourcePath}
              </code>
              . This route will render inline once the branch merges to main.
            </p>

            <p className="text-xs">
              Until then,{" "}
              <a
                className="underline decoration-dotted hover:text-foreground"
                href={sourceUrl}
                target="_blank"
                rel="noreferrer"
              >
                read it on GitHub
              </a>
              . No placeholder prose is shown here — NO_SHADOW_STATE applies to
              the public face too.
            </p>
          </CardContent>
        </Card>

        <footer className="text-center text-[11px] text-muted-foreground/70 italic pt-8 border-t border-border/50">
          <p>Close it. Prove it. Reveal it.</p>
        </footer>
      </div>
    </div>
  );
}
