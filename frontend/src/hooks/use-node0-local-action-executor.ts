"use client";

import { useCallback, useState } from "react";

import {
  createNode0ActionIntent,
  recordNode0LocalActionReceipt,
  type Node0ActionIntentResult,
  type Node0LocalActionReceipt,
} from "@/hooks/use-sovereign-api";

type LocalActionStatus = "idle" | "executing" | "receipted" | "error";
type LocalActionType = "copy_text" | "open_url";

interface ExecuteLocalActionArgs {
  actionType: LocalActionType;
  target: string;
  label?: string;
  userGestureConfirmed?: boolean;
}

async function sha256Hex(value: string): Promise<string> {
  if (!window.crypto?.subtle) {
    throw new Error("Browser crypto is unavailable");
  }
  const digest = await window.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function executeBrowserAction(
  actionType: LocalActionType,
  target: string,
): Promise<"executed" | "blocked"> {
  if (actionType === "copy_text") {
    const writer = navigator.clipboard?.writeText;
    if (!writer) {
      return "blocked";
    }
    await writer.call(navigator.clipboard, target);
    return "executed";
  }

  const opened = window.open(target, "_blank", "noopener,noreferrer");
  return opened ? "executed" : "blocked";
}

export function useNode0LocalActionExecutor() {
  const [status, setStatus] = useState<LocalActionStatus>("idle");
  const [lastIntent, setLastIntent] = useState<Node0ActionIntentResult | null>(null);
  const [lastReceipt, setLastReceipt] = useState<Node0LocalActionReceipt | null>(null);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async (args: ExecuteLocalActionArgs) => {
    setStatus("executing");
    setError(null);

    try {
      if (args.userGestureConfirmed !== true) {
        throw new Error("Explicit user gesture required");
      }
      const intent = await createNode0ActionIntent({
        actionType: args.actionType,
        target: args.target,
        label: args.label,
      });
      setLastIntent(intent);

      const computedHash = await sha256Hex(intent.target);
      if (computedHash !== intent.target_hash) {
        throw new Error("Validated target hash mismatch");
      }
      const result = await executeBrowserAction(intent.action_type, intent.target);
      const receipt = await recordNode0LocalActionReceipt({
        actionId: intent.action_id,
        actionType: intent.action_type,
        result,
        targetPreview: intent.target_preview,
        targetHash: intent.target_hash,
        error: result === "blocked" ? "Browser client blocked the local action" : "",
      });
      setLastReceipt(receipt);
      setStatus(result === "executed" ? "receipted" : "error");
      if (result === "blocked") {
        setError("Browser client blocked the local action");
      }
      return receipt;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Local action failed";
      setError(message);
      setStatus("error");
      return null;
    }
  }, []);

  return {
    status,
    lastIntent,
    lastReceipt,
    error,
    execute,
    clear: () => {
      setStatus("idle");
      setLastIntent(null);
      setLastReceipt(null);
      setError(null);
    },
  };
}
