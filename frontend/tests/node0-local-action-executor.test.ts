import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useNode0LocalActionExecutor } from "../src/hooks/use-node0-local-action-executor";
import {
  createNode0ActionIntent,
  recordNode0LocalActionReceipt,
} from "@/hooks/use-sovereign-api";

vi.mock("@/hooks/use-sovereign-api", () => ({
  createNode0ActionIntent: vi.fn(),
  recordNode0LocalActionReceipt: vi.fn(),
}));

const createIntentMock = vi.mocked(createNode0ActionIntent);
const recordReceiptMock = vi.mocked(recordNode0LocalActionReceipt);

async function sha256Hex(value: string): Promise<string> {
  const digest = await window.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("useNode0LocalActionExecutor", () => {
  it("executes the backend-validated copy target and records its hash", async () => {
    const targetHash = await sha256Hex("backend validated target");
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    createIntentMock.mockResolvedValue({
      action_id: "action-1",
      accepted: true,
      status: "ready_for_user_handoff",
      action_type: "copy_text",
      label: "Copy",
      target: "backend validated target",
      target_preview: "backend validated target",
      target_hash: targetHash,
      execution_mode: "client_handoff_only",
      handoff_method: "clipboard_write",
      server_executed: false,
      requires_user_confirmation: true,
      truth_label: "[ENFORCEMENT: WIRED]",
      source_label: "user_confirmed_action_intent",
      next_action: "confirm action in the local browser",
    });
    recordReceiptMock.mockResolvedValue({
      receipt_id: "receipt-1",
      action_id: "action-1",
      recorded: true,
      status: "executed",
      action_type: "copy_text",
      execution_channel: "browser_client",
      server_executed: false,
      target_preview: "backend validated target",
      target_hash: targetHash,
      recorded_at: "2026-04-30T12:00:00Z",
      truth_label: "[ENFORCEMENT: WIRED]",
      source_label: "browser_client_local_action",
      next_action: "inspect receipt or submit next mission",
    });

    const { result } = renderHook(() => useNode0LocalActionExecutor());

    await act(async () => {
      await result.current.execute({
        actionType: "copy_text",
        target: "unvalidated caller target",
        label: "Copy",
        userGestureConfirmed: true,
      });
    });

    expect(writeText).toHaveBeenCalledWith("backend validated target");
    expect(recordReceiptMock).toHaveBeenCalledWith({
      actionId: "action-1",
      actionType: "copy_text",
      result: "executed",
      targetPreview: "backend validated target",
      targetHash,
      error: "",
    });
    expect(result.current.status).toBe("receipted");
    expect(result.current.lastReceipt?.receipt_id).toBe("receipt-1");
  });

  it("fails closed before execution when the backend target hash mismatches", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    createIntentMock.mockResolvedValue({
      action_id: "action-2",
      accepted: true,
      status: "ready_for_user_handoff",
      action_type: "copy_text",
      label: "Copy",
      target: "tampered target",
      target_preview: "tampered target",
      target_hash: "0".repeat(64),
      execution_mode: "client_handoff_only",
      handoff_method: "clipboard_write",
      server_executed: false,
      requires_user_confirmation: true,
      truth_label: "[ENFORCEMENT: WIRED]",
      source_label: "user_confirmed_action_intent",
      next_action: "confirm action in the local browser",
    });

    const { result } = renderHook(() => useNode0LocalActionExecutor());

    await act(async () => {
      await result.current.execute({
        actionType: "copy_text",
        target: "tampered target",
        userGestureConfirmed: true,
      });
    });

    expect(writeText).not.toHaveBeenCalled();
    expect(recordReceiptMock).not.toHaveBeenCalled();
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("Validated target hash mismatch");
  });

  it("requires an explicit user gesture marker before preparing intent", async () => {
    const { result } = renderHook(() => useNode0LocalActionExecutor());

    await act(async () => {
      await result.current.execute({
        actionType: "copy_text",
        target: "no gesture",
      });
    });

    expect(createIntentMock).not.toHaveBeenCalled();
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("Explicit user gesture required");
  });
});
