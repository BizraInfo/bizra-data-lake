/**
 * BIZRA Self-Healing PostToolUse Hook
 *
 * Detects errors from tool executions and triggers self-healing recovery.
 * Security violations are NEVER auto-recovered.
 *
 * Created: 2026-02-01 | BIZRA Remediation v2.2.1
 *
 * @type {import('@anthropic/claude-code').Hook}
 */

// Security violations that NEVER auto-recover
const SECURITY_VIOLATIONS = new Set([
  "SANDBOX_VIOLATION",
  "REJECT_SIGNATURE",
  "REJECT_POLICY_MISMATCH",
  "REJECT_IHSAN_BELOW_MIN",
  "REJECT_SNR_BELOW_MIN",
  "IHSAN_BELOW_MIN",
  "SNR_BELOW_MIN",
  "REJECT_FATE_VIOLATION",
  "REJECT_INVARIANT_FAILED",
]);

// Error patterns for classification
const ERROR_PATTERNS = {
  // Security (escalate immediately)
  security: [
    /SANDBOX_VIOLATION/i,
    /REJECT_SIGNATURE/i,
    /REJECT_POLICY_MISMATCH/i,
    /REJECT_IHSAN_BELOW_MIN/i,
    /REJECT_SNR_BELOW_MIN/i,
    /REJECT_FATE_VIOLATION/i,
    /unauthorized|forbidden/i,
  ],

  // Recoverable - missing dependency
  missingDep: [
    /ModuleNotFoundError.*No module named/,
    /ImportError.*cannot import name/,
    /command not found/i,
  ],

  // Recoverable - connection issues
  connection: [
    /ConnectionRefusedError/,
    /ConnectionResetError/,
    /ECONNREFUSED/,
    /ETIMEDOUT/,
    /timeout/i,
  ],

  // Recoverable - transient
  transient: [
    /temporary failure/i,
    /try again/i,
    /rate limit/i,
  ],

  // Not recoverable - escalate
  escalate: [
    /PermissionError/,
    /FileNotFoundError/,
    /MemoryError/,
    /fatal/i,
  ],
};

/**
 * Check if output contains a security violation
 * @param {string} output - Tool output to check
 * @returns {boolean}
 */
function isSecurityViolation(output) {
  for (const code of SECURITY_VIOLATIONS) {
    if (output.includes(code)) {
      return true;
    }
  }
  return ERROR_PATTERNS.security.some((pattern) => pattern.test(output));
}

/**
 * Classify the error type
 * @param {string} output - Tool output to classify
 * @returns {{type: string, recoverable: boolean, action: string}}
 */
function classifyError(output) {
  if (isSecurityViolation(output)) {
    return {
      type: "security",
      recoverable: false,
      action: "ESCALATE - Security violation detected. DO NOT retry.",
    };
  }

  if (ERROR_PATTERNS.missingDep.some((p) => p.test(output))) {
    return {
      type: "missing_dependency",
      recoverable: true,
      action: "Consider installing the missing package before retrying.",
    };
  }

  if (ERROR_PATTERNS.connection.some((p) => p.test(output))) {
    return {
      type: "connection",
      recoverable: true,
      action: "Retry after a brief delay. Check if service is running.",
    };
  }

  if (ERROR_PATTERNS.transient.some((p) => p.test(output))) {
    return {
      type: "transient",
      recoverable: true,
      action: "Retry with exponential backoff.",
    };
  }

  if (ERROR_PATTERNS.escalate.some((p) => p.test(output))) {
    return {
      type: "escalate",
      recoverable: false,
      action: "ESCALATE - Human intervention may be required.",
    };
  }

  return {
    type: "unknown",
    recoverable: false,
    action: "Unknown error. Review output carefully.",
  };
}

/**
 * Extract error message from output
 * @param {string} output
 * @returns {string}
 */
function extractErrorMessage(output) {
  // Look for common error patterns
  const patterns = [
    /Error: (.+)/i,
    /Exception: (.+)/i,
    /Traceback[\s\S]*?(\w+Error: .+)/,
    /fatal: (.+)/i,
    /"message":\s*"([^"]+)"/,
  ];

  for (const pattern of patterns) {
    const match = output.match(pattern);
    if (match) {
      return match[1].trim();
    }
  }

  // Return first non-empty line as fallback
  const lines = output.split("\n").filter((l) => l.trim());
  return lines[0] || "Unknown error";
}

/**
 * PostToolUse hook handler
 * @param {Object} context - Hook context
 * @returns {Promise<Object>}
 */
async function postToolUse(context) {
  const { tool_name, output, exit_code } = context;

  // Only process if there's an error
  const hasError =
    exit_code !== 0 ||
    /error|exception|fail|traceback/i.test(output) ||
    isSecurityViolation(output);

  if (!hasError) {
    return { continue: true };
  }

  // Classify the error
  const classification = classifyError(output);
  const errorMessage = extractErrorMessage(output);

  // Build response message
  const response = {
    continue: true,
    message: null,
  };

  // Security violations block further action
  if (classification.type === "security") {
    response.continue = false;
    response.message = `
SECURITY VIOLATION DETECTED

Tool: ${tool_name}
Type: ${classification.type}
Error: ${errorMessage}

Action: ${classification.action}

This error type is flagged as a security violation and cannot be auto-recovered.
Please review the error and take appropriate action.
`.trim();
    return response;
  }

  // For recoverable errors, provide guidance
  if (classification.recoverable) {
    response.message = `
Self-Healing Advisory

Tool: ${tool_name}
Type: ${classification.type}
Error: ${errorMessage}

Recommended Action: ${classification.action}
`.trim();
  }

  return response;
}

// Export for Claude Code hook system
module.exports = {
  name: "self-healing-hook",
  version: "2.2.1",
  description: "BIZRA Self-Healing PostToolUse Hook",
  events: ["PostToolUse"],
  handler: postToolUse,

  // Utility exports for testing
  isSecurityViolation,
  classifyError,
  extractErrorMessage,
};
