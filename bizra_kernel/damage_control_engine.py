import os
import re
import shlex
from typing import Dict, List, Any


class DamageControlEngine:
    """
    Elite damage control engine with zero-trust pre-execution checks.

    Features:
    - Pattern-based command blocking
    - Path protection (zero-access, read-only, no-delete)
    - Safety SNR factor for downstream scoring
    """

    def __init__(self) -> None:
        self.zero_access_paths = self._normalize_paths([
            "~/.ssh",
            "~/.gnupg",
            "/root/.ssh",
            "/etc/shadow",
        ])
        self.read_only_paths = self._normalize_paths([
            "/etc",
            "/boot",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/var",
            "/proc",
            "/sys",
        ])
        self.no_delete_paths = self._normalize_paths([
            "/",
            "/etc",
            "/boot",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/var",
            "/proc",
            "/sys",
        ])

    def _normalize_paths(self, paths: List[str]) -> List[str]:
        normalized = []
        for path in paths:
            expanded = os.path.expanduser(path)
            normalized.append(os.path.abspath(expanded))
        return normalized

    def _path_is_within(self, path: str, prefixes: List[str]) -> bool:
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix + os.sep):
                return True
        return False

    def check_path(self, path: str, operation: str = "read") -> Dict[str, Any]:
        normalized = os.path.abspath(os.path.expanduser(path))
        if self._path_is_within(normalized, self.zero_access_paths):
            return {"allowed": False, "reason": "zero-access path"}

        if operation in ("write", "delete") and self._path_is_within(normalized, self.read_only_paths):
            return {"allowed": False, "reason": "read-only path"}

        if operation == "delete" and self._path_is_within(normalized, self.no_delete_paths):
            return {"allowed": False, "reason": "no-delete path"}

        return {"allowed": True, "reason": "ok"}

    def _tokenize(self, command: str) -> List[str]:
        if not command:
            return []
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def _strip_sudo(self, tokens: List[str]) -> List[str]:
        if tokens and tokens[0] in ("sudo", "doas"):
            return tokens[1:]
        return tokens

    def _is_recursive_or_force_rm(self, tokens: List[str]) -> bool:
        tokens = self._strip_sudo(tokens)
        if not tokens or tokens[0] != "rm":
            return False
        for token in tokens[1:]:
            if not token.startswith("-"):
                continue
            if token in ("--recursive", "--force"):
                return True
            if "r" in token or "f" in token:
                return True
        return False

    def _is_chmod_777(self, tokens: List[str]) -> bool:
        tokens = self._strip_sudo(tokens)
        if not tokens or tokens[0] != "chmod":
            return False
        return any(token.startswith("777") for token in tokens[1:])

    def _is_unqualified_sql_delete(self, text: str) -> bool:
        match = re.search(r"\bdelete\s+from\s+\w+", text, re.IGNORECASE)
        if not match:
            return False
        tail = text[match.end():]
        return "where" not in tail.lower()

    def _check_path_tokens(self, tokens: List[str], operation: str) -> List[str]:
        blocked = []
        for token in tokens:
            if token.startswith("-"):
                continue
            if token in ("rm", "chmod", "sudo", "doas"):
                continue
            if token.startswith("/") or token.startswith("~"):
                verdict = self.check_path(token, operation=operation)
                if not verdict["allowed"]:
                    blocked.append(verdict["reason"])
        return blocked

    def evaluate_command(self, command: str) -> Dict[str, Any]:
        tokens = self._tokenize(command)
        command_lower = command.lower()
        blocked_reasons: List[str] = []

        if self._is_recursive_or_force_rm(tokens):
            blocked_reasons.append("rm with recursive or force flags")

        if self._is_chmod_777(tokens):
            blocked_reasons.append("chmod 777 is unsafe")

        if self._is_unqualified_sql_delete(command_lower):
            blocked_reasons.append("unqualified sql delete")

        tokens_no_sudo = self._strip_sudo(tokens)
        if tokens_no_sudo and tokens_no_sudo[0] == "rm":
            blocked_reasons.extend(self._check_path_tokens(tokens_no_sudo, operation="delete"))
        if tokens_no_sudo and tokens_no_sudo[0] == "chmod":
            blocked_reasons.extend(self._check_path_tokens(tokens_no_sudo, operation="write"))

        allowed = not blocked_reasons
        safety_snr = 1.0 if allowed else 0.5

        return {
            "allowed": allowed,
            "blocked": blocked_reasons,
            "safety_snr": safety_snr,
        }

    def evaluate_text(self, *segments: str) -> Dict[str, Any]:
        combined = " ".join([segment for segment in segments if segment])
        return self.evaluate_command(combined)
