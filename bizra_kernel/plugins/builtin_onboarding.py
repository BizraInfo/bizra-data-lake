"""Builtin onboarding plugin scaffold.

This is intentionally lightweight: it proves the plugin pathway and defines
the contract for full Week-1 onboarding implementation.
"""

from __future__ import annotations

from typing import Any, Dict


class BuiltinOnboardingPlugin:
    name = "onboarding"

    def run(self, action: str, payload: Dict[str, Any], context) -> Dict[str, Any]:
        if action not in {"start", "status"}:
            return {
                "ok": False,
                "plugin": self.name,
                "action": action,
                "error": f"Unsupported action: {action}",
            }

        architect_name = "Architect"
        node_id = None
        identity = getattr(context, "identity", None)
        if identity is not None:
            architect = getattr(identity, "architect", None)
            if architect is not None:
                architect_name = getattr(architect, "name", architect_name)
            node_id = getattr(identity, "node_id", None)

        return {
            "ok": True,
            "plugin": self.name,
            "action": action,
            "message": "Onboarding plugin scaffold is active and ready for full implementation.",
            "identity": {
                "architect": architect_name,
                "node_id": node_id,
            },
            "checklist": [
                "Generate keypair",
                "Mint identity",
                "Activate PAT+SAT roster",
                "Store credentials",
                "Run first query",
            ],
            "next_hook": "Replace builtin plugin with production onboarding adapter.",
        }


def build_plugin() -> BuiltinOnboardingPlugin:
    return BuiltinOnboardingPlugin()

