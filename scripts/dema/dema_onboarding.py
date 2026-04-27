"""Dema onboarding — capture preferred name, languages, persona, consent.

Truth label: MEASURED (writes a real local profile + receipt).

Usage:
    python scripts/dema/dema_onboarding.py --init
    python scripts/dema/dema_onboarding.py --init --root <path>

Inputs (env, optional):
    DEMA_PREFERRED_NAME, DEMA_MOTHER_LANGUAGE, DEMA_WORK_LANGUAGE,
    DEMA_PERSONA_TONE, DEMA_MEMORY_CONSENT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import (  # noqa: E402
    DailyLog,
    DailyLogEntry,
    DemaReceipt,
    ProfileStore,
    ReceiptWriter,
)

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"


def init(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    profile_store = ProfileStore(root)
    profile = profile_store.init_from_env_or_defaults()
    profile_path = profile_store.save(profile)

    receipt = DemaReceipt(
        action="dema.onboarding.init",
        truth_label="MEASURED",
        touched_paths=[str(profile_path)],
        not_touched_paths=[
            "network",
            "desktop",
            "MEMORY.md",
            "docs/canon/",
            "frontend/public/",
        ],
        approval_required=False,
        approval_status="n/a",
        payload={
            "preferred_name": profile.preferred_name,
            "mother_language": profile.mother_language,
            "work_language": profile.work_language,
            "persona_tone": profile.persona_tone,
            "memory_consent": profile.memory_consent,
            "schema_version": profile.schema_version,
        },
    )
    rid, receipt_path = ReceiptWriter(root).write(receipt)

    DailyLog(root).append(
        DailyLogEntry(
            timestamp=profile.created_at,
            kind="onboarding",
            summary=f"profile initialized for {profile.preferred_name}",
            receipt_id=rid,
            metadata={"persona_tone": profile.persona_tone},
        )
    )

    return {
        "ok": True,
        "profile_path": str(profile_path),
        "receipt_id": rid,
        "receipt_path": str(receipt_path),
        "truth_label": "MEASURED",
        "next_step": "Run scripts/dema/dema_status.py --json to see Current/Ideal/Gap/Next.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="Initialize Dema profile.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    if not args.init:
        parser.error("Pass --init to initialize the Dema profile.")

    result = init(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
