"""CLI for executing and displaying canonical loop proofs."""

import json
import subprocess
import sys
import time
from pathlib import Path

from core.proof_engine.loop_proof import execute_loop_proof


def gather_evidence(question: str) -> tuple[str, list[str], str]:
    """Gather local evidence for the proof mission."""
    repo = Path("/data/bizra/repos/bizra-data-lake")
    evidence_parts = []
    refs = []

    # Git evidence
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep=spearpoint", "-n", "5"],
            capture_output=True, text=True, timeout=10, cwd=str(repo),
        )
        if r.stdout.strip():
            evidence_parts.append(r.stdout.strip())
            refs.append("git-log:spearpoint")
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["git", "log", "--format=%H %s", "-1", "b08f2208"],
            capture_output=True, text=True, timeout=10, cwd=str(repo),
        )
        if r.stdout.strip():
            evidence_parts.append(r.stdout.strip())
            refs.append("git-show:b08f2208")
    except Exception:
        pass

    # File evidence
    for f in ["core/zpk/kernel.py", "core/proof_engine/fate_gate.py", "core/proof_engine/sat_validator.py"]:
        if (repo / f).exists():
            refs.append(f"file:{f}")

    evidence_text = "\n".join(evidence_parts)

    # Call PAT model
    import urllib.request
    try:
        payload = json.dumps({
            "model": "gemma4:e4b",
            "messages": [
                {"role": "system", "content": "Answer using only the evidence provided. Be precise."},
                {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}\n\nANSWER:"},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            msg = data.get("message", {})
            answer = msg.get("content", "") or msg.get("thinking", "")
    except Exception as e:
        answer = f"PAT unavailable: {e}"

    confidence = "high" if len(refs) >= 3 else "medium"
    return answer, refs, confidence


def main():
    mission = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is the Spearpoint seal (commit b08f2208) and why does it matter?"
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = Path(f"/data/bizra/logs/loop-proof-{timestamp}.json")

    print(f"=== BIZRA Canonical Loop Proof ===")
    print(f"Mission: {mission}")
    print(f"Output:  {output}")
    print()

    print("Gathering evidence...", flush=True)
    answer, refs, confidence = gather_evidence(mission)

    print(f"PAT: {len(answer)} chars, {len(refs)} refs, confidence={confidence}")
    print("Executing loop proof...", flush=True)

    t0 = time.time()
    proof = execute_loop_proof(
        mission=mission,
        pat_answer=answer,
        evidence_refs=refs,
        confidence=confidence,
        output_path=output,
    )
    elapsed = time.time() - t0

    # Display
    print()
    for step in proof.steps:
        verdict = step.evidence.get("verdict", "")
        status_icon = "✓" if step.status in ("ok", "completed", "pass") else "✗" if step.status in ("fail", "blocked") else "○"
        print(f"  {status_icon} [{step.seq}] {step.actor:40s} {step.action:30s} {step.status}")
        if verdict:
            print(f"         verdict={verdict}  ihsan={step.evidence.get('ihsan_score', '')}")

    print()
    print(f"Chain valid:   {proof.verify_chain()}")
    print(f"Manifest hash: {proof.manifest_hash[:24]}...")
    print(f"Steps:         {len(proof.steps)}")
    print(f"Elapsed:       {elapsed:.1f}s")
    print(f"FATE verdict:  {proof.fate_result.get('verdict', {}).get('verdict', 'N/A')}")
    print(f"Artifact:      {output}")

    sys.exit(0 if proof.fate_result.get("passed") else 1)


if __name__ == "__main__":
    main()
