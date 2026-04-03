# core/autopoietic_ci_trigger.py - Autopoietic CI Trigger
# Standing on Shoulders of Giants Protocol: CI/CD, GitHub Actions integration
# Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO_OWNER = os.getenv("REPO_OWNER", "")
REPO_NAME = os.getenv("REPO_NAME", "")

WORKFLOW_FILE = ".github/workflows/autopoietic.yml"
BUILD_TIMEOUT_SECONDS = 300


class AutopoieticCITrigger:
    def __init__(
        self,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.owner = owner or REPO_OWNER
        self.repo = repo or REPO_NAME
        self.token = token or GITHUB_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _compute_build_hash(self, payload: Dict[str, Any]) -> str:
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_workflow_runs(self, workflow_id: str) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API_URL}/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/runs"
        response = self.session.get(url, params={"per_page": 5})
        response.raise_for_status()
        return response.json().get("workflow_runs", [])

    def _trigger_workflow(
        self,
        workflow_id: str,
        ref: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{GITHUB_API_URL}/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches"
        payload = {
            "ref": ref,
            "inputs": inputs or {},
        }
        response = self.session.post(url, json=payload)
        
        if response.status_code == 204:
            return {
                "success": True,
                "workflow_id": workflow_id,
                "ref": ref,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        response.raise_for_status()
        return response.json()

    def _cancel_workflow_run(self, run_id: int) -> Dict[str, Any]:
        url = f"{GITHUB_API_URL}/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/cancel"
        response = self.session.post(url)
        response.raise_for_status()
        return {"success": True, "run_id": run_id}

    def _rerun_workflow(self, run_id: int) -> Dict[str, Any]:
        url = f"{GITHUB_API_URL}/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/rerun"
        response = self.session.post(url)
        response.raise_for_status()
        return {"success": True, "run_id": run_id}

    def check_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        runs = self._get_workflow_runs(workflow_id)
        
        if not runs:
            return {
                "status": "no_runs",
                "workflow_id": workflow_id,
            }
        
        latest_run = runs[0]
        return {
            "status": latest_run.get("status"),
            "conclusion": latest_run.get("conclusion"),
            "run_number": latest_run.get("run_number"),
            "head_branch": latest_run.get("head_branch"),
            "created_at": latest_run.get("created_at"),
        }

    def trigger_autopoietic_cycle(
        self,
        cycle_type: str = "full",
        branch: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        build_hash = self._compute_build_hash({
            "cycle_type": cycle_type,
            "branch": branch,
            "timestamp": time.time(),
            **(metadata or {}),
        })

        inputs = {
            "cycle_type": cycle_type,
            "build_hash": build_hash,
            "metadata": json.dumps(metadata or {}),
            "triggered_by": "autopoietic_ci_trigger",
        }

        result = self._trigger_workflow("autopoietic.yml", branch, inputs)

        return {
            "triggered": result.get("success", False),
            "build_hash": build_hash,
            "branch": branch,
            "cycle_type": cycle_type,
            "timestamp": result.get("timestamp"),
        }

    def is_build_stale(self, workflow_id: str, max_age_seconds: int = 3600) -> bool:
        runs = self._get_workflow_runs(workflow_id)
        
        if not runs:
            return True
        
        latest_run = runs[0]
        created_at = datetime.fromisoformat(
            latest_run.get("created_at", "").replace("Z", "+00:00")
        )
        
        now = datetime.now(timezone.utc)
        age_seconds = (now - created_at).total_seconds()
        
        return age_seconds > max_age_seconds

    def wait_for_completion(
        self,
        workflow_id: str,
        timeout: int = BUILD_TIMEOUT_SECONDS,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_workflow_status(workflow_id)
            
            if status.get("status") in ["completed", "success", "failure", "cancelled"]:
                return {
                    "completed": True,
                    "conclusion": status.get("conclusion"),
                    "run_number": status.get("run_number"),
                }
            
            time.sleep(poll_interval)
        
        return {
            "completed": False,
            "timeout": True,
            "elapsed_seconds": time.time() - start_time,
        }

    def trigger_with_retry(
        self,
        workflow_id: str,
        branch: str = "main",
        max_retries: int = 3,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                result = self._trigger_workflow(workflow_id, branch, inputs)
                return {
                    "success": True,
                    "attempt": attempt + 1,
                    "result": result,
                }
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
                time.sleep(2 ** attempt)
        
        return {
            "success": False,
            "error": "Max retries exceeded",
        }

    def get_build_metrics(self, workflow_id: str, limit: int = 10) -> Dict[str, Any]:
        runs = self._get_workflow_runs(workflow_id)
        runs = runs[:limit]
        
        total_duration = 0
        success_count = 0
        failure_count = 0
        
        for run in runs:
            duration = run.get("run_started_at") and run.get("updated_at")
            if duration:
                start = datetime.fromisoformat(
                    run.get("run_started_at", "").replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    run.get("updated_at", "").replace("Z", "+00:00")
                )
                total_duration += (end - start).total_seconds()
            
            if run.get("conclusion") == "success":
                success_count += 1
            elif run.get("conclusion") == "failure":
                failure_count += 1
        
        return {
            "total_runs": len(runs),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(runs) if runs else 0,
            "avg_duration_seconds": total_duration / len(runs) if runs else 0,
        }


def trigger_full_cycle(
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trigger = AutopoieticCITrigger(owner, repo, token)
    return trigger.trigger_autopoietic_cycle(cycle_type="full", metadata=metadata)


def trigger_eval_cycle(
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    test_suite: str = "full",
) -> Dict[str, Any]:
    trigger = AutopoieticCITrigger(owner, repo, token)
    return trigger.trigger_autopoietic_cycle(
        cycle_type="eval",
        metadata={"test_suite": test_suite},
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Autopoietic CI Trigger")
    parser.add_argument("--type", choices=["full", "eval"], default="full")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    
    trigger = AutopoieticCITrigger()
    result = trigger.trigger_autopoietic_cycle(
        cycle_type=args.type,
        branch=args.branch,
    )
    
    print(json.dumps(result, indent=2))
    
    if args.wait:
        completion = trigger.wait_for_completion("autopoietic.yml")
        print(json.dumps(completion, indent=2))