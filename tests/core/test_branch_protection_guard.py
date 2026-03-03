from scripts.ops.branch_protection_guard import (
    BranchProtectionPolicy,
    bool_field,
    build_protection_payload,
    evaluate_drift,
)


def _policy() -> BranchProtectionPolicy:
    return BranchProtectionPolicy.from_dict(
        {
            "branches": ["main", "master", "develop"],
            "ignore_missing_branches": True,
            "strict_status_checks": True,
            "required_status_checks": ["CI", "Phase56 Security Gate"],
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "require_conversation_resolution": True,
            "enforce_admins": True,
            "require_linear_history": True,
            "allow_force_pushes": False,
            "allow_deletions": False,
            "block_creations": False,
            "lock_branch": False,
            "allow_fork_syncing": True,
        }
    )


def _actual_ok() -> dict:
    return {
        "required_status_checks": {
            "strict": True,
            "contexts": ["CI", "Phase56 Security Gate"],
        },
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "required_approving_review_count": 1,
        },
        "required_conversation_resolution": {"enabled": True},
        "required_linear_history": {"enabled": True},
        "allow_force_pushes": {"enabled": False},
        "allow_deletions": {"enabled": False},
        "block_creations": {"enabled": False},
        "lock_branch": {"enabled": False},
        "allow_fork_syncing": {"enabled": True},
        "enforce_admins": {"enabled": True},
    }


def test_build_payload_includes_required_contexts():
    payload = build_protection_payload(_policy())
    assert payload["required_status_checks"]["strict"] is True
    assert payload["required_status_checks"]["contexts"] == [
        "CI",
        "Phase56 Security Gate",
    ]


def test_evaluate_drift_passes_for_matching_policy():
    drifts = evaluate_drift(_actual_ok(), _policy())
    assert drifts == []


def test_evaluate_drift_detects_missing_status_check():
    actual = _actual_ok()
    actual["required_status_checks"]["contexts"] = ["CI"]
    drifts = evaluate_drift(actual, _policy())
    assert any("Missing required status checks" in item for item in drifts)


def test_bool_field_handles_enabled_dict_and_scalar():
    assert bool_field({"enabled": True}) is True
    assert bool_field({"enabled": False}) is False
    assert bool_field(True) is True
    assert bool_field(False) is False
