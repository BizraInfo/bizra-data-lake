#!/usr/bin/env bash
#
# Codex Linux sandbox preflight/remediation for bubblewrap user namespaces.
#
# Why this exists:
# - Codex `sandbox_mode = "workspace-write"` uses bubblewrap (`bwrap`) on Linux.
# - bwrap needs user namespace support, or startup fails with errors like:
#   "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
#   "No permissions to create new namespace"
#
# Usage:
#   ./scripts/ops/codex_linux_userns_preflight.sh
#   ./scripts/ops/codex_linux_userns_preflight.sh --apply

set -euo pipefail

UNPRIV_USERNS_KEY="kernel.unprivileged_userns_clone"
MAX_USERNS_KEY="user.max_user_namespaces"
MIN_USERNS_VALUE=28633
PERSIST_FILE="/etc/sysctl.d/99-codex-userns.conf"

APPLY=false

usage() {
    cat <<'EOF'
Codex Linux sandbox preflight for bubblewrap user namespaces.

Checks:
  - kernel.unprivileged_userns_clone == 1 (if present on this distro)
  - user.max_user_namespaces > 0

Options:
  --apply    Apply and persist safe defaults using sudo/root privileges.
  -h, --help Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            APPLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "SKIP: non-Linux host ($(uname -s)); bubblewrap userns preflight not needed."
    exit 0
fi

read_sysctl_value() {
    local key="$1"
    local path="/proc/sys/${key//./\/}"
    if [[ -r "$path" ]]; then
        tr -d '[:space:]' < "$path"
    else
        printf '%s' ""
    fi
}

is_int() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

need_unpriv_fix=false
need_max_userns_fix=false
need_bwrap_runtime_fix=false

unpriv_value="$(read_sysctl_value "$UNPRIV_USERNS_KEY")"
max_userns_value="$(read_sysctl_value "$MAX_USERNS_KEY")"

bwrap_path=""
bwrap_probe_output=""

probe_bwrap() {
    bwrap --unshare-user --unshare-net --uid 0 --gid 0 \
        --ro-bind / / --proc /proc --dev /dev /bin/true
}

echo "== Codex bubblewrap preflight =="
if command -v bwrap >/dev/null 2>&1; then
    bwrap_path="$(command -v bwrap)"
    echo "bwrap: $bwrap_path"
    if probe_bwrap >/dev/null 2>&1; then
        echo "OK:   bubblewrap namespace probe succeeded"
    else
        bwrap_probe_output="$(probe_bwrap 2>&1 || true)"
        echo "FAIL: bubblewrap namespace probe failed"
        echo "      ${bwrap_probe_output}"
        need_bwrap_runtime_fix=true
    fi
else
    echo "WARN: bwrap not found on PATH. Install bubblewrap for workspace-write sandbox mode."
fi

if [[ -z "$unpriv_value" ]]; then
    echo "INFO: $UNPRIV_USERNS_KEY not exposed on this kernel/distro; skipping this check."
else
    if [[ "$unpriv_value" == "1" ]]; then
        echo "OK:   $UNPRIV_USERNS_KEY=$unpriv_value"
    else
        echo "FAIL: $UNPRIV_USERNS_KEY=$unpriv_value (must be 1 for unprivileged user namespaces)"
        need_unpriv_fix=true
    fi
fi

if [[ -z "$max_userns_value" ]]; then
    echo "FAIL: $MAX_USERNS_KEY is unavailable; user namespaces appear unsupported."
    need_max_userns_fix=true
else
    if ! is_int "$max_userns_value"; then
        echo "FAIL: $MAX_USERNS_KEY=$max_userns_value (non-integer value)"
        need_max_userns_fix=true
    elif (( max_userns_value <= 0 )); then
        echo "FAIL: $MAX_USERNS_KEY=$max_userns_value (must be > 0)"
        need_max_userns_fix=true
    else
        echo "OK:   $MAX_USERNS_KEY=$max_userns_value"
        if (( max_userns_value < MIN_USERNS_VALUE )); then
            echo "WARN: $MAX_USERNS_KEY is low; recommend at least $MIN_USERNS_VALUE"
            need_max_userns_fix=true
        fi
    fi
fi

if [[ "$need_unpriv_fix" == false && "$need_max_userns_fix" == false && "$need_bwrap_runtime_fix" == false ]]; then
    echo
    echo "PASS: Linux user namespace prerequisites look good for Codex bubblewrap sandbox."
    exit 0
fi

echo
if [[ "$APPLY" == false ]]; then
    echo "Preflight failed."
    if [[ "$need_unpriv_fix" == true || "$need_max_userns_fix" == true ]]; then
        echo "Re-run with --apply to remediate and persist sysctl settings:"
        echo "  ./scripts/ops/codex_linux_userns_preflight.sh --apply"
    fi
    if [[ "$need_bwrap_runtime_fix" == true ]]; then
        echo
        echo "bubblewrap still cannot create runtime namespaces on this host."
        echo "This commonly happens in nested/containerized environments that block"
        echo "network namespace operations even when userns sysctls are enabled."
        echo
        echo "Fallback (trusted repo only): set Codex sandbox mode to danger-full-access."
    fi
    if [[ "$need_unpriv_fix" == false && "$need_max_userns_fix" == false ]]; then
        exit 2
    fi
    echo
    echo "Then verify with:"
    echo "  ./scripts/ops/codex_linux_userns_preflight.sh"
    exit 1
fi

need_sysctl_fix=false
if [[ "$need_unpriv_fix" == true || "$need_max_userns_fix" == true ]]; then
    need_sysctl_fix=true
fi

if [[ "$need_sysctl_fix" == false ]]; then
    echo "No sysctl changes required."
    if [[ "$need_bwrap_runtime_fix" == true ]]; then
        echo "bubblewrap is still blocked by host/container runtime restrictions."
        echo "Fallback (trusted repo only): set Codex sandbox mode to danger-full-access."
        exit 2
    fi
    exit 0
fi

if [[ "$need_max_userns_fix" == true && -z "$max_userns_value" ]]; then
    echo "ERROR: $MAX_USERNS_KEY is unavailable on this kernel; cannot remediate automatically."
    exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
    sudo_cmd=()
elif command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
else
    echo "ERROR: --apply requires root privileges (run as root or install sudo)." >&2
    exit 1
fi

declare -a persist_lines
persist_lines+=("# Codex bubblewrap prerequisites")

if [[ -n "$unpriv_value" && "$unpriv_value" != "1" ]]; then
    "${sudo_cmd[@]}" sysctl -w "${UNPRIV_USERNS_KEY}=1" >/dev/null
    persist_lines+=("${UNPRIV_USERNS_KEY}=1")
    echo "APPLY: set ${UNPRIV_USERNS_KEY}=1"
fi

if [[ -n "$max_userns_value" ]] && is_int "$max_userns_value"; then
    target_max="$max_userns_value"
    if (( max_userns_value < MIN_USERNS_VALUE )); then
        target_max="$MIN_USERNS_VALUE"
    fi
    if (( target_max != max_userns_value )); then
        "${sudo_cmd[@]}" sysctl -w "${MAX_USERNS_KEY}=${target_max}" >/dev/null
        echo "APPLY: set ${MAX_USERNS_KEY}=${target_max}"
    fi
    persist_lines+=("${MAX_USERNS_KEY}=${target_max}")
else
    # Fallback when current value couldn't be parsed/read but key exists via sysctl.
    "${sudo_cmd[@]}" sysctl -w "${MAX_USERNS_KEY}=${MIN_USERNS_VALUE}" >/dev/null
    persist_lines+=("${MAX_USERNS_KEY}=${MIN_USERNS_VALUE}")
    echo "APPLY: set ${MAX_USERNS_KEY}=${MIN_USERNS_VALUE}"
fi

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT
printf '%s\n' "${persist_lines[@]}" > "$tmp_file"
"${sudo_cmd[@]}" install -m 0644 "$tmp_file" "$PERSIST_FILE"
"${sudo_cmd[@]}" sysctl --load "$PERSIST_FILE" >/dev/null

# Re-probe bubblewrap after applying sysctls.
if [[ -n "$bwrap_path" ]]; then
    if probe_bwrap >/dev/null 2>&1; then
        need_bwrap_runtime_fix=false
        echo "OK:   bubblewrap namespace probe succeeded after apply"
    else
        bwrap_probe_output="$(probe_bwrap 2>&1 || true)"
        echo "FAIL: bubblewrap namespace probe still failing after apply"
        echo "      ${bwrap_probe_output}"
        need_bwrap_runtime_fix=true
    fi
fi

echo
echo "DONE: persisted settings at $PERSIST_FILE"
echo "Re-run without --apply to verify:"
echo "  ./scripts/ops/codex_linux_userns_preflight.sh"
if [[ "$need_bwrap_runtime_fix" == true ]]; then
    echo
    echo "NOTE: sysctl values were applied, but bubblewrap is still blocked."
    echo "In nested/containerized environments, use Codex danger-full-access mode as fallback."
    exit 2
fi
