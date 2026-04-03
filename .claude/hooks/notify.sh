#!/bin/bash
# BIZRA Notification Hook
# Sends desktop notifications for long-running task completion

set -euo pipefail

INPUT=$(cat)
STOP_REASON=$(echo "$INPUT" | jq -r '.stop_hook_reason // "completed"')

# Extract task summary from conversation context if available
TASK_SUMMARY=$(echo "$INPUT" | jq -r '.task_summary // "Task completed"' | head -c 100)

# Determine notification title based on stop reason
case "$STOP_REASON" in
  "end_turn")
    TITLE="BIZRA: Task Completed"
    URGENCY="normal"
    ;;
  "error"|"tool_error")
    TITLE="BIZRA: Error Occurred"
    URGENCY="critical"
    ;;
  "max_turns")
    TITLE="BIZRA: Max Turns Reached"
    URGENCY="normal"
    ;;
  *)
    TITLE="BIZRA: Session Update"
    URGENCY="low"
    ;;
esac

# Try multiple notification methods for cross-platform support
notify_desktop() {
  local title="$1"
  local message="$2"
  local urgency="${3:-normal}"

  # WSL: Try Windows toast notification via PowerShell
  if command -v powershell.exe &>/dev/null; then
    powershell.exe -NoProfile -Command "
      [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
      [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
      \$template = @'
<toast>
  <visual>
    <binding template=\"ToastGeneric\">
      <text>$title</text>
      <text>$message</text>
    </binding>
  </visual>
  <audio silent=\"true\"/>
</toast>
'@
      \$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
      \$xml.LoadXml(\$template)
      \$toast = [Windows.UI.Notifications.ToastNotification]::new(\$xml)
      [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('BIZRA').Show(\$toast)
    " 2>/dev/null && return 0
  fi

  # Linux: Try notify-send
  if command -v notify-send &>/dev/null; then
    notify-send -u "$urgency" "$title" "$message" 2>/dev/null && return 0
  fi

  # macOS: Try osascript
  if command -v osascript &>/dev/null; then
    osascript -e "display notification \"$message\" with title \"$title\"" 2>/dev/null && return 0
  fi

  # Fallback: Terminal bell
  printf '\a'
  return 0
}

# Send notification (don't fail hook if notification fails)
notify_desktop "$TITLE" "$TASK_SUMMARY" "$URGENCY" || true

exit 0
