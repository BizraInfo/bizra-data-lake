// DEMA Utility Functions
import { formatDistanceToNow, format } from "date-fns";

export function timeAgo(date: string | Date): string {
  return formatDistanceToNow(new Date(date), { addSuffix: true });
}

export function formatTimestamp(date: string | Date): string {
  return format(new Date(date), "MMM d, yyyy · HH:mm");
}

export function formatShortDate(date: string | Date): string {
  return format(new Date(date), "MMM d");
}

export function formatId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}

export function trustLevelColor(level: string): string {
  switch (level) {
    case "admin": return "text-trust";
    case "operator": return "text-receipt";
    case "citizen": return "text-warning";
    case "visitor": return "text-muted-foreground";
    default: return "text-muted-foreground";
  }
}

export function trustLevelBadge(level: string): { label: string; variant: "default" | "secondary" | "destructive" | "outline" } {
  switch (level) {
    case "admin": return { label: "Admin", variant: "default" };
    case "operator": return { label: "Operator", variant: "secondary" };
    case "citizen": return { label: "Citizen", variant: "outline" };
    case "visitor": return { label: "Visitor", variant: "outline" };
    default: return { label: "Unknown", variant: "outline" };
  }
}

export function receiptStatusColor(status: string): string {
  switch (status) {
    case "verified": return "text-success";
    case "pending": return "text-warning";
    case "rejected": return "text-destructive";
    case "expired": return "text-muted-foreground";
    default: return "text-muted-foreground";
  }
}

export function receiptStatusDot(status: string): string {
  switch (status) {
    case "verified": return "bg-success";
    case "pending": return "bg-warning";
    case "rejected": return "bg-destructive";
    case "expired": return "bg-muted-foreground";
    default: return "bg-muted-foreground";
  }
}

export function receiptTypeIcon(type: string): string {
  switch (type) {
    case "completion": return "check-circle-2";
    case "action": return "zap";
    case "verification": return "shield-check";
    case "delegation": return "arrow-right-left";
    case "error": return "alert-circle";
    default: return "file-text";
  }
}

export function actionStatusColor(status: string): string {
  switch (status) {
    case "completed": return "text-success";
    case "executing": return "text-trust";
    case "pending": return "text-warning";
    case "approved": return "text-receipt";
    case "failed": return "text-destructive";
    case "denied": return "text-destructive";
    case "stopped": return "text-muted-foreground";
    default: return "text-muted-foreground";
  }
}

export function resourceTypeIcon(type: string): string {
  switch (type) {
    case "file": return "file";
    case "url": return "globe";
    case "credential": return "key-round";
    case "service": return "server";
    case "knowledge": return "brain";
    case "browser": return "monitor";
    case "terminal": return "terminal";
    default: return "box";
  }
}

export function resourceStatusColor(status: string): string {
  switch (status) {
    case "active": return "text-success";
    case "registered": return "text-trust";
    case "revoked": return "text-destructive";
    default: return "text-muted-foreground";
  }
}

export function urgencyColor(urgency: string): string {
  switch (urgency) {
    case "critical": return "bg-destructive/10 text-destructive border-destructive/20";
    case "high": return "bg-warning/10 text-warning border-warning/20";
    case "medium": return "bg-trust/10 text-trust border-trust/20";
    case "low": return "bg-muted text-muted-foreground border-border";
    default: return "bg-muted text-muted-foreground border-border";
  }
}

export function memoryCategoryIcon(category: string): string {
  switch (category) {
    case "preference": return "sliders-horizontal";
    case "context": return "layers";
    case "knowledge": return "brain";
    case "poi": return "bookmark";
    default: return "database";
  }
}

export function memoryCategoryColor(category: string): string {
  switch (category) {
    case "preference": return "bg-action/10 text-action-foreground";
    case "context": return "bg-manifest/10 text-manifest-foreground";
    case "knowledge": return "bg-trust/10 text-trust-foreground";
    case "poi": return "bg-receipt/10 text-receipt-foreground";
    default: return "bg-muted text-muted-foreground";
  }
}

export function confidenceBarColor(confidence: number): string {
  if (confidence >= 0.9) return "bg-success";
  if (confidence >= 0.7) return "bg-trust";
  if (confidence >= 0.5) return "bg-warning";
  return "bg-destructive";
}

export function modeIcon(mode: string): string {
  switch (mode) {
    case "browser": return "globe";
    case "computer": return "monitor";
    case "code": return "code-2";
    case "research": return "search";
    default: return "zap";
  }
}
