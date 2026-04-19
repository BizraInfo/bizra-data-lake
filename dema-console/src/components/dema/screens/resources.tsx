"use client";

import { useState } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Box,
  Plus,
  Search,
  File,
  Globe,
  KeyRound,
  Server,
  Brain,
  Monitor,
  Terminal,
  ExternalLink,
  Trash2,
  Activity,
  Wifi,
  WifiOff,
  MoreHorizontal,
  Grid3X3,
  List,
} from "lucide-react";
import { timeAgo, resourceTypeIcon, resourceStatusColor } from "@/lib/helpers/dema";
import { cn } from "@/lib/utils";
import type { Resource, ResourceType } from "@/lib/types";

const TYPE_ICONS: Record<string, React.ElementType> = {
  file: File,
  url: Globe,
  credential: KeyRound,
  service: Server,
  knowledge: Brain,
  browser: Monitor,
  terminal: Terminal,
};

const TYPE_LABELS: Record<string, string> = {
  file: "File",
  url: "URL",
  credential: "Credential",
  service: "Service",
  knowledge: "Knowledge",
  browser: "Browser",
  terminal: "Terminal",
};

function ResourceCard({
  resource,
  viewMode,
  onRemove,
}: {
  resource: Resource;
  viewMode: "grid" | "list";
  onRemove: () => void;
}) {
  const Icon = TYPE_ICONS[resource.type] || Box;
  const isListView = viewMode === "list";

  return (
    <Card
      className={cn(
        "border-border/30 hover:border-border/60 transition-all group",
        isListView ? "" : ""
      )}
    >
      <CardContent className={cn("p-4", isListView ? "" : "")}>
        {isListView ? (
          <div className="flex items-center gap-3">
            <div className={cn("p-1.5 rounded-md bg-action/5 shrink-0")}>
              <Icon className={cn("h-4 w-4", resourceStatusColor(resource.status))} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium truncate">{resource.name}</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 shrink-0">
                  {TYPE_LABELS[resource.type] || resource.type}
                </Badge>
              </div>
              {resource.path && (
                <p className="text-[11px] text-muted-foreground font-mono truncate mt-0.5">
                  {resource.path}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <div className={cn("flex items-center gap-1 text-[10px]", resourceStatusColor(resource.status))}>
                {resource.status === "active" ? (
                  <Wifi className="h-3 w-3" />
                ) : (
                  <WifiOff className="h-3 w-3" />
                )}
                {resource.status}
              </div>
              <span className="text-[10px] text-muted-foreground">
                {timeAgo(resource.createdAt)}
              </span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-destructive"
                    onClick={onRemove}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent className="text-xs">Remove resource</TooltipContent>
              </Tooltip>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-start justify-between">
              <div className={cn("p-2 rounded-md bg-action/5")}>
                <Icon className={cn("h-5 w-5", resourceStatusColor(resource.status))} />
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-destructive"
                    onClick={onRemove}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent className="text-xs">Remove resource</TooltipContent>
              </Tooltip>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium truncate">{resource.name}</span>
              </div>
              {resource.path && (
                <p className="text-[10px] text-muted-foreground font-mono truncate">
                  {resource.path}
                </p>
              )}
            </div>
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                {TYPE_LABELS[resource.type]}
              </Badge>
              <div className={cn("flex items-center gap-1 text-[10px]", resourceStatusColor(resource.status))}>
                {resource.status === "active" ? (
                  <Wifi className="h-2.5 w-2.5" />
                ) : (
                  <WifiOff className="h-2.5 w-2.5" />
                )}
                {resource.status}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ResourcesScreen() {
  const { resources, addResource, removeResource } = useDEMAStore();
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState<ResourceType>("service");
  const [newPath, setNewPath] = useState("");

  const filtered = resources.filter((r) => {
    const matchesSearch =
      r.name.toLowerCase().includes(search.toLowerCase()) ||
      (r.path && r.path.toLowerCase().includes(search.toLowerCase()));
    const matchesType = typeFilter === "all" || r.type === typeFilter;
    return matchesSearch && matchesType;
  });

  const handleAdd = () => {
    if (!newName.trim()) return;
    addResource({
      id: `res-${Date.now()}`,
      name: newName.trim(),
      type: newType,
      path: newPath.trim() || null,
      status: "registered",
      metadata: null,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
    setNewName("");
    setNewPath("");
    setShowAddDialog(false);
  };

  const activeCount = resources.filter((r) => r.status === "active").length;
  const typeCounts = resources.reduce(
    (acc, r) => {
      acc[r.type] = (acc[r.type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-4 p-6 max-w-6xl mx-auto dema-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Resources</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Registered resources, services, and local nodes. {activeCount} active across {Object.keys(typeCounts).length} types.
          </p>
        </div>
        <Button
          size="sm"
          onClick={() => setShowAddDialog(true)}
          className="h-8 text-xs"
        >
          <Plus className="h-3.5 w-3.5 mr-1.5" />
          Register Resource
        </Button>
      </div>

      {/* Type badges */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(typeCounts).map(([type, count]) => {
          const Icon = TYPE_ICONS[type] || Box;
          return (
            <Badge
              key={type}
              variant="outline"
              className="text-[11px] px-2.5 py-1 cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => setTypeFilter(typeFilter === type ? "all" : type)}
            >
              <Icon className="h-3 w-3 mr-1" />
              {TYPE_LABELS[type] || type} ({count})
            </Badge>
          );
        })}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search resources..."
            className="h-8 text-xs pl-8"
          />
        </div>
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="h-8 text-xs w-[130px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            {Object.keys(TYPE_LABELS).map((type) => (
              <SelectItem key={type} value={type}>
                {TYPE_LABELS[type]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex items-center gap-0.5 ml-auto border border-border/50 rounded-md overflow-hidden">
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "h-8 w-8 p-0 rounded-none",
              viewMode === "list" ? "bg-accent" : ""
            )}
            onClick={() => setViewMode("list")}
          >
            <List className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "h-8 w-8 p-0 rounded-none",
              viewMode === "grid" ? "bg-accent" : ""
            )}
            onClick={() => setViewMode("grid")}
          >
            <Grid3X3 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Resource List/Grid */}
      <div className={cn(viewMode === "grid" ? "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3" : "space-y-2")}>
        {filtered.map((resource) => (
          <ResourceCard
            key={resource.id}
            resource={resource}
            viewMode={viewMode}
            onRemove={() => removeResource(resource.id)}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Box className="h-8 w-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No resources match your filters</p>
        </div>
      )}

      {/* Add Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="text-base">Register New Resource</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-xs">Name</Label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Resource name"
                className="text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Type</Label>
              <Select value={newType} onValueChange={(v) => setNewType(v as ResourceType)}>
                <SelectTrigger className="text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(TYPE_LABELS).map(([key, label]) => (
                    <SelectItem key={key} value={key}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Path / URL (optional)</Label>
              <Input
                value={newPath}
                onChange={(e) => setNewPath(e.target.value)}
                placeholder="/path/to/resource or https://..."
                className="text-sm font-mono"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" size="sm" onClick={() => setShowAddDialog(false)}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleAdd} disabled={!newName.trim()}>
              Register
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
