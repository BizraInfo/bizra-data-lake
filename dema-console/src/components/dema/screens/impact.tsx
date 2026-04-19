"use client";

import { useState, useMemo, useCallback } from "react";
import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { timeAgo } from "@/lib/helpers/dema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import {
  Network,
  GitBranch,
  ArrowRight,
  Target,
  Activity,
  Zap,
  Eye,
  BarChart3,
  Circle,
  ChevronRight,
  Play,
  AlertTriangle,
  CheckCircle2,
  Clock,
  XCircle,
  Info,
  Layers,
  Shield,
  Cpu,
  FileCheck,
  Brain,
  Lock,
  Search,
} from "lucide-react";
import type {
  GraphNode,
  GraphEdge,
  GraphNodeType,
  EdgeType,
  ImpactPropagation,
} from "@/lib/types";

// ─── Color & Style Configs ──────────────────────────────────────

const NODE_TYPE_CONFIG: Record<
  GraphNodeType,
  { color: string; bg: string; border: string; icon: React.ElementType }
> = {
  agent: { color: "text-trust", bg: "fill-trust/20", border: "stroke-trust", icon: Cpu },
  resource: {
    color: "text-receipt",
    bg: "fill-receipt/20",
    border: "stroke-receipt",
    icon: FileCheck,
  },
  mission: {
    color: "text-manifest",
    bg: "fill-manifest/20",
    border: "stroke-manifest",
    icon: Target,
  },
  receipt: {
    color: "text-success",
    bg: "fill-success/20",
    border: "stroke-success",
    icon: FileCheck,
  },
  boundary: {
    color: "text-destructive",
    bg: "fill-destructive/20",
    border: "stroke-destructive",
    icon: Lock,
  },
  action: {
    color: "text-warning",
    bg: "fill-warning/20",
    border: "stroke-warning",
    icon: Zap,
  },
  memory: {
    color: "text-action",
    bg: "fill-action/20",
    border: "stroke-action",
    icon: Brain,
  },
};

const EDGE_TYPE_CONFIG: Record<
  EdgeType,
  { color: string; dash: string; label: string }
> = {
  depends_on: { color: "stroke-muted-foreground", dash: "8 4", label: "Depends On" },
  produces: { color: "stroke-receipt", dash: "none", label: "Produces" },
  verifies: { color: "stroke-trust", dash: "none", label: "Verifies" },
  delegates: { color: "stroke-manifest", dash: "none", label: "Delegates" },
  blocks: { color: "stroke-destructive", dash: "none", label: "Blocks" },
  informs: { color: "stroke-muted-foreground", dash: "3 3", label: "Informs" },
};

const STATUS_DOT: Record<string, string> = {
  active: "fill-success",
  busy: "fill-warning",
  idle: "fill-muted-foreground",
  error: "fill-destructive",
  sleeping: "fill-muted-foreground",
  verified: "fill-success",
  enforced: "fill-trust",
  terminated: "fill-destructive",
};

function nodeRadius(weight: number): number {
  return Math.max(30, Math.min(60, 25 + weight * 35));
}

function impactColor(score: number): string {
  if (score >= 0.9) return "text-destructive";
  if (score >= 0.7) return "text-warning";
  if (score >= 0.4) return "text-trust";
  return "text-success";
}

function impactBarColor(score: number): string {
  if (score >= 0.9) return "[&>div]:bg-destructive";
  if (score >= 0.7) return "[&>div]:bg-warning";
  if (score >= 0.4) return "[&>div]:bg-trust";
  return "[&>div]:bg-success";
}

function statusBadgeProps(status: string): {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
  className: string;
} {
  switch (status) {
    case "completed":
      return { label: "Completed", variant: "default", className: "bg-success/15 text-success border-success/25" };
    case "calculating":
      return { label: "Calculating", variant: "secondary", className: "bg-trust/15 text-trust border-trust/25" };
    case "failed":
      return { label: "Failed", variant: "destructive", className: "" };
    default:
      return { label: status, variant: "outline", className: "" };
  }
}

// ─── SVG Grid Pattern ───────────────────────────────────────────

function GraphGrid() {
  return (
    <defs>
      <pattern id="graph-grid" width="40" height="40" patternUnits="userSpaceOnUse">
        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.3" opacity="0.15" />
      </pattern>
      <pattern id="graph-grid-large" width="200" height="200" patternUnits="userSpaceOnUse">
        <path d="M 200 0 L 0 0 0 200" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.08" />
      </pattern>
      <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" floodColor="currentColor" floodOpacity="0.15" />
      </filter>
      <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" className="fill-muted-foreground/50" />
      </marker>
      <marker id="arrowhead-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" className="fill-receipt" />
      </marker>
      <marker id="arrowhead-amber" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" className="fill-trust" />
      </marker>
      <marker id="arrowhead-teal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" className="fill-manifest" />
      </marker>
      <marker id="arrowhead-red" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" className="fill-destructive" />
      </marker>
    </defs>
  );
}

function edgeMarker(type: EdgeType): string {
  switch (type) {
    case "produces": return "url(#arrowhead-green)";
    case "verifies": return "url(#arrowhead-amber)";
    case "delegates": return "url(#arrowhead-teal)";
    case "blocks": return "url(#arrowhead-red)";
    default: return "url(#arrowhead)";
  }
}

// ─── Legend ──────────────────────────────────────────────────────

function GraphLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2 px-4 py-3">
      <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider self-center mr-1">
        Nodes
      </div>
      {(Object.entries(NODE_TYPE_CONFIG) as [GraphNodeType, typeof NODE_TYPE_CONFIG[GraphNodeType]][]).map(
        ([type, cfg]) => {
          const Icon = cfg.icon;
          return (
            <div key={type} className="flex items-center gap-1.5">
              <Circle className={cn("h-2.5 w-2.5", cfg.color)} fill="currentColor" />
              <span className="text-[11px] text-muted-foreground capitalize">{type}</span>
            </div>
          );
        }
      )}
      <Separator orientation="vertical" className="h-4 mx-1" />
      <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider self-center mr-1">
        Edges
      </div>
      {(Object.entries(EDGE_TYPE_CONFIG) as [EdgeType, typeof EDGE_TYPE_CONFIG[EdgeType]][]).map(
        ([type, cfg]) => (
          <div key={type} className="flex items-center gap-1.5">
            <svg width="20" height="8" className="shrink-0">
              <line
                x1="0" y1="4" x2="20" y2="4"
                strokeWidth="2"
                className={cfg.color}
                strokeDasharray={cfg.dash === "none" ? undefined : cfg.dash}
              />
            </svg>
            <span className="text-[11px] text-muted-foreground">{cfg.label}</span>
          </div>
        )
      )}
    </div>
  );
}

// ─── Tab 1: Dependency Graph ────────────────────────────────────

interface GraphState {
  hoveredNode: string | null;
  selectedNode: string | null;
  search: string;
  filterType: GraphNodeType | "all";
  scale: number;
}

function DependencyGraphTab() {
  const { graphSnapshot } = useDEMAStore();
  const [state, setState] = useState<GraphState>({
    hoveredNode: null,
    selectedNode: null,
    search: "",
    filterType: "all",
    scale: 1,
  });

  const filteredNodes = useMemo(() => {
    let nodes = graphSnapshot.nodes;
    if (state.filterType !== "all") {
      nodes = nodes.filter((n) => n.type === state.filterType);
    }
    if (state.search) {
      const q = state.search.toLowerCase();
      nodes = nodes.filter((n) => n.label.toLowerCase().includes(q) || n.id.toLowerCase().includes(q));
    }
    return nodes;
  }, [graphSnapshot.nodes, state.filterType, state.search]);

  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  const filteredEdges = useMemo(
    () =>
      graphSnapshot.edges.filter(
        (e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)
      ),
    [graphSnapshot.edges, filteredNodeIds]
  );

  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    graphSnapshot.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [graphSnapshot.nodes]);

  const connectedNodeIds = useMemo(() => {
    if (!state.selectedNode) return new Set<string>();
    const ids = new Set<string>();
    ids.add(state.selectedNode);
    graphSnapshot.edges.forEach((e) => {
      if (e.source === state.selectedNode) ids.add(e.target);
      if (e.target === state.selectedNode) ids.add(e.source);
    });
    return ids;
  }, [state.selectedNode, graphSnapshot.edges]);

  // Compute bounding box for auto-fit
  const viewBox = useMemo(() => {
    if (filteredNodes.length === 0) return "0 0 800 450";
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    filteredNodes.forEach((n) => {
      minX = Math.min(minX, n.x - 80);
      minY = Math.min(minY, n.y - 80);
      maxX = Math.max(maxX, n.x + 80);
      maxY = Math.max(maxY, n.y + 80);
    });
    const pad = 60;
    return `${minX - pad} ${minY - pad} ${maxX - minX + pad * 2} ${maxY - minY + pad * 2}`;
  }, [filteredNodes]);

  const highlightNodes = state.selectedNode ? connectedNodeIds : null;

  const handleNodeHover = useCallback((id: string | null) => {
    setState((s) => ({ ...s, hoveredNode: id }));
  }, []);

  const handleNodeClick = useCallback((id: string) => {
    setState((s) => ({ ...s, selectedNode: s.selectedNode === id ? null : id }));
  }, []);

  return (
    <div className="space-y-3">
      {/* Stats bar */}
      <div className="flex flex-wrap items-center gap-3 px-1">
        <div className="flex items-center gap-1.5">
          <Circle className="h-3 w-3 text-muted-foreground" fill="currentColor" />
          <span className="text-xs text-muted-foreground">
            <span className="font-mono font-medium text-foreground">{graphSnapshot.totalNodes}</span> nodes
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <GitBranch className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            <span className="font-mono font-medium text-foreground">{graphSnapshot.totalEdges}</span> edges
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <Activity className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            Density: <span className="font-mono font-medium text-foreground">{graphSnapshot.density.toFixed(2)}</span>
          </span>
        </div>
        {state.selectedNode && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-[10px] px-2 ml-auto"
            onClick={() => setState((s) => ({ ...s, selectedNode: null }))}
          >
            Clear selection
          </Button>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[160px] max-w-[260px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search nodes..."
            value={state.search}
            onChange={(e) => setState((s) => ({ ...s, search: e.target.value }))}
            className="h-8 text-xs pl-8"
          />
        </div>
        <Select
          value={state.filterType}
          onValueChange={(v) => setState((s) => ({ ...s, filterType: v as GraphNodeType | "all" }))}
        >
          <SelectTrigger className="h-8 w-[130px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="agent">Agent</SelectItem>
            <SelectItem value="resource">Resource</SelectItem>
            <SelectItem value="mission">Mission</SelectItem>
            <SelectItem value="receipt">Receipt</SelectItem>
            <SelectItem value="boundary">Boundary</SelectItem>
            <SelectItem value="memory">Memory</SelectItem>
            <SelectItem value="action">Action</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-1 ml-auto">
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => setState((s) => ({ ...s, scale: Math.min(s.scale + 0.15, 2) }))}
          >
            <span className="text-xs font-bold">+</span>
          </Button>
          <span className="text-[10px] text-muted-foreground font-mono w-10 text-center">
            {Math.round(state.scale * 100)}%
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => setState((s) => ({ ...s, scale: Math.max(s.scale - 0.15, 0.4) }))}
          >
            <span className="text-xs font-bold">−</span>
          </Button>
        </div>
      </div>

      {/* SVG Graph */}
      <Card className="border-border/50 overflow-hidden">
        <div className="bg-card relative">
          <svg
            viewBox={viewBox}
            className="w-full text-muted-foreground"
            style={{ minHeight: 420 }}
          >
            <GraphGrid />
            <rect width="100%" height="100%" fill="url(#graph-grid)" />
            <rect width="100%" height="100%" fill="url(#graph-grid-large)" />

            {/* Render edges */}
            {filteredEdges.map((edge) => {
              const source = nodeMap.get(edge.source);
              const target = nodeMap.get(edge.target);
              if (!source || !target) return null;

              const sR = nodeRadius(source.weight);
              const tR = nodeRadius(target.weight);
              const dx = target.x - source.x;
              const dy = target.y - source.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              const nx = dx / (dist || 1);
              const ny = dy / (dist || 1);

              const x1 = source.x + nx * sR;
              const y1 = source.y + ny * sR;
              const x2 = target.x - nx * (tR + 6);
              const y2 = target.y - ny * (tR + 6);

              const cfg = EDGE_TYPE_CONFIG[edge.type];
              const isHighlighted =
                state.selectedNode &&
                (edge.source === state.selectedNode || edge.target === state.selectedNode);
              const isDimmed = highlightNodes !== null && !isHighlighted;

              return (
                <g key={edge.id} opacity={isDimmed ? 0.15 : 1}>
                  <line
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    strokeWidth={isHighlighted ? 2.5 : 1.5}
                    className={cn(
                      cfg.color,
                      isHighlighted && "opacity-100"
                    )}
                    strokeDasharray={cfg.dash === "none" ? undefined : cfg.dash}
                    markerEnd={edgeMarker(edge.type)}
                  />
                  {/* Edge label */}
                  {edge.label && isHighlighted && (
                    <text
                      x={(x1 + x2) / 2}
                      y={(y1 + y2) / 2 - 6}
                      textAnchor="middle"
                      className="fill-muted-foreground text-[9px] font-mono"
                    >
                      {edge.label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Render nodes */}
            {filteredNodes.map((node) => {
              const cfg = NODE_TYPE_CONFIG[node.type];
              const r = nodeRadius(node.weight);
              const isHovered = state.hoveredNode === node.id;
              const isSelected = state.selectedNode === node.id;
              const isConnected = highlightNodes ? highlightNodes.has(node.id) : true;
              const isDimmed = highlightNodes !== null && !isConnected;

              return (
                <g
                  key={node.id}
                  className="cursor-pointer transition-all duration-150"
                  opacity={isDimmed ? 0.2 : 1}
                  onMouseEnter={() => handleNodeHover(node.id)}
                  onMouseLeave={() => handleNodeHover(null)}
                  onClick={() => handleNodeClick(node.id)}
                  transform={`scale(${isHovered ? 1.08 : 1})`}
                  style={{ transformOrigin: `${node.x}px ${node.y}px` }}
                >
                  {/* Outer glow on hover/selection */}
                  {(isHovered || isSelected) && (
                    <circle
                      cx={node.x} cy={node.y} r={r + 8}
                      className={cn(cfg.bg)}
                      opacity={0.4}
                    />
                  )}

                  {/* Node body */}
                  <circle
                    cx={node.x} cy={node.y} r={r}
                    className={cn(cfg.bg)}
                    stroke={isSelected ? "currentColor" : "transparent"}
                    strokeWidth={isSelected ? 2 : 0}
                    filter="url(#node-shadow)"
                  />

                  {/* Inner ring */}
                  <circle
                    cx={node.x} cy={node.y} r={r - 4}
                    className={cfg.border}
                    fill="none"
                    strokeWidth={1.5}
                    opacity={0.5}
                  />

                  {/* Status dot */}
                  {node.status && (
                    <circle
                      cx={node.x + r - 8}
                      cy={node.y - r + 8}
                      r={4}
                      className={STATUS_DOT[node.status] || "fill-muted-foreground"}
                      stroke="currentColor"
                      strokeWidth={1.5}
                    />
                  )}

                  {/* Label */}
                  <text
                    x={node.x} y={node.y + 1}
                    textAnchor="middle"
                    dominantBaseline="central"
                    className={cn(
                      "font-medium pointer-events-none select-none",
                      cfg.color,
                      r < 36 ? "text-[9px]" : "text-[11px]"
                    )}
                    style={{ maxWidth: r * 1.6 }}
                  >
                    {node.label.length > 14 ? node.label.slice(0, 12) + "…" : node.label}
                  </text>

                  {/* Weight label below node */}
                  {isHovered && (
                    <text
                      x={node.x} y={node.y + r + 14}
                      textAnchor="middle"
                      className="fill-muted-foreground text-[9px] font-mono pointer-events-none"
                    >
                      w: {node.weight.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Floating tooltip for hovered node */}
          {state.hoveredNode && nodeMap.has(state.hoveredNode) && (
            <NodeTooltip node={nodeMap.get(state.hoveredNode)!} />
          )}
        </div>

        <GraphLegend />
      </Card>
    </div>
  );
}

function NodeTooltip({ node }: { node: GraphNode }) {
  const cfg = NODE_TYPE_CONFIG[node.type];
  const Icon = cfg.icon;

  return (
    <div className="absolute top-3 right-3 z-10">
      <Card className="border-border/80 bg-card/95 backdrop-blur-sm shadow-lg p-3 w-56">
        <div className="flex items-center gap-2 mb-2">
          <div className={cn("p-1 rounded-md bg-muted", cfg.color)}>
            <Icon className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-semibold truncate">{node.label}</div>
            <div className="text-[10px] text-muted-foreground font-mono">{node.id}</div>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">Type</span>
            <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 capitalize">
              {node.type}
            </Badge>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">Weight</span>
            <span className="font-mono font-medium">{node.weight.toFixed(2)}</span>
          </div>
          {node.status && (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground">Status</span>
              <div className="flex items-center gap-1">
                <div className={cn("w-1.5 h-1.5 rounded-full", STATUS_DOT[node.status]?.replace("fill-", "bg-") || "bg-muted-foreground")} />
                <span className="capitalize text-[10px]">{node.status}</span>
              </div>
            </div>
          )}
          {Object.keys(node.metadata).length > 0 && (
            <Separator className="my-1" />
          )}
          {Object.entries(node.metadata).map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground">{k}</span>
              <span className="font-mono text-[10px] truncate max-w-[100px]">{String(v)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─── Tab 2: Impact Propagation ──────────────────────────────────

function ImpactPropagationTab() {
  const { graphSnapshot } = useDEMAStore();
  const propagations = graphSnapshot.propagations;

  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    graphSnapshot.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [graphSnapshot.nodes]);

  return (
    <ScrollArea className="h-[calc(100vh-260px)] dema-scrollbar">
      <div className="space-y-4 pr-4">
        {/* Summary header */}
        <div className="flex items-center gap-3 px-1">
          <Activity className="h-4 w-4 text-trust" />
          <span className="text-xs text-muted-foreground">
            <span className="font-mono font-medium text-foreground">{propagations.length}</span> propagation
            analyses
          </span>
          <Separator orientation="vertical" className="h-4" />
          <span className="text-xs text-muted-foreground">
            Avg impact:{" "}
            <span className="font-mono font-medium text-foreground">
              {propagations.length > 0
                ? (propagations.reduce((sum, p) => sum + p.impactScore, 0) / propagations.length).toFixed(2)
                : "—"}
            </span>
          </span>
        </div>

        {/* Propagation cards */}
        {propagations.map((prop) => (
          <PropagationCard key={prop.id} propagation={prop} nodeMap={nodeMap} />
        ))}

        {propagations.length === 0 && (
          <Card className="border-border/50">
            <CardContent className="py-12 text-center">
              <GitBranch className="h-8 w-8 mx-auto text-muted-foreground/40 mb-3" />
              <p className="text-sm text-muted-foreground">No propagation analyses yet.</p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                Run a simulation from the Graph Analytics tab.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </ScrollArea>
  );
}

function PropagationCard({
  propagation,
  nodeMap,
}: {
  propagation: ImpactPropagation;
  nodeMap: Map<string, GraphNode>;
}) {
  const sourceNode = nodeMap.get(propagation.sourceNodeId);
  const sb = statusBadgeProps(propagation.status);

  return (
    <Card className="border-border/50 bg-card/50 hover:border-border transition-colors">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <div className="p-1.5 rounded-md bg-trust/5 shrink-0">
              <GitBranch className="h-4 w-4 text-trust" />
            </div>
            <div className="min-w-0">
              <CardTitle className="text-sm font-medium truncate">
                {sourceNode?.label ?? propagation.sourceNodeId}
              </CardTitle>
              <p className="text-[10px] text-muted-foreground mt-0.5 font-mono">
                {propagation.id}
              </p>
            </div>
          </div>
          <Badge variant={sb.variant} className={cn("text-[10px] px-2 py-0 shrink-0", sb.className)}>
            {sb.label}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Impact metrics row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {/* Impact score */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Impact Score
            </div>
            <div className={cn("text-lg font-bold font-mono", impactColor(propagation.impactScore))}>
              {(propagation.impactScore * 100).toFixed(0)}
              <span className="text-xs font-normal text-muted-foreground">%</span>
            </div>
            <Progress value={propagation.impactScore * 100} className={cn("h-1.5", impactBarColor(propagation.impactScore))} />
          </div>

          {/* Target type */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Target Type
            </div>
            <Badge variant="outline" className="text-[11px] capitalize">
              {propagation.targetType}
            </Badge>
          </div>

          {/* Depth */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Depth
            </div>
            <div className="text-lg font-bold font-mono">{propagation.depth}</div>
            <div className="text-[10px] text-muted-foreground">
              {propagation.depth === 1 ? "Direct" : `${propagation.depth - 1} hop${propagation.depth - 1 > 1 ? "s" : ""}`}
            </div>
          </div>

          {/* Affected */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Affected
            </div>
            <div className="text-lg font-bold font-mono">{propagation.affectedNodes.length}</div>
            <div className="text-[10px] text-muted-foreground">nodes impacted</div>
          </div>
        </div>

        <Separator />

        {/* Affected nodes */}
        <div className="space-y-2">
          <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            Affected Nodes
          </div>
          <div className="flex flex-wrap gap-1.5">
            {propagation.affectedNodes.map((nodeId) => {
              const n = nodeMap.get(nodeId);
              const cfg = n ? NODE_TYPE_CONFIG[n.type] : null;
              return (
                <Tooltip key={nodeId}>
                  <TooltipTrigger asChild>
                    <Badge
                      variant="outline"
                      className={cn(
                        "text-[10px] px-2 py-0.5 cursor-default transition-colors",
                        cfg?.color
                      )}
                    >
                      {n?.label ?? nodeId}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="text-xs">
                    {n?.label ?? nodeId} · {n?.type ?? "unknown"}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        </div>

        {/* Propagation path */}
        <div className="space-y-2">
          <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            Propagation Path
          </div>
          <div className="flex items-center gap-1 flex-wrap">
            {propagation.propagationPath.map((nodeId, i) => {
              const n = nodeMap.get(nodeId);
              return (
                <div key={nodeId} className="flex items-center gap-1">
                  <Badge variant="secondary" className="text-[10px] px-2 py-0.5 font-mono">
                    {n?.label ?? nodeId}
                  </Badge>
                  {i < propagation.propagationPath.length - 1 && (
                    <ChevronRight className="h-3 w-3 text-muted-foreground/50 shrink-0" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Insights */}
        {propagation.insights.length > 0 && (
          <div className="space-y-2">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Insights
            </div>
            <ul className="space-y-1.5">
              {propagation.insights.map((insight, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                  <Info className="h-3 w-3 text-trust shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Timestamp */}
        <div className="text-[10px] text-muted-foreground/60 text-right">
          {timeAgo(propagation.timestamp)}
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Tab 3: Graph Analytics ─────────────────────────────────────

function GraphAnalyticsTab() {
  const { graphSnapshot, addOrchestrationEvent } = useDEMAStore();
  const [simulating, setSimulating] = useState(false);

  // Computed analytics
  const analytics = useMemo(() => {
    const nodes = graphSnapshot.nodes;
    const edges = graphSnapshot.edges;

    // Node type distribution
    const nodeTypeCounts: Partial<Record<GraphNodeType, number>> = {};
    nodes.forEach((n) => {
      nodeTypeCounts[n.type] = (nodeTypeCounts[n.type] || 0) + 1;
    });

    // Edge type distribution
    const edgeTypeCounts: Partial<Record<EdgeType, number>> = {};
    edges.forEach((e) => {
      edgeTypeCounts[e.type] = (edgeTypeCounts[e.type] || 0) + 1;
    });

    // Connection counts per node
    const connectionCounts: Record<string, number> = {};
    nodes.forEach((n) => (connectionCounts[n.id] = 0));
    edges.forEach((e) => {
      connectionCounts[e.source] = (connectionCounts[e.source] || 0) + 1;
      connectionCounts[e.target] = (connectionCounts[e.target] || 0) + 1;
    });

    // Most connected node
    let mostConnected = "";
    let maxConnections = 0;
    Object.entries(connectionCounts).forEach(([id, count]) => {
      if (count > maxConnections) {
        mostConnected = id;
        maxConnections = count;
      }
    });

    // Average weight
    const avgWeight =
      nodes.length > 0
        ? nodes.reduce((sum, n) => sum + n.weight, 0) / nodes.length
        : 0;

    // Critical path (longest weighted path approximation)
    const criticalPath = graphSnapshot.propagations.length > 0
      ? graphSnapshot.propagations.reduce((max, p) =>
          p.impactScore > max.impactScore ? p : max, graphSnapshot.propagations[0]
        )
      : null;

    return {
      nodeTypeCounts,
      edgeTypeCounts,
      mostConnected,
      maxConnections,
      avgWeight,
      criticalPath,
      totalNodes: graphSnapshot.totalNodes,
      totalEdges: graphSnapshot.totalEdges,
      density: graphSnapshot.density,
    };
  }, [graphSnapshot]);

  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    graphSnapshot.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [graphSnapshot.nodes]);

  const handleSimulation = useCallback(() => {
    setSimulating(true);

    // Simulate a propagation calculation
    setTimeout(() => {
      addOrchestrationEvent({
        id: `evt-sim-${Date.now()}`,
        type: "coordination",
        agentId: "agt-coord-01",
        message: "Impact propagation simulation triggered from Graph Analytics",
        metadata: {
          source: "impact_screen",
          action: "propagation_simulation",
          timestamp: new Date().toISOString(),
        },
        timestamp: new Date().toISOString(),
        severity: "info",
      });
      setSimulating(false);
    }, 1500);
  }, [addOrchestrationEvent]);

  return (
    <ScrollArea className="h-[calc(100vh-260px)] dema-scrollbar">
      <div className="space-y-4 pr-4">
        {/* Stats cards */}
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
          <StatCard
            label="Total Nodes"
            value={analytics.totalNodes.toString()}
            icon={Network}
            color="text-trust"
            bg="bg-trust/5"
          />
          <StatCard
            label="Total Edges"
            value={analytics.totalEdges.toString()}
            icon={GitBranch}
            color="text-manifest"
            bg="bg-manifest/5"
          />
          <StatCard
            label="Density"
            value={analytics.density.toFixed(3)}
            icon={Activity}
            color="text-receipt"
            bg="bg-receipt/5"
          />
          <StatCard
            label="Avg Weight"
            value={analytics.avgWeight.toFixed(3)}
            icon={BarChart3}
            color="text-warning"
            bg="bg-warning/5"
          />
          <StatCard
            label="Most Connected"
            value={nodeMap.get(analytics.mostConnected)?.label ?? "—"}
            sub={`${analytics.maxConnections} links`}
            icon={Zap}
            color="text-action"
            bg="bg-action/5"
          />
          <StatCard
            label="Critical Path"
            value={analytics.criticalPath ? `${(analytics.criticalPath.impactScore * 100).toFixed(0)}%` : "—"}
            sub={analytics.criticalPath ? nodeMap.get(analytics.criticalPath.sourceNodeId)?.label : "none"}
            icon={AlertTriangle}
            color="text-destructive"
            bg="bg-destructive/5"
          />
        </div>

        {/* Distribution panels */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Node type distribution */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-manifest" />
                <CardTitle className="text-sm font-medium">Node Type Distribution</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {(Object.entries(analytics.nodeTypeCounts) as [GraphNodeType, number][]).map(
                ([type, count]) => {
                  const cfg = NODE_TYPE_CONFIG[type];
                  const pct = (count / analytics.totalNodes) * 100;
                  return (
                    <div key={type} className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Circle className={cn("h-2.5 w-2.5", cfg.color)} fill="currentColor" />
                          <span className="text-xs capitalize">{type}</span>
                        </div>
                        <span className="text-xs font-mono text-muted-foreground">
                          {count} <span className="text-[10px]">({pct.toFixed(0)}%)</span>
                        </span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all duration-500",
                            type === "agent" ? "bg-trust" :
                            type === "resource" ? "bg-receipt" :
                            type === "mission" ? "bg-manifest" :
                            type === "receipt" ? "bg-success" :
                            type === "boundary" ? "bg-destructive" :
                            type === "action" ? "bg-warning" :
                            "bg-action"
                          )}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                }
              )}
            </CardContent>
          </Card>

          {/* Edge type distribution */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-trust" />
                <CardTitle className="text-sm font-medium">Edge Type Distribution</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {(Object.entries(analytics.edgeTypeCounts) as [EdgeType, number][]).map(
                ([type, count]) => {
                  const cfg = EDGE_TYPE_CONFIG[type];
                  const pct = (count / analytics.totalEdges) * 100;
                  return (
                    <div key={type} className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <svg width="16" height="8" className="shrink-0">
                            <line
                              x1="0" y1="4" x2="16" y2="4"
                              strokeWidth="2"
                              className={cfg.color}
                              strokeDasharray={cfg.dash === "none" ? undefined : cfg.dash}
                            />
                          </svg>
                          <span className="text-xs">{cfg.label}</span>
                        </div>
                        <span className="text-xs font-mono text-muted-foreground">
                          {count} <span className="text-[10px]">({pct.toFixed(0)}%)</span>
                        </span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all duration-500",
                            type === "depends_on" ? "bg-muted-foreground" :
                            type === "produces" ? "bg-receipt" :
                            type === "verifies" ? "bg-trust" :
                            type === "delegates" ? "bg-manifest" :
                            type === "blocks" ? "bg-destructive" :
                            "bg-muted-foreground"
                          )}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                }
              )}
            </CardContent>
          </Card>
        </div>

        {/* Simulation button */}
        <Card className="border-border/50 bg-card/50">
          <CardContent className="py-5">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <Play className="h-4 w-4 text-trust" />
                  <span className="text-sm font-medium">Run Propagation Simulation</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Trigger a new impact propagation analysis from a random source node.
                  Results will be published to the orchestration event stream.
                </p>
              </div>
              <Button
                onClick={handleSimulation}
                disabled={simulating}
                className={cn(
                  "shrink-0 gap-2",
                  simulating && "animate-pulse"
                )}
              >
                {simulating ? (
                  <>
                    <Activity className="h-4 w-4 animate-spin" />
                    Simulating...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    Run Simulation
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Connection matrix (compact) */}
        <Card className="border-border/50 bg-card/50">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm font-medium">Node Connectivity Matrix</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="w-full dema-scrollbar">
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr>
                      <th className="text-left p-1.5 text-muted-foreground font-medium sticky left-0 bg-card" />
                      {graphSnapshot.nodes.map((n) => (
                        <th key={n.id} className="p-1 text-muted-foreground text-center font-normal min-w-[60px]" title={n.label}>
                          {n.label.slice(0, 8)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {graphSnapshot.nodes.map((row) => (
                      <tr key={row.id}>
                        <td className="p-1.5 text-left sticky left-0 bg-card font-medium" title={row.label}>
                          <span className={cn(NODE_TYPE_CONFIG[row.type].color)}>
                            {row.label.slice(0, 8)}
                          </span>
                        </td>
                        {graphSnapshot.nodes.map((col) => {
                          const hasEdge = graphSnapshot.edges.some(
                            (e) =>
                              (e.source === row.id && e.target === col.id) ||
                              (e.target === row.id && e.source === col.id)
                          );
                          return (
                            <td key={col.id} className="p-1 text-center">
                              {row.id === col.id ? (
                                <div className="w-3 h-3 mx-auto rounded-sm bg-border/30" />
                              ) : hasEdge ? (
                                <div className="w-3 h-3 mx-auto rounded-sm bg-trust/60" />
                              ) : (
                                <div className="w-3 h-3 mx-auto rounded-sm bg-muted/20" />
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}

function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  color,
  bg,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: React.ElementType;
  color: string;
  bg: string;
}) {
  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className={cn("p-1.5 rounded-md", bg)}>
            <Icon className={cn("h-3.5 w-3.5", color)} />
          </div>
        </div>
        <div className={cn("text-lg font-bold font-mono tracking-tight truncate", value.length > 10 && "text-sm")}>
          {value}
        </div>
        <div className="text-[10px] text-muted-foreground mt-0.5">{label}</div>
        {sub && (
          <div className="text-[10px] text-muted-foreground/60 font-mono mt-0.5">{sub}</div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Main Screen ────────────────────────────────────────────────

export function ImpactScreen() {
  const { graphSnapshot } = useDEMAStore();

  return (
    <div className="space-y-4 p-6 max-w-7xl mx-auto dema-fade-in">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-md bg-trust/5">
            <Network className="h-5 w-5 text-trust" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Impact & Graph</h1>
            <p className="text-sm text-muted-foreground mt-0.5">
              Dependency graph reasoning, impact propagation analysis, and network topology.
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="graph" className="w-full">
        <TabsList className="bg-muted/80">
          <TabsTrigger value="graph" className="gap-1.5 text-xs">
            <Network className="h-3.5 w-3.5" />
            Dependency Graph
          </TabsTrigger>
          <TabsTrigger value="propagation" className="gap-1.5 text-xs">
            <GitBranch className="h-3.5 w-3.5" />
            Impact Propagation
          </TabsTrigger>
          <TabsTrigger value="analytics" className="gap-1.5 text-xs">
            <BarChart3 className="h-3.5 w-3.5" />
            Graph Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="mt-3">
          <DependencyGraphTab />
        </TabsContent>

        <TabsContent value="propagation" className="mt-3">
          <ImpactPropagationTab />
        </TabsContent>

        <TabsContent value="analytics" className="mt-3">
          <GraphAnalyticsTab />
        </TabsContent>
      </Tabs>

      {/* Footer timestamp */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground/50 px-1 pb-2">
        <span>
          Snapshot: <span className="font-mono">{graphSnapshot.id}</span>
        </span>
        <span>{timeAgo(graphSnapshot.timestamp)}</span>
      </div>
    </div>
  );
}
