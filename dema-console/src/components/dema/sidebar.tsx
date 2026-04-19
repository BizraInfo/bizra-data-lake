"use client";

import { useDEMAStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  LayoutDashboard,
  MessageSquare,
  FileCheck,
  Box,
  Zap,
  Settings,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Network,
  GitBranch,
  Shield,
  Brain,
  Activity,
  Factory,
} from "lucide-react";
import type { Screen } from "@/lib/types";

const NAV_ITEMS: {
  id: Screen;
  label: string;
  icon: React.ElementType;
  shortcut?: string;
  section?: string;
}[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, shortcut: "⌘1", section: "CORE" },
  { id: "ask", label: "Ask / Research", icon: MessageSquare, shortcut: "⌘2", section: "CORE" },
  { id: "receipts", label: "Receipts & Manifest", icon: FileCheck, shortcut: "⌘3", section: "CORE" },
  { id: "resources", label: "Resources", icon: Box, shortcut: "⌘4", section: "CORE" },
  { id: "actions", label: "Actions", icon: Zap, shortcut: "⌘5", section: "CORE" },
  { id: "orchestration", label: "Orchestration", icon: Network, shortcut: "⌘6", section: "BIZRA" },
  { id: "impact", label: "Impact / Graph", icon: GitBranch, shortcut: "⌘7", section: "BIZRA" },
  { id: "governance", label: "Governance", icon: Shield, shortcut: "⌘8", section: "BIZRA" },
  { id: "autopilot", label: "Autopilot", icon: Brain, shortcut: "⌘9", section: "BIZRA" },
  { id: "operations", label: "Operations", icon: Activity, shortcut: "⌘0", section: "BIZRA" },
  { id: "adk-factory", label: "ADK Factory", icon: Factory, section: "BIZRA" },
];

export function DEMA_Sidebar() {
  const { currentScreen, setScreen, sidebarOpen, setSidebarOpen, trustState } =
    useDEMAStore();

  // Group items by section
  const coreItems = NAV_ITEMS.filter((i) => i.section === "CORE");
  const bizraItems = NAV_ITEMS.filter((i) => i.section === "BIZRA");

  const renderNavButton = (item: typeof NAV_ITEMS[number]) => {
    const Icon = item.icon;
    const isActive = currentScreen === item.id;

    const navButton = (
      <Button
        key={item.id}
        variant="ghost"
        onClick={() => setScreen(item.id)}
        className={cn(
          "w-full h-8 justify-start gap-2 relative transition-all",
          isActive
            ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
            : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
          sidebarOpen ? "px-2.5" : "px-0 justify-center"
        )}
      >
        {isActive && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r bg-trust" />
        )}
        <Icon className={cn("h-3.5 w-3.5 shrink-0", isActive && "text-trust")} />
        {sidebarOpen && (
          <span className="truncate text-xs dema-fade-in">
            {item.label}
          </span>
        )}
      </Button>
    );

    if (sidebarOpen) {
      return navButton;
    }
    return (
      <Tooltip key={item.id}>
        <TooltipTrigger asChild>{navButton}</TooltipTrigger>
        <TooltipContent side="right" className="text-xs">
          {item.label}
          {item.shortcut && (
            <kbd className="ml-2 text-[10px] text-muted-foreground">
              {item.shortcut}
            </kbd>
          )}
        </TooltipContent>
      </Tooltip>
    );
  };

  return (
    <aside
      className={cn(
        "h-full flex flex-col bg-sidebar border-r border-sidebar-border transition-all duration-200 ease-in-out shrink-0",
        sidebarOpen ? "w-52" : "w-12"
      )}
    >
      {/* Logo / Brand */}
      <div className="flex items-center gap-2 px-3 h-12 border-b border-sidebar-border">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-trust/10 shrink-0">
          <Sparkles className="h-3.5 w-3.5 text-trust" />
        </div>
        {sidebarOpen && (
          <div className="flex flex-col min-w-0 dema-fade-in">
            <span className="text-sm font-bold tracking-tight truncate">
              DEMA
            </span>
            <span className="text-[10px] text-muted-foreground truncate">
              Sovereign Operator
            </span>
          </div>
        )}
      </div>

      {/* Scrollable Navigation */}
      <nav className="flex-1 py-1.5 px-2 space-y-0.5 overflow-y-auto dema-scrollbar">
        {/* Core Section */}
        {sidebarOpen && (
          <div className="flex items-center gap-1 px-1 pt-1 pb-0.5">
            <span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground/60">
              Core
            </span>
          </div>
        )}
        {coreItems.map(renderNavButton)}

        <Separator className="my-1.5" />

        {/* BIZRA Section */}
        {sidebarOpen && (
          <div className="flex items-center gap-1 px-1 pt-0.5 pb-0.5">
            <span className="text-[9px] font-semibold uppercase tracking-widest text-manifest">
              Bizra
            </span>
            <div className="flex-1 h-px bg-manifest/20" />
          </div>
        )}
        {bizraItems.map(renderNavButton)}
      </nav>

      <Separator />

      {/* Footer */}
      <div className="p-2 space-y-1">
        {/* Trust Level Indicator */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "flex items-center gap-2 h-8 rounded-md px-2 transition-colors",
                sidebarOpen ? "" : "justify-center",
                "text-sidebar-foreground/50 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
              )}
            >
              <div
                className={cn(
                  "w-2 h-2 rounded-full shrink-0",
                  trustState.isActive ? "bg-success dema-pulse" : "bg-muted-foreground"
                )}
              />
              {sidebarOpen && (
                <span className="text-[11px] truncate dema-fade-in">
                  {trustState.level} · {trustState.score}/{trustState.maxScore}
                </span>
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="right" className="text-xs">
            Trust: {trustState.level} ({trustState.score}/{trustState.maxScore})
          </TooltipContent>
        </Tooltip>

        {/* Collapse Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className={cn(
            "w-full h-7 transition-colors text-sidebar-foreground/50 hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
            sidebarOpen ? "justify-end px-2" : "justify-center px-0"
          )}
        >
          {sidebarOpen ? (
            <>
              <ChevronLeft className="h-3 w-3" />
              <span className="text-[10px] ml-1">Collapse</span>
            </>
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </Button>
      </div>
    </aside>
  );
}
