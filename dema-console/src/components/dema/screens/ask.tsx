"use client";

import { useState, useRef, useEffect } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Send,
  Sparkles,
  Search,
  Link,
  Shield,
  ArrowRight,
  Quote,
  RotateCcw,
  Bookmark,
  Lightbulb,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTimestamp } from "@/lib/helpers/dema";
import type { AskMessage, ResearchCitation } from "@/lib/types";

function CitationChip({ citation }: { citation: ResearchCitation }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-trust/5 border border-trust/10 hover:bg-trust/10 transition-colors text-[11px] text-trust-foreground max-w-[200px] truncate"
        >
          <Link className="h-3 w-3 shrink-0" />
          <span className="truncate">{citation.title}</span>
        </a>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="text-xs max-w-xs">
        <p className="font-medium">{citation.title}</p>
        <p className="text-muted-foreground mt-1">{citation.snippet}</p>
        <p className="text-[10px] text-muted-foreground mt-1">
          Credibility: {Math.round(citation.credibility * 100)}%
        </p>
      </TooltipContent>
    </Tooltip>
  );
}

function MessageBubble({ message }: { message: AskMessage }) {
  const isDema = message.role === "dema";

  return (
    <div
      className={cn(
        "flex gap-3 dema-slide-in",
        isDema ? "flex-row" : "flex-row-reverse"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5",
          isDema ? "bg-trust/10" : "bg-accent"
        )}
      >
        {isDema ? (
          <Sparkles className="h-3.5 w-3.5 text-trust" />
        ) : (
          <div className="h-3.5 w-3.5 rounded-full bg-foreground/20" />
        )}
      </div>

      {/* Content */}
      <div className={cn("max-w-[70%] space-y-2", isDema ? "" : "text-right")}>
        <div
          className={cn(
            "text-sm leading-relaxed whitespace-pre-wrap",
            isDema
              ? "bg-card border border-border/50 rounded-xl rounded-tl-sm p-4 text-left"
              : "bg-primary text-primary-foreground rounded-xl rounded-tr-sm p-4 text-left"
          )}
        >
          {message.content}
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c) => (
              <CitationChip key={c.id} citation={c} />
            ))}
          </div>
        )}

        {/* Metadata row */}
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span>{formatTimestamp(message.timestamp)}</span>
          {message.confidence !== undefined && isDema && (
            <>
              <span>·</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="flex items-center gap-0.5">
                    <Shield className="h-2.5 w-2.5" />
                    {Math.round(message.confidence * 100)}%
                  </span>
                </TooltipTrigger>
                <TooltipContent className="text-xs">Answer confidence</TooltipContent>
              </Tooltip>
            </>
          )}
          {message.trustState && isDema && (
            <>
              <span>·</span>
              <Badge variant="outline" className="text-[9px] px-1 py-0 h-3">
                {message.trustState}
              </Badge>
            </>
          )}
        </div>

        {/* Next action suggestion */}
        {message.nextAction && isDema && (
          <div className="flex items-center gap-1.5 px-2 py-1.5 rounded-md bg-warning/5 border border-warning/10 w-fit">
            <ArrowRight className="h-3 w-3 text-warning" />
            <span className="text-[11px] text-warning">{message.nextAction}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function AskScreen() {
  const { askMessages, addAskMessage } = useDEMAStore();
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [mode, setMode] = useState<"ask" | "research">("ask");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [askMessages]);

  const [sessionId] = useState(() => `sess-${crypto.randomUUID()}`);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: AskMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    addAskMessage(userMessage);
    const query = input.trim();
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query, mode, sessionId }),
      });

      const json = await res.json();

      if (json.success && json.data) {
        const d = json.data;
        const demaResponse: AskMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "dema",
          content: d.content || "No response generated.",
          confidence: d.confidence,
          trustState: d.trustState,
          nextAction: d.nextAction,
          timestamp: new Date().toISOString(),
          citations: d.citations?.length > 0 ? d.citations : undefined,
        };
        addAskMessage(demaResponse);
      } else {
        addAskMessage({
          id: `msg-${Date.now() + 1}`,
          role: "dema",
          content: "I encountered an error processing your request. Please try again.",
          confidence: 0,
          timestamp: new Date().toISOString(),
        });
      }
    } catch {
      addAskMessage({
        id: `msg-${Date.now() + 1}`,
        role: "dema",
        content: "Connection error. Please check the server and try again.",
        confidence: 0,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border/50">
        <div className="flex items-center gap-3">
          <Sparkles className="h-4 w-4 text-trust" />
          <div>
            <h1 className="text-sm font-semibold">
              {mode === "ask" ? "Ask DEMA" : "Research Mode"}
            </h1>
            <p className="text-[11px] text-muted-foreground">
              {mode === "ask"
                ? "Calm answers with citations, confidence, and next actions"
                : "Deep cited research with source-backed analysis"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Select value={mode} onValueChange={(v) => setMode(v as "ask" | "research")}>
            <SelectTrigger className="h-7 text-xs w-[130px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ask">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3" />
                  Ask Mode
                </div>
              </SelectItem>
              <SelectItem value="research">
                <div className="flex items-center gap-1.5">
                  <Search className="h-3 w-3" />
                  Research Mode
                </div>
              </SelectItem>
            </SelectContent>
          </Select>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
                <Bookmark className="h-3.5 w-3.5 text-muted-foreground" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="text-xs">Saved research</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => useDEMAStore.getState().clearAskMessages()}
              >
                <RotateCcw className="h-3.5 w-3.5 text-muted-foreground" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="text-xs">Reset conversation</TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 px-6 py-4" ref={scrollRef}>
        <div className="space-y-4 max-w-3xl mx-auto">
          {askMessages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isTyping && (
            <div className="flex gap-3 dema-slide-in">
              <div className="w-7 h-7 rounded-lg bg-trust/10 flex items-center justify-center shrink-0">
                <Sparkles className="h-3.5 w-3.5 text-trust dema-pulse" />
              </div>
              <div className="bg-card border border-border/50 rounded-xl rounded-tl-sm p-4">
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 dema-pulse" />
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 dema-pulse [animation-delay:0.2s]" />
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 dema-pulse [animation-delay:0.4s]" />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border/50 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  mode === "ask"
                    ? "Ask anything — cite, verify, act..."
                    : "Enter a research query for cited analysis..."
                }
                className="min-h-[44px] max-h-[120px] resize-none pr-12 text-sm"
                rows={1}
              />
              <Button
                size="sm"
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
                className="absolute right-2 bottom-2 h-8 w-8 p-0 rounded-md"
              >
                <Send className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between mt-2 px-1">
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <Info className="h-2.5 w-2.5" />
                Press Enter to send, Shift+Enter for new line
              </span>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-0.5">
                <Quote className="h-2.5 w-2.5" />
                {mode === "research" ? "Citations enabled" : "Ask mode"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
