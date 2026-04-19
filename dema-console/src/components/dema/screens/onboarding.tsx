"use client";

import { useState } from "react";
import { useDEMAStore } from "@/lib/store";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Sparkles,
  Globe,
  User,
  Monitor,
  Shield,
  ShieldCheck,
  Heart,
  Scale,
  Cpu,
  HardDrive,
  Network,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  Sun,
  Moon,
  Languages,
  Briefcase,
  Users,
  GraduationCap,
  Wrench,
  HandHeart,
  Lock,
  Eye,
  EyeOff,
  BookOpen,
  Star,
  Fingerprint,
  Rocket,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ═══════════════════════════════════════════════════════════════
// First Citizen Path — DEMA Onboarding Protocol v1
// From Stranger to Sovereign User
// ═══════════════════════════════════════════════════════════════

const LANGUAGES = [
  { code: "ar", name: "العربية", native: "Arabic", rtl: true },
  { code: "en", name: "English", native: "English", rtl: false },
  { code: "zh", name: "中文", native: "Chinese", rtl: false },
  { code: "hi", name: "हिन्दी", native: "Hindi", rtl: false },
  { code: "es", name: "Español", native: "Spanish", rtl: false },
  { code: "fr", name: "Français", native: "French", rtl: false },
  { code: "pt", name: "Português", native: "Portuguese", rtl: false },
  { code: "bn", name: "বাংলা", native: "Bengali", rtl: false },
  { code: "ur", name: "اردو", native: "Urdu", rtl: true },
  { code: "id", name: "Bahasa Indonesia", native: "Indonesian", rtl: false },
  { code: "tr", name: "Türkçe", native: "Turkish", rtl: false },
  { code: "de", name: "Deutsch", native: "German", rtl: false },
  { code: "ja", name: "日本語", native: "Japanese", rtl: false },
  { code: "ko", name: "한국어", native: "Korean", rtl: false },
  { code: "ms", name: "Bahasa Melayu", native: "Malay", rtl: false },
  { code: "sw", name: "Kiswahili", native: "Swahili", rtl: false },
  { code: "ru", name: "Русский", native: "Russian", rtl: false },
  { code: "fa", name: "فارسی", native: "Persian", rtl: true },
];

const TOTAL_STAGES = 10;

// ─── Stage 0: Entry Gate ──────────────────────────────────────
function StageEntryGate({ onContinue }: { onContinue: () => void }) {
  return (
    <div className="space-y-8 text-center max-w-lg mx-auto">
      {/* DEMA Mark */}
      <div className="space-y-4">
        <div className="w-20 h-20 rounded-3xl bg-trust/10 flex items-center justify-center mx-auto ring-1 ring-trust/20">
          <Sparkles className="h-10 w-10 text-trust" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">DEMA</h1>
          <p className="text-sm text-trust font-medium mt-1">
            The Sovereign Operator
          </p>
        </div>
      </div>

      {/* The Core Message */}
      <div className="space-y-4 py-2">
        <p className="text-base text-foreground/80 leading-relaxed">
          DEMA does not begin by asking for your data.
        </p>
        <p className="text-base text-foreground/80 leading-relaxed">
          DEMA begins by learning <strong className="text-foreground">how to speak to you</strong>,
          then asks <strong className="text-foreground">who you are</strong>,
          what you hope to build, and only then asks what parts of your world
          you want it to help steward.
        </p>
        <div className="pt-2">
          <p className="text-sm text-muted-foreground italic leading-relaxed">
            &ldquo;The more I learned, the more I realized my ignorance —
            and that what I see as correct may carry error,
            and what I see as wrong in another may carry truth.&rdquo;
          </p>
          <p className="text-xs text-muted-foreground/60 mt-2">
            — The spirit that guides DEMA
          </p>
        </div>
      </div>

      <Separator className="max-w-xs mx-auto" />

      {/* Purpose Cards */}
      <div className="grid grid-cols-3 gap-3 max-w-md mx-auto">
        {[
          {
            icon: Scale,
            label: "Truth",
            desc: "Fight destructive assumptions",
            color: "text-trust",
            bg: "bg-trust/10",
          },
          {
            icon: Shield,
            label: "Dignity",
            desc: "Resist extraction & riba",
            color: "text-receipt",
            bg: "bg-receipt/10",
          },
          {
            icon: HandHeart,
            label: "Empower",
            desc: "Every human, equally",
            color: "text-manifest",
            bg: "bg-manifest/10",
          },
        ].map((item) => {
          const I = item.icon;
          return (
            <Tooltip key={item.label}>
              <TooltipTrigger asChild>
                <div className="flex flex-col items-center gap-2 p-3 rounded-xl border border-border/30 hover:border-border/60 transition-colors cursor-default">
                  <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", item.bg)}>
                    <I className={cn("h-5 w-5", item.color)} />
                  </div>
                  <span className="text-xs font-semibold">{item.label}</span>
                  <span className="text-[10px] text-muted-foreground leading-tight text-center">
                    {item.desc}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs max-w-[200px]">
                {item.desc}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>

      {/* Enter */}
      <Button size="lg" onClick={onContinue} className="mt-4">
        Begin Your Journey
        <ChevronRight className="h-4 w-4 ml-2" />
      </Button>

      <p className="text-[10px] text-muted-foreground/50">
        DEMA serves with humility, truth, dignity, and care — never extraction, never deception, never domination.
      </p>
    </div>
  );
}

// ─── Stage 1: Language ─────────────────────────────────────────
function StageLanguage({
  motherLang,
  setMotherLang,
  secondLang,
  setSecondLang,
  onContinue,
  onBack,
}: {
  motherLang: string;
  setMotherLang: (l: string) => void;
  secondLang: string;
  setSecondLang: (l: string) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-manifest/10 flex items-center justify-center mx-auto">
          <Languages className="h-7 w-7 text-manifest" />
        </div>
        <h2 className="text-xl font-bold">How should DEMA speak to you?</h2>
        <p className="text-sm text-muted-foreground">
          Language is the first trust layer. Choose how DEMA communicates.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-5">
          <div className="space-y-2">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Sun className="h-3.5 w-3.5 text-trust" />
              Your mother tongue
            </Label>
            <div className="grid grid-cols-3 gap-2 max-h-40 overflow-y-auto dema-scrollbar pr-1">
              {LANGUAGES.map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => setMotherLang(lang.code)}
                  className={cn(
                    "text-left px-3 py-2 rounded-lg text-xs border transition-all",
                    motherLang === lang.code
                      ? "border-trust bg-trust/5 text-trust-foreground font-medium"
                      : "border-border/30 hover:border-border/60 text-muted-foreground"
                  )}
                >
                  <div className="font-medium">{lang.name}</div>
                  <div className="text-[10px] opacity-60">{lang.native}</div>
                </button>
              ))}
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Moon className="h-3.5 w-3.5 text-manifest" />
              Second language
              <span className="text-[10px] text-muted-foreground font-normal">(optional)</span>
            </Label>
            <div className="grid grid-cols-3 gap-2 max-h-28 overflow-y-auto dema-scrollbar pr-1">
              {LANGUAGES.filter((l) => l.code !== motherLang).map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => setSecondLang(secondLang === lang.code ? "" : lang.code)}
                  className={cn(
                    "text-left px-3 py-2 rounded-lg text-xs border transition-all",
                    secondLang === lang.code
                      ? "border-manifest bg-manifest/5 text-manifest-foreground font-medium"
                      : "border-border/30 hover:border-border/60 text-muted-foreground"
                  )}
                >
                  <div className="font-medium">{lang.name}</div>
                  <div className="text-[10px] opacity-60">{lang.native}</div>
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} disabled={!motherLang} className="text-xs">
          Continue
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 2: Human Profile ────────────────────────────────────
function StageHumanProfile({
  name,
  setName,
  work,
  setWork,
  goal,
  setGoal,
  techLevel,
  setTechLevel,
  onContinue,
  onBack,
}: {
  name: string;
  setName: (n: string) => void;
  work: string;
  setWork: (w: string) => void;
  goal: string;
  setGoal: (g: string) => void;
  techLevel: string;
  setTechLevel: (l: string) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-trust/10 flex items-center justify-center mx-auto">
          <User className="h-7 w-7 text-trust" />
        </div>
        <h2 className="text-xl font-bold">Who are you?</h2>
        <p className="text-sm text-muted-foreground">
          DEMA learns about you to serve you better. This feels like respectful calibration, not interrogation.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="user-name" className="text-sm font-medium flex items-center gap-2">
              <User className="h-3.5 w-3.5 text-trust" />
              Your name
            </Label>
            <Input
              id="user-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="What should DEMA call you?"
              className="text-sm h-10"
              autoFocus
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="user-work" className="text-sm font-medium flex items-center gap-2">
              <Briefcase className="h-3.5 w-3.5 text-manifest" />
              Your work or role
              <span className="text-[10px] text-muted-foreground font-normal">(optional)</span>
            </Label>
            <Input
              id="user-work"
              value={work}
              onChange={(e) => setWork(e.target.value)}
              placeholder="e.g. Software Engineer, Student, Entrepreneur..."
              className="text-sm h-10"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="user-goal" className="text-sm font-medium flex items-center gap-2">
              <Star className="h-3.5 w-3.5 text-trust" />
              What do you want DEMA to help with first?
            </Label>
            <Input
              id="user-goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Organize my projects, Research a topic, Build something..."
              className="text-sm h-10"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium flex items-center gap-2">
              <GraduationCap className="h-3.5 w-3.5 text-action" />
              Technical comfort level
            </Label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: "beginner", label: "Beginner", icon: Users },
                { value: "intermediate", label: "Intermediate", icon: Wrench },
                { value: "advanced", label: "Advanced", icon: Cpu },
              ].map((level) => {
                const I = level.icon;
                return (
                  <button
                    key={level.value}
                    onClick={() => setTechLevel(level.value)}
                    className={cn(
                      "flex flex-col items-center gap-1.5 p-3 rounded-lg border transition-all",
                      techLevel === level.value
                        ? "border-action bg-action/5 text-action-foreground"
                        : "border-border/30 hover:border-border/60 text-muted-foreground"
                    )}
                  >
                    <I className="h-4 w-4" />
                    <span className="text-[11px] font-medium">{level.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} disabled={!name.trim()} className="text-xs">
          Continue
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 3: Device Topology ──────────────────────────────────
function StageDeviceTopology({
  deviceCount,
  setDeviceCount,
  onContinue,
  onBack,
}: {
  deviceCount: number;
  setDeviceCount: (n: number) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-action/10 flex items-center justify-center mx-auto">
          <Monitor className="h-7 w-7 text-action" />
        </div>
        <h2 className="text-xl font-bold">Your Devices</h2>
        <p className="text-sm text-muted-foreground">
          How many personal devices do you own that you want to use as node assets?
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium">Number of devices</Label>
            <div className="grid grid-cols-5 gap-2">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  onClick={() => setDeviceCount(n)}
                  className={cn(
                    "h-12 rounded-lg border text-lg font-bold transition-all",
                    deviceCount === n
                      ? "border-action bg-action/5 text-action"
                      : "border-border/30 hover:border-border/60 text-muted-foreground"
                  )}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Your device types</Label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { icon: Monitor, label: "Desktop / Laptop", os: "Windows, macOS, Linux" },
                { icon: Cpu, label: "Server / NAS", os: "Linux, BSD" },
                { icon: Network, label: "Single Board", os: "Raspberry Pi, etc." },
                { icon: HardDrive, label: "External Storage", os: "USB, SSD, HDD" },
              ].map((device) => {
                const I = device.icon;
                return (
                  <div
                    key={device.label}
                    className="flex items-start gap-2.5 p-3 rounded-lg border border-border/30 bg-muted/10"
                  >
                    <I className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                    <div>
                      <div className="text-xs font-medium">{device.label}</div>
                      <div className="text-[10px] text-muted-foreground">{device.os}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <p className="text-[10px] text-muted-foreground bg-muted/20 rounded-lg p-3 leading-relaxed">
            <Lock className="h-3 w-3 inline mr-1" />
            DEMA will never scan your devices without your explicit, staged consent.
            You control what is inspected, what is touched, and what remains local.
          </p>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} disabled={deviceCount === 0} className="text-xs">
          Continue
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 4: Permissioned Scan ────────────────────────────────
function StagePermissionedScan({
  scanApproved,
  setScanApproved,
  onContinue,
  onBack,
}: {
  scanApproved: boolean;
  setScanApproved: (a: boolean) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-receipt/10 flex items-center justify-center mx-auto">
          <Shield className="h-7 w-7 text-receipt" />
        </div>
        <h2 className="text-xl font-bold">Permission & Privacy</h2>
        <p className="text-sm text-muted-foreground">
          DEMA asks before looking. Nothing is touched without your explicit approval.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-4">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">What DEMA wants to inspect</h3>
            <div className="space-y-2">
              {[
                { label: "Operating system & version", icon: Monitor, desc: "To match config to your environment" },
                { label: "CPU, RAM, GPU specs", icon: Cpu, desc: "To assess compute capability" },
                { label: "Available disk space", icon: HardDrive, desc: "To plan local storage for your node" },
                { label: "Network connectivity", icon: Network, desc: "To understand your connectivity" },
              ].map((item) => {
                const I = item.icon;
                return (
                  <div key={item.label} className="flex items-center gap-3 p-2.5 rounded-lg bg-muted/10">
                    <I className="h-4 w-4 text-muted-foreground shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium">{item.label}</div>
                      <div className="text-[10px] text-muted-foreground">{item.desc}</div>
                    </div>
                    <CheckCircle2 className="h-3.5 w-3.5 text-success shrink-0" />
                  </div>
                );
              })}
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <h3 className="text-sm font-semibold">What DEMA will NOT do</h3>
            <div className="space-y-2">
              {[
                { label: "Read your personal files", denied: true },
                { label: "Access your credentials", denied: true },
                { label: "Install anything without consent", denied: true },
                { label: "Upload your data to any cloud", denied: true },
                { label: "Modify your system settings", denied: true },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3 p-2 rounded-lg">
                  <EyeOff className="h-3.5 w-3.5 text-destructive shrink-0" />
                  <span className="text-xs text-muted-foreground flex-1">{item.label}</span>
                  <Badge variant="outline" className="text-[10px] text-destructive border-destructive/20">
                    Never
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button
          size="sm"
          onClick={() => { setScanApproved(true); onContinue(); }}
          className="text-xs bg-receipt hover:bg-receipt/90 text-white"
        >
          <ShieldCheck className="h-3.5 w-3.5 mr-1" />
          Grant Permission & Continue
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 5: Node Readiness ───────────────────────────────────
function StageNodeReadiness({
  onContinue,
  onBack,
}: {
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-success/10 flex items-center justify-center mx-auto">
          <Cpu className="h-7 w-7 text-success" />
        </div>
        <h2 className="text-xl font-bold">Node Readiness</h2>
        <p className="text-sm text-muted-foreground">
          Your &ldquo;you are here&rdquo; report — detected capabilities and recommendations.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-4">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-success" />
              Detected Hardware
            </h3>
            {[
              { label: "OS", value: "Browser Environment", status: "detected" },
              { label: "CPU Cores", value: "Available", status: "detected" },
              { label: "Memory", value: "Sufficient", status: "sufficient" },
              { label: "Network", value: "Connected", status: "healthy" },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between p-2 rounded-lg bg-muted/10">
                <span className="text-xs text-muted-foreground">{item.label}</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium">{item.value}</span>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-[9px]",
                      item.status === "sufficient" ? "text-success border-success/20" : "text-manifest border-manifest/20"
                    )}
                  >
                    {item.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>

          <Separator />

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Compatibility Score</h3>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold text-success">87</div>
              <div className="flex-1">
                <Progress value={87} className="h-2" />
                <p className="text-[10px] text-muted-foreground mt-1">
                  Your node is ready for local-first operation
                </p>
              </div>
            </div>
          </div>

          <div className="bg-trust/5 border border-trust/10 rounded-lg p-3">
            <p className="text-xs text-trust-foreground font-medium">
              Recommendation
            </p>
            <p className="text-[11px] text-muted-foreground mt-1">
              For optimal performance, consider installing the DEMA desktop app on your primary device.
              The web version runs perfectly for most tasks.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} className="text-xs">
          Continue
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 6: BIZRA / DEMA Introduction ────────────────────────
function StageIntroduction({
  onContinue,
  onBack,
}: {
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-lg mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-trust/10 flex items-center justify-center mx-auto">
          <BookOpen className="h-7 w-7 text-trust" />
        </div>
        <h2 className="text-xl font-bold">What is BIZRA? What is DEMA?</h2>
      </div>

      <div className="space-y-4">
        {/* BIZRA */}
        <Card className="border-border/30">
          <CardContent className="p-5 space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-trust/10 flex items-center justify-center">
                <Sparkles className="h-4 w-4 text-trust" />
              </div>
              <div>
                <h3 className="text-sm font-bold">BIZRA</h3>
                <p className="text-[10px] text-muted-foreground">The Ecosystem</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              BIZRA stands on three parallel pillars that fuse into something new:
            </p>
            <div className="space-y-2">
              {[
                {
                  icon: Heart,
                  label: "Ideology",
                  desc: "Meaning, dignity, anti-riba, anti-assumption. The why.",
                  color: "text-trust",
                  bg: "bg-trust/10",
                },
                {
                  icon: Brain,
                  label: "AI",
                  desc: "Cognition, agency, and interface. The how.",
                  color: "text-manifest",
                  bg: "bg-manifest/10",
                },
                {
                  icon: ShieldCheck,
                  label: "Blockchain",
                  desc: "Proof, persistence, and value. The truth layer.",
                  color: "text-receipt",
                  bg: "bg-receipt/10",
                },
              ].map((pillar) => {
                const I = pillar.icon;
                return (
                  <div key={pillar.label} className="flex items-center gap-3 p-2.5 rounded-lg bg-muted/10">
                    <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center shrink-0", pillar.bg)}>
                      <I className={cn("h-4 w-4", pillar.color)} />
                    </div>
                    <div>
                      <div className="text-xs font-semibold">{pillar.label}</div>
                      <div className="text-[10px] text-muted-foreground">{pillar.desc}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* DEMA */}
        <Card className="border-border/30 border-trust/20">
          <CardContent className="p-5 space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-trust/10 flex items-center justify-center">
                <Sparkles className="h-4 w-4 text-trust" />
              </div>
              <div>
                <h3 className="text-sm font-bold">DEMA</h3>
                <p className="text-[10px] text-trust">Your personal sovereign guide</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              DEMA is the one visible face of BIZRA — built by a human being, not a Silicon Valley startup,
              not a venture-backed extraction machine. It was born from pain, meaning, and a covenant
              to serve humanity — not to extract from it.
            </p>
            <div className="bg-trust/5 rounded-lg p-3 border border-trust/10">
              <p className="text-xs text-trust-foreground leading-relaxed">
                DEMA exists to confront the silent killers of human flourishing:
                assumption without truth, extraction without justice,
                and despair that makes a person believe they are powerless.
              </p>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              DEMA gives each person a personal think tank and task force —
              regardless of belief, color, wealth, or status — to help them
              think better, act better, and rise beyond what they thought was possible.
            </p>
          </CardContent>
        </Card>

        {/* The Philosophy */}
        <Card className="border-border/30 bg-muted/5">
          <CardContent className="p-5 text-center space-y-2">
            <p className="text-sm leading-loose" dir="rtl" style={{ fontFamily: "serif" }}>
              كلما ازددت علماً ازددت يقيناً بجهلي
            </p>
            <p className="text-sm leading-loose" dir="rtl" style={{ fontFamily: "serif" }}>
              وأن رأيي صواب يحتمل الخطأ
            </p>
            <p className="text-sm leading-loose" dir="rtl" style={{ fontFamily: "serif" }}>
              وأن رأي غيري خطأ يحتمل الصواب
            </p>
            <Separator className="my-2" />
            <p className="text-[10px] text-muted-foreground italic">
              &ldquo;The more I learned, the more I realized my ignorance —
              and that what I see as correct may carry error,
              and what I see as wrong in another may carry truth.&rdquo;
            </p>
            <p className="text-[10px] text-muted-foreground/60">
              This is DEMA&apos;s permanent spirit.
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} className="text-xs">
          I Understand
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 7: Resource Contribution ────────────────────────────
function StageResourceContribution({
  contributionLevel,
  setContributionLevel,
  onContinue,
  onBack,
}: {
  contributionLevel: string;
  setContributionLevel: (l: string) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-warning/10 flex items-center justify-center mx-auto">
          <HandHeart className="h-7 w-7 text-warning" />
        </div>
        <h2 className="text-xl font-bold">Sharing & Contribution</h2>
        <p className="text-sm text-muted-foreground">
          This is entirely optional. You stay in full control.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium">Choose your contribution level</Label>
            <div className="space-y-2">
              {[
                {
                  value: "private",
                  label: "Private Only",
                  desc: "All resources stay on your device. No sharing.",
                  icon: Lock,
                  color: "text-success",
                  bg: "bg-success/10",
                },
                {
                  value: "local",
                  label: "Local First",
                  desc: "Use DEMA locally. May opt-in to sharing later.",
                  icon: Monitor,
                  color: "text-manifest",
                  bg: "bg-manifest/10",
                  recommended: true,
                },
                {
                  value: "share",
                  label: "Contributor",
                  desc: "Share some resources for impact & optional future income as tokens.",
                  icon: Network,
                  color: "text-trust",
                  bg: "bg-trust/10",
                },
              ].map((option) => {
                const I = option.icon;
                return (
                  <button
                    key={option.value}
                    onClick={() => setContributionLevel(option.value)}
                    className={cn(
                      "w-full flex items-start gap-3 p-4 rounded-xl border transition-all text-left",
                      contributionLevel === option.value
                        ? "border-trust bg-trust/5 ring-1 ring-trust/20"
                        : "border-border/30 hover:border-border/60"
                    )}
                  >
                    <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center shrink-0", option.bg)}>
                      <I className={cn("h-5 w-5", option.color)} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold">{option.label}</span>
                        {option.recommended && (
                          <Badge className="text-[9px] bg-trust/10 text-trust border-trust/20">
                            Recommended
                          </Badge>
                        )}
                      </div>
                      <p className="text-[11px] text-muted-foreground mt-0.5">{option.desc}</p>
                    </div>
                    {contributionLevel === option.value && (
                      <CheckCircle2 className="h-5 w-5 text-trust shrink-0 mt-1" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <p className="text-[10px] text-muted-foreground bg-muted/20 rounded-lg p-3 leading-relaxed">
            <Shield className="h-3 w-3 inline mr-1" />
            You can change this choice at any time from Settings. All contribution is opt-in, consent-based,
            and you retain full ownership of your resources and data.
          </p>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} disabled={!contributionLevel} className="text-xs">
          Continue
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 8: Identity Mint ────────────────────────────────────
function StageIdentityMint({
  name,
  onContinue,
  onBack,
}: {
  name: string;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-trust/10 flex items-center justify-center mx-auto">
          <Fingerprint className="h-7 w-7 text-trust" />
        </div>
        <h2 className="text-xl font-bold">Your Sovereign Identity</h2>
        <p className="text-sm text-muted-foreground">
          DEMA is now creating your local identity, node identity, and first trust state.
        </p>
      </div>

      <Card className="border-trust/20 bg-trust/5">
        <CardContent className="p-5 space-y-4">
          <div className="text-center space-y-3">
            <div className="w-16 h-16 rounded-full bg-trust/10 flex items-center justify-center mx-auto ring-2 ring-trust/20">
              <span className="text-2xl font-bold text-trust">
                {name.charAt(0).toUpperCase()}
              </span>
            </div>
            <div>
              <p className="text-lg font-bold">{name}</p>
              <p className="text-xs text-trust">Sovereign Operator</p>
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            {[
              { label: "Principal ID", value: `prin-${Math.random().toString(36).slice(2, 10)}`, icon: Fingerprint },
              { label: "Node Identity", value: `node-0-${Math.random().toString(36).slice(2, 6)}`, icon: Monitor },
              { label: "Trust Level", value: "Citizen", icon: Shield },
              { label: "Session ID", value: `sess-${Math.random().toString(36).slice(2, 10)}`, icon: Lock },
              { label: "First Receipt", value: "Identity Activation", icon: CheckCircle2 },
            ].map((item) => {
              const I = item.icon;
              return (
                <div key={item.label} className="flex items-center justify-between p-2 rounded-lg">
                  <div className="flex items-center gap-2">
                    <I className="h-3.5 w-3.5 text-trust" />
                    <span className="text-xs text-muted-foreground">{item.label}</span>
                  </div>
                  <span className="text-xs font-mono font-medium">{item.value}</span>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] text-muted-foreground text-center">
            All identity data is stored locally on your device. DEMA does not transmit your identity to any external server.
          </p>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} className="text-xs bg-trust hover:bg-trust/90 text-trust-foreground">
          <Fingerprint className="h-3.5 w-3.5 mr-1" />
          Mint Identity & Continue
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 9: First Mission ────────────────────────────────────
function StageFirstMission({
  missionChoice,
  setMissionChoice,
  onContinue,
  onBack,
}: {
  missionChoice: string;
  setMissionChoice: (m: string) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-action/10 flex items-center justify-center mx-auto">
          <Rocket className="h-7 w-7 text-action" />
        </div>
        <h2 className="text-xl font-bold">Your First Mission</h2>
        <p className="text-sm text-muted-foreground">
          Onboarding should end with one concrete action that improves your life immediately.
        </p>
      </div>

      <Card className="border-border/30">
        <CardContent className="p-5 space-y-3">
          <Label className="text-sm font-medium">Choose your first mission</Label>
          <div className="space-y-2">
            {[
              {
                value: "organize",
                label: "Organize My Space",
                desc: "Map your workspace structure and create a clean inventory of your files and projects.",
                icon: HardDrive,
                color: "text-receipt",
              },
              {
                value: "explore",
                label: "Explore DEMA",
                desc: "Take a guided tour of all operator modes and discover what DEMA can do for you.",
                icon: Compass,
                color: "text-manifest",
              },
              {
                value: "ask",
                label: "Ask Me Anything",
                desc: "Start a conversation. Ask about your projects, research a topic, or get help with code.",
                icon: Sparkles,
                color: "text-trust",
              },
            ].map((mission) => {
              const I = mission.icon;
              return (
                <button
                  key={mission.value}
                  onClick={() => setMissionChoice(mission.value)}
                  className={cn(
                    "w-full flex items-start gap-3 p-4 rounded-xl border transition-all text-left",
                    missionChoice === mission.value
                      ? "border-action bg-action/5 ring-1 ring-action/20"
                      : "border-border/30 hover:border-border/60"
                  )}
                >
                  <div className="w-10 h-10 rounded-xl bg-action/10 flex items-center justify-center shrink-0">
                    <I className={cn("h-5 w-5", mission.color)} />
                  </div>
                  <div className="flex-1">
                    <span className="text-sm font-semibold">{mission.label}</span>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{mission.desc}</p>
                  </div>
                  {missionChoice === mission.value && (
                    <CheckCircle2 className="h-5 w-5 text-action shrink-0 mt-1" />
                  )}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Back
        </Button>
        <Button size="sm" onClick={onContinue} disabled={!missionChoice} className="text-xs">
          Begin Mission
          <ArrowRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// ─── Stage 10: Activation Complete ─────────────────────────────
function StageActivationComplete({
  name,
  onComplete,
}: {
  name: string;
  onComplete: () => void;
}) {
  return (
    <div className="space-y-8 text-center max-w-lg mx-auto">
      <div className="space-y-4">
        <div className="w-20 h-20 rounded-3xl bg-success/10 flex items-center justify-center mx-auto ring-2 ring-success/20">
          <CheckCircle2 className="h-10 w-10 text-success" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Welcome, {name}</h1>
          <p className="text-sm text-success font-medium mt-1">
            Your sovereign session is now active
          </p>
        </div>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed max-w-md mx-auto">
        You are now a sovereign operator. Your trust strip is visible, your identity is minted,
        and your first receipt has been issued. Everything stays on your device.
      </p>

      {/* What you see now */}
      <Card className="border-border/30 max-w-sm mx-auto">
        <CardContent className="p-5 space-y-3">
          <h3 className="text-sm font-semibold text-left">Your persistent home shows:</h3>
          <div className="space-y-2">
            {[
              { label: "Who you are", icon: User, color: "text-trust" },
              { label: "Node identity & trust state", icon: Fingerprint, color: "text-trust" },
              { label: "Current → Ideal state gap", icon: ArrowRight, color: "text-gap" },
              { label: "Next admissible action", icon: Star, color: "text-trust" },
              { label: "Recent receipts & evidence", icon: CheckCircle2, color: "text-receipt" },
              { label: "Local resources & memory", icon: HardDrive, color: "text-manifest" },
            ].map((item) => {
              const I = item.icon;
              return (
                <div key={item.label} className="flex items-center gap-3 p-2 rounded-lg bg-muted/10">
                  <I className={cn("h-4 w-4", item.color)} />
                  <span className="text-xs">{item.label}</span>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Button size="lg" onClick={onComplete} className="bg-trust hover:bg-trust/90 text-trust-foreground">
        Enter DEMA
        <ArrowRight className="h-4 w-4 ml-2" />
      </Button>

      <p className="text-[10px] text-muted-foreground/50 italic">
        &ldquo;DEMA exists to help the user see more clearly, choose more wisely, and rise more fully.&rdquo;
      </p>
    </div>
  );
}

// ─── Main Onboarding Screen ────────────────────────────────────

export function OnboardingScreen() {
  const { completeOnboarding, onboardingStep, setOnboardingStep } = useDEMAStore();
  const [name, setName] = useState("");
  const [work, setWork] = useState("");
  const [goal, setGoal] = useState("");
  const [techLevel, setTechLevel] = useState("");
  const [motherLang, setMotherLang] = useState("");
  const [secondLang, setSecondLang] = useState("");
  const [deviceCount, setDeviceCount] = useState(0);
  const [scanApproved, setScanApproved] = useState(false);
  const [contributionLevel, setContributionLevel] = useState("");
  const [missionChoice, setMissionChoice] = useState("");

  const progress = onboardingStep === 0
    ? 0
    : ((onboardingStep) / TOTAL_STAGES) * 100;

  const handleContinue = () => {
    if (onboardingStep === TOTAL_STAGES) {
      completeOnboarding(name.trim() || "Operator");
    } else {
      setOnboardingStep(onboardingStep + 1);
    }
  };

  const handleBack = () => {
    if (onboardingStep > 0) {
      setOnboardingStep(onboardingStep - 1);
    }
  };

  const renderStage = () => {
    switch (onboardingStep) {
      case 0:
        return <StageEntryGate onContinue={handleContinue} />;
      case 1:
        return (
          <StageLanguage
            motherLang={motherLang} setMotherLang={setMotherLang}
            secondLang={secondLang} setSecondLang={setSecondLang}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 2:
        return (
          <StageHumanProfile
            name={name} setName={setName}
            work={work} setWork={setWork}
            goal={goal} setGoal={setGoal}
            techLevel={techLevel} setTechLevel={setTechLevel}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 3:
        return (
          <StageDeviceTopology
            deviceCount={deviceCount} setDeviceCount={setDeviceCount}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 4:
        return (
          <StagePermissionedScan
            scanApproved={scanApproved} setScanApproved={setScanApproved}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 5:
        return (
          <StageNodeReadiness
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 6:
        return (
          <StageIntroduction
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 7:
        return (
          <StageResourceContribution
            contributionLevel={contributionLevel} setContributionLevel={setContributionLevel}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 8:
        return (
          <StageIdentityMint
            name={name}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 9:
        return (
          <StageFirstMission
            missionChoice={missionChoice} setMissionChoice={setMissionChoice}
            onContinue={handleContinue} onBack={handleBack}
          />
        );
      case 10:
        return (
          <StageActivationComplete
            name={name}
            onComplete={() => completeOnboarding(name.trim() || "Operator")}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Progress bar — hidden on first and last stage */}
      {onboardingStep > 0 && onboardingStep < TOTAL_STAGES && (
        <div className="px-6 pt-4">
          <div className="max-w-lg mx-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-muted-foreground">
                Stage {onboardingStep} of {TOTAL_STAGES}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {Math.round(progress)}%
              </span>
            </div>
            <Progress value={progress} className="h-1" />
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-8">
        <div
          className="w-full max-w-2xl dema-fade-in"
          key={onboardingStep}
        >
          {renderStage()}
        </div>
      </div>

      {/* Stage dots — hidden on first and last stage */}
      {onboardingStep > 0 && onboardingStep < TOTAL_STAGES && (
        <div className="pb-6 flex justify-center">
          <div className="flex items-center gap-1.5">
            {Array.from({ length: TOTAL_STAGES + 1 }, (_, i) => (
              <div
                key={i}
                className={cn(
                  "w-1.5 h-1.5 rounded-full transition-all",
                  i === onboardingStep
                    ? "bg-trust w-3"
                    : i < onboardingStep
                      ? "bg-trust/40"
                      : "bg-muted-foreground/20"
                )}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
