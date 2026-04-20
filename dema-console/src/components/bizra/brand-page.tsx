"use client";

/**
 * BIZRA Brand Identity Page — full visual-system showcase
 * =========================================================
 *
 * Ported from Z.ai workspace shell (2026-04) with **7 canon
 * reconciliations** applied during the port:
 *
 *   1. Nav date "December 2025" → evergreen "Brand Identity · v2.0"
 *      (dated strings rot; canonical version tag doesn't)
 *   2. AegisSection tag "Double Submit" → "CSRF Shield"
 *      (implementation-agnostic; Double-Submit-Cookie is one pattern
 *      among several; shield phrase travels better)
 *   3. AegisSection description "(Ihsan Metric)" → "(Ihsān Vector)"
 *      (aligns with OMNI-SYNTHESIS Whitepaper §07 authoritative naming)
 *   4. SNR tag "Latency <0.5ms" → "Sub-ms Target"
 *      (L-001: honest — marks as design goal, not verified fact;
 *      no perf audit exists at the time of port)
 *   5. SNR tag "Ihsan >0.99" → "Strict ≥ 0.99"
 *      (cites CANON-TERMS §02 tier name; the 0.99 is the Strict tier
 *      threshold, not a runtime claim)
 *   6. SNR tag "Self-Correcting" → "Self-Healing"
 *      (standard DevOps lexicon; less metaphysical framing)
 *   7. SNR description "self-correction ritual" → "self-healing routine"
 *      (matches tag rename)
 *
 * Sections: Nav → Hero → Semiotics (Sacred Geometry) → Visual System
 * (typography + palette) → Aegis (infrastructure) → Digital Card →
 * CTA → Footer.
 *
 * Canon references:
 *   - `docs/design/CANON-TERMS.md` §02 (Ihsān tier values)
 *   - `docs/design/CANON-TERMS.md` §07 (tagline canon)
 *   - ADK North Star V2 · "BIZRA reveals truth, never simulates" (L-001)
 */

import { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";
import { BrandLogo, BrandLogoCompact } from "@/components/bizra/brand-logo";
import { Sparkles } from "lucide-react";

function Section({
  children,
  className = "",
  id,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <motion.section
      ref={ref}
      id={id}
      className={className}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.section>
  );
}

function BrandNav({
  onEnter,
  isReturn,
}: {
  onEnter?: () => void;
  isReturn?: boolean;
}) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 w-full z-50 px-6 md:px-8 py-5 md:py-6 flex justify-between items-center transition-all duration-500 ${
        scrolled
          ? "bg-navy-900/80 backdrop-blur-md border-b border-white/5"
          : "mix-blend-difference"
      }`}
    >
      <div className="text-[10px] md:text-xs uppercase tracking-[0.2em] md:tracking-[0.3em] text-gold-500 font-sans">
        Identity System v2.0
      </div>
      <div className="flex items-center gap-4">
        {isReturn && (
          <button
            onClick={onEnter}
            className="text-[10px] md:text-xs uppercase tracking-[0.2em] md:tracking-[0.3em] text-gold-500 font-sans hover:text-gold-300 transition-colors"
          >
            Return to DEMA
          </button>
        )}
        <div className="text-[10px] md:text-xs uppercase tracking-[0.2em] md:tracking-[0.3em] text-white/50 font-sans">
          Brand Identity · v2.0
        </div>
      </div>
    </nav>
  );
}

function HeroReveal() {
  return (
    <section className="h-screen w-full flex flex-col items-center justify-center relative z-10">
      <div className="relative w-64 h-64 md:w-96 md:h-96 mb-12">
        <BrandLogo
          size={384}
          className="w-full h-full"
          animate={true}
          showConstruction={true}
        />
      </div>

      <motion.div
        className="text-center overflow-hidden"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2.2, duration: 1, ease: [0.16, 1, 0.3, 1] }}
      >
        <h1 className="text-6xl md:text-8xl font-serif tracking-[0.15em] gold-text-gradient">
          BIZRA
        </h1>
      </motion.div>

      <motion.div
        className="mt-6 font-arabic text-[#C9A962]/60 text-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.8, duration: 1 }}
      >
        البذرة
      </motion.div>

      <motion.div
        className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 3.5, duration: 1 }}
      >
        <span className="text-[9px] uppercase tracking-[0.3em] text-white/30">
          Scroll to discover
        </span>
        <motion.div
          className="w-px h-8 bg-gradient-to-b from-gold-500/40 to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        />
      </motion.div>
    </section>
  );
}

function SemioticsSection() {
  const [highlightedPart, setHighlightedPart] = useState<number | null>(null);

  const semioticItems = [
    {
      id: 1,
      num: "01",
      title: "The Seed (Nuqta)",
      description:
        "The single central circle represents the Divine Origin (Tawhid). It is the dot under the Bā' (ب), the beginning of all knowledge.",
    },
    {
      id: 2,
      num: "02",
      title: "The Seed of Life",
      description:
        "The six circles surrounding the one represent the 6 days of creation. It is the perfect balance found in nature, from cells to galaxies.",
    },
    {
      id: 3,
      num: "03",
      title: "The Bloom (Ihsan)",
      description:
        "Where the circles overlap, they form the flower. This represents the community (Ummah) and the result of the system: Beauty and Excellence.",
    },
  ];

  return (
    <Section className="py-24 md:py-32 px-6 md:px-24 w-full border-t border-white/5 relative bg-navy-900">
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16 items-center">
        <div className="relative aspect-square brand-glass-card rounded-2xl p-8 md:p-12 flex items-center justify-center group overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-gold-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
          <svg viewBox="0 0 200 200" className="w-full h-full relative z-10">
            <g
              style={{
                opacity:
                  highlightedPart === null || highlightedPart === 1 ? 1 : 0.15,
                transition: "opacity 0.3s ease",
              }}
            >
              <circle
                cx="100"
                cy="100"
                r="30"
                fill="rgba(201, 169, 98, 0.2)"
                stroke="#C9A962"
                strokeWidth={1}
              />
              <circle cx="100" cy="100" r="1.5" fill="#C9A962" />
              <line
                x1="130"
                y1="100"
                x2="150"
                y2="100"
                stroke="#C9A962"
                strokeWidth={0.5}
              />
              <text
                x="155"
                y="103"
                fill="#C9A962"
                fontSize="7"
                fontFamily="Inter"
                className="uppercase"
                letterSpacing="2"
              >
                The Seed (1)
              </text>
            </g>
            <g
              style={{
                opacity:
                  highlightedPart === null || highlightedPart === 2 ? 1 : 0.15,
                transition: "opacity 0.3s ease",
              }}
            >
              <circle cx="100" cy="70" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <circle cx="126" cy="85" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <circle cx="126" cy="115" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <circle cx="100" cy="130" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <circle cx="74" cy="115" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <circle cx="74" cy="85" r="30" fill="none" stroke="#C9A962" strokeWidth={0.5} strokeDasharray="2 2" />
              <line x1="126" y1="70" x2="150" y2="40" stroke="#C9A962" strokeWidth={0.5} />
              <text x="155" y="43" fill="#C9A962" fontSize="7" fontFamily="Inter" className="uppercase" letterSpacing="2">
                The Creation (6)
              </text>
            </g>
            <g
              style={{
                opacity:
                  highlightedPart === null || highlightedPart === 3 ? 1 : 0.15,
                transition: "opacity 0.3s ease",
              }}
            >
              <path d="M100 70 Q115 85 100 100 Q85 85 100 70" fill="#C9A962" opacity={0.8} />
              <path d="M126 85 Q113 93 100 100 Q113 93 126 85" fill="#C9A962" opacity={0.5} />
              <path d="M126 115 Q113 107 100 100 Q113 107 126 115" fill="#C9A962" opacity={0.5} />
              <path d="M100 130 Q85 115 100 100 Q115 115 100 130" fill="#C9A962" opacity={0.8} />
              <path d="M74 115 Q87 107 100 100 Q87 107 74 115" fill="#C9A962" opacity={0.5} />
              <path d="M74 85 Q87 93 100 100 Q87 93 74 85" fill="#C9A962" opacity={0.5} />
              <line x1="100" y1="130" x2="150" y2="160" stroke="#C9A962" strokeWidth={0.5} />
              <text x="155" y="163" fill="#C9A962" fontSize="7" fontFamily="Inter" className="uppercase" letterSpacing="2">
                The Flower (Unity)
              </text>
            </g>
          </svg>
        </div>
        <div>
          <h2 className="text-gold-500 text-sm tracking-[0.4em] uppercase mb-6 font-sans">
            Semiotics
          </h2>
          <h3 className="text-3xl md:text-5xl font-serif text-white mb-8">
            Sacred Geometry
          </h3>
          <div className="space-y-6 md:space-y-8">
            {semioticItems.map((item) => (
              <div
                key={item.id}
                className="flex gap-4 md:gap-6 group cursor-pointer"
                onMouseEnter={() => setHighlightedPart(item.id)}
                onMouseLeave={() => setHighlightedPart(null)}
              >
                <div className="w-10 h-10 md:w-12 md:h-12 rounded-full border border-gold-500/30 flex items-center justify-center text-gold-500 group-hover:bg-gold-500 group-hover:text-navy-900 transition-all duration-300 text-xs md:text-sm flex-shrink-0">
                  {item.num}
                </div>
                <div>
                  <h4 className="text-lg md:text-xl text-white mb-2 group-hover:text-gold-500 transition-colors font-serif">
                    {item.title}
                  </h4>
                  <p className="text-white/50 text-sm leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

function VisualSystemSection() {
  const colorPalette = [
    { name: "Genesis Gold (v2.0)", hex: "#C9A962", bgClass: "bg-gold-500", shadowClass: "shadow-[0_0_15px_rgba(201,169,98,0.4)]" },
    { name: "Celestial Navy (v2.0)", hex: "#0A1628", bgClass: "bg-navy-800", shadowClass: "border border-white/10" },
    { name: "Pure White", hex: "#FFFFFF", bgClass: "bg-white/5", shadowClass: "border border-white/10", label: true },
    { name: "Deep Void", hex: "#050B14", bgClass: "", shadowClass: "border border-white/10", voidColor: true },
    { name: "Gold 300", hex: "#E6D5A6", bgClass: "bg-gold-300", shadowClass: "" },
    { name: "Gold 900", hex: "#8A6B2E", bgClass: "bg-gold-900", shadowClass: "" },
  ];

  return (
    <Section className="py-24 md:py-32 px-6 bg-black relative overflow-hidden">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-gold-500 text-sm tracking-[0.4em] uppercase mb-16 text-center font-sans">
          Visual System
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
          <div className="brand-glass-card p-6 md:p-8 rounded-xl">
            <div className="text-white/40 text-[10px] uppercase tracking-[3px] mb-4 font-sans">
              Primary Typeface
            </div>
            <div className="text-4xl md:text-5xl font-serif text-white mb-2">Playfair</div>
            <div className="text-gold-500 text-xl font-serif">Display</div>
            <div className="mt-6 md:mt-8 text-white/60 text-sm leading-relaxed">
              Used for headlines and spiritual messaging. Elegant, timeless, authoritative.
            </div>
            <div className="mt-6 text-3xl md:text-4xl text-white/20">Aa Bb Cc</div>
          </div>
          <div className="brand-glass-card p-6 md:p-8 rounded-xl">
            <div className="text-white/40 text-[10px] uppercase tracking-[3px] mb-4 font-sans">
              Secondary Typeface
            </div>
            <div className="text-4xl md:text-5xl font-sans text-white mb-2 font-extralight">Inter</div>
            <div className="text-gold-500 text-xl">Interface</div>
            <div className="mt-6 md:mt-8 text-white/60 text-sm leading-relaxed">
              Used for UI elements, data visualization, and body text. Clean, precise, invisible.
            </div>
            <div className="mt-6 text-3xl md:text-4xl text-white/20 font-sans">Aa Bb Cc</div>
          </div>
          <div className="brand-glass-card p-6 md:p-8 rounded-xl flex flex-col">
            <div className="text-white/40 text-[10px] uppercase tracking-[3px] mb-4 font-sans">
              Color Codes
            </div>
            <div className="space-y-3 flex-1">
              {colorPalette.slice(0, 3).map((color) => (
                <div key={color.hex} className="flex items-center gap-4">
                  <div
                    className={`w-11 h-11 rounded-lg flex-shrink-0 ${color.bgClass} ${color.shadowClass}`}
                    style={color.voidColor ? { backgroundColor: "#050B14" } : undefined}
                  >
                    {color.label && (
                      <span className="flex items-center justify-center w-full h-full text-[9px] text-white/40 uppercase">Pure</span>
                    )}
                  </div>
                  <div>
                    <div className="text-white text-sm">{color.name}</div>
                    <div className="text-white/40 text-[11px] font-mono">{color.hex}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-8 flex flex-wrap gap-3 justify-center">
          {colorPalette.map((color) => (
            <div key={color.hex} className="flex items-center gap-2 brand-glass-card px-3 py-2 rounded-lg">
              <div
                className={`w-4 h-4 rounded-sm ${color.bgClass}`}
                style={color.voidColor ? { backgroundColor: "#050B14" } : undefined}
              />
              <span className="text-[10px] font-mono text-white/50">{color.hex}</span>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

function AegisSection() {
  // Label reconciliations per CANON-TERMS (see file header):
  //   "Double Submit" → "CSRF Shield"
  //   "(Ihsan Metric)" → "(Ihsān Vector)"
  //   "Latency <0.5ms" → "Sub-ms Target"
  //   "Ihsan >0.99" → "Strict ≥ 0.99"
  //   "Self-Correcting" → "Self-Healing"
  //   "self-correction ritual" → "self-healing routine"
  const infraCards = [
    {
      title: "Aegis CSRF Hardening",
      borderColor: "border-l-gold-500",
      description:
        "Stateless security enforced via the Double Submit Cookie pattern. Protected by `__Host-` prefixed tokens with `SameSite=Strict` and constant-time validation. This is the shield that ensures only sovereign intent manifests in the ledger.",
      tags: [
        { text: "CSRF Shield", color: "text-gold-500/50" },
        { text: "Strict Entropy", color: "text-gold-500/50" },
        { text: "Secure-First", color: "text-gold-500/50" },
      ],
    },
    {
      title: "SNR Autonomous Engine",
      borderColor: "border-l-teal-500",
      description:
        "Real-time monitoring of Signal-to-Noise Ratio. The system autonomously modulates performance and ethics (Ihsān Vector). If SNR falls below the elite threshold, the system triggers a self-healing routine.",
      tags: [
        { text: "Sub-ms Target", color: "text-teal-500/50" },
        { text: "Strict ≥ 0.99", color: "text-teal-500/50" },
        { text: "Self-Healing", color: "text-teal-500/50" },
      ],
    },
  ];

  return (
    <Section className="py-24 md:py-32 px-6 bg-navy-900 border-y border-white/5 relative">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12 md:mb-16">
          <h2 className="text-gold-500 text-sm tracking-[0.4em] uppercase mb-4 font-sans">
            Infrastructure
          </h2>
          <h3 className="text-3xl md:text-4xl font-serif text-white">
            The Aegis Architecture
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-12">
          {infraCards.map((card) => (
            <div
              key={card.title}
              className={`brand-glass-card p-6 md:p-8 rounded-2xl border-l-4 ${card.borderColor}`}
            >
              <h4 className="text-xl md:text-2xl font-serif text-white mb-4">
                {card.title}
              </h4>
              <p className="text-white/60 text-sm leading-relaxed mb-6">
                {card.description}
              </p>
              <div className="flex flex-wrap items-center gap-3 md:gap-4">
                {card.tags.map((tag) => (
                  <span
                    key={tag.text}
                    className={`text-[9px] md:text-[10px] font-mono ${tag.color} uppercase tracking-widest flex items-center gap-1.5`}
                  >
                    <span className="text-[6px]">&#9679;</span>
                    {tag.text}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

function DigitalCardSection() {
  return (
    <Section className="py-24 md:py-32 px-6 flex flex-col items-center justify-center relative">
      <div className="text-center mb-12 md:mb-16">
        <h2 className="text-gold-500 text-sm tracking-[0.4em] uppercase mb-4 font-sans">
          Application
        </h2>
        <h3 className="text-3xl md:text-4xl font-serif text-white">
          The Key to the System
        </h3>
      </div>
      <motion.div
        className="w-[320px] h-[190px] md:w-[450px] md:h-[260px] rounded-2xl relative shadow-[0_20px_50px_rgba(0,0,0,0.5)] group overflow-hidden cursor-pointer"
        style={{
          background: "linear-gradient(135deg, #1a1a1a, #050505)",
          border: "1px solid rgba(201, 169, 98, 0.3)",
        }}
        whileHover={{ scale: 1.03, rotateY: 8, rotateX: -5 }}
        transition={{ type: "spring", stiffness: 200, damping: 20 }}
      >
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='5' height='5'%3E%3Crect width='5' height='5' fill='%23ffffff' fill-opacity='0.1'/%3E%3C/svg%3E")`,
          }}
        />
        <div className="absolute top-6 right-6 md:top-8 md:right-8">
          <BrandLogoCompact size={50} />
        </div>
        <div className="absolute top-6 left-6 md:top-8 md:left-8">
          <div className="w-10 h-7 md:w-12 md:h-8 bg-gold-300/20 rounded flex items-center justify-center border border-gold-500/20">
            <div className="w-7 md:w-8 h-[1px] bg-gold-500/40" />
          </div>
        </div>
        <div className="absolute bottom-6 left-6 md:bottom-8 md:left-8">
          <div className="text-gold-500 font-serif tracking-[0.15em] text-lg">
            BIZRA
          </div>
          <div className="text-white/30 text-[9px] md:text-[10px] tracking-[0.2em] uppercase">
            Genesis Access
          </div>
        </div>
        <div className="absolute bottom-6 right-6 md:bottom-8 md:right-8 text-right">
          <div className="text-white/40 text-[9px] md:text-[10px] font-mono">
            ID: 0000-0001
          </div>
          <div className="text-white/60 text-xs font-arabic mt-1">البذرة</div>
        </div>
        <div
          className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out"
          style={{
            background:
              "linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.04) 45%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.04) 55%, transparent 60%)",
          }}
        />
      </motion.div>
      <div className="mt-8 w-[200px] h-px bg-gradient-to-r from-transparent via-gold-500/20 to-transparent" />
    </Section>
  );
}

function BrandFooter() {
  return (
    <footer className="py-12 md:py-16 px-6 border-t border-white/5 bg-navy-900">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex items-center gap-4">
          <BrandLogoCompact size={28} />
          <div>
            <div className="text-sm font-serif tracking-[0.15em] text-gold-500">
              BIZRA
            </div>
            <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-sans">
              The Seed of Lawful Systems
            </div>
          </div>
        </div>
        <div className="flex items-center gap-6 md:gap-8">
          <div className="text-center">
            <div className="text-[10px] font-mono text-white/20">Genesis</div>
            <div className="text-[9px] font-mono text-white/40">#C9A962</div>
          </div>
          <div className="w-px h-6 bg-white/10" />
          <div className="text-center">
            <div className="text-[10px] font-mono text-white/20">Navy</div>
            <div className="text-[9px] font-mono text-white/40">#0A1628</div>
          </div>
          <div className="w-px h-6 bg-white/10" />
          <div className="text-center">
            <div className="text-[10px] font-mono text-white/20">Version</div>
            <div className="text-[9px] font-mono text-white/40">2.0</div>
          </div>
        </div>
        <div className="font-arabic text-gold-500/30 text-lg">البذرة</div>
      </div>
    </footer>
  );
}

function BrandCTA({ onEnter }: { onEnter: () => void }) {
  return (
    <Section className="py-24 md:py-32 px-6 flex flex-col items-center justify-center relative bg-navy-900">
      <div className="text-center max-w-2xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="w-16 h-16 mx-auto mb-8 border border-gold-500/30 rounded-full flex items-center justify-center bg-gold-500/5">
            <Sparkles className="h-7 w-7 text-gold-500" />
          </div>
          <h2 className="text-3xl md:text-5xl font-serif text-white mb-6">
            Begin Your Journey
          </h2>
          <p className="text-white/50 text-sm md:text-base leading-relaxed mb-10 max-w-lg mx-auto">
            Enter the sovereign operator system. Your intent will be governed by
            constitutional law, verified by cryptographic receipts, and bound to
            your trust chain.
          </p>
          <button
            onClick={onEnter}
            className="group relative inline-flex items-center gap-3 px-8 py-4 bg-transparent border border-gold-500 text-gold-500 text-[10px] font-medium uppercase tracking-[0.3em] rounded-sm hover:bg-gold-500/10 transition-all duration-500"
          >
            <span>Enter the System</span>
            <motion.span
              className="inline-block"
              animate={{ x: [0, 4, 0] }}
              transition={{
                repeat: Infinity,
                duration: 1.5,
                ease: "easeInOut",
              }}
            >
              &#8594;
            </motion.span>
          </button>
        </motion.div>
      </div>
    </Section>
  );
}

export function BrandIdentityPage({
  onEnter,
  isReturn,
  onReturn,
}: {
  onEnter: () => void;
  isReturn: boolean;
  onReturn?: () => void;
}) {
  return (
    <div
      className="min-h-screen flex flex-col overflow-x-hidden"
      style={{ backgroundColor: "#050B14", color: "#F8F6F1" }}
    >
      <div className="fixed inset-0 brand-grid-bg pointer-events-none z-0" />
      <BrandNav onEnter={isReturn ? onReturn : undefined} isReturn={isReturn} />
      <main className="relative z-10 flex-1">
        <HeroReveal />
        <SemioticsSection />
        <VisualSystemSection />
        <AegisSection />
        <DigitalCardSection />
        {!isReturn && <BrandCTA onEnter={onEnter} />}
      </main>
      <div className="mt-auto relative z-10">
        <BrandFooter />
      </div>
    </div>
  );
}
