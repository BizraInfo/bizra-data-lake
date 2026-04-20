"use client";

/**
 * BIZRA Landing Page — consumer-facing entry with architecture story
 * ====================================================================
 *
 * Ported from Z.ai workspace shell (2026-04). No retired-canon labels
 * detected in source; ported verbatim with canon-comment header.
 *
 * Sections: Hero (animated Seed of Life) → Covenant → Architecture
 * (Three Pillars) → Threshold CTA → Footer.
 *
 * Canon:
 *   - "One face. One law. One sovereign jurisdiction." (North Star V2)
 *   - "The Seed of Lawful Systems" — this doc's tagline; candidate for
 *     CANON-TERMS §07 promotion.
 *   - Arabic quote: "كلما ازددت علماً ازددت يقيناً بجهلي" — Al-Shāfiʿī
 *     epistemology (per memory `project_bizra_epistemology.md`).
 */

import { useRef, useState, useEffect } from "react";
import { motion, useInView, useScroll, useTransform } from "framer-motion";
import { BrandLogo, BrandLogoCompact } from "@/components/bizra/brand-logo";
import { Shield, Hash, Lock } from "lucide-react";

function Section({
  children,
  className = "",
  id,
  style,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
  style?: React.CSSProperties;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <motion.section
      ref={ref}
      id={id}
      className={className}
      style={style}
      initial={{ opacity: 0, y: 24 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.section>
  );
}

function LandingNav({
  onReturn,
  isReturn,
}: {
  onReturn?: () => void;
  isReturn?: boolean;
}) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 w-full z-50 px-6 md:px-10 py-5 flex justify-between items-center transition-all duration-500 ${
        scrolled ? "bg-navy-900/70 backdrop-blur-md border-b border-white/[0.04]" : ""
      }`}
    >
      <div className="flex items-center gap-2.5">
        <BrandLogoCompact size={16} />
        <span className="text-[11px] uppercase tracking-[0.25em] text-gold-500/70 font-sans">
          BIZRA
        </span>
      </div>
      <div className="flex items-center gap-5">
        {isReturn && (
          <button
            onClick={onReturn}
            className="text-[10px] uppercase tracking-[0.2em] text-white/40 font-sans hover:text-gold-500 transition-colors duration-300"
          >
            Return
          </button>
        )}
      </div>
    </nav>
  );
}

function HeroSection() {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.15], [1, 0]);
  const y = useTransform(scrollYProgress, [0, 0.15], [0, -40]);

  return (
    <section className="relative h-screen w-full flex flex-col items-center justify-center overflow-hidden">
      <motion.div
        className="relative w-56 h-56 sm:w-72 sm:h-72 md:w-96 md:h-96 mb-10 md:mb-14"
        style={{ opacity, y }}
      >
        <BrandLogo
          size={384}
          className="w-full h-full"
          animate={true}
          showConstruction={true}
        />
      </motion.div>

      <motion.div
        className="text-center overflow-hidden"
        style={{ opacity, y }}
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2.2, duration: 1, ease: [0.16, 1, 0.3, 1] }}
      >
        <h1 className="text-5xl sm:text-6xl md:text-8xl font-serif tracking-[0.12em] gold-text-gradient">
          BIZRA
        </h1>
      </motion.div>

      <motion.div
        className="mt-5 font-arabic text-gold-500/40 text-xl md:text-2xl"
        style={{ opacity, y }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.8, duration: 1 }}
      >
        البذرة
      </motion.div>

      <motion.p
        className="mt-6 text-white/30 text-sm md:text-base max-w-md text-center leading-relaxed"
        style={{ opacity }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 3.2, duration: 1 }}
      >
        The Seed of Lawful Systems
      </motion.p>

      <motion.div
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 3.8, duration: 1 }}
      >
        <span className="text-[8px] uppercase tracking-[0.35em] text-white/20 font-sans">
          Scroll
        </span>
        <motion.div
          className="w-px h-6 bg-gradient-to-b from-gold-500/30 to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2.5, ease: "easeInOut" }}
        />
      </motion.div>
    </section>
  );
}

function CovenantSection() {
  return (
    <Section
      className="py-20 md:py-32 px-6 md:px-16 w-full border-t border-white/[0.03]"
      style={{ backgroundColor: "#040910" }}
    >
      <div className="max-w-2xl mx-auto text-center space-y-6 md:space-y-8">
        <div className="w-10 h-px bg-gradient-to-r from-transparent via-gold-500/30 to-transparent mx-auto" />
        <h2 className="text-xl sm:text-2xl md:text-3xl font-serif text-white/90 leading-snug">
          In a world built on extraction,{" "}
          <br className="hidden sm:block" />
          <span className="text-gold-500">BIZRA</span> is a covenant.
        </h2>
        <p className="text-sm md:text-[15px] text-white/30 leading-relaxed max-w-lg mx-auto">
          Every action governed by constitutional law. Every receipt sealed with
          cryptographic truth. No extraction. No intermediation. Your intent,
          verified and immutable.
        </p>
        <p className="font-arabic text-gold-500/25 text-lg">
          كلما ازددت علماً ازددت يقيناً بجهلي
        </p>
        <div className="w-10 h-px bg-gradient-to-r from-transparent via-gold-500/30 to-transparent mx-auto" />
      </div>
    </Section>
  );
}

function ArchitectureSection() {
  const pillars = [
    {
      icon: Shield,
      title: "Constitutional Law",
      subtitle: "Five invariants",
      description:
        "Every intent passes through five constitutional gates — verified by reasoning, not bureaucracy. No action proceeds without compliance.",
      accent: "gold",
    },
    {
      icon: Hash,
      title: "Cryptographic Truth",
      subtitle: "BLAKE3 bound",
      description:
        "Every completed action receives a cryptographic receipt. Immutable, verifiable, chained to your trust history. Proof, not promise.",
      accent: "emerald",
    },
    {
      icon: Lock,
      title: "Zero Extraction",
      subtitle: "Your sovereignty",
      description:
        "Your data. Your intent. Your sovereignty. The system works for you — not on you. What you build here stays yours.",
      accent: "white",
    },
  ];

  return (
    <Section className="py-20 md:py-32 px-6 md:px-16 w-full">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14 md:mb-20">
          <span className="text-[9px] font-mono uppercase tracking-[0.4em] text-white/20 block mb-4">
            Architecture
          </span>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-serif text-white">
            How it works
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 md:gap-6">
          {pillars.map((pillar, i) => (
            <motion.div
              key={pillar.title}
              className="group relative p-6 md:p-8 rounded-xl border border-white/[0.04] bg-white/[0.015] hover:bg-white/[0.03] hover:border-white/[0.08] transition-all duration-500"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{
                delay: i * 0.12,
                duration: 0.6,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              <div
                className={`absolute top-0 left-6 right-6 h-px transition-all duration-500 ${
                  pillar.accent === "gold"
                    ? "bg-gold-500/20 group-hover:bg-gold-500/50"
                    : pillar.accent === "emerald"
                      ? "bg-bizra-emerald/20 group-hover:bg-bizra-emerald/50"
                      : "bg-white/10 group-hover:bg-white/30"
                }`}
              />

              <div
                className={`w-10 h-10 rounded-lg border flex items-center justify-center mb-5 transition-all duration-500 ${
                  pillar.accent === "gold"
                    ? "border-gold-500/20 bg-gold-500/[0.04] group-hover:border-gold-500/40 group-hover:bg-gold-500/[0.08]"
                    : pillar.accent === "emerald"
                      ? "border-bizra-emerald/20 bg-bizra-emerald/[0.04] group-hover:border-bizra-emerald/40 group-hover:bg-bizra-emerald/[0.08]"
                      : "border-white/10 bg-white/[0.02] group-hover:border-white/20 group-hover:bg-white/[0.04]"
                }`}
              >
                <pillar.icon
                  className={`w-4 h-4 transition-colors duration-500 ${
                    pillar.accent === "gold"
                      ? "text-gold-500/60 group-hover:text-gold-500"
                      : pillar.accent === "emerald"
                        ? "text-bizra-emerald/60 group-hover:text-bizra-emerald"
                        : "text-white/30 group-hover:text-white/60"
                  }`}
                />
              </div>

              <span className="text-[9px] font-mono uppercase tracking-[0.3em] text-white/20 block mb-2">
                {pillar.subtitle}
              </span>

              <h3 className="text-lg md:text-xl font-serif text-white/90 mb-3 group-hover:text-white transition-colors">
                {pillar.title}
              </h3>

              <p className="text-sm text-white/25 leading-relaxed group-hover:text-white/40 transition-colors duration-500">
                {pillar.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </Section>
  );
}

function ThresholdSection({ onEnter }: { onEnter: () => void }) {
  return (
    <Section
      className="py-24 md:py-36 px-6 flex flex-col items-center justify-center relative border-t border-white/[0.03]"
      style={{ backgroundColor: "#040910" }}
    >
      <div className="text-center max-w-xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="mx-auto mb-8 flex justify-center opacity-40">
            <BrandLogoCompact size={32} />
          </div>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-serif text-white mb-4">
            Ready?
          </h2>
          <p className="text-sm text-white/25 leading-relaxed mb-10 max-w-md mx-auto">
            Enter the sovereign operator system. Your intent will be governed by
            law, verified by truth, and sealed to your chain.
          </p>
          <button
            onClick={onEnter}
            className="group relative inline-flex items-center gap-3 px-8 py-3.5 bg-transparent border border-gold-500/50 text-gold-500 text-[10px] font-medium uppercase tracking-[0.3em] rounded-sm hover:bg-gold-500/[0.06] hover:border-gold-500/80 transition-all duration-500"
          >
            <span>Enter the System</span>
            <motion.span
              className="inline-block"
              animate={{ x: [0, 4, 0] }}
              transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
            >
              &#8594;
            </motion.span>
          </button>
        </motion.div>
      </div>
    </Section>
  );
}

function LandingFooter() {
  return (
    <footer className="py-8 px-6 border-t border-white/[0.03]">
      <div className="max-w-5xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-2.5">
          <BrandLogoCompact size={14} />
          <span className="text-[10px] text-white/15 font-sans">BIZRA</span>
        </div>
        <p className="text-[9px] text-white/10 font-mono tracking-wider">
          The Seed of Lawful Systems
        </p>
        <p className="font-arabic text-gold-500/15 text-sm">البذرة</p>
      </div>
    </footer>
  );
}

export function LandingPage({
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
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: "#050B14", color: "#F8F6F1" }}
    >
      <div className="fixed inset-0 brand-grid-bg pointer-events-none z-0" />

      <LandingNav onReturn={onReturn} isReturn={isReturn} />

      <main className="relative z-10 flex-1">
        <HeroSection />
        <CovenantSection />
        <ArchitectureSection />
        {!isReturn && <ThresholdSection onEnter={onEnter} />}
      </main>

      <div className="mt-auto relative z-10">
        <LandingFooter />
      </div>
    </div>
  );
}
